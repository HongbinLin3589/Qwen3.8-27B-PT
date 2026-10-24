"""Offline transport checks; no real endpoint or credentials are contacted."""

import io
import json
import os
import unittest
from unittest.mock import patch
from urllib.error import HTTPError, URLError

from qwenpt.eval.api_backend import APIError, ChatClient, endpoint_metadata, resolve_endpoint


CONFIG = {"base_url": "http://localhost:8000/v1", "model": "local-candidate",
          "api_key_env": "BENCHMARK_TEST_KEY", "retries": 0}
SECRET = "private-test-value-do-not-log"


def response(text="Answer: B", **changes):
    value = {"model": "local-candidate-revision", "usage": {"completion_tokens": 4},
             "choices": [{"message": {"role": "assistant", "content": text,
                                      "reasoning_content": "private reasoning"},
                          "finish_reason": "stop"}]}
    value.update(changes)
    return io.BytesIO(json.dumps(value).encode())


class TestEndpoint(unittest.TestCase):
    def test_environment_override_and_safe_metadata(self):
        config = {**CONFIG, "base_url_env": "BENCHMARK_TEST_URL", "model_env": "BENCHMARK_TEST_MODEL"}
        with patch.dict(os.environ, {"BENCHMARK_TEST_KEY": SECRET, "BENCHMARK_TEST_URL": "https://example.test/v1/",
                                     "BENCHMARK_TEST_MODEL": "new-model"}):
            endpoint = resolve_endpoint(config)
            self.assertEqual(endpoint["api_key"], SECRET)
            metadata = endpoint_metadata(config)
        self.assertEqual(metadata["base_url"], "https://example.test/v1")
        self.assertEqual(metadata["model"], "new-model")
        self.assertNotIn(SECRET, json.dumps(metadata))
        self.assertNotIn("api_key", metadata)

    def test_missing_key_fails_before_request(self):
        with patch.dict(os.environ, {}, clear=True), patch("qwenpt.eval.api_backend._open_request") as request:
            with self.assertRaisesRegex(ValueError, "missing or empty"):
                ChatClient(CONFIG).complete([{"role": "user", "content": "hello"}], {})
            request.assert_not_called()

    def test_local_optional_key(self):
        with patch.dict(os.environ, {}, clear=True):
            endpoint = resolve_endpoint({**CONFIG, "api_key_required": False})
        self.assertEqual(endpoint["api_key"], "")

    def test_secret_in_url_or_literal_config_is_rejected_without_echo(self):
        for url in [f"https://user:{SECRET}@example.test/v1", f"https://example.test/v1?api_key={SECRET}",
                    f"https://example.test/v1#{SECRET}", "file:///tmp/model", "http://x:bad/v1"]:
            with self.subTest(url=url), self.assertRaises(ValueError) as error:
                endpoint_metadata({**CONFIG, "base_url": url})
            self.assertNotIn(SECRET, str(error.exception))
        with self.assertRaisesRegex(ValueError, "api_key_env"):
            endpoint_metadata({**CONFIG, "api_key": SECRET})


class TestChatClient(unittest.TestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {"BENCHMARK_TEST_KEY": SECRET})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_messages_generation_and_reasoning_are_preserved(self):
        image = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}
        messages = [{"role": "user", "content": [{"type": "text", "text": "What is shown?"}, image]}]
        with patch("qwenpt.eval.api_backend._open_request", return_value=response()) as request:
            result = ChatClient(CONFIG).complete(messages, {"max_completion_tokens": 100, "reasoning_effort": "low"})
        req = request.call_args.args[0]
        body = json.loads(req.data)
        self.assertEqual(req.full_url, "http://localhost:8000/v1/chat/completions")
        self.assertEqual(req.get_header("Authorization"), "Bearer " + SECRET)
        self.assertEqual(body["messages"], messages)
        self.assertEqual(body["n"], 1)
        self.assertFalse(body["stream"])
        self.assertEqual(result["text"], "Answer: B")
        self.assertEqual(result["reasoning"], "private reasoning")
        self.assertEqual(result["usage"], {"completion_tokens": 4})

    def test_retry_only_transient_http_statuses_and_keep_request_identical(self):
        for status, retries in [(429, 1), (503, 1), (401, 0), (400, 0)]:
            error = HTTPError("http://x/" + SECRET, status, SECRET, {}, io.BytesIO(SECRET.encode()))
            with self.subTest(status=status), patch("qwenpt.eval.api_backend.time.sleep"), patch(
                "qwenpt.eval.api_backend._open_request", side_effect=[error, response()]
            ) as request:
                client = ChatClient({**CONFIG, "retries": 2})
                if retries:
                    self.assertEqual(client.complete([{"role": "user", "content": "q"}], {})["text"], "Answer: B")
                    self.assertEqual(request.call_args_list[0].args[0].data, request.call_args_list[1].args[0].data)
                else:
                    with self.assertRaises(APIError) as captured:
                        client.complete([{"role": "user", "content": "q"}], {})
                    self.assertNotIn(SECRET, str(captured.exception))
                self.assertEqual(request.call_count, retries + 1)

    def test_network_retries_are_bounded_and_do_not_leak(self):
        with patch("qwenpt.eval.api_backend.time.sleep"), patch(
            "qwenpt.eval.api_backend._open_request", side_effect=URLError(SECRET)
        ) as request:
            with self.assertRaises(APIError) as captured:
                ChatClient({**CONFIG, "retries": 2}).complete([{"role": "user", "content": "q"}], {})
        self.assertEqual(request.call_count, 3)
        self.assertNotIn(SECRET, str(captured.exception))

    def test_invalid_payload_and_unexpected_transport_error_do_not_leak(self):
        for source in [io.BytesIO(SECRET.encode()), RuntimeError(SECRET)]:
            kwargs = {"side_effect": source} if isinstance(source, Exception) else {"return_value": source}
            with patch("qwenpt.eval.api_backend._open_request", **kwargs), self.assertRaises(APIError) as captured:
                ChatClient(CONFIG).complete([{"role": "user", "content": "q"}], {})
            self.assertNotIn(SECRET, str(captured.exception))

    def test_invalid_response_fields_are_rejected(self):
        cases = [response(text=""), response(text=None), response(choices=[]), response(choices=[{
            "message": {"content": "done", "tool_calls": [{"function": SECRET}]}, "finish_reason": "tool_calls"}]),
            response(choices=[{"message": {"content": "done"}, "finish_reason": None}])]
        for source in cases:
            with patch("qwenpt.eval.api_backend._open_request", return_value=source), self.assertRaises(APIError):
                ChatClient(CONFIG).complete([{"role": "user", "content": "q"}], {})

    def test_parameters_cannot_change_model_or_enable_agent(self):
        for parameters in [{"model": "override"}, {"n": 2}, {"stream": True}, {"tools": []},
                           {"max_tokens": 10, "max_completion_tokens": 10}, {"temperature": float("nan")}]:
            with self.subTest(parameters=parameters), patch("qwenpt.eval.api_backend._open_request") as request:
                with self.assertRaises(ValueError):
                    ChatClient(CONFIG).complete([{"role": "user", "content": "q"}], parameters)
                request.assert_not_called()

    def test_redirect_does_not_forward_key_to_another_host(self):
        from qwenpt.eval.api_backend import _NoRedirect
        self.assertIsNone(_NoRedirect().redirect_request(None, None, 302, "", {}, "http://other.example"))


if __name__ == "__main__":
    unittest.main()
