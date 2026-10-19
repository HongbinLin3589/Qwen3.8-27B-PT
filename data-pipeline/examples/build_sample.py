"""Build a small sample (function tasks + tiny SWE repos + an eval set) that
exercises every drop path in the pipeline. Output: examples/sample/.

    python examples/build_sample.py
    python -m datapipe run --config examples/sample.yaml --input examples/sample/tasks.jsonl --out out/sample --overwrite
"""
import json
import os
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "sample")
IMAGE = "python:3.12-slim"
TEST_CMD = "python -m unittest discover -s tests -q"


def fn_task(tid, prompt, ref, tests, *, created="2026-03-01", source="synthetic-math", fn="solve", **meta):
    return {"id": tid, "source": source, "domain": "code", "prompt": prompt, "created_at": created,
            "verifier": {"type": "python_function", "function": fn,
                         "tests": [{"input": list(i), "output": o} for i, o in tests]},
            "reference": {"code": ref}, "meta": meta}


PHI = "def solve(n):\n    r, p = n, 2\n    while p * p <= n:\n        if n % p == 0:\n            while n % p == 0:\n                n //= p\n            r -= r // p\n        p += 1\n    if n > 1:\n        r -= r // n\n    return r\n"
PHI_TESTS = [((1,), 1), ((10,), 4), ((36,), 12), ((97,), 96), ((10**12,), 400000000000)]
PHI_PROMPT = "给定正整数 n（1 ≤ n ≤ 10^12），求 1 到 n 中与 n 互素的整数个数。"


def function_tasks():
    T = []
    T.append(fn_task("phi", PHI_PROMPT, PHI, PHI_TESTS, mock_p_correct=0.5))
    T.append(fn_task("phi_copy", PHI_PROMPT + " ", PHI, PHI_TESTS, source="other-dataset"))          # exact dup
    T.append(fn_task("phi_reworded", "给定正整数 n（1 ≤ n ≤ 10^12），请求出 1 到 n 中与 n 互素的整数的个数。",
                     PHI, PHI_TESTS, source="other-dataset"))                                          # near dup
    T.append(fn_task("sigma", "给定 n（1 ≤ n ≤ 10^12），求 n 的所有正因数之和。",
                     "def solve(n):\n    s, d = 0, 1\n    while d * d <= n:\n        if n % d == 0:\n            s += d\n            if d != n // d:\n                s += n // d\n        d += 1\n    return s\n",
                     [((1,), 1), ((12,), 28), ((97,), 98), ((10**12,), 2499694822171)], mock_p_correct=0.75))
    T.append(fn_task("fib_mod", "求第 n 个斐波那契数对 1000000007 取模的结果，F(0)=0，F(1)=1，n ≤ 10^18。",
                     "def solve(n):\n    M = 10**9 + 7\n    def f(k):\n        if k == 0:\n            return (0, 1)\n        a, b = f(k >> 1)\n        c = a * (2 * b - a) % M\n        d = (a * a + b * b) % M\n        return (d, (c + d) % M) if k & 1 else (c, d)\n    return f(n)[0]\n",
                     [((0,), 0), ((1,), 1), ((10,), 55), ((10**18,), 209783453)], mock_p_correct=0.25))
    T.append(fn_task("after_cutoff", "给定 n，求 n 的二进制表示中 1 的个数，n ≤ 10^18。",
                     "def solve(n):\n    return bin(n).count('1')\n", [((0,), 0), ((7,), 3)], created="2026-09-10"))
    T.append(fn_task("contaminated", "Given an integer n, return the number of trailing zeros in n factorial. Constraints: 0 <= n <= 10^18.",
                     "def solve(n):\n    c = 0\n    while n:\n        n //= 5\n        c += n\n    return c\n",
                     [((0,), 0), ((25,), 6), ((100,), 24)]))
    # weak tests: a wrong solution that returns 1 passes -> false positive
    T.append(fn_task("weak_tests", "给定 n（1 ≤ n ≤ 10^9），求 n 的不同质因子个数加一。",
                     "def solve(n):\n    c, p = 0, 2\n    while p * p <= n:\n        if n % p == 0:\n            c += 1\n            while n % p == 0:\n                n //= p\n        p += 1\n    return c + (n > 1) + 1\n",
                     [((1,), 1)], mock_p_correct=0.5,
                     mock_wrong_solutions=[{"code": "def solve(n):\n    return 1\n"}]))
    # over-specified tests: statement allows any order, tests require ascending -> false negative
    T.append(fn_task("strict_order", "给定 n（1 ≤ n ≤ 10^6），返回 n 的所有正因数组成的列表，顺序不限。",
                     "def solve(n):\n    return [d for d in range(1, n + 1) if n % d == 0]\n",
                     [((12,), [1, 2, 3, 4, 6, 12]), ((7,), [1, 7])], mock_p_correct=0.75,
                     mock_correct_solutions=[{"code": "def solve(n):\n    return [d for d in range(n, 0, -1) if n % d == 0]\n"}]))
    T.append(fn_task("bad_reference", "给定 n，求 1 到 n 的平方和，n ≤ 10^18。",
                     "def solve(n):\n    return n * (n + 1) // 2\n", [((3,), 14), ((10,), 385)]))
    T.append(fn_task("very_hard", "给定 n（n ≤ 10^18），求 n 以内的素数个数。",
                     "def solve(n):\n    return [0, 0, 1, 2, 2, 3, 3, 4, 4, 4, 4][n] if n <= 10 else 25\n",
                     [((10,), 4), ((100,), 25)], mock_p_correct=0.0))
    easy = [
        ("easy_sum", "给定 n，返回 1 到 n 的整数之和。", "def solve(n):\n    return n * (n + 1) // 2\n", [((3,), 6), ((100,), 5050)]),
        ("easy_max", "给定一个整数列表 xs，返回其中的最大值。", "def solve(xs):\n    return max(xs)\n", [(([3, 9, 2],), 9)]),
        ("easy_rev", "Reverse the given string s and return it.", "def solve(s):\n    return s[::-1]\n", [(("abc",), "cba")]),
        ("easy_vowel", "Count how many vowels (a, e, i, o, u) appear in the lowercase word w.", "def solve(w):\n    return sum(c in 'aeiou' for c in w)\n", [(("banana",), 3)]),
        ("easy_pal", "判断字符串 s 是否为回文，返回 True 或 False。", "def solve(s):\n    return s == s[::-1]\n", [(("aba",), True), (("ab",), False)]),
        ("easy_gcd", "Return the greatest common divisor of two positive integers a and b.", "def solve(a, b):\n    import math\n    return math.gcd(a, b)\n", [((12, 18), 6)]),
        ("easy_digits", "给定非负整数 n，返回它的十进制位数（0 视为 1 位）。", "def solve(n):\n    return len(str(n))\n", [((0,), 1), ((12345,), 5)]),
        ("easy_even", "Given a list of integers, return how many of them are even.", "def solve(xs):\n    return sum(x % 2 == 0 for x in xs)\n", [(([1, 2, 4],), 2)]),
    ]
    for tid, p, ref, tests in easy:
        T.append(fn_task(tid, p, ref, tests, mock_p_correct=1.0))
    return T


STRUTILS_INIT = 'def slugify(text):\n    """Lowercase, keep [a-z0-9], join words with "-"."""\n    raise NotImplementedError\n'
STRUTILS_REF = ('import re\n\n\ndef slugify(text):\n    """Lowercase, keep [a-z0-9], join words with "-"."""\n'
                '    words = re.findall(r"[a-z0-9]+", text.lower())\n    return "-".join(words)\n')
STRUTILS_TEST = ('import unittest\nfrom strutils import slugify\n\n\nclass T(unittest.TestCase):\n'
                 '    def test_basic(self):\n        self.assertEqual(slugify("Hello, World!"), "hello-world")\n\n'
                 '    def test_spaces(self):\n        self.assertEqual(slugify("  a  b "), "a-b")\n')


def write_repo(name, files):
    root = os.path.join(OUT, "repos", name)
    for p, c in files.items():
        full = os.path.join(root, p)
        os.makedirs(os.path.dirname(full), exist_ok=True)
        with open(full, "w") as f:
            f.write(c)
    return os.path.join("repos", name)


SWE_PROMPTS = {
    "swe_good": "博客系统需要从文章标题生成 URL 片段。请在 strutils.py 里补全 slugify：全部转小写，只留字母数字，词与词之间用连字符相连。",
    "swe_flaky": "The CMS crashes when exporting pages because slugify is a stub. Implement it so titles become lowercase hyphen-joined ASCII words.",
    "swe_already_solved": "Issue #41: page anchors are generated inconsistently. Make sure slugify lowercases input and joins alphanumeric runs with dashes.",
    "swe_leaky": "我们的静态站点生成器缺少 slugify 实现，导致构建失败。实现它：输入任意标题，输出由小写字母数字片段和短横线组成的标识。",
    "swe_build_copy_only": "Tag pages need stable identifiers. Write strutils.slugify returning the lowercase alphanumeric tokens of a string separated by '-'.",
    "swe_lint_flag": "slugify should reject non-string input with a TypeError and otherwise produce lowercase dash-separated alphanumeric slugs.",
    "swe_eval_repo": "Implement slugify in the holdout library (lowercase, alnum tokens, dash separator).",
}


def swe_task(tid, repo, files, *, created="2026-05-01", ref=STRUTILS_REF, **meta):
    return {"id": tid, "source": "synthetic-swe", "domain": "swe", "repo": repo, "created_at": created,
            "prompt": SWE_PROMPTS[tid],
            "env": {"image": IMAGE, "repo_dir": write_repo(tid, files), "test_cmd": TEST_CMD,
                    "timeout_sec": 60, "target_files": ["strutils.py"]},
            "verifier": {"type": "command"},
            "reference": {"file_edits": {"strutils.py": ref}}, "meta": meta}


def swe_tasks():
    base = {"strutils.py": STRUTILS_INIT, "tests/test_strutils.py": STRUTILS_TEST, "tests/__init__.py": ""}
    T = [swe_task("swe_good", "acme/strutils", base, mock_p_correct=0.5)]
    flaky_test = STRUTILS_TEST.replace("    def test_spaces", "    def test_flaky(self):\n        import random\n"
                                       "        self.assertEqual(slugify('x'), 'x')\n        self.assertLess(random.random(), 0.5)\n\n    def test_spaces")
    T.append(swe_task("swe_flaky", "acme/flaky", {**base, "tests/test_strutils.py": flaky_test}))
    T.append(swe_task("swe_already_solved", "acme/solved", {**base, "strutils.py": STRUTILS_REF}))
    T.append(swe_task("swe_leaky", "acme/leaky", {**base, "docs/old_impl/strutils.py": STRUTILS_REF,
                                                  "build/lib/strutils.py": STRUTILS_REF}))
    T.append(swe_task("swe_build_copy_only", "acme/buildcopy", {**base, "build/lib/strutils.py": STRUTILS_REF},
                      mock_p_correct=0.75))
    lint_test = STRUTILS_TEST + ('\n    def test_error(self):\n        with self.assertRaisesRegex(TypeError, "expected str, got int"):\n'
                                 '            slugify(3)\n')
    lint_ref = STRUTILS_REF.replace('    words =', '    if not isinstance(text, str):\n        raise TypeError(f"expected str, got {type(text).__name__}")\n    words =')
    T.append(swe_task("swe_lint_flag", "acme/lintflag", {**base, "tests/test_strutils.py": lint_test}, ref=lint_ref,
                      mock_p_correct=0.5))
    T.append(swe_task("swe_eval_repo", "evalorg/holdout-lib", base))
    return T


def eval_set():
    return [
        {"id": "eval-1", "prompt": "Given an integer n, return the number of trailing zeros in n factorial.  Constraints: 0 <= n <= 10^18."},
        {"id": "eval-2", "prompt": "Find the length of the longest increasing subsequence of an array."},
        {"id": "eval-swe-1", "repo": "evalorg/holdout-lib", "problem_statement": "Fix slugify for unicode input."},
    ]


def main():
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(os.path.join(OUT, "eval"))
    tasks = function_tasks() + swe_tasks()
    with open(os.path.join(OUT, "tasks.jsonl"), "w") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "eval", "eval_set.jsonl"), "w") as f:
        for e in eval_set():
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {len(tasks)} tasks to {OUT}")


if __name__ == "__main__":
    main()
