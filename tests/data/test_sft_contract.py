"""Regression tests for the supplied reasoning / computer-use SFT contract."""
import copy
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

ROOT = Path(__file__).resolve().parents[2]
from qwenpt.data.export import export_sft_files, validate_file
from qwenpt.data.export.verl import drop_nulls, read_parquet, validate_sft_row, write_parquet

GUI = json.loads((ROOT / 'examples/data/verl_format/computer_use_sft.json').read_text())
MATH = {'messages': [
    {'role': 'system', 'content': '你是数学助手。'},
    {'role': 'user', 'content': '有 3 个苹果，再买 2 个，一共有多少个？'},
    {'role': 'assistant', 'content': '3 + 2 = 5。\n#### 5'},
]}


def png(pixel):
    """Distinct, decodable 1x1 test images; no original FinanceGym media needed."""
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(b'\0' + bytes(pixel))) + chunk(b'IEND', b''))


class TestStrictSFT(unittest.TestCase):
    def test_supplied_examples_and_final_parallel_calls(self):
        for row in (MATH, GUI, {'messages': MATH['messages'][1:]}):
            self.assertEqual(validate_sft_row(row), [])

    def test_boolean_is_not_a_string_or_integer(self):
        for value in ('false', 'true', 0, 1, None, [], {}):
            with self.subTest(value=value):
                self.assertIn('boolean', ' '.join(validate_sft_row({**MATH, 'enable_thinking': value})))
        for value in (True, False):
            self.assertEqual(validate_sft_row({**MATH, 'enable_thinking': value}), [])

    def test_tool_declaration_is_required(self):
        for tools in (None, [], 'computer_use', [{}]):
            row = copy.deepcopy(GUI)
            if tools is None:
                row.pop('tools')
            else:
                row['tools'] = tools
            self.assertTrue(validate_sft_row(row))

    def test_arguments_follow_the_declared_schema(self):
        for args in ({}, {'action': 'not-an-action'}, {'action': 42}, {'action': 'scroll', 'pixels': 'seven'}):
            with self.subTest(args=args):
                row = copy.deepcopy(GUI)
                row['messages'][2]['tool_calls'][0]['function']['arguments'] = json.dumps(args)
                self.assertTrue(validate_sft_row(row))

    def test_coordinate_bounds_and_shape(self):
        for xy in ([-1, 20], [1000, 20], [True, 20], [1], [1, 2, 3], '1,2', [float('nan'), 0], [10**1000, 0]):
            with self.subTest(xy=xy):
                row = copy.deepcopy(GUI)
                row['messages'][2]['tool_calls'][0]['function']['arguments'] = json.dumps({'action': 'left_click', 'coordinate': xy})
                self.assertTrue(validate_sft_row(row))
        for xy in ([0, 999], [999, 0]):
            row = copy.deepcopy(GUI)
            row['messages'][2]['tool_calls'][0]['function']['arguments'] = json.dumps({'action': 'left_click', 'coordinate': xy})
            self.assertEqual(validate_sft_row(row), [])

    def test_bad_json_arguments(self):
        for args in ('{', '[]', 'null', '42', '{"action": "wait", "time": NaN}', {'action': 'wait'}):
            row = copy.deepcopy(GUI)
            row['messages'][2]['tool_calls'][0]['function']['arguments'] = args
            self.assertTrue(validate_sft_row(row))

    def test_invalid_and_remote_parameter_schema(self):
        for schema in ({'type': 'array'}, {'type': 'object', 'required': 42}, {'$ref': 'https://example.invalid/schema'}):
            row = copy.deepcopy(GUI)
            row['tools'][0]['function']['parameters'] = schema
            self.assertTrue(validate_sft_row(row))

    def test_duplicate_and_out_of_order_tool_returns(self):
        variants = []
        row = copy.deepcopy(GUI)
        row['messages'][4]['tool_calls'][0]['id'] = 'call_1_0'
        variants.append(row)
        row = copy.deepcopy(GUI)
        row['messages'][3]['tool_call_id'] = 'missing'
        variants.append(row)
        row = copy.deepcopy(GUI)
        row['messages'][3], row['messages'][4] = row['messages'][4], row['messages'][3]
        variants.append(row)
        row = copy.deepcopy(GUI)
        row['messages'].insert(4, copy.deepcopy(row['messages'][3]))
        variants.append(row)
        for row in variants:
            self.assertTrue(validate_sft_row(row))

    def test_bad_shapes_report_errors_instead_of_crashing(self):
        for row in (None, [], 3, {}, {'messages': 'hi'}, {'messages': [None]},
                    {'messages': [{'role': [], 'content': 4}]},
                    {'messages': [{'role': 'assistant', 'content': '', 'tool_calls': [None]}]}):
            with self.subTest(row=row):
                self.assertTrue(validate_sft_row(row))

    def test_empty_or_nonassistant_target(self):
        for last in ({'role': 'assistant', 'content': ''}, {'role': 'user', 'content': 'hi'}):
            self.assertTrue(validate_sft_row({'messages': MATH['messages'][:-1] + [last]}))

    def test_media_source_types_and_count(self):
        for media in ('file.jpg', [{'image': ''}], [{'image': 42}], [{'bytes': ''}],
                      [{'image': 'x', 'bytes': b'abc'}], [{'bytes': b''}], [{}], []):
            row = {'messages': [{'role': 'user', 'content': '<image>'}, {'role': 'assistant', 'content': 'ok'}],
                   'images': media}
            self.assertTrue(validate_sft_row(row))


class TestSFTBatchIO(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.media = self.root / 'images'
        self.media.mkdir()
        self.images = [png((20 + i * 50, 20, 30)) for i in range(4)]
        for i, data in enumerate(self.images):
            (self.media / f'screen_{i}.jpg').write_bytes(data)

    def tearDown(self):
        self.tmp.cleanup()

    def save(self, rows, name='samples.json'):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.parquet':
            write_parquet(rows, str(path))
        elif path.suffix == '.jsonl':
            path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
        else:
            path.write_text(json.dumps(rows))
        return path

    def export(self, paths, **fmt):
        out = self.root / 'out'
        report = export_sft_files([str(p) for p in paths], str(out), {
            'sft': {'media_root': str(self.root), 'media': 'absolute_path', 'on_invalid': 'error', **fmt},
        })
        files = sorted((out / 'sft').glob('*.parquet'))
        return out, report, [(p, read_parquet(str(p))) for p in files]

    def test_write_parquet_unions_all_rows_in_both_orders(self):
        for rows in ([MATH, GUI], [GUI, MATH]):
            path = self.root / 'union.parquet'
            write_parquet(rows, str(path))
            self.assertEqual([drop_nulls(r) for r in read_parquet(str(path))], rows)

    def test_mixed_image_batch_keeps_later_tool_and_boolean_fields(self):
        simple = {'messages': [{'role': 'user', 'content': '<image> describe'},
                               {'role': 'assistant', 'content': 'ok'}],
                  'images': [{'image': 'images/screen_0.jpg'}]}
        for rows in ([simple, GUI], [GUI, simple]):
            out, _, files = self.export([self.save(rows)], media='path')
            restored = [drop_nulls(r) for r in files[0][1]]
            # Missing media lists are deliberately filled to prevent len(None).
            expected = copy.deepcopy(rows)
            for row in expected:
                row.setdefault('videos', [])
            self.assertEqual(restored, expected)
            self.assertEqual(validate_file(str(files[0][0]), media_root=str(self.root))['invalid'], 0)
            report = json.loads((out / 'export_report.json').read_text())
            self.assertEqual(next(iter(report['verl_read_back'].values()))['rows_changed_beyond_nulls'], 0)

    def test_json_jsonl_and_parquet_inputs(self):
        for extension in ('json', 'jsonl', 'parquet'):
            path = self.save([GUI], f'gui.{extension}')
            _, _, files = self.export([path], media='bytes')
            row = drop_nulls(files[0][1][0])
            self.assertEqual(row['messages'], GUI['messages'])
            self.assertEqual(row['tools'], GUI['tools'])
            self.assertEqual([x['bytes'] for x in row['images']], self.images)
            self.assertEqual(row['videos'], [])
            self.assertIs(row['enable_thinking'], False)
            self.assertEqual(validate_file(str(files[0][0]))['invalid'], 0)

    def test_different_tool_schemas_are_kept_separate(self):
        other = copy.deepcopy(GUI)
        other['tools'][0]['function']['parameters']['properties']['extra'] = {'type': 'string'}
        _, _, files = self.export([self.save([GUI, other])], media='bytes')
        self.assertEqual(len(files), 2)
        restored_tools = [drop_nulls(rows[0])['tools'] for _, rows in files]
        self.assertIn(GUI['tools'], restored_tools)
        self.assertIn(other['tools'], restored_tools)

    def test_same_basename_inputs_are_combined(self):
        paths = [self.save([MATH], 'one/same.json'), self.save([MATH], 'two/same.json')]
        _, _, files = self.export(paths)
        self.assertEqual(len(files[0][1]), 2)

    def test_invalid_samples_are_reported_or_raise(self):
        bad = copy.deepcopy(GUI)
        bad.pop('tools')
        path = self.save([bad, MATH, {'messages': 123}, None])
        with self.assertRaisesRegex(ValueError, 'undeclared tool'):
            self.export([path])
        out, report, files = self.export([path], on_invalid='drop')
        self.assertEqual(report['sft/invalid.jsonl'], 3)
        self.assertEqual(files[0][1], [MATH])
        self.assertEqual(len((out / 'sft/invalid.jsonl').read_text().splitlines()), 3)

    def test_boolean_override_does_not_hide_invalid_input(self):
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.export([self.save([{**MATH, 'enable_thinking': 'false'}])], enable_thinking=False)

    def test_null_external_rows_are_not_skipped_for_reserved_basenames(self):
        for name in ('reference.json', 'trajectories.json'):
            with self.assertRaisesRegex(ValueError, 'sample is not an object'):
                self.export([self.save([None], name)])

    def test_missing_input_file_is_not_silently_ignored(self):
        with self.assertRaises(FileNotFoundError):
            self.export([self.root / 'missing.jsonl'])

    def test_image_and_video_columns_stay_aligned(self):
        video = self.root / 'clip.mp4'
        video.write_bytes(b'test-video-fixture')  # Existence check, not a video codec test.
        row = {'messages': [{'role': 'user', 'content': '<video> Describe'},
                             {'role': 'assistant', 'content': 'a clip'}],
               'videos': [{'video': str(video)}]}
        _, _, files = self.export([self.save([GUI, row])], media='bytes')
        self.assertEqual(len(files), 1)
        rows = [drop_nulls(r) for r in files[0][1]]
        self.assertEqual(rows[0]['videos'], [])
        self.assertEqual(rows[1]['images'], [])
        self.assertEqual(rows[1]['videos'], row['videos'])
        self.assertEqual(validate_file(str(files[0][0]))['invalid'], 0)

    def test_failed_roundtrip_keeps_previous_target_file(self):
        path = self.root / 'atomic.parquet'
        write_parquet([MATH], str(path))
        original = path.read_bytes()
        with self.assertRaisesRegex(ValueError, 'round-trip'):
            write_parquet([{'value': float('nan')}], str(path))
        self.assertEqual(path.read_bytes(), original)

    def test_required_structured_columns_cannot_be_json_strings(self):
        with self.assertRaisesRegex(ValueError, 'structured'):
            export_sft_files([str(self.save([MATH]))], str(self.root / 'bad'), {'json_columns': ['messages']})

    def test_media_existence_checks_and_structure_only_mode(self):
        path = self.save([GUI])
        self.assertEqual(validate_file(str(path), media_root=str(self.root))['invalid'], 0)
        (self.media / 'screen_0.jpg').unlink()
        self.assertEqual(validate_file(str(path), media_root=str(self.root))['invalid'], 1)
        self.assertEqual(validate_file(str(path), check_media_exists=False)['invalid'], 0)
        with self.assertRaisesRegex(ValueError, 'media'):
            self.export([path])

    def test_cli_sft_import_export_validate(self):
        inp = self.save([MATH, GUI], 'input.jsonl')
        out = self.root / 'cli'
        args = [sys.executable, '-m', 'qwenpt.data', 'sft', '--input', str(inp), '--out', str(out),
                '--set', f'export.sft.media_root={self.root}', '--set', 'export.sft.media=bytes',
                '--set', 'export.sft.on_invalid=error']
        result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        files = list((out / 'sft').glob('*.parquet'))
        self.assertEqual(len(files), 2)
        for path in files:
            result = subprocess.run([sys.executable, '-m', 'qwenpt.data', 'validate', str(path)],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr + result.stdout)
        # Re-import the produced Parquet files without losing image bytes or tool calls.
        out2 = self.root / 'cli2'
        result = subprocess.run([sys.executable, '-m', 'qwenpt.data', 'sft', '--input', *map(str, files),
                                 '--out', str(out2), '--set', 'export.sft.on_invalid=error'],
                                cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sum(len(read_parquet(str(p))) for p in (out2 / 'sft').glob('*.parquet')), 2)

    def test_cli_rejects_bad_boolean_and_missing_media(self):
        for row in ({**MATH, 'enable_thinking': 'false'}, GUI):
            path = self.save([row], 'invalid.json')
            result = subprocess.run([sys.executable, '-m', 'qwenpt.data', 'validate', str(path), '--kind', 'sft'],
                                    cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 1, result.stderr + result.stdout)
            self.assertEqual(json.loads(result.stdout)['invalid'], 1)


if __name__ == '__main__':
    unittest.main()
