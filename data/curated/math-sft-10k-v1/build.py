"""Build a pinned, verified OpenR1 SFT subset using the repository's real loader.

Run from the repository root with math-verify==0.9.0 installed:
    python data/curated/math-sft-10k-v1/build.py --raw-dir /path/to/raw
The raw directory contains openr1/, tokenizer/, and eval/ downloaded from the
immutable URLs in source_manifest.json. No model inference is performed.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
from pathlib import Path
import random
import re
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'data-pipeline')]
SEED = 20261019
MAX_LENGTH = 8192
DATA_REVISION = 'e4e141ec9dea9f8326f4d347be56105859b2bd68'
TOKENIZER_REVISION = '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0'
SUFFIX = '\n\nPlease reason step by step and put your final answer within \\boxed{}.'
TOKENIZER = None


def stable_hash(value):
    return hashlib.sha256(value.encode()).hexdigest()


def compact_parquet(path):
    import pyarrow.parquet as pq
    table = pq.ParquetFile(path).read()
    temporary = path.with_suffix('.zstd.tmp')
    try:
        pq.write_table(table, temporary, compression='zstd', compression_level=9)
        if not pq.ParquetFile(temporary).read().equals(table):
            raise ValueError('Parquet compression changed rows')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def init_worker(tokenizer_path):
    global TOKENIZER
    from transformers import AutoTokenizer
    TOKENIZER = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
    logging.getLogger('math_verify').setLevel(logging.ERROR)


def verify_candidate(row):
    from math_verify import LatexExtractionConfig, parse, verify
    from qwenpt.train.verl.template import tokenize_sft_sample
    from datapipe.export.verl import validate_sft_row
    sample = {'messages': [{'role': 'user', 'content': row['problem'] + SUFFIX},
                           {'role': 'assistant', 'content': row['generation']}],
              'enable_thinking': True}
    try:
        errors = validate_sft_row(sample)
        if errors:
            return 'format_error', row['sample_id']
        inputs = tokenize_sft_sample(sample, TOKENIZER)
        length = len(inputs['input_ids'])
        if length > MAX_LENGTH:
            return 'too_long', row['sample_id']
        if length < 1024:
            return 'short_reasoning', row['sample_id']
        supervised = int(inputs['loss_mask'].sum())
        if supervised <= 0:
            return 'no_supervision', row['sample_id']
        final = row['generation'].split('</think>', 1)[1]
        extraction = [LatexExtractionConfig()]
        answer = row['gold_answer']
        gold = parse(answer if '\\boxed' in answer else '\\boxed{' + answer + '}',
                     extraction_config=extraction, fallback_mode='no_fallback', parsing_timeout=3)
        prediction = parse(final, extraction_config=extraction, fallback_mode='no_fallback', parsing_timeout=3)
        if not gold or not prediction:
            return 'unparseable_answer', row['sample_id']
        if not verify(gold, prediction, timeout_seconds=3):
            return 'answer_mismatch', row['sample_id']
        sample.update({key: value for key, value in row.items() if key not in ('problem', 'generation')})
        sample.update(token_count=length, supervised_tokens=supervised,
                      verification='math-verify==0.9.0; final-answer equivalence')
        return 'accepted', sample
    except Exception as error:
        return 'verification_exception:' + type(error).__name__, row['sample_id']


def load_evals(raw, output):
    import pyarrow.parquet as pq
    result = []
    for path in sorted((raw / 'eval').iterdir()):
        if path.suffix == '.parquet':
            rows = pq.ParquetFile(path).read().to_pylist()
        elif path.suffix == '.jsonl':
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        else:
            continue
        for index, row in enumerate(rows):
            prompt = row.get('problem') or row.get('question') or row.get('prompt')
            if isinstance(prompt, list):
                prompt = '\n'.join(message['content'] for message in prompt if message.get('role') == 'user')
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f'Unreadable eval prompt: {path.name}:{index}')
            prompt = re.sub(r"\s*Let's think step by step and output the final answer within \\boxed\{\}\.?(?:\s*)$", '', prompt)
            result.append({'id': f'{path.stem}:{index}', 'prompt': prompt, '_file': path.stem})
    target = output / 'eval_prompts.internal.jsonl'
    target.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in result))
    return result, target


def main():
    import numpy as np
    import pyarrow.parquet as pq
    from datapipe.schema import Task
    from datapipe.stages.base import Context
    from datapipe.stages.dedup import Dedup, DecontamSplit
    from datapipe.textutil import exact_key
    from datapipe.export.verl import write_parquet
    from datapipe.export import validate_file
    from transformers import AutoTokenizer
    from qwenpt.train.verl.dataset import QwenSFTDataset

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    raw, output = args.raw_dir.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    scratch = output / 'audit'
    scratch.mkdir(exist_ok=True)
    if any(output.glob('*.parquet')):
        raise ValueError('Output already contains Parquet files; use a fresh output directory')
    started = time.time()
    stats = Counter()
    candidates = []
    for path in sorted((raw / 'openr1').glob('train-*.parquet')):
        rows = pq.ParquetFile(path).read().to_pylist()
        print(f'Reading {path.name}: {len(rows)} problems', flush=True)
        for row_index, row in enumerate(rows):
            stats['source_problems'] += 1
            problem, answer = row['problem'].strip(), str(row['answer']).strip()
            if (not answer or answer.lower() in ('proof', 'notfound') or row.get('question_type') == 'proof'
                or re.search(r'\b(prove|show that)\b', problem, re.I)
                or re.search(r'<image>|<video>|\[asy\]|diagram|figure|shown below', problem, re.I)):
                stats['drop:proof_or_missing_visual_or_answer'] += 1
                continue
            generations = row.get('generations') or []
            correctness = row.get('correctness_math_verify') or []
            complete = row.get('is_reasoning_complete') or []
            choices = []
            for index, text in enumerate(generations):
                if index >= len(correctness) or correctness[index] is not True:
                    continue
                if index >= len(complete) or complete[index] is not True:
                    continue
                if not isinstance(text, str):
                    continue
                text = text.strip()
                if (not text.startswith('<think>') or text.count('<think>') != 1
                    or text.count('</think>') != 1 or '\\boxed' not in text.split('</think>', 1)[1]
                    or any(token in text + problem for token in ('<|im_start|>', '<|im_end|>', '<|endoftext|>'))):
                    continue
                choices.append((len(text), index, text))
            if not choices:
                stats['drop:no_complete_math_verified_trace'] += 1
                continue
            _, index, text = min(choices)
            correct = sum(flag is True for flag in correctness)
            candidates.append({
                'problem': problem, 'generation': text, 'gold_answer': answer,
                'sample_id': 'openr1-' + exact_key(problem)[:24],
                'source': row['source'], 'dataset': 'open-r1/OpenR1-Math-220k',
                'dataset_revision': DATA_REVISION, 'source_file': path.name,
                'source_row': row_index, 'source_uuid': row['uuid'], 'generation_index': index,
                'math_category': row['problem_type'] or 'Other', 'question_type': row['question_type'],
                'teacher_correct_generations': correct, 'teacher_generations': len(generations),
                'teacher_success_rate': correct / len(generations),
            })
    print(f'Complete, upstream math-verified candidates: {len(candidates)}', flush=True)
    candidates.sort(key=lambda row: (row['teacher_success_rate'], stable_hash(str(SEED) + row['source_uuid'])))
    # Keep one candidate for each normalized prompt before creating the ID map.
    seen, unique = set(), []
    for row in candidates:
        if row['sample_id'] in seen:
            stats['drop:exact_duplicate'] += 1
        else:
            seen.add(row['sample_id'])
            unique.append(row)
    candidates = unique
    by_id = {row['sample_id']: row for row in candidates}
    tasks = [Task.from_dict({'id': row['sample_id'], 'source': row['source'], 'domain': 'math',
                            'prompt': row['problem'], 'verifier': {'type': 'answer_match', 'answer': row['gold_answer']}})
             for row in candidates]
    evals, eval_file = load_evals(raw, scratch)
    eval_keys = {exact_key(row['prompt']) for row in evals}
    tasks = [task for task in tasks if exact_key(task.prompt) not in eval_keys]
    stats['drop:exact_eval_overlap'] = len(candidates) - len(tasks)
    ctx = Context({}, str(scratch), {}, 'math-sft-10k-v1')
    dedup_cfg = {'answer_guard': False,
                 'minhash': {'enabled': True, 'threshold': 0.8, 'ngram': 5, 'num_perm': 128, 'bands': 16, 'seed': SEED},
                 'embedding': {'enabled': False}}
    tasks = Dedup(dedup_cfg, ctx).run(tasks)
    print(f'After exact / MinHash dedup: {len(tasks)}', flush=True)
    decontam_cfg = {'eval_sets': [str(eval_file)],
                   'ngram': {'n': 13, 'min_shared': 1, 'word_tokens': True, 'max_eval_df': 3,
                             'max_train_df': 50, 'min_overlap': 0.2},
                   'embedding': {'enabled': False}, 'time_split': {'enabled': False}, 'repo_split': {'enabled': False}}
    tasks = DecontamSplit(decontam_cfg, ctx).run(tasks)
    ctx.close()
    print(f'After benchmark decontamination: {len(tasks)}', flush=True)
    kept = []
    with (scratch / 'verification_rejections.jsonl').open('w') as rejection_file:
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=init_worker, initargs=(str(raw / 'tokenizer'),)) as pool:
            for count, (status, value) in enumerate(pool.map(verify_candidate, (by_id[t.id] for t in tasks), chunksize=16), 1):
                stats[status] += 1
                if status == 'accepted':
                    kept.append(value)
                else:
                    rejection_file.write(json.dumps({'sample_id': value, 'reason': status}) + '\n')
                if count % 1000 == 0:
                    print(f'Checked {count}/{len(tasks)}; accepted {len(kept)}', flush=True)
    if len(kept) < 10000:
        (output / 'build_partial_report.json').write_text(json.dumps(dict(stats), indent=2))
        raise ValueError(f'Only {len(kept)} qualified samples; add candidate shards rather than relax checks')
    # Round-robin categories while taking lower teacher success rate first in each.
    # This is a teacher proxy, not a measurement of the target Qwen model.
    groups = defaultdict(deque)
    for row in kept:
        groups[row['math_category']].append(row)
    selected = []
    while len(selected) < 10000:
        for category in sorted(groups):
            if groups[category] and len(selected) < 10000:
                selected.append(groups[category].popleft())
    random.Random(SEED).shuffle(selected)
    validation, train = selected[:500], selected[500:]
    artifacts = []
    for index in range(10):
        path = output / f'data-{index:05d}-of-00010.parquet'
        rows = [dict(row, split='train') for row in train[index * 950:(index + 1) * 950]]
        rows += [dict(row, split='validation') for row in validation[index * 50:(index + 1) * 50]]
        write_parquet(rows, str(path))
        compact_parquet(path)
        artifacts.append((path, rows, 'mixed'))
    tokenizer = AutoTokenizer.from_pretrained(raw / 'tokenizer', local_files_only=True)
    verification = []
    for path, rows, split in artifacts:
        report = validate_file(str(path))
        assert report['invalid'] == 0, report
        loader = QwenSFTDataset(str(path), tokenizer, {'max_length': MAX_LENGTH, 'truncation': 'error',
                                                      'pad_mode': 'no_padding', 'loss_scope': 'all_assistant'})
        for index in range(len(loader)):
            item = loader[index]
            assert len(item['input_ids']) == rows[index]['token_count']
            assert int(item['loss_mask'].sum()) == rows[index]['supervised_tokens']
            assert item['input_ids'].shape == item['loss_mask'].shape == item['position_ids'].shape
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        verification.append({'file': path.name, 'split': split, 'rows': len(rows), 'bytes': path.stat().st_size,
                             'sha256': digest, 'format_invalid': 0, 'real_loader_passed': len(loader),
                             'train_rows': 950, 'validation_rows': 50})
        print(f'Validated {path.name}: {len(rows)} rows; {path.stat().st_size / 1e6:.1f} MB', flush=True)
    lengths = [row['token_count'] for row in selected]
    train_ids = {row['sample_id'] for row in train}
    val_ids = {row['sample_id'] for row in validation}
    assert not train_ids & val_ids
    assert len(train_ids | val_ids) == 10000
    report = {
        'dataset_id': 'math-sft-10k-v1', 'total_rows': 10000, 'train_rows': 9500, 'validation_rows': 500,
        'seed': SEED, 'source_revision': DATA_REVISION, 'tokenizer_id': 'Qwen/Qwen3.8-27B',
        'tokenizer_revision': TOKENIZER_REVISION, 'max_length': MAX_LENGTH,
        'sequence_length': {'min': min(lengths), 'median': int(np.median(lengths)),
                            'p95': int(np.percentile(lengths, 95)), 'max': max(lengths), 'total': sum(lengths)},
        'categories': dict(Counter(row['math_category'] for row in selected)),
        'sources': dict(Counter(row['source'] for row in selected)),
        'teacher_success_rate_counts': dict(Counter(str(row['teacher_success_rate']) for row in selected)),
        'filter_counts': dict(stats), 'pipeline_counts': {k: dict(v) for k, v in ctx.stats.items()},
        'eval_sets': dict(Counter(row['_file'] for row in evals)),
        'dedup_config': dedup_cfg, 'decontam_config': {k: v for k, v in decontam_cfg.items() if k != 'eval_sets'},
        'train_validation_exact_overlap': 0, 'artifacts': verification,
        'base_repository_commit': 'cd2f77933639cf31b3dcbb86507020daf6efff93',
        'checks': {'all_final_answers_reverified': True, 'all_rows_loaded_with_real_QwenSFTDataset': True,
                   'truncated_rows': 0, 'target_model_difficulty_measured': False,
                   'semantic_decontamination_performed': False, 'gpu_training_run': False},
        'limitations': ['Final-answer equivalence does not prove every reasoning step.',
                        'Only the listed public eval sets were checked using exact / lexical overlap.',
                        'MinHash is approximate; it cannot exclude all semantic paraphrases.',
                        'Teacher success is a DeepSeek-R1 proxy, not target-model pass rate.',
                        'This is math reasoning SFT, not SWE/terminal/tool-use training or capability-retention replay.'],
        'build_seconds': round(time.time() - started, 1),
        'compression': {'codec': 'zstd', 'level': 9, 'all_rows_equal_to_loader_validated_tables': True},
        'build_environment': {name: importlib.metadata.version(name) for name in
                              ('transformers', 'torch', 'pyarrow', 'numpy', 'math-verify', 'jsonschema')},
    }
    (output / 'validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
