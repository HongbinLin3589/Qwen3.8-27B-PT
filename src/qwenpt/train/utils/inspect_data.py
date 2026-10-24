"""Inspect rendered model inputs and supervision from prepared SFT Parquet."""

import argparse
import json


def main() -> None:
    """Print rendered inputs and supervised tokens for a bounded set of rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", nargs="+", required=True)
    parser.add_argument("--tokenizer", required=True, help="Tokenizer/model ID or local snapshot path")
    parser.add_argument("--processor", help="Optional multimodal processor ID or local snapshot path")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--limit", type=int, default=3)
    parser.add_argument("--loss-scope", choices=("all_assistant", "last_assistant"), default="all_assistant")
    args = parser.parse_args()
    if args.limit <= 0:
        parser.error("--limit must be positive")
    from transformers import AutoProcessor, AutoTokenizer

    from qwenpt.train.verl.dataset import QwenSFTDataset

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=args.local_files_only)
    processor = (AutoProcessor.from_pretrained(args.processor, local_files_only=args.local_files_only)
                 if args.processor else None)
    dataset = QwenSFTDataset(args.data, tokenizer, {
        "max_length": args.max_length, "loss_scope": args.loss_scope,
    }, processor=processor)
    for index in range(min(args.limit, len(dataset))):
        item = dataset[index]
        ids = item["input_ids"]
        mask = item["loss_mask"].bool()
        thinking = dataset.rows[index].get("enable_thinking")
        if thinking is None:
            thinking = dataset.enable_thinking_default
        print(json.dumps({
            "index": index, "sample_id": dataset.rows[index].get("sample_id"),
            "tokens": len(ids), "supervised_tokens": int(mask.sum()),
            "effort_label": None, "template_effort": "xhigh" if thinking else None,
            "text": tokenizer.decode(ids, skip_special_tokens=False),
            "supervised_text": tokenizer.decode(ids[mask], skip_special_tokens=False),
            "position_ids_shape": list(item["position_ids"].shape),
        }, ensure_ascii=False))


if __name__ == "__main__":
    main()
