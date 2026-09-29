"""Local E7 input audit, adapting Transformers 5 chat-template return type."""

import json
from pathlib import Path
import sys

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
from fragmented_facts.io import read_json, read_jsonl
from project_plan.feasibility.e7_same_count_run_20260928 import validated_grid
from project_plan.feasibility.e7_same_count_run_20260928 import FACTS, TEMPLATES, E6, PROTOCOL


def main():
    tokenizer = AutoTokenizer.from_pretrained(
        str(ROOT / "data/raw/aya_23_8b"), use_fast=True,
        local_files_only=True, trust_remote_code=False)
    original = tokenizer.apply_chat_template

    def compatibility(*args, **kwargs):
        value = original(*args, **kwargs)
        if kwargs.get("tokenize") and isinstance(value, list) and len(value) == 1 and hasattr(value[0], "ids"):
            return value[0].ids
        if kwargs.get("tokenize") and hasattr(value, "input_ids"):
            return value["input_ids"]
        return value

    tokenizer.apply_chat_template = compatibility
    facts = {cohort: {row["fact_id"]: row for row in read_jsonl(path)}
             for cohort, path in FACTS.items()}
    templates = read_json(TEMPLATES)
    e6 = {(row["cohort"], row["fact_id"], row["language"]): row
          for row in read_json(E6)["rows"]}
    rows = read_json(PROTOCOL)["rows"]
    for row in rows:
        fact = facts[row["cohort"]][row["fact_id"]]
        frozen = e6[(row["cohort"], row["fact_id"], row["language"])]
        validated_grid(tokenizer, fact, row, frozen, templates)
    print(json.dumps({"status": "local_grid_valid", "rows": len(rows)}))


if __name__ == "__main__":
    main()
