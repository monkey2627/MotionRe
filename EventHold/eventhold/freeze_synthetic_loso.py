"""Create explicit leave-one-subject-out exploration splits for an extension set."""
import argparse
import csv
import json
from pathlib import Path

from .audit import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--candidate-csv", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError("Use a new LOSO output root")
    manifest = json.loads(args.manifest.read_text())
    with args.candidate_csv.open(encoding="utf-8-sig") as handle:
        candidates = list(csv.DictReader(handle))
    candidate_by_source = {row["source"]: row for row in candidates}
    rows = []
    for row in manifest["manifest"]:
        if row["source"] not in candidate_by_source:
            raise ValueError(f"Missing candidate for {row['source']}")
        rows.append({**row, "subject_key": row["subject_key"]})
    subjects = sorted({row["subject_key"] for row in rows})
    if len(subjects) < 2:
        raise ValueError("LOSO requires at least two subject groups")
    args.output_root.mkdir(parents=True)
    for index, held_out in enumerate(subjects):
        output = args.output_root / f"fold_{index:02d}_{held_out.rsplit('/', 1)[-1]}"
        output.mkdir()
        split_rows = []
        for row in rows:
            split_rows.append(
                {
                    **row,
                    "split": "holdout_exploration" if row["subject_key"] == held_out else "train_exploration",
                }
            )
        payload = {
            "status": "frozen_loso_exploration_split_not_final_benchmark",
            "manifest_sha256": sha256(args.manifest),
            "candidate_csv_sha256": sha256(args.candidate_csv),
            "group_rule": "subject_key",
            "held_out_group": held_out,
            "groups": len(subjects),
            "rows": split_rows,
            "no_test_claim": True,
        }
        (output / "split.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"folds": len(subjects), "subjects": subjects}), flush=True)


if __name__ == "__main__":
    main()
