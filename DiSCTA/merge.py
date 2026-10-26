from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from discta.data import load_records, save_records, save_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", type=Path, required=True)
    p.add_argument(
        "--manifest", type=Path, help="Expected IDs; defaults to all dataset images"
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    records, sources = ({}, {})
    config = None
    for run in args.runs:
        current = json.loads((run / "config.json").read_text())
        if config is not None and config != current:
            raise ValueError("Shard configs differ")
        config = current
        ids = json.loads((run / "selected_ids.json").read_text())
        rows = load_records(run)
        if set(rows) != set(ids) or len(ids) != len(set(ids)):
            raise ValueError("Shard records do not match manifest")
        for i, row in rows.items():
            if i in records or row["status"] != "ok":
                raise ValueError(f"Duplicate or incomplete sample: {i}")
            for relative in [
                f"{i}_clean.png",
                f"adv_images/{i}.png",
                f"debug/{i}_mask.png",
            ]:
                if not (run / relative).is_file():
                    raise FileNotFoundError(run / relative)
            records[i], sources[i] = (row, run)
    expected = (
        json.loads(args.manifest.read_text())
        if args.manifest
        else sorted(p.stem for p in Path(config["paths"]["images_root"]).glob("*.jpg"))
    )
    if len(expected) != len(set(expected)):
        raise ValueError("Duplicate IDs in expected manifest")
    if set(records) != set(expected):
        raise ValueError("Merged sample set differs from full manifest")
    args.output.mkdir(parents=True, exist_ok=False)
    for folder in ["adv_images", "debug"]:
        (args.output / folder).mkdir()
    for i in expected:
        for relative in [
            f"{i}_clean.png",
            f"adv_images/{i}.png",
            f"debug/{i}_mask.png",
        ]:
            shutil.copy2(sources[i] / relative, args.output / relative)
        records[i]["clean_path"] = str((args.output / f"{i}_clean.png").resolve())
        records[i]["adv_path"] = str(
            (args.output / "adv_images" / f"{i}.png").resolve()
        )
        records[i]["source_run"] = str(sources[i].resolve())
    save_json(args.output / "config.json", config)
    save_json(args.output / "selected_ids.json", expected)
    save_records(args.output, {i: records[i] for i in expected})
    save_json(
        args.output / "run_summary.json",
        dict(
            num_samples=len(expected), status_counts={"ok": len(expected), "failed": 0}
        ),
    )
    print(f"Merged {len(expected)} samples into {args.output}")


if __name__ == "__main__":
    main()
