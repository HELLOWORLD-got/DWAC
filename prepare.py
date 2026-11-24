"""One entry point: raw MDMT -> audited train/val/test manifests."""

import argparse
from pathlib import Path
from tools.prepare_data import prepare, fetch_official_test_gt, prepare_official_test, audit_native_categories
import json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="data/raw/MDMT", help="Original MDMT directory")
    parser.add_argument(
        "--output", help="Generated manifests and audit; defaults depend on class mode"
    )
    parser.add_argument(
        "--verify-images", action="store_true", help="Check every JPEG, not just metadata"
    )
    parser.add_argument("--class-mode", choices=["car_only", "agnostic", "native"], default="car_only")
    parser.add_argument("--max-instances", type=int, help="Keep complete groups within this GT limit; car_only defaults to 25")
    parser.add_argument("--xml-only", action="store_true", help="Skip official TXT test manifest")
    parser.add_argument("--audit-only", action="store_true", help="Audit native categories without writing manifests")
    args = parser.parse_args()
    if args.max_instances is not None and args.max_instances < 1:
        parser.error("--max-instances must be positive")
    if args.class_mode == "car_only" and args.max_instances is None:
        args.max_instances = 25
    default_output = (f"data/processed/MDMT_car_q{args.max_instances}" if args.class_mode == "car_only" else
                      "data/processed/MDMT_native" if args.class_mode == "native" else "data/processed/MDMT")
    args.output = args.output or default_output
    if args.audit_only:
        if args.class_mode != "native":
            parser.error("--audit-only checks native labels")
        report = audit_native_categories(args.root)
        output = Path(args.output);output.mkdir(parents=True, exist_ok=True)
        (output/"native_category_audit.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:v for k,v in report.items() if k != "conflicts"},indent=2))
        return
    prepare(args.root, args.output, args.class_mode, args.verify_images, args.max_instances)
    if args.class_mode == "agnostic" and not args.xml_only:
        output = Path(args.output)
        fetch_official_test_gt(output / "official_test")
        report = prepare_official_test(
            output / "test.jsonl", output / "official_test", output / "test_official.jsonl"
        )
        print("Official test manifest:", report)
    elif args.class_mode == "native":
        print("Native XML test labels retained. Official TXT is not used to overwrite categories.")
    elif args.class_mode == "car_only":
        print("Car-only XML labels, whole groups filtered by capacity; no car GT truncated. Official TXT not substituted.")


if __name__ == "__main__":
    main()
