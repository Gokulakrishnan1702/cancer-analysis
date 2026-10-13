"""
Main CLI entry point for the Medical Image Download Pipeline.
Supports:
  python main.py --list-sources
  python main.py --cancer <name>
  python main.py --all
  python main.py --failed-only
  python main.py --status
"""

import argparse
import sys
from pathlib import Path

# Add current directory to sys.path so imports work smoothly
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from config import CANCER_TYPES, SOURCES, DATA_DIR, MANIFEST_PATH, SUMMARY_PATH
from manifest import manifest_manager
from sources import SOURCE_DOWNLOADERS


def list_sources():
    """Print verified sources for all 15 cancer types."""
    print("=" * 80)
    print("VERIFIED DATASET SOURCES (15 CANCER TYPES)")
    print("=" * 80)
    for idx, c in enumerate(CANCER_TYPES, 1):
        print(f"\n[{idx}] {c.upper()}")
        c_cfg = SOURCES.get(c, {})
        for mod, details in c_cfg.items():
            ds_name = details.get("dataset_name", "N/A")
            src = details.get("source", "N/A")
            lic = details.get("license", "N/A")
            auth = "Auth Required" if details.get("auth_required") else "Direct Open Access"
            print(f"    - Modality/Branch: {mod:<18} | Source: {src:<12} | Dataset: {ds_name:<20} | License: {lic} | Access: {auth}")
    print("=" * 80)


def show_status():
    """Display current download status from manifest.csv and disk."""
    records = manifest_manager.read_records()
    print("=" * 80)
    print("CURRENT PIPELINE DOWNLOAD STATUS")
    print("=" * 80)

    if not records:
        print("No manifest records found. Run downloads first via --cancer <name> or --all.")
        return

    # Count by cancer and modality
    counts = {}
    auth_req = []
    failed = []

    for r in records:
        c = r["cancer"]
        m = r["modality"]
        st = r["download_status"]

        if c not in counts:
            counts[c] = {}
        if m not in counts[c]:
            counts[c][m] = 0

        if st == "DOWNLOADED":
            counts[c][m] += 1
        elif st == "AUTH_REQUIRED":
            auth_req.append((c, m, r["dataset_name"], r["notes"]))
        elif st in ("DOWNLOAD_FAILED", "VALIDATION_FAILED"):
            failed.append((c, m, r["file_path"], r["notes"]))

    print(f"{'Cancer':<15} | {'Downloaded Counts by Modality':<45} | {'Total Files'}")
    print("-" * 80)
    grand_total = 0
    for c in CANCER_TYPES:
        c_counts = counts.get(c, {})
        breakdown = ", ".join(f"{m.upper()}: {cnt}" for m, cnt in c_counts.items() if cnt > 0) or "None"
        total_c = sum(c_counts.values())
        grand_total += total_c
        print(f"{c.capitalize():<15} | {breakdown:<45} | {total_c}")

    print("-" * 80)
    print(f"Grand Total Validated Image Files: {grand_total}")

    if auth_req:
        print("\n--- SOURCES REQUIRING AUTHENTICATION / TOKENS ---")
        for c, m, ds, note in auth_req:
            print(f"  * {c.capitalize()} ({m}): {ds} -> {note}")

    if failed:
        print("\n--- FAILED DOWNLOADS ---")
        for c, m, fp, note in failed:
            print(f"  * {c.capitalize()} ({m}): {fp} -> {note}")

    # Regenerate summary CSV
    manifest_manager.generate_availability_summary()
    print(f"\nAvailability summary updated: {SUMMARY_PATH}")
    print("=" * 80)


def run_cancer(cancer_name: str, target_total: int = 5):
    """Run download for a single cancer type."""
    cancer_name = cancer_name.lower()
    if cancer_name not in SOURCE_DOWNLOADERS:
        print(f"Error: Unknown cancer type '{cancer_name}'. Supported: {', '.join(CANCER_TYPES)}")
        sys.exit(1)

    print(f"\n>>> Starting download for {cancer_name.upper()} (target: {target_total} files) <<<")
    downloader = SOURCE_DOWNLOADERS[cancer_name]
    results = downloader(target_total=target_total)
    print(f">>> Completed {cancer_name.upper()}: {len(results)} valid files acquired <<<")
    manifest_manager.generate_availability_summary()
    return results


def run_failed_only():
    """Re-attempt sources marked as failed in manifest.csv."""
    records = manifest_manager.read_records()
    failed_cancers = set()
    for r in records:
        if r.get("download_status") in ("DOWNLOAD_FAILED", "VALIDATION_FAILED"):
            failed_cancers.add(r.get("cancer"))

    if not failed_cancers:
        print("No failed records found in manifest.csv.")
        return

    print(f"Re-attempting failed sources for cancers: {', '.join(failed_cancers)}")
    for c in failed_cancers:
        if c in SOURCE_DOWNLOADERS:
            run_cancer(c)


def main():
    parser = argparse.ArgumentParser(
        description="Modular Medical Image Download Pipeline for 15 Cancer Types"
    )
    parser.add_argument("--list-sources", action="store_true", help="List all verified sources and access requirements")
    parser.add_argument("--cancer", type=str, help="Download samples for a specific cancer (e.g. lung, breast)")
    parser.add_argument("--all", action="store_true", help="Download small samples across all 15 cancer types")
    parser.add_argument("--status", action="store_true", help="Print download progress, file counts, and manifest status")
    parser.add_argument("--failed-only", action="store_true", help="Re-attempt downloads only for failed records")
    parser.add_argument("--target-samples", type=int, default=5, help="Target number of image files per cancer (default: 5)")

    args = parser.parse_args()

    if args.list_sources:
        list_sources()
    elif args.status:
        show_status()
    elif args.failed_only:
        run_failed_only()
    elif args.cancer:
        run_cancer(args.cancer, target_total=args.target_samples)
    elif args.all:
        print("Starting full pipeline run for all 15 cancer types...")
        for c in CANCER_TYPES:
            run_cancer(c, target_total=args.target_samples)
        show_status()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
