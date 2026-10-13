"""
Manifest manager module.
Maintains manifest.csv with exactly the required columns:
cancer, modality, file_path, source, dataset_name, license, download_status, notes
"""

import csv
from pathlib import Path
from typing import Dict, List, Optional
from config import MANIFEST_PATH, SUMMARY_PATH

REQUIRED_COLUMNS = [
    "cancer",
    "modality",
    "file_path",
    "source",
    "dataset_name",
    "license",
    "download_status",
    "notes",
]

class ManifestManager:
    def __init__(self, manifest_path: Path = MANIFEST_PATH):
        self.manifest_path = manifest_path
        self.init_manifest()

    def init_manifest(self):
        """Initialize the manifest CSV with required columns if it doesn't exist."""
        if not self.manifest_path.exists():
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(REQUIRED_COLUMNS)

    def read_records(self) -> List[Dict[str, str]]:
        """Read all existing manifest records."""
        if not self.manifest_path.exists():
            return []
        records = []
        with open(self.manifest_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                records.append(dict(row))
        return records

    def record_entry(
        self,
        cancer: str,
        modality: str,
        file_path: str,
        source: str,
        dataset_name: str,
        license_str: str,
        download_status: str,
        notes: str,
    ):
        """
        Record or update a single entry in manifest.csv.
        Avoids duplicates by matching (cancer, modality, file_path, source).
        """
        records = self.read_records()
        updated = False
        new_row = {
            "cancer": cancer.lower(),
            "modality": modality.lower(),
            "file_path": str(file_path).replace("\\", "/"),
            "source": source,
            "dataset_name": dataset_name,
            "license": license_str,
            "download_status": download_status,
            "notes": notes,
        }

        for i, r in enumerate(records):
            # Check for identical key
            if (
                r.get("cancer") == new_row["cancer"]
                and r.get("modality") == new_row["modality"]
                and r.get("file_path") == new_row["file_path"]
                and r.get("source") == new_row["source"]
            ):
                records[i] = new_row
                updated = True
                break

        if not updated:
            records.append(new_row)

        with open(self.manifest_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=REQUIRED_COLUMNS)
            writer.writeheader()
            writer.writerows(records)

    def generate_availability_summary(self, summary_path: Path = SUMMARY_PATH):
        """
        Generate dataset_availability_summary.csv detailing counts and statuses per cancer and modality.
        """
        records = self.read_records()
        summary: Dict[str, Dict[str, any]] = {}

        for r in records:
            c = r["cancer"]
            m = r["modality"]
            status = r["download_status"]
            if c not in summary:
                summary[c] = {
                    "cancer": c,
                    "modalities_verified": set(),
                    "downloaded_files": 0,
                    "auth_required_count": 0,
                    "failed_count": 0,
                    "modalities_breakdown": {},
                }
            summary[c]["modalities_verified"].add(m)
            if m not in summary[c]["modalities_breakdown"]:
                summary[c]["modalities_breakdown"][m] = 0

            if status == "DOWNLOADED":
                summary[c]["downloaded_files"] += 1
                summary[c]["modalities_breakdown"][m] += 1
            elif status == "AUTH_REQUIRED":
                summary[c]["auth_required_count"] += 1
            else:
                summary[c]["failed_count"] += 1

        summary_rows = []
        for c, data in summary.items():
            breakdown_str = "; ".join(f"{mod}: {cnt}" for mod, cnt in data["modalities_breakdown"].items())
            summary_rows.append({
                "cancer": c,
                "verified_modalities": ", ".join(sorted(data["modalities_verified"])),
                "total_downloaded_files": data["downloaded_files"],
                "modalities_breakdown": breakdown_str,
                "auth_required_sources": data["auth_required_count"],
                "failed_sources": data["failed_count"],
            })

        fieldnames = [
            "cancer",
            "verified_modalities",
            "total_downloaded_files",
            "modalities_breakdown",
            "auth_required_sources",
            "failed_sources",
        ]
        with open(summary_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)

manifest_manager = ManifestManager()
