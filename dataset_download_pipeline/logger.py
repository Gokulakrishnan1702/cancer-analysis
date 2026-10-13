"""
Logger module for the Medical Image Download Pipeline.
Maintains structured logging to download_log.txt and console.
"""

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional
from config import LOG_PATH

class PipelineLogger:
    def __init__(self, log_path: Path = LOG_PATH):
        self.log_path = log_path
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.log_path.exists():
            with open(self.log_path, "w", encoding="utf-8") as f:
                f.write("=== MEDICAL IMAGE DOWNLOAD PIPELINE LOG ===\n")
                f.write(f"Initialized: {datetime.utcnow().isoformat()}Z\n\n")

    def log_event(
        self,
        cancer: str,
        modality: str,
        source: str,
        identifier: str,
        action: str,
        status: str,
        reason: Optional[str] = None,
        auth_requirements: Optional[str] = None,
        retries: int = 0,
        extra_notes: Optional[str] = None
    ):
        """
        Record a structured download event into download_log.txt.
        """
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%SZ")
        lines = [
            f"[{timestamp}] CANCER: {cancer.upper()} | MODALITY: {modality.upper()}",
            f"  Source/Dataset : {source}",
            f"  Identifier     : {identifier}",
            f"  Action         : {action}",
            f"  Status         : {status}",
            f"  Retries        : {retries}",
        ]
        if reason:
            lines.append(f"  Reason         : {reason}")
        if auth_requirements:
            lines.append(f"  Auth / Terms   : {auth_requirements}")
        if extra_notes:
            lines.append(f"  Notes          : {extra_notes}")
        lines.append("-" * 70 + "\n")

        entry = "\n".join(lines)
        print(f"[{status}] {cancer.capitalize()} ({modality}): {identifier} - {reason or action}")

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(entry)

    def log_missing_modality(self, cancer: str, modality: str, reason: str):
        """
        Explicitly log a missing or unavailable modality.
        """
        self.log_event(
            cancer=cancer,
            modality=modality,
            source="N/A",
            identifier="MISSING_MODALITY",
            action="Check modality availability",
            status="NOT_FOUND",
            reason=reason,
            extra_notes="Modality not offered as open direct benchmark for this cancer."
        )

# Global default logger instance
pipeline_logger = PipelineLogger()
