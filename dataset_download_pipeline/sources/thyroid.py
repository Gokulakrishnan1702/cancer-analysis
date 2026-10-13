"""
Thyroid Cancer Download Module.
Sources:
- Pathology: TCGA-THCA (GDC) - Confirmed papillary thyroid carcinoma SVS slides
- Imaging Modality: Explicitly reported as UNAVAILABLE / Pathology Only in open public benchmarks.
"""

from pathlib import Path
from config import DATA_DIR, SOURCES
from logger import pipeline_logger
from sources.common import download_gdc_svs_sample

def download_thyroid(target_total: int = 5):
    cancer = "thyroid"
    pathology_dir = DATA_DIR / cancer / "pathology"

    cfg = SOURCES[cancer]

    # 1. Log missing CT / MRI modality explicitly
    pipeline_logger.log_missing_modality(
        cancer=cancer,
        modality="ct",
        reason="No direct open-access CT benchmark for thyroid carcinoma; public TCGA cohort is pathology-focused.",
    )
    pipeline_logger.log_missing_modality(
        cancer=cancer,
        modality="mri",
        reason="No direct open-access MRI benchmark for thyroid carcinoma; public TCGA cohort is pathology-focused.",
    )

    # 2. Download Pathology from TCGA-THCA (GDC)
    pipeline_logger.log_event(
        cancer=cancer,
        modality="pathology",
        source="GDC",
        identifier="TCGA-THCA",
        action="Initiate pathology download",
        status="IN_PROGRESS",
        extra_notes="Confirmed papillary thyroid carcinoma diagnostic SVS slides."
    )
    path_files = download_gdc_svs_sample(
        cancer=cancer,
        project_id=cfg["pathology"]["project_id"],
        dataset_name=cfg["pathology"]["dataset_name"],
        license_str=cfg["pathology"]["license"],
        dest_dir=pathology_dir,
        limit=target_total,
    )

    return path_files
