"""
Melanoma Download Module.
Sources:
- Pathology: TCGA-SKCM (GDC) - Confirmed skin cutaneous melanoma SVS slides
- Imaging Modality: Explicitly reported as UNAVAILABLE / Pathology Only in public benchmark.
"""

from pathlib import Path
from config import DATA_DIR, SOURCES
from logger import pipeline_logger
from sources.common import download_gdc_svs_sample

def download_melanoma(target_total: int = 5):
    cancer = "melanoma"
    pathology_dir = DATA_DIR / cancer / "pathology"

    cfg = SOURCES[cancer]
    downloaded = []

    # 1. Log missing CT / MRI modality explicitly
    pipeline_logger.log_missing_modality(
        cancer=cancer,
        modality="ct",
        reason="No direct public CT benchmark for melanoma; disease staged clinically and via histopathology.",
    )
    pipeline_logger.log_missing_modality(
        cancer=cancer,
        modality="mri",
        reason="No direct public MRI benchmark for melanoma; macroscopic imaging is dermoscopy (ISIC) rather than MRI.",
    )

    # 2. Download Pathology from TCGA-SKCM (GDC)
    pipeline_logger.log_event(
        cancer=cancer,
        modality="pathology",
        source="GDC",
        identifier="TCGA-SKCM",
        action="Initiate pathology download",
        status="IN_PROGRESS",
        extra_notes="Confirmed malignant melanoma diagnostic SVS slides."
    )
    path_files = download_gdc_svs_sample(
        cancer=cancer,
        project_id=cfg["pathology"]["project_id"],
        dataset_name=cfg["pathology"]["dataset_name"],
        license_str=cfg["pathology"]["license"],
        dest_dir=pathology_dir,
        limit=target_total,
    )
    downloaded.extend(path_files)

    return downloaded
