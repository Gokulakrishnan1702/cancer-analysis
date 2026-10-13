"""
Ovarian Cancer Download Module.
Sources:
- CT/MRI: TCGA-OV (TCIA) - Confirmed ovarian serous cystadenocarcinoma
- Pathology: TCGA-OV (GDC) - Confirmed ovarian serous cystadenocarcinoma SVS slides
- Note: CPTAC-OV on TCIA contains Slide Microscopy (SM) histopathology, not radiology.
"""

from pathlib import Path
from config import DATA_DIR, SOURCES
from logger import pipeline_logger
from sources.common import download_gdc_svs_sample, download_tcia_dicom_sample

def download_ovarian(target_total: int = 5):
    cancer = "ovarian"
    ct_dir = DATA_DIR / cancer / "ct"
    pathology_dir = DATA_DIR / cancer / "pathology"

    cfg = SOURCES[cancer]
    downloaded = []

    # 1. CT from TCGA-OV (TCIA)
    pipeline_logger.log_event(
        cancer=cancer,
        modality="ct",
        source="TCIA",
        identifier="TCGA-OV",
        action="Initiate CT download",
        status="IN_PROGRESS",
        extra_notes="TCGA-OV verified CT radiology on TCIA (note: CPTAC-OV is Slide Microscopy, not CT)."
    )
    ct_files = download_tcia_dicom_sample(
        cancer=cancer,
        collection=cfg["ct"]["collection"],
        modality="CT",
        dataset_name=cfg["ct"]["dataset_name"],
        license_str=cfg["ct"]["license"],
        dest_dir=ct_dir,
        limit=3,
        label_info=cfg["ct"]["label_type"],
    )
    downloaded.extend(ct_files)

    # 2. Pathology from TCGA-OV (GDC)
    path_limit = max(2, target_total - len(downloaded))
    path_files = download_gdc_svs_sample(
        cancer=cancer,
        project_id=cfg["pathology"]["project_id"],
        dataset_name=cfg["pathology"]["dataset_name"],
        license_str=cfg["pathology"]["license"],
        dest_dir=pathology_dir,
        limit=path_limit,
    )
    downloaded.extend(path_files)

    return downloaded
