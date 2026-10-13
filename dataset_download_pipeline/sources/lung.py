"""
Lung Cancer Download Module.
Sources:
- CT: LIDC-IDRI (TCIA) - Screening / indeterminate nodules
- Pathology: TCGA-LUAD (GDC) - Confirmed lung adenocarcinoma
"""

from pathlib import Path
from config import DATA_DIR, SOURCES
from logger import pipeline_logger
from sources.common import download_gdc_svs_sample, download_tcia_dicom_sample

def download_lung(target_total: int = 5):
    cancer = "lung"
    ct_dir = DATA_DIR / cancer / "ct"
    pathology_dir = DATA_DIR / cancer / "pathology"

    cfg = SOURCES[cancer]
    downloaded = []

    # 1. CT from LIDC-IDRI (TCIA)
    pipeline_logger.log_event(
        cancer=cancer,
        modality="ct",
        source="TCIA",
        identifier="LIDC-IDRI",
        action="Initiate CT download",
        status="IN_PROGRESS",
        extra_notes="LIDC-IDRI thoracic nodule studies (malignancy rated 1-5; indeterminate screening/diagnostic nodules)."
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

    # 2. Pathology from TCGA-LUAD (GDC)
    pipeline_logger.log_event(
        cancer=cancer,
        modality="pathology",
        source="GDC",
        identifier="TCGA-LUAD",
        action="Initiate pathology download",
        status="IN_PROGRESS",
        extra_notes="Confirmed lung adenocarcinoma diagnostic whole-slide images (.svs)."
    )
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
