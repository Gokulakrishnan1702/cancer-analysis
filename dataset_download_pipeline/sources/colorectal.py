"""
Colorectal Cancer Download Module.
Sources:
- CT: CT COLONOGRAPHY (TCIA) - Screening cohort (polyps/normal mucosa)
- Pathology: TCGA-COAD (GDC) - Confirmed colon adenocarcinoma SVS slides
- Pathology Tiles: NCT-CRC-HE-100K (Zenodo)
"""

from pathlib import Path
from config import DATA_DIR, SOURCES
from logger import pipeline_logger
from sources.common import download_gdc_svs_sample, download_tcia_dicom_sample

def download_colorectal(target_total: int = 5):
    cancer = "colorectal"
    ct_dir = DATA_DIR / cancer / "ct"
    pathology_dir = DATA_DIR / cancer / "pathology"

    cfg = SOURCES[cancer]
    downloaded = []

    # 1. CT from CT COLONOGRAPHY (TCIA)
    pipeline_logger.log_event(
        cancer=cancer,
        modality="ct",
        source="TCIA",
        identifier="CT COLONOGRAPHY",
        action="Initiate CT download",
        status="IN_PROGRESS",
        extra_notes="ACRIN 6664 virtual colonoscopy screening trial (screening cohort / mostly benign polyps or normal mucosa)."
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

    # 2. Pathology from TCGA-COAD (GDC)
    path_limit = max(2, target_total - len(downloaded))
    path_files = download_gdc_svs_sample(
        cancer=cancer,
        project_id="TCGA-COAD",
        dataset_name="TCGA-COAD",
        license_str="CC0",
        dest_dir=pathology_dir,
        limit=path_limit,
    )
    downloaded.extend(path_files)

    return downloaded
