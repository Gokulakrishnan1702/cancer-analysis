"""
Breast Cancer Download Module.
Sources:
- MRI: TCGA-BRCA (TCIA) - Confirmed breast invasive carcinoma MRI
- Pathology: TCGA-BRCA (GDC) - Confirmed breast invasive carcinoma SVS slides
- Optional: BACH / CAMELYON16 (Grand Challenge) - Authentication handling
"""

import os
from pathlib import Path
from config import DATA_DIR, SOURCES
from logger import pipeline_logger
from sources.common import download_gdc_svs_sample, download_tcia_dicom_sample, record_auth_required_source

def download_breast(target_total: int = 5):
    cancer = "breast"
    mri_dir = DATA_DIR / cancer / "mri"
    pathology_dir = DATA_DIR / cancer / "pathology"

    cfg = SOURCES[cancer]
    downloaded = []

    # 1. MRI from TCGA-BRCA (TCIA)
    pipeline_logger.log_event(
        cancer=cancer,
        modality="mri",
        source="TCIA",
        identifier="TCGA-BRCA",
        action="Initiate MRI download",
        status="IN_PROGRESS",
        extra_notes="Verified MR modality series (DCE/T1/T2) on TCIA."
    )
    mri_files = download_tcia_dicom_sample(
        cancer=cancer,
        collection=cfg["mri"]["collection"],
        modality="MR",
        dataset_name=cfg["mri"]["dataset_name"],
        license_str=cfg["mri"]["license"],
        dest_dir=mri_dir,
        limit=3,
        label_info=cfg["mri"]["label_type"],
    )
    downloaded.extend(mri_files)

    # 2. Pathology from TCGA-BRCA (GDC)
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

    # 3. Check / Record authentication-required candidate sources (BACH, CAMELYON16)
    if not os.environ.get("GRAND_CHALLENGE_API_TOKEN"):
        record_auth_required_source(
            cancer=cancer,
            modality="pathology",
            dataset_name=cfg["pathology_bach"]["dataset_name"],
            source=cfg["pathology_bach"]["source"],
            setup_instructions="Register at https://iciar2018-challenge.grand-challenge.org/ to receive access credentials.",
            required_credential="GRAND_CHALLENGE_API_TOKEN or challenge credentials",
            license_str=cfg["pathology_bach"]["license"],
        )
        record_auth_required_source(
            cancer=cancer,
            modality="pathology",
            dataset_name=cfg["pathology_camelyon16"]["dataset_name"],
            source=cfg["pathology_camelyon16"]["source"],
            setup_instructions="Create account at https://camelyon16.grand-challenge.org/ and request cloud mirror link.",
            required_credential="GRAND_CHALLENGE_ACCOUNT or cloud mirror credentials",
            license_str=cfg["pathology_camelyon16"]["license"],
        )

    return downloaded
