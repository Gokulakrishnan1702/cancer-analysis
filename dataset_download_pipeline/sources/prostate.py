"""
Prostate Cancer Download Module.
Sources:
- MRI: PROSTATEx (TCIA) - Multi-parametric MRI (T2W, ADC, DCE) with lesion labels
- Pathology: TCGA-PRAD (GDC) - Confirmed prostate adenocarcinoma SVS slides
- Optional: PANDA (Kaggle) - Authentication handling for Kaggle competition
"""

import os
from pathlib import Path
from config import DATA_DIR, SOURCES
from logger import pipeline_logger
from sources.common import download_gdc_svs_sample, download_tcia_dicom_sample, record_auth_required_source

def download_prostate(target_total: int = 5):
    cancer = "prostate"
    mri_dir = DATA_DIR / cancer / "mri"
    pathology_dir = DATA_DIR / cancer / "pathology"

    cfg = SOURCES[cancer]
    downloaded = []

    # 1. MRI from PROSTATEx (TCIA)
    pipeline_logger.log_event(
        cancer=cancer,
        modality="mri",
        source="TCIA",
        identifier="PROSTATEx",
        action="Initiate MRI download",
        status="IN_PROGRESS",
        extra_notes="SPIE-AAPM-NCI PROSTATEx multiparametric MRI (T2W/ADC/DCE) with lesion significance ground truth."
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

    # 2. Pathology from TCGA-PRAD (GDC)
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

    # 3. Check / Record authentication-required candidate source (PANDA Challenge)
    kaggle_creds_exist = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    kaggle_file = Path.home() / ".kaggle" / "kaggle.json"
    if not (kaggle_creds_exist or kaggle_file.exists()):
        record_auth_required_source(
            cancer=cancer,
            modality="pathology",
            dataset_name=cfg["pathology_panda"]["dataset_name"],
            source=cfg["pathology_panda"]["source"],
            setup_instructions=(
                "1. Create account at https://www.kaggle.com\n"
                "2. Accept rules at https://www.kaggle.com/c/prostate-cancer-grade-assessment\n"
                "3. Place kaggle.json in ~/.kaggle/kaggle.json or set KAGGLE_USERNAME & KAGGLE_KEY env vars."
            ),
            required_credential="~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY",
            license_str=cfg["pathology_panda"]["license"],
        )

    return downloaded
