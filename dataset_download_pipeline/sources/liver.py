"""
Liver Cancer Download Module.
Sources:
- CT/MRI: TCGA-LIHC (TCIA) - Confirmed liver hepatocellular carcinoma
- Pathology: TCGA-LIHC (GDC) - Confirmed liver hepatocellular carcinoma SVS slides
- Optional: LiTS (Grand Challenge / CodaLab) - Authentication handling
"""

import os
from pathlib import Path
from config import DATA_DIR, SOURCES
from logger import pipeline_logger
from sources.common import download_gdc_svs_sample, download_tcia_dicom_sample, record_auth_required_source

def download_liver(target_total: int = 5):
    cancer = "liver"
    ct_dir = DATA_DIR / cancer / "ct"
    pathology_dir = DATA_DIR / cancer / "pathology"

    cfg = SOURCES[cancer]
    downloaded = []

    # 1. CT from TCGA-LIHC (TCIA)
    pipeline_logger.log_event(
        cancer=cancer,
        modality="ct",
        source="TCIA",
        identifier="TCGA-LIHC",
        action="Initiate CT download",
        status="IN_PROGRESS",
        extra_notes="TCGA-LIHC multiphase liver CT series on TCIA."
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

    # 2. Pathology from TCGA-LIHC (GDC)
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

    # 3. Check / Record authentication-required candidate source (LiTS Challenge)
    if not os.environ.get("CODALAB_API_TOKEN"):
        record_auth_required_source(
            cancer=cancer,
            modality="ct",
            dataset_name=cfg["ct_lits"]["dataset_name"],
            source=cfg["ct_lits"]["source"],
            setup_instructions="Register at https://competitions.codalab.org/competitions/17094 to access LiTS NIfTI files.",
            required_credential="CODALAB_API_TOKEN or competition credentials",
            license_str=cfg["ct_lits"]["license"],
        )

    return downloaded
