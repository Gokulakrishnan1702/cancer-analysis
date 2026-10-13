"""
Configuration module for the Medical Image Download Pipeline.
Contains official endpoints, license terms, dataset metadata, and output configurations.
"""

from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MANIFEST_PATH = BASE_DIR / "manifest.csv"
LOG_PATH = BASE_DIR / "download_log.txt"
SUMMARY_PATH = BASE_DIR / "dataset_availability_summary.csv"

# API Endpoints
TCIA_API_BASE = "https://services.cancerimagingarchive.net/nbia-api/services/v1"
GDC_API_BASE = "https://api.gdc.cancer.gov"
ZENODO_API_BASE = "https://zenodo.org/api"

# Default download limits
TARGET_SAMPLES_PER_CANCER = 5
MAX_SVS_SIZE_BYTES = 50 * 1024 * 1024  # 50 MB limit for full SVS slides in small sample
REQUEST_TIMEOUT_SECONDS = 45

# Supported Cancer Types
CANCER_TYPES = [
    "lung",
    "breast",
    "colorectal",
    "melanoma",
    "ovarian",
    "prostate",
    "pancreatic",
    "renal",
    "gastric",
    "endometrial",
    "bladder",
    "esophageal",
    "thyroid",
    "lymphoma",
    "liver",
]

# Source Definitions
SOURCES = {
    "lung": {
        "ct": {
            "source": "TCIA",
            "dataset_name": "LIDC-IDRI",
            "collection": "LIDC-IDRI",
            "url": "https://www.cancerimagingarchive.net/collection/lidc-idri/",
            "license": "CC BY 3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "auth_required": False,
            "label_type": "Screening / Indeterminate Pulmonary Nodules (Malignancy 1-5)",
            "notes": "LIDC-IDRI contains lung CT nodule studies; not an automatic confirmed cancer cohort.",
        },
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-LUAD / TCGA-LUSC",
            "project_id": "TCGA-LUAD",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-LUAD",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed primary lung adenocarcinoma",
            "notes": "Open-access diagnostic and tissue whole-slide images (.svs).",
        },
    },
    "breast": {
        "mri": {
            "source": "TCIA",
            "dataset_name": "TCGA-BRCA",
            "collection": "TCGA-BRCA",
            "url": "https://www.cancerimagingarchive.net/collection/tcga-brca/",
            "license": "CC BY 3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "auth_required": False,
            "label_type": "Confirmed breast invasive carcinoma (MRI DCE/T1/T2)",
            "notes": "Verified MR modality availability on TCIA (1,858 series).",
        },
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-BRCA",
            "project_id": "TCGA-BRCA",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-BRCA",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed breast invasive carcinoma",
            "notes": "Open-access diagnostic whole-slide images (.svs).",
        },
        "pathology_bach": {
            "source": "Grand Challenge",
            "dataset_name": "BACH (ICIAR 2018)",
            "url": "https://iciar2018-challenge.grand-challenge.org/",
            "license": "CC BY-NC-ND",
            "auth_required": True,
            "auth_type": "Registration credentials via email",
            "notes": "Requires Grand Challenge account registration.",
        },
        "pathology_camelyon16": {
            "source": "Grand Challenge",
            "dataset_name": "CAMELYON16",
            "url": "https://camelyon16.grand-challenge.org/",
            "license": "CC BY-NC-ND 4.0",
            "auth_required": True,
            "auth_type": "Grand Challenge account and cloud mirror access",
            "notes": "Requires account and cloud mirror download.",
        },
    },
    "colorectal": {
        "ct": {
            "source": "TCIA",
            "dataset_name": "CT COLONOGRAPHY",
            "collection": "CT COLONOGRAPHY",
            "url": "https://www.cancerimagingarchive.net/collection/ct-colonography/",
            "license": "CC BY 3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "auth_required": False,
            "label_type": "Screening trial (mostly benign polyps or normal mucosa)",
            "notes": "ACRIN 6664 virtual colonoscopy screening trial.",
        },
        "pathology": {
            "source": "Zenodo",
            "dataset_name": "NCT-CRC-HE-100K",
            "zenodo_record": "1214456",
            "url": "https://doi.org/10.5281/zenodo.1214456",
            "license": "CC BY 4.0",
            "license_url": "https://creativecommons.org/licenses/by/4.0/",
            "auth_required": False,
            "label_type": "Confirmed colorectal adenocarcinoma vs normal colon mucosa (tiles)",
            "notes": "Standard 224x224 histological tiles; also backed by TCGA-COAD on GDC.",
        },
    },
    "melanoma": {
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-SKCM",
            "project_id": "TCGA-SKCM",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-SKCM",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed skin cutaneous melanoma",
            "notes": "Open-access diagnostic SVS slides; pathology only.",
        },
    },
    "ovarian": {
        "ct": {
            "source": "TCIA",
            "dataset_name": "TCGA-OV",
            "collection": "TCGA-OV",
            "url": "https://www.cancerimagingarchive.net/collection/tcga-ov/",
            "license": "CC BY 3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "auth_required": False,
            "label_type": "Confirmed ovarian serous cystadenocarcinoma (CT/MRI)",
            "notes": "Verified CT/MR radiology on TCIA; note CPTAC-OV is Slide Microscopy only.",
        },
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-OV",
            "project_id": "TCGA-OV",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-OV",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed ovarian serous cystadenocarcinoma",
            "notes": "Open-access diagnostic SVS slides.",
        },
    },
    "prostate": {
        "mri": {
            "source": "TCIA",
            "dataset_name": "PROSTATEx",
            "collection": "PROSTATEx",
            "url": "https://www.cancerimagingarchive.net/collection/prostatex/",
            "license": "CC BY 3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "auth_required": False,
            "label_type": "Confirmed lesions with clinical significance labels (T2W/ADC/DCE)",
            "notes": "Multi-parametric MRI studies with biopsy ground truth.",
        },
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-PRAD",
            "project_id": "TCGA-PRAD",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-PRAD",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed prostate adenocarcinoma (Gleason score)",
            "notes": "Open-access diagnostic SVS slides.",
        },
        "pathology_panda": {
            "source": "Kaggle",
            "dataset_name": "PANDA Challenge",
            "url": "https://www.kaggle.com/c/prostate-cancer-grade-assessment",
            "license": "CC BY-SA-NC 4.0",
            "auth_required": True,
            "auth_type": "Kaggle API token (kaggle.json) and competition rules acceptance",
            "notes": "Requires Kaggle account credentials.",
        },
    },
    "pancreatic": {
        "ct": {
            "source": "TCIA",
            "dataset_name": "Pancreas-CT",
            "collection": "Pancreas-CT",
            "url": "https://www.cancerimagingarchive.net/collection/pancreas-ct/",
            "license": "CC BY 3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "auth_required": False,
            "label_type": "Normal pancreas anatomy / Healthy kidney donors",
            "notes": "Pancreas-CT does NOT contain pancreatic cancer lesions.",
        },
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-PAAD",
            "project_id": "TCGA-PAAD",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-PAAD",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed pancreatic ductal adenocarcinoma",
            "notes": "Open-access diagnostic SVS slides.",
        },
    },
    "renal": {
        "ct": {
            "source": "TCIA",
            "dataset_name": "C4KC-KiTS (KiTS19)",
            "collection": "C4KC-KiTS",
            "url": "https://www.cancerimagingarchive.net/collection/c4kc-kits/",
            "license": "CC BY 3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "auth_required": False,
            "label_type": "Confirmed nephrectomy renal tumors with segmentation masks",
            "notes": "KiTS19 challenge CT series hosted on TCIA.",
        },
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-KIRC",
            "project_id": "TCGA-KIRC",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-KIRC",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed kidney clear cell renal carcinoma",
            "notes": "Open-access diagnostic SVS slides.",
        },
    },
    "gastric": {
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-STAD",
            "project_id": "TCGA-STAD",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-STAD",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed stomach adenocarcinoma",
            "notes": "Open-access diagnostic SVS slides; pathology only.",
        },
    },
    "endometrial": {
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-UCEC",
            "project_id": "TCGA-UCEC",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-UCEC",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed uterine corpus endometrial carcinoma",
            "notes": "Open-access diagnostic SVS slides; pathology only.",
        },
    },
    "bladder": {
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-BLCA",
            "project_id": "TCGA-BLCA",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-BLCA",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed bladder urothelial carcinoma",
            "notes": "Open-access diagnostic SVS slides; pathology only.",
        },
    },
    "esophageal": {
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-ESCA",
            "project_id": "TCGA-ESCA",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-ESCA",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed esophageal carcinoma",
            "notes": "Open-access diagnostic SVS slides; pathology only.",
        },
    },
    "thyroid": {
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-THCA",
            "project_id": "TCGA-THCA",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-THCA",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed papillary thyroid carcinoma",
            "notes": "Open-access diagnostic SVS slides; pathology only.",
        },
    },
    "lymphoma": {
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-DLBC",
            "project_id": "TCGA-DLBC",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-DLBC",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed diffuse large B-cell lymphoma",
            "notes": "Open-access diagnostic SVS slides; verified lymphoma public dataset.",
        },
        "pathology_lynsec": {
            "source": "Zenodo",
            "dataset_name": "LyNSeC",
            "url": "https://doi.org/10.5281/zenodo.8064560",
            "license": "CC BY 4.0",
            "auth_required": False,
            "label_type": "Confirmed lymphoma subtypes (DLBCL, FL, MCL)",
            "notes": "Microscopy patches for nuclear classification.",
        },
    },
    "liver": {
        "ct": {
            "source": "TCIA",
            "dataset_name": "TCGA-LIHC",
            "collection": "TCGA-LIHC",
            "url": "https://www.cancerimagingarchive.net/collection/tcga-lihc/",
            "license": "CC BY 3.0",
            "license_url": "https://creativecommons.org/licenses/by/3.0/",
            "auth_required": False,
            "label_type": "Confirmed liver hepatocellular carcinoma (CT/MR)",
            "notes": "Multiphase liver CT and MR series hosted on TCIA.",
        },
        "pathology": {
            "source": "GDC",
            "dataset_name": "TCGA-LIHC",
            "project_id": "TCGA-LIHC",
            "url": "https://portal.gdc.cancer.gov/projects/TCGA-LIHC",
            "license": "CC0",
            "license_url": "https://gdc.cancer.gov/about-data/data-analysis-policies",
            "auth_required": False,
            "label_type": "Confirmed liver hepatocellular carcinoma",
            "notes": "Open-access diagnostic SVS slides.",
        },
        "ct_lits": {
            "source": "Grand Challenge / CodaLab",
            "dataset_name": "LiTS (Liver Tumor Segmentation)",
            "url": "https://competitions.codalab.org/competitions/17094",
            "license": "CC BY-NC-SA 4.0",
            "auth_required": True,
            "auth_type": "CodaLab registration or Academic Torrents",
            "notes": "Large 3D volume benchmarks; requires account.",
        },
    },
}
