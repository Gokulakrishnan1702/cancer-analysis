# Modular Medical Image Download Pipeline for 15 Cancer Types

This package provides a robust, modular, and authenticated medical imaging and histopathology download pipeline for 15 cancer types. It interfaces directly with official public APIs (**The Cancer Imaging Archive / NBIA API** and **NCI Genomic Data Commons API**), enforces strict binary file signature and checksum verification, maintains truthful audit trails in `manifest.csv` and `download_log.txt`, and gracefully handles authentication barriers without silent failures.

---

## 1. Verified Dataset Sources & Licensing

| # | Cancer | Verified Datasets | Modality | Format | License | Access Terms |
|---|---|---|---|---|---|---|
| 1 | **Lung** | LIDC-IDRI (TCIA); TCGA-LUAD (GDC) | CT; Pathology | DICOM (`.dcm`), SVS (`.svs`) | CC BY 3.0 / CC0 | Direct Open Access |
| 2 | **Breast** | TCGA-BRCA (TCIA MRI); TCGA-BRCA (GDC) | MRI; Pathology | DICOM (`.dcm`), SVS (`.svs`) | CC BY 3.0 / CC0 | Direct Open Access |
| 3 | **Colorectal** | CT COLONOGRAPHY (TCIA); TCGA-COAD (GDC) | CT; Pathology | DICOM (`.dcm`), SVS (`.svs`) | CC BY 3.0 / CC0 | Direct Open Access |
| 4 | **Melanoma** | TCGA-SKCM (GDC) | Pathology | Aperio SVS (`.svs`) | CC0 | Direct Open Access |
| 5 | **Ovarian** | TCGA-OV (TCIA); TCGA-OV (GDC) | CT/MRI; Pathology | DICOM (`.dcm`), SVS (`.svs`) | CC BY 3.0 / CC0 | Direct Open Access |
| 6 | **Prostate** | PROSTATEx (TCIA); TCGA-PRAD (GDC) | mpMRI; Pathology | DICOM (`.dcm`), SVS (`.svs`) | CC BY 3.0 / CC0 | Direct Open Access |
| 7 | **Pancreatic** | Pancreas-CT (TCIA); TCGA-PAAD (GDC) | CT; Pathology | DICOM (`.dcm`), SVS (`.svs`) | CC BY 3.0 / CC0 | Direct Open Access |
| 8 | **Renal** | C4KC-KiTS (TCIA); TCGA-KIRC (GDC) | CT; Pathology | DICOM (`.dcm`), SVS (`.svs`) | CC BY 3.0 / CC0 | Direct Open Access |
| 9 | **Gastric** | TCGA-STAD (GDC) | Pathology | Aperio SVS (`.svs`) | CC0 | Direct Open Access |
| 10 | **Endometrial** | TCGA-UCEC (GDC) | Pathology | Aperio SVS (`.svs`) | CC0 | Direct Open Access |
| 11 | **Bladder** | TCGA-BLCA (GDC) | Pathology | Aperio SVS (`.svs`) | CC0 | Direct Open Access |
| 12 | **Esophageal** | TCGA-ESCA (GDC) | Pathology | Aperio SVS (`.svs`) | CC0 | Direct Open Access |
| 13 | **Thyroid** | TCGA-THCA (GDC) | Pathology | Aperio SVS (`.svs`) | CC0 | Direct Open Access |
| 14 | **Lymphoma** | TCGA-DLBC (GDC); LyNSeC (Zenodo) | Pathology | Aperio SVS (`.svs`), PNG | CC0 / CC BY 4.0 | Direct Open Access |
| 15 | **Liver** | TCGA-LIHC (TCIA); TCGA-LIHC (GDC) | CT/MRI; Pathology | DICOM (`.dcm`), SVS (`.svs`) | CC BY 3.0 / CC0 | Direct Open Access |

---

## 2. Directory Layout & Output Structure

Downloaded files are saved in strict accordance with the oncology modality hierarchy:
```text
data/
  ├── lung/
  │   ├── ct/            # DICOM (.dcm) files
  │   └── pathology/     # Aperio SVS (.svs) files
  ├── breast/
  │   ├── mri/           # DICOM (.dcm) files
  │   └── pathology/     # Aperio SVS (.svs) files
  ├── colorectal/
  │   ├── ct/            # DICOM (.dcm) files
  │   └── pathology/     # Aperio SVS (.svs) files
  ├── melanoma/
  │   └── pathology/     # Aperio SVS (.svs) files (Pathology only)
  ...
```

Project root logs and records:
* `manifest.csv`: Machine-readable tracking table with columns:
  `cancer, modality, file_path, source, dataset_name, license, download_status, notes`
* `download_log.txt`: Timestamped event audit log capturing retries, errors, and missing modalities.
* `dataset_availability_summary.csv`: Aggregated matrix of verified modalities and file counts.

---

## 3. Installation & Dependencies

Requires Python 3.9+:
```bash
cd dataset_download_pipeline
pip install -r requirements.txt
```

---

## 4. Usage Commands

### List All Verified Sources & Terms
```bash
python main.py --list-sources
```

### Download a Small Sample for a Single Cancer Type
```bash
python main.py --cancer lung
python main.py --cancer breast
python main.py --cancer prostate
```

### Download Small Samples Across All 15 Cancers
```bash
python main.py --all
```

### Check Current Status & Counts
```bash
python main.py --status
```

### Re-attempt Failed Downloads Only
```bash
python main.py --failed-only
```

---

## 5. Authentication Handling for Restricted Datasets

If you wish to download from sources requiring authentication (e.g. **PANDA Challenge**, **BACH**, **CAMELYON16**, **LiTS**):
1. **PANDA Challenge (Kaggle)**:
   * Accept the rules on [Kaggle PANDA](https://www.kaggle.com/c/prostate-cancer-grade-assessment).
   * Save your API token to `~/.kaggle/kaggle.json` or export `KAGGLE_USERNAME` and `KAGGLE_KEY`.
2. **BACH (Grand Challenge)**:
   * Register on the [BACH Challenge Portal](https://iciar2018-challenge.grand-challenge.org/) and obtain access tokens.
3. **LiTS (CodaLab)**:
   * Register at [CodaLab LiTS](https://competitions.codalab.org/competitions/17094) and set `CODALAB_API_TOKEN`.

*Note: The pipeline never prompts for secrets in chat or stores credentials in code.*
