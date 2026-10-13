"""
Image Validation & Security Service for Oncology CDSS
======================================================
Provides server-side medical image validation:
  1. File extension & MIME type checks (.png, .jpg, .jpeg)
  2. Magic bytes and file integrity verification
  3. Image decoding & corruption checks
  4. Dimensionality and minimum quality standards
  5. Modality consistency checking (Pathology vs CT Scan vs MRI)
  6. Model compatibility verification (Stage 2 DL supports CT Scan & Pathology)
"""

import os
import io
import uuid
import logging
from typing import Dict, Any, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger("image_validator")

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".dcm", ".tif", ".tiff"}
MAX_FILE_SIZE_BYTES = 30 * 1024 * 1024  # 30 MB
MIN_DIMENSION = 64
SUPPORTED_ANALYSIS_MODALITIES = {"CT Scan", "Pathology"}
UNSUPPORTED_ANALYSIS_MODALITIES = {"MRI"}

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
UPLOADS_DIR = os.path.join(WORKSPACE_ROOT, "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


def classify_radiology_modality(gray_arr: np.ndarray, original_filename: str = "") -> Tuple[str, float, str]:
    """
    Distinguishes CT scans (Computed Tomography / Radiographs) from MRI scans (Magnetic Resonance Imaging).
    Uses radiological markers:
      1. Bone attenuation: CT shows bright hyperdense bone cortex/calcification (>215 HU equivalent)
         with sharp edge gradients. In contrast, cortical bone in MRI is a dark signal void (<25).
      2. Thoracic / pulmonary patterns: Inverted lung windows, rib cages, and multi-panel axial montages.
      3. Background characteristics: MRI scans feature broad black non-anatomical background with
         soft-tissue signal distribution in parenchyma.
      4. Filename keywords: 'ct', 'axial_ct', 'cat_scan', 'hrct' vs 'mri', 't1', 't2', 'flair', 'dwi'.
    """
    fname = (original_filename or "").lower()

    # 1. Filename explicit overrides
    ct_keywords = ["ct_slice", "chest_ct", "axial_ct", "cat_scan", "computed_tomography", "_ct.", "-ct.", "ct_", "_ct_"]
    mri_keywords = ["mri", "t1_weighted", "t2_weighted", "flair", "dwi", "adc", "brain_mri", "_mr.", "-mr."]

    for kw in ct_keywords:
        if kw in fname:
            return "CT Scan", 0.99, "Filename specifies Computed Tomography (CT)"
    for kw in mri_keywords:
        if kw in fname:
            return "MRI", 0.99, "Filename specifies Magnetic Resonance Imaging (MRI)"

    # 2. Pixel & Radiological feature extraction
    arr_f = gray_arr.astype(float)
    mean_val = float(np.mean(arr_f))
    high_frac = float(np.mean(arr_f > 215))       # Hyperdense bone cortex / calcification / inverted lung
    very_high_frac = float(np.mean(arr_f > 235))
    dark_frac = float(np.mean(arr_f < 25))        # Air cavity or black background

    # Edge detection for bone-cortex boundaries
    try:
        from scipy import ndimage
        sx = ndimage.sobel(arr_f, axis=0)
        sy = ndimage.sobel(arr_f, axis=1)
        edge_mag = np.hypot(sx, sy)
        high_mask = arr_f > 200
        high_edge_mean = float(np.mean(edge_mag[high_mask])) if high_mask.any() else 0.0
    except Exception:
        diff_y = np.abs(np.diff(arr_f, axis=0))
        diff_x = np.abs(np.diff(arr_f, axis=1))
        high_mask = arr_f[:-1, :-1] > 200
        high_edge_mean = float(np.mean(diff_x[:-1, :] + diff_y[:, :-1])) if high_mask.any() else 0.0

    # Grid / multi-panel layout cuts (common in clinical CT journal figures/montages)
    row_diff = np.abs(np.diff(arr_f.mean(axis=1)))
    col_diff = np.abs(np.diff(arr_f.mean(axis=0)))
    grid_cuts = int(np.sum(row_diff > 12) + np.sum(col_diff > 12))

    ct_score = 0.0
    mri_score = 0.0
    reasons = []

    # Evidence for CT Scan:
    if mean_val > 130.0:
        ct_score += 4.0
        reasons.append("High mean radiodensity characteristic of chest radiography/CT lung window")

    if high_frac > 0.08 or very_high_frac > 0.04:
        ct_score += 3.5
        reasons.append("Prominent hyperdense skeletal/calcified attenuation (>215 HU equivalent)")

    if grid_cuts >= 3:
        ct_score += 2.5
        reasons.append("Multi-panel axial CT scan/radiograph montage layout")

    if high_edge_mean > 20.0:
        ct_score += 2.0
        reasons.append("Sharp bone-cortex transition gradients characteristic of CT attenuation")

    # Evidence for MRI Scan:
    if dark_frac > 0.25 and mean_val < 115.0 and high_frac < 0.04:
        mri_score += 4.0
        reasons.append("Dark non-anatomical background with soft-tissue contrast and cortical bone signal void")

    if ct_score > mri_score:
        conf = min(0.99, 0.65 + ct_score * 0.06)
        return "CT Scan", conf, "; ".join(reasons)
    elif mri_score > ct_score:
        conf = min(0.99, 0.65 + mri_score * 0.06)
        return "MRI", conf, "; ".join(reasons)
    else:
        if high_frac > 0.05 or mean_val > 120.0:
            return "CT Scan", 0.70, "Radiographic density pattern consistent with CT"
        return "Indeterminate", 0.5, "Equivocal radiology features"


def validate_and_save_medical_image(
    file_bytes: bytes,
    original_filename: str,
    image_type: str
) -> Dict[str, Any]:
    """
    Validates uploaded medical image bytes according to clinical CDSS requirements.
    Saves the file securely if valid.
    """
    try:
        # 1. Image presence
        if not file_bytes or len(file_bytes) == 0:
            return {
                "valid": False,
                "error": "Please upload an image.",
                "message": "Please upload an image."
            }

        # 2. Image Type selection check
        if not image_type or image_type.strip() not in ["MRI", "CT Scan", "Pathology"]:
            return {
                "valid": False,
                "error": "Please select an image type (MRI, CT Scan, or Pathology).",
                "message": "Please select an image type before uploading or validating."
            }

        image_type = image_type.strip()

        # 3. File size limit
        if len(file_bytes) > MAX_FILE_SIZE_BYTES:
            return {
                "valid": False,
                "error": "File size exceeds the 15MB limit. Please upload a smaller image.",
                "message": "File size exceeds the 15MB limit."
            }

        # 4. File extension check
        ext = os.path.splitext(original_filename or "")[1].lower()
        if ext not in ALLOWED_EXTENSIONS:
            return {
                "valid": False,
                "error": "Unsupported file format. Please upload PNG, JPG, or JPEG.",
                "message": "Unsupported file format. Please upload PNG, JPG, or JPEG."
            }

        # 5. Magic bytes & Format Identification
        is_png = file_bytes.startswith(b"\x89PNG\r\n\x1a\n")
        is_jpg = file_bytes.startswith(b"\xff\xd8")
        is_tiff = file_bytes.startswith(b"II*\x00") or file_bytes.startswith(b"MM\x00*")
        is_dicom = (len(file_bytes) > 132 and file_bytes[128:132] == b"DICM") or ext == ".dcm"

        if not (is_png or is_jpg or is_tiff or is_dicom):
            return {
                "valid": False,
                "error": "Unsupported file format. Please upload PNG, JPG, or JPEG.",
                "message": "Unsupported file format. Please upload PNG, JPG, or JPEG."
            }

        dicom_metadata = None
        img = None

        # 6. Reading & Image Decoding
        if is_dicom:
            try:
                import pydicom
                ds = pydicom.dcmread(io.BytesIO(file_bytes), force=True)
                if hasattr(ds, "pixel_array") and ds.pixel_array is not None:
                    pix = ds.pixel_array.astype(np.float32)
                    p_min, p_max = float(pix.min()), float(pix.max())
                    if p_max > p_min:
                        pix_norm = ((pix - p_min) / (p_max - p_min) * 255.0).astype(np.uint8)
                    else:
                        pix_norm = np.zeros_like(pix, dtype=np.uint8)
                    
                    if pix_norm.ndim == 2:
                        img = Image.fromarray(pix_norm, mode="L")
                    elif pix_norm.ndim == 3:
                        img = Image.fromarray(pix_norm)
                    else:
                        raise ValueError(f"Unsupported DICOM pixel dimension: {pix_norm.ndim}")
                    
                    dicom_metadata = {
                        "modality": str(getattr(ds, "Modality", "UNKNOWN")).upper(),
                        "patient_id": str(getattr(ds, "PatientID", "ANONYMOUS")),
                        "study_description": str(getattr(ds, "StudyDescription", "N/A")),
                        "series_description": str(getattr(ds, "SeriesDescription", "N/A")),
                        "body_part": str(getattr(ds, "BodyPartExamined", "N/A")),
                        "rows": int(getattr(ds, "Rows", 0)),
                        "columns": int(getattr(ds, "Columns", 0)),
                    }
                else:
                    return {
                        "valid": False,
                        "error": "This DICOM file contains no viewable pixel matrix.",
                        "message": "This image cannot be opened. Please upload another image."
                    }
            except Exception as e:
                logger.warning(f"DICOM parsing error: {e}")
                return {
                    "valid": False,
                    "error": "This image cannot be opened. Please upload another image.",
                    "message": "This image cannot be opened. Please upload another image."
                }
        else:
            try:
                img_verify = Image.open(io.BytesIO(file_bytes))
                img_verify.verify()

                img = Image.open(io.BytesIO(file_bytes))
                img.load()
            except Exception as e:
                logger.warning(f"Image read error: {e}")
                return {
                    "valid": False,
                    "error": "This image cannot be opened. Please upload another image.",
                    "message": "This image cannot be opened. Please upload another image."
                }

        # 7. Dimensions & Minimum Resolution check
        width, height = img.size
        if width < MIN_DIMENSION or height < MIN_DIMENSION:
            return {
                "valid": False,
                "error": "Image quality is insufficient for analysis. Minimum resolution is 64x64.",
                "message": "Image quality is insufficient for analysis. Please upload a clearer image."
            }

        # 8. Empty / Flat / Low-Quality check
        rgb_img = img.convert("RGB")
        arr = np.array(rgb_img).astype(float)
        pixel_std = float(np.std(arr))
        if pixel_std < 2.0:
            return {
                "valid": False,
                "error": "Image quality is insufficient for analysis. Image appears blank or uninformative.",
                "message": "Image quality is insufficient for analysis. Please upload a clearer image."
            }

        # 9. Modality Verification
        # Calculate color chromatic difference across R, G, B channels
        diff_rg = np.abs(arr[:, :, 0] - arr[:, :, 1])
        diff_rb = np.abs(arr[:, :, 0] - arr[:, :, 2])
        color_diff = float((np.mean(diff_rg) + np.mean(diff_rb)) / 2.0)
        gray_arr = np.mean(arr, axis=2)

        # Modality specific checks
        if dicom_metadata and dicom_metadata.get("modality"):
            d_mod = dicom_metadata["modality"]
            if image_type == "CT Scan":
                if d_mod in ["MR", "MRI"]:
                    return {
                        "valid": False,
                        "error": "Invalid image for CT analysis. Authoritative DICOM header specifies MR (Magnetic Resonance Imaging).",
                        "message": "Invalid image for CT analysis. Authoritative DICOM header specifies MR. Please upload a CT scan or use the MRI slot.",
                        "detected_modality": "MRI",
                        "requested_modality": "CT Scan",
                        "dicom_metadata": dicom_metadata
                    }
                success_msg = f"✓ Valid CT image (DICOM {d_mod} - Patient: {dicom_metadata.get('patient_id')})"
            elif image_type == "MRI":
                if d_mod == "CT":
                    return {
                        "valid": False,
                        "error": "Invalid image for MRI analysis. Authoritative DICOM header specifies CT (Computed Tomography).",
                        "message": "Invalid image for MRI analysis. Authoritative DICOM header specifies CT. Please upload an MRI scan or use the CT Scan slot.",
                        "detected_modality": "CT Scan",
                        "requested_modality": "MRI",
                        "dicom_metadata": dicom_metadata
                    }
                success_msg = f"✓ Valid MRI image (DICOM {d_mod} - Patient: {dicom_metadata.get('patient_id')})"
            elif image_type == "Pathology":
                if d_mod in ["CT", "MR", "MRI"]:
                    return {
                        "valid": False,
                        "error": f"Invalid image for pathology analysis. Uploaded image is a {d_mod} radiological scan.",
                        "message": f"Invalid image for pathology analysis. Uploaded image is a {d_mod} radiological scan.",
                        "detected_modality": d_mod,
                        "requested_modality": "Pathology"
                    }
                success_msg = "✓ Valid Pathology image (DICOM Whole Slide / SM)"
        elif image_type == "Pathology":
            # Pathology images (H&E stains) are color stained with significant chromatic divergence
            if color_diff < 8.0:
                return {
                    "valid": False,
                    "error": "Invalid image for pathology analysis. Please upload a valid pathology image.",
                    "message": "Invalid image for pathology analysis. Please upload a valid pathology image."
                }
            fname_lower = (original_filename or "").lower()
            if "colonca" in fname_lower or "colon_aca" in fname_lower or "colon" in fname_lower:
                success_msg = "✓ Valid Pathology image (Colon Adenocarcinoma H&E Tile)"
            else:
                success_msg = "✓ Valid Pathology image"

        elif image_type == "CT Scan":
            # CT scans are monochrome / grayscale radiology cross-sections
            if color_diff >= 8.0:
                return {
                    "valid": False,
                    "error": "Invalid image for CT analysis. Please upload a valid CT image.",
                    "message": "Invalid image for CT analysis. Please upload a valid CT image."
                }
            detected_modality, confidence, modality_reason = classify_radiology_modality(gray_arr, original_filename)
            if detected_modality == "MRI":
                return {
                    "valid": False,
                    "error": "Invalid image for CT analysis. Detected MRI Scan (Magnetic Resonance Imaging) modality.",
                    "message": f"Invalid image for CT analysis. Detected MRI Scan modality ({modality_reason}). Please upload a valid CT scan or use the MRI Scan slot.",
                    "detected_modality": "MRI",
                    "requested_modality": "CT Scan"
                }
            success_msg = "✓ Valid CT image"

        elif image_type == "MRI":
            # MRI scans are monochrome / grayscale radiology scans
            if color_diff >= 8.0:
                return {
                    "valid": False,
                    "error": "Invalid image for MRI analysis. Please upload a valid MRI image.",
                    "message": "Invalid image for MRI analysis. Please upload a valid MRI image."
                }
            detected_modality, confidence, modality_reason = classify_radiology_modality(gray_arr, original_filename)
            if detected_modality == "CT Scan":
                return {
                    "valid": False,
                    "error": "Invalid image for MRI analysis. This image is a CT scan (Computed Tomography) / radiograph. Please upload a valid MRI scan or use the CT Scan slot.",
                    "message": f"Invalid image for MRI analysis. Detected CT Scan modality ({modality_reason}). Please upload a valid MRI scan or use the CT Scan slot.",
                    "detected_modality": "CT Scan",
                    "requested_modality": "MRI"
                }
            success_msg = "✓ Valid MRI image"

        else:
            return {
                "valid": False,
                "error": "Image modality could not be verified.",
                "message": "Image modality could not be verified."
            }

        # 10. Save validated image safely to uploads/
        # Always save as PNG for reliable browser rendering & CNN ingestion
        save_ext = ".png" if is_dicom or ext in [".tif", ".tiff"] else ext
        safe_filename = f"med_{uuid.uuid4().hex[:12]}_{abs(hash(original_filename)) % 10000}{save_ext}"
        saved_path = os.path.join(UPLOADS_DIR, safe_filename)
        
        # Save decoded PIL image
        if is_dicom or ext in [".tif", ".tiff"]:
            img.save(saved_path, format="PNG")
        else:
            with open(saved_path, "wb") as f:
                f.write(file_bytes)

        # Relative path for frontend serving and pipeline processing
        relative_web_url = f"/api/pipeline/uploaded-image/{safe_filename}"

        result_dict = {
            "valid": True,
            "status": "success",
            "image_type": image_type,
            "filename": os.path.basename(original_filename or safe_filename),
            "file_size_bytes": len(file_bytes),
            "file_size_formatted": f"{len(file_bytes) / 1024:.1f} KB",
            "dimensions": [width, height],
            "format": "DICOM" if is_dicom else (img.format or ext.replace(".", "").upper()),
            "color_mode": img.mode,
            "message": success_msg,
            "saved_filename": safe_filename,
            "saved_path": saved_path,
            "web_url": relative_web_url,
            "model_supported": image_type in SUPPORTED_ANALYSIS_MODALITIES,
            "model_support_note": (
                "Compatible with Stage 2 Multimodal Deep Learning Model"
                if image_type in SUPPORTED_ANALYSIS_MODALITIES
                else "This image type is not currently supported by the existing model."
            )
        }
        if dicom_metadata:
            result_dict["dicom_metadata"] = dicom_metadata
        return result_dict

    except Exception as e:
        logger.error(f"Image validation exception: {e}", exc_info=True)
        return {
            "valid": False,
            "error": "Unable to process the image right now. Please try again.",
            "message": "Unable to process the image right now. Please try again."
        }
