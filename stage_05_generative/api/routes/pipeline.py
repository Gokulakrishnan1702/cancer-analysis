"""
FastAPI Routes for Unified 5-Stage Clinical AI Pipeline
======================================================
Provides endpoints for:
  - Full sequential 5-stage pipeline execution
  - Individual stage execution (ML, DL, NLP, SLM, GenAI)
  - Pre-curated patient presets
  - CSV upload and parsing
  - Historical patient analyses
  - Hospital analytics and CDSS performance metrics
  - Medical scan and pathology image streaming
"""

import os
import io
import csv
import json
import glob
import logging
from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Response
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from api.services.unified_pipeline_service import get_unified_pipeline_service
from api.services.image_validator import validate_and_save_medical_image
from api.services.patient_validator import validate_patient_input
from api.services.gradcam_explainer import get_gradcam_explainer

logger = logging.getLogger("pipeline_routes")
router = APIRouter(prefix="/api/pipeline", tags=["Unified Clinical AI Pipeline"])

WORKSPACE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
STAGE2_IMG_DIR = os.path.join(WORKSPACE_ROOT, "Stage2", "ct images") if os.path.exists(os.path.join(WORKSPACE_ROOT, "Stage2", "ct images")) else os.path.join(WORKSPACE_ROOT, "Stage2", "2d_images_large")
STAGE2_DATA_TEST_SET_DIR = os.path.join(WORKSPACE_ROOT, "Stage2", "data", "mri images") if os.path.exists(os.path.join(WORKSPACE_ROOT, "Stage2", "data", "mri images")) else os.path.join(WORKSPACE_ROOT, "Stage2", "data", "Test Set")



class PatientAnalysisRequest(BaseModel):
    patient_id: Optional[str] = "PAT-0001"
    age: Optional[float] = 60.0
    sex: Optional[str] = "M"
    cancer_type: Optional[str] = "Lung (LUAD)"
    cancer_stage: Optional[str] = "Stage IV"
    treatment_name: Optional[str] = "Targeted Therapy"
    treatment_cycle: Optional[int] = 3
    dosage_mg: Optional[float] = 400.0
    previous_dose_mg: Optional[float] = 400.0
    dose_change_percentage: Optional[float] = 0.0
    treatment_duration_days: Optional[float] = 21.0
    mutation_profile: Optional[str] = "EGFR L858R"
    mutation_count: Optional[int] = 1
    biomarker_status: Optional[str] = "Positive"
    ctDNA_level: Optional[float] = 0.45
    tumor_marker_level: Optional[float] = 14.2
    WBC: Optional[float] = 6.5
    hemoglobin: Optional[float] = 12.5
    platelet_count: Optional[float] = 250.0
    creatinine: Optional[float] = 1.1
    bilirubin: Optional[float] = 1.2
    ALT: Optional[float] = 45.0
    AST: Optional[float] = 42.0
    heart_rate: Optional[float] = 76.0
    systolic_bp: Optional[float] = 130.0
    diastolic_bp: Optional[float] = 82.0
    spo2: Optional[float] = 96.0
    temperature: Optional[float] = 37.0
    respiratory_rate: Optional[float] = 16.0
    previous_adverse_reaction: Optional[str] = "No"
    previous_treatment_response: Optional[str] = "Partial Response"
    previous_toxicity: Optional[str] = "Low"
    treatment_response: Optional[str] = "Partial Response"
    clinical_notes: Optional[str] = None
    image_file: Optional[str] = None
    image_type: Optional[str] = None
    ct_image_file: Optional[str] = None
    pathology_image_file: Optional[str] = None
    mri_image_file: Optional[str] = None


@router.post("/validate-image", summary="Validate Uploaded Medical Image")
async def validate_image_endpoint(
    file: UploadFile = File(...),
    image_type: str = Form(...)
):
    """
    Validates uploaded medical image format, integrity, dimensions, and modality consistency.
    Enforces user-selected modality (MRI, CT Scan, Pathology).
    """
    try:
        file_bytes = await file.read()
        res = validate_and_save_medical_image(
            file_bytes=file_bytes,
            original_filename=file.filename or "uploaded_scan.png",
            image_type=image_type
        )
        return res
    except Exception as e:
        logger.error(f"[VALIDATE-IMAGE ERROR] {e}", exc_info=True)
        return {
            "valid": False,
            "error": "Unable to process the image right now. Please try again.",
            "message": "Unable to process the image right now. Please try again."
        }


@router.get("/uploaded-image/{image_name}", summary="Serve Uploaded Medical Image")
async def serve_uploaded_medical_image(image_name: str):
    """Serves validated medical images saved in uploads or test datasets with proper fallback."""
    safe_name = os.path.basename(image_name)
    uploads_dir = os.path.join(WORKSPACE_ROOT, "uploads")
    target_path = os.path.join(uploads_dir, safe_name)

    if not os.path.exists(target_path):
        # Look across Stage2 image repositories
        candidates = [
            os.path.join(STAGE2_IMG_DIR, "luad", safe_name),
            os.path.join(STAGE2_IMG_DIR, safe_name),
            os.path.join(STAGE2_DATA_TEST_SET_DIR, "colon_aca", safe_name),
            os.path.join(WORKSPACE_ROOT, "data", "lung", "ct", safe_name),
        ]
        for c in candidates:
            if os.path.exists(c):
                target_path = c
                break

    # Recursive search if still not found
    if not os.path.exists(target_path) and os.path.exists(STAGE2_IMG_DIR):
        for root, _, files in os.walk(STAGE2_IMG_DIR):
            if safe_name in files:
                target_path = os.path.join(root, safe_name)
                break

    # Graceful fallback to default validated scan
    if not os.path.exists(target_path):
        fallback = os.path.join(STAGE2_IMG_DIR, "luad", "ct_slice_0000.png")
        if os.path.exists(fallback):
            return FileResponse(fallback, media_type="image/png")
        raise HTTPException(status_code=404, detail="Uploaded image not found")

    media_type = "image/png" if safe_name.lower().endswith(".png") else "image/jpeg"
    return FileResponse(target_path, media_type=media_type)


@router.post("/run-all", summary="Run Complete 6-Stage AI Analysis")
async def run_complete_analysis(req: PatientAnalysisRequest):
    """Executes ML -> DL -> NLP -> SLM -> GenAI -> Agentic sequentially and returns unified report."""
    has_actionable = bool(req.ct_image_file or req.pathology_image_file or (req.image_file and req.image_type in ["CT Scan", "Pathology"]))
    
    # 1. Check Modality Support Interlock
    if (req.image_type == "MRI" and not has_actionable) or (req.mri_image_file and not has_actionable):
        raise HTTPException(
            status_code=400,
            detail="MRI analysis is not currently supported: Modality outside validated DL scope. Status: Unable to provide a reliable result."
        )

    # 2. Check Image Presence
    active_image = req.image_file or req.ct_image_file or req.pathology_image_file
    if not active_image and not req.mri_image_file:
        return {
            "status": "Unable to provide a reliable result",
            "reliable": False,
            "reliability_state": "MISSING_REQUIRED_INPUTS",
            "reason": "Medical scan or pathology image is required for multimodal feature extraction and clinical risk inference.",
            "safety_interlock_active": True,
            "validation_errors": ["Missing required medical image input."],
            "validation_warnings": ["Please upload and validate a medical scan before starting analysis."],
            "patient_id": req.patient_id,
            "clinical_guidance": "Please upload a validated CT scan or pathology image to proceed."
        }

    # 3. Patient Numerical Data Validation & Safety Ranges
    patient_dict = req.model_dump()
    val_res = validate_patient_input(patient_dict)
    if not val_res.is_reliable:
        val_d = val_res.to_dict()
        val_errors = [f"{item['field']}: {item['value']} outside safe bounds [{item.get('valid_range', '')}]" for item in val_res.out_of_range_fields] + [f"Missing critical field: {f}" for f in val_res.missing_critical_fields]
        return {
            "status": "Unable to provide a reliable result",
            "reliable": False,
            "reliability_state": "UNRELIABLE_PHYSIOLOGICAL_MEASUREMENTS",
            "reason": val_res.reason or "Clinical input parameters violate safety validation constraints or contain out-of-range physiological measurements.",
            "safety_interlock_active": True,
            "validation_errors": val_errors,
            "validation_warnings": val_res.warnings,
            "missing_critical_fields": val_res.missing_critical_fields,
            "out_of_range_fields": val_res.out_of_range_fields,
            "patient_id": req.patient_id,
            "clinical_guidance": "System cannot compute a reliable clinical risk estimate. Please review input values against standard physiological ranges."
        }

    try:
        service = get_unified_pipeline_service()
        result = service.run_full_pipeline(patient_dict, image_filename=active_image)
        result["patient_validation"] = val_res.to_dict()

        # 4. Attach Grad-CAM Convolutional Explanation for Supported Scans
        if active_image and (req.image_type in ["CT Scan", "Pathology"] or not req.image_type):
            try:
                explainer = get_gradcam_explainer()
                modality_str = req.image_type or ("CT Scan" if "ct" in active_image.lower() else "Pathology")
                
                # Resolve local filesystem path for image
                from api.services.gradcam_explainer import resolve_image_path
                resolved_img_path = resolve_image_path(active_image)

                if resolved_img_path:
                    cam_explanation = explainer.generate_explanation(
                        image_path=resolved_img_path,
                        modality=modality_str
                    )
                    result["gradcam_explanation"] = cam_explanation
            except Exception as cam_err:
                logger.warning(f"[GRAD-CAM IN RUN-ALL] Non-blocking warning: {cam_err}")

        return result
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"[RUN-ALL ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/stage/{stage_id}", summary="Run Individual AI Stage")
async def run_single_stage(stage_id: str, req: PatientAnalysisRequest):
    """Runs a single isolated stage (1=ML, 2=DL, 3=NLP, 4=SLM, 5=GenAI, 6=Agentic)."""
    if req.image_type == "MRI" and stage_id.lower() in ["2", "dl", "stage2"]:
        raise HTTPException(status_code=400, detail="This image type is not currently supported by the existing model.")
    service = get_unified_pipeline_service()
    patient_dict = req.model_dump()
    st = stage_id.lower()

    if st in ["1", "ml", "stage1"]:
        return service.run_stage1_ml(patient_dict)
    elif st in ["2", "dl", "stage2"]:
        img = req.ct_image_file or req.pathology_image_file or req.image_file
        return service.run_stage2_dl(patient_dict, image_filename=img)
    elif st in ["3", "nlp", "stage3"]:
        return service.run_stage3_nlp(patient_dict)
    elif st in ["4", "slm", "stage4"]:
        return service.run_stage4_slm(patient_dict)
    elif st in ["5", "genai", "stage5"]:
        return service.run_stage5_genai(patient_dict)
    elif st in ["6", "agentic", "stage6"]:
        return service.run_stage6_agentic(patient_dict)
    else:
        raise HTTPException(status_code=400, detail=f"Unknown stage '{stage_id}'. Valid values: 1, 2, 3, 4, 5, 6.")


@router.get("/presets", summary="Get Pre-configured Clinical Patient Presets")
async def get_patient_presets():
    """Returns realistic clinical cases (High Risk, Moderate Risk, Low Risk, Critical DILI Edge Case)."""
    service = get_unified_pipeline_service()
    return service.get_sample_presets()


@router.get("/history", summary="Get Patient Analysis History")
async def get_patient_history(limit: int = 50):
    """Fetches previously analyzed patient cases from SQLite database."""
    service = get_unified_pipeline_service()
    return await service.get_history(limit=limit)


@router.get("/analytics", summary="Get CDSS Hospital Analytics & Model Metrics")
async def get_cdss_analytics():
    """Serves high-level hospital decision-support performance analytics and charts."""
    service = get_unified_pipeline_service()
    return await service.get_analytics()


@router.post("/upload-csv", summary="Upload and Parse Patient Data CSV")
async def upload_patient_csv(file: UploadFile = File(...)):
    """Parses a patient/lab CSV file and maps fields into structured clinical input schema."""
    try:
        contents = await file.read()
        text_stream = io.StringIO(contents.decode("utf-8", errors="ignore"))
        reader = csv.DictReader(text_stream)
        rows = list(reader)

        if not rows:
            raise HTTPException(status_code=400, detail="CSV file is empty or has invalid formatting.")

        first_row = rows[0]

        # Case-insensitive mapping helper
        def get_field(keys, default=None):
            for k in keys:
                for row_k, v in first_row.items():
                    if row_k.strip().lower() == k.lower() and v != "":
                        return v
            return default

        mapped_data = {
            "patient_id": get_field(["patient_id", "id", "mrn"], "PAT-UPLOAD-01"),
            "age": float(get_field(["age"], 62.0)),
            "sex": get_field(["sex", "gender"], "M"),
            "cancer_type": get_field(["cancer_type", "cancer", "histology"], "Lung (LUAD)"),
            "cancer_stage": get_field(["cancer_stage", "stage", "stage_numeric"], "Stage IV"),
            "treatment_name": get_field(["treatment_name", "treatment_type", "drug"], "Targeted Therapy"),
            "treatment_cycle": int(float(get_field(["treatment_cycle", "cycle"], 3))),
            "dosage_mg": float(get_field(["dosage_mg", "dose"], 400.0)),
            "previous_dose_mg": float(get_field(["previous_dose_mg"], 400.0)),
            "dose_change_percentage": float(get_field(["dose_change_percentage"], 0.0)),
            "treatment_duration_days": float(get_field(["treatment_duration_days", "duration"], 21.0)),
            "mutation_profile": get_field(["mutation_profile", "genomics", "mutations"], "EGFR L858R"),
            "mutation_count": int(float(get_field(["mutation_count"], 1))),
            "biomarker_status": get_field(["biomarker_status"], "Positive"),
            "ctDNA_level": float(get_field(["ctDNA_level", "baseline_cdna", "ctdna"], 0.45)),
            "tumor_marker_level": float(get_field(["tumor_marker_level", "baseline_cea"], 14.2)),
            "WBC": float(get_field(["WBC", "baseline_wbc"], 6.5)),
            "hemoglobin": float(get_field(["hemoglobin", "baseline_hgb"], 12.5)),
            "platelet_count": float(get_field(["platelet_count", "baseline_plt"], 250.0)),
            "creatinine": float(get_field(["creatinine", "baseline_creatinine"], 1.1)),
            "bilirubin": float(get_field(["bilirubin", "baseline_bilirubin"], 1.2)),
            "ALT": float(get_field(["ALT", "baseline_alt"], 45.0)),
            "AST": float(get_field(["AST", "baseline_ast"], 42.0)),
            "heart_rate": float(get_field(["heart_rate"], 76.0)),
            "systolic_bp": float(get_field(["systolic_bp"], 130.0)),
            "diastolic_bp": float(get_field(["diastolic_bp"], 82.0)),
            "spo2": float(get_field(["spo2"], 96.0)),
            "temperature": float(get_field(["temperature"], 37.0)),
            "respiratory_rate": float(get_field(["respiratory_rate"], 16.0)),
            "clinical_notes": get_field(["clinical_notes", "notes", "symptoms"], "Uploaded via CSV panel.")
        }

        return {
            "status": "success",
            "message": f"Successfully parsed CSV with {len(rows)} record(s). Loaded patient {mapped_data['patient_id']}.",
            "patient_data": mapped_data,
            "total_records_found": len(rows)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[CSV UPLOAD ERROR] {e}")
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV file: {str(e)}")


@router.get("/sample-images", summary="List Sample Medical Images")
async def list_sample_images():
    """Lists available CT slices and pathology tiles grouped by cancer type, including test set datasets."""
    results = {}
    if os.path.exists(STAGE2_IMG_DIR):
        for ctype in os.listdir(STAGE2_IMG_DIR):
            cpath = os.path.join(STAGE2_IMG_DIR, ctype)
            if os.path.isdir(cpath):
                files = os.listdir(cpath)
                cts = [f for f in files if "ct_slice" in f][:5]
                tiles = [f for f in files if "tile" in f][:5]
                results[ctype] = {
                    "ct_scans": cts,
                    "pathology_tiles": tiles
                }

    # Add colon adenocarcinoma test set images
    colon_test_dir = os.path.join(STAGE2_DATA_TEST_SET_DIR, "colon_aca")
    if os.path.exists(colon_test_dir):
        colon_files = [f for f in os.listdir(colon_test_dir) if f.lower().endswith((".jpeg", ".jpg", ".png"))][:10]
        results["colon_aca"] = {
            "ct_scans": [],
            "pathology_tiles": colon_files
        }
        if "coad" in results:
            results["coad"]["test_set_pathology"] = colon_files[:5]
        else:
            results["coad"] = {
                "ct_scans": [],
                "pathology_tiles": colon_files[:5],
                "test_set_pathology": colon_files[:5]
            }

    return results


@router.get("/image/{cancer_type}/{image_name}", summary="Serve Medical Image Scan / Tile")
async def serve_medical_image(cancer_type: str, image_name: str):
    """Serves real CT scan slices and Pathology tiles from 2d_images_large, Test Set, and uploads."""
    safe_ctype = os.path.basename(cancer_type).lower()
    safe_img = os.path.basename(image_name)

    # 1. Look in Stage2/2d_images_large/{cancer_type}/{image_name}
    target_path = os.path.join(STAGE2_IMG_DIR, safe_ctype, safe_img)

    # 2. Look in Stage2/data/Test Set/{cancer_type}/{image_name}
    if not os.path.exists(target_path):
        target_path = os.path.join(STAGE2_DATA_TEST_SET_DIR, safe_ctype, safe_img)

    # 3. Look in Stage2/data/Test Set/colon_aca/{image_name}
    if not os.path.exists(target_path) and ("colon" in safe_ctype or "coad" in safe_ctype or safe_img.startswith("colonca")):
        target_path = os.path.join(STAGE2_DATA_TEST_SET_DIR, "colon_aca", safe_img)

    # 4. Search recursively inside data/Test Set
    if not os.path.exists(target_path) and os.path.exists(STAGE2_DATA_TEST_SET_DIR):
        for root, _, files in os.walk(STAGE2_DATA_TEST_SET_DIR):
            if safe_img in files:
                target_path = os.path.join(root, safe_img)
                break

    # 5. Look in uploads directory
    if not os.path.exists(target_path):
        uploads_path = os.path.join(WORKSPACE_ROOT, "uploads", safe_img)
        if os.path.exists(uploads_path):
            target_path = uploads_path

    # 6. Fallback if still missing
    if not os.path.exists(target_path):
        fallback = os.path.join(STAGE2_IMG_DIR, "luad", "ct_slice_0000.png")
        if os.path.exists(fallback):
            return FileResponse(fallback, media_type="image/png")
        raise HTTPException(status_code=404, detail="Image not found")

    media_type = "image/jpeg" if safe_img.lower().endswith((".jpg", ".jpeg")) else "image/png"
    return FileResponse(target_path, media_type=media_type)


class PatientPdfExportRequest(BaseModel):
    patient: Dict[str, Any]
    results: Optional[Dict[str, Any]] = None


@router.post("/export-pdf", summary="Export Clinical Tumor Board PDF Report")
async def export_patient_pdf(req: PatientPdfExportRequest):
    """Generates an oncologist-grade, downloadable PDF report for the active patient case."""
    try:
        from api.services.pdf_report_service import generate_patient_pdf_report
        pdf_bytes = generate_patient_pdf_report(req.patient, req.results)
        pid = req.patient.get("patient_id", "PAT-0001")
        filename = f"TumorBoard_Report_{pid}.pdf"
        return Response(
            content=pdf_bytes,
            media_type="application/pdf",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition"
            }
        )
    except Exception as e:
        logger.error(f"[PDF EXPORT ERROR] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to generate clinical PDF: {str(e)}")


@router.post("/validate-patient", summary="Validate Patient Numerical Measurements & Safety Constraints")
async def validate_patient_endpoint(req: PatientAnalysisRequest):
    """Validates patient tabular measurements against physiological thresholds and completeness criteria."""
    return validate_patient_input(req.model_dump()).to_dict()


class GradCamRequest(BaseModel):
    image_filename: str
    modality: Optional[str] = "CT Scan"
    target_class: Optional[int] = 0


@router.post("/explain/gradcam", summary="Generate PyTorch Grad-CAM Convolutional Heatmap Overlay")
async def generate_gradcam_endpoint(req: GradCamRequest):
    """
    Computes genuine PyTorch Grad-CAM class activation maps for CT or Pathology scans.
    Overlays Jet colormap on original scan, computes spatial quadrant attributions, and returns artifact URL.
    """
    if req.modality and req.modality.strip().upper() == "MRI":
        return {
            "status": "unsupported",
            "modality": "MRI",
            "error": "Grad-CAM is not available for MRI scans. Deep learning vision models are currently trained on CT Scans and Histopathology slides.",
            "disclaimer": (
                "EXPLAINABLE AI NOTICE: Visual models require validated convolutional architectures. "
                "MRI modality is currently outside the trained model scope."
            )
        }

    from api.services.gradcam_explainer import resolve_image_path, get_gradcam_explainer
    resolved_path = resolve_image_path(req.image_filename)

    if not resolved_path:
        raise HTTPException(
            status_code=404,
            detail=f"Medical scan file '{req.image_filename}' not found for Grad-CAM generation."
        )

    try:
        explainer = get_gradcam_explainer()
        res = explainer.generate_explanation(
            image_path=resolved_path,
            modality=req.modality or "CT Scan",
            target_class=req.target_class or 0
        )
        return res
    except Exception as e:
        logger.error(f"[GRAD-CAM ERROR] {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Grad-CAM generation failed: {str(e)}")


@router.get("/datasets/provenance", summary="Get 15-Cohort Dataset Provenance and Patient-Level Integrity Report")
async def get_dataset_provenance():
    """Serves the complete audit of 15 target cancer types, genuine public datasets, split counts, and provenance."""
    report_path = os.path.join(WORKSPACE_ROOT, "dataset_audit_report.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read cached dataset report: {e}")

    try:
        import sys
        if WORKSPACE_ROOT not in sys.path:
            sys.path.insert(0, WORKSPACE_ROOT)
        from generate_dataset_report import build_dataset_audit_report
        return build_dataset_audit_report()
    except Exception as e:
        logger.error(f"Failed to generate dataset report: {e}")
        raise HTTPException(status_code=500, detail=f"Dataset provenance generation failed: {str(e)}")


@router.get("/models/reliability", summary="Get Verifiable Multi-Stage Model Evaluation and Reliability Metrics")
async def get_model_reliability():
    """Serves authentic, reproducible evaluation metrics across Stages 1 to 6 without fabricated values."""
    report_path = os.path.join(WORKSPACE_ROOT, "model_reliability_audit.json")
    if os.path.exists(report_path):
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not read cached model reliability report: {e}")

    try:
        import sys
        if WORKSPACE_ROOT not in sys.path:
            sys.path.insert(0, WORKSPACE_ROOT)
        from evaluate_all_models import run_full_reliability_audit
        return run_full_reliability_audit()
    except Exception as e:
        logger.error(f"Failed to generate model reliability report: {e}")
        raise HTTPException(status_code=500, detail=f"Model evaluation report failed: {str(e)}")

