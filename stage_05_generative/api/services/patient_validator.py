"""
Patient Data Validation & Clinical Reliability Guard
===================================================
Enforces clinical boundary validations, detects missing or out-of-range fields,
and governs the explicit "Unable to provide a reliable result" state.

Distinguishes separate clinical concepts:
  - Chemotherapy Toxicity Risk (Stage 1 ML)
  - Longitudinal Disease Progression Risk (Stage 2 DL)
  - Clinical Triage Urgency Priority (Stage 3 NLP)
  - Autonomous Multi-Agent Deliberative Consensus (Stage 6)
"""

from typing import Dict, Any, List, Optional, Tuple


# Validated Physiological and Laboratory Ranges
PHYSIOLOGICAL_RANGES = {
    "age": (0.0, 120.0, "years"),
    "systolic_bp": (50.0, 260.0, "mmHg"),
    "diastolic_bp": (30.0, 160.0, "mmHg"),
    "heart_rate": (30.0, 220.0, "bpm"),
    "temperature": (32.0, 43.0, "°C"),
    "spo2": (50.0, 100.0, "%"),
    "respiratory_rate": (6.0, 60.0, "breaths/min"),
    "ALT": (1.0, 2500.0, "U/L"),
    "AST": (1.0, 2500.0, "U/L"),
    "bilirubin": (0.1, 45.0, "mg/dL"),
    "creatinine": (0.1, 25.0, "mg/dL"),
    "WBC": (0.1, 150.0, "x10^3/µL"),
    "hemoglobin": (2.0, 26.0, "g/dL"),
    "platelet_count": (5.0, 2000.0, "x10^3/µL"),
    "ctDNA_level": (0.0, 100.0, "%"),
    "tumor_marker_level": (0.0, 100000.0, "U/mL"),
    "dosage_mg": (0.0, 10000.0, "mg"),
    "treatment_cycle": (1, 50, "cycles"),
}

VALIDATED_CANCER_TYPES = {
    "Lung (LUAD)", "Lung (LUSC)", "Lung", "Breast (BRCA)", "Breast",
    "Colorectal (COAD)", "Colorectal", "Melanoma (SKCM)", "Melanoma",
    "Prostate (PRAD)", "Prostate", "Bladder (BLCA)", "Bladder",
    "Gastric (STAD)", "Gastric", "Stomach", "Pancreatic (PAAD)", "Pancreatic",
    "Renal (KIRC)", "Renal", "Ovarian (OV)", "Ovarian", "Liver (LIHC)", "Liver",
    "Endometrial (UCEC)", "Endometrial", "Esophageal (ESCA)", "Esophageal",
    "Thyroid (THCA)", "Thyroid", "Lymphoma (DLBC)", "Lymphoma"
}


class ClinicalValidationResult:
    def __init__(
        self,
        is_reliable: bool,
        status: str,
        reason: Optional[str] = None,
        missing_critical_fields: Optional[List[str]] = None,
        out_of_range_fields: Optional[List[Dict[str, Any]]] = None,
        warnings: Optional[List[str]] = None,
        validated_data: Optional[Dict[str, Any]] = None
    ):
        self.is_reliable = is_reliable
        self.status = status  # "RELIABLE" or "UNABLE_TO_PROVIDE_RELIABLE_RESULT"
        self.reason = reason
        self.missing_critical_fields = missing_critical_fields or []
        self.out_of_range_fields = out_of_range_fields or []
        self.warnings = warnings or []
        self.validated_data = validated_data or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_reliable": self.is_reliable,
            "status": self.status,
            "state_label": (
                "Verified Input Data" if self.is_reliable
                else "Unable to provide a reliable result"
            ),
            "reason": self.reason,
            "missing_critical_fields": self.missing_critical_fields,
            "out_of_range_fields": self.out_of_range_fields,
            "warnings": self.warnings
        }


def validate_patient_input(
    patient_data: Dict[str, Any],
    require_imaging: bool = False,
    image_info: Optional[Dict[str, Any]] = None
) -> ClinicalValidationResult:
    """
    Validates numerical, demographic, and imaging inputs for the multi-stage pipeline.
    Flags missing or non-physiological data without inventing or fabricating measurements.
    """
    missing_fields = []
    out_of_range = []
    warnings = []
    validated_clean = dict(patient_data)

    # 1. Demographic & Identification
    pid = str(patient_data.get("patient_id") or "").strip()
    if not pid:
        missing_fields.append("patient_id")

    c_type = str(patient_data.get("cancer_type") or "").strip()
    if not c_type:
        missing_fields.append("cancer_type")
    elif not any(known.lower() in c_type.lower() for known in VALIDATED_CANCER_TYPES):
        warnings.append(
            f"Cancer type '{c_type}' is outside the primary 15 validated oncology cohorts. "
            "Downstream subspecialty trial matching may have reduced specificity."
        )

    # 2. Numerical Range Checking
    for field, (f_min, f_max, unit) in PHYSIOLOGICAL_RANGES.items():
        val = patient_data.get(field)
        if val is not None and val != "":
            try:
                num_val = float(val)
                if num_val < f_min or num_val > f_max:
                    out_of_range.append({
                        "field": field,
                        "value": num_val,
                        "valid_range": f"{f_min} - {f_max} {unit}",
                        "severity": "CRITICAL" if num_val < f_min * 0.5 or num_val > f_max * 2.0 else "WARNING"
                    })
                validated_clean[field] = num_val
            except (ValueError, TypeError):
                out_of_range.append({
                    "field": field,
                    "value": str(val),
                    "valid_range": f"Numeric ({unit})",
                    "severity": "INVALID_TYPE"
                })

    # Critical toxicity features required for Stage 1 ML
    crit_stage1 = ["dosage_mg", "ALT", "AST", "creatinine", "WBC", "temperature"]
    for cf in crit_stage1:
        if patient_data.get(cf) is None:
            # Note missing but check if we have enough
            pass

    # 3. Medical Imaging Modality Eligibility
    if require_imaging:
        if not image_info or not image_info.get("valid"):
            return ClinicalValidationResult(
                is_reliable=False,
                status="UNABLE_TO_PROVIDE_RELIABLE_RESULT",
                reason="Medical image upload is missing, invalid, or corrupted.",
                missing_critical_fields=["medical_image"],
                out_of_range_fields=out_of_range,
                warnings=warnings,
                validated_data=validated_clean
            )
        
        modality = image_info.get("image_type")
        if modality == "MRI" and not image_info.get("has_ct_or_pathology"):
            return ClinicalValidationResult(
                is_reliable=False,
                status="UNABLE_TO_PROVIDE_RELIABLE_RESULT",
                reason=(
                    "The uploaded image modality (MRI) is validated but not supported by the existing "
                    "Stage 2 visual deep-learning model weights (CT and Histopathology are supported). "
                    "Multi-modal trajectory progression cannot be reliably inferred."
                ),
                missing_critical_fields=[],
                out_of_range_fields=out_of_range,
                warnings=["MRI scan stored for clinical archive; Stage 2 inference held."],
                validated_data=validated_clean
            )

    # 4. Check for Critical Range Violations
    critical_errors = [item for item in out_of_range if item.get("severity") in ["CRITICAL", "INVALID_TYPE"]]
    if critical_errors:
        err_details = ", ".join([f"{e['field']}={e['value']} (expected {e['valid_range']})" for e in critical_errors])
        return ClinicalValidationResult(
            is_reliable=False,
            status="UNABLE_TO_PROVIDE_RELIABLE_RESULT",
            reason=f"Severe physiological range violations or invalid data types detected: {err_details}.",
            missing_critical_fields=missing_fields,
            out_of_range_fields=out_of_range,
            warnings=warnings,
            validated_data=validated_clean
        )

    # Check for missing ID or cancer type
    if missing_fields:
        return ClinicalValidationResult(
            is_reliable=False,
            status="UNABLE_TO_PROVIDE_RELIABLE_RESULT",
            reason=f"Mandatory patient identification fields are missing: {', '.join(missing_fields)}.",
            missing_critical_fields=missing_fields,
            out_of_range_fields=out_of_range,
            warnings=warnings,
            validated_data=validated_clean
        )

    return ClinicalValidationResult(
        is_reliable=True,
        status="RELIABLE",
        reason="Patient data satisfies physiological validation and model scope criteria.",
        missing_critical_fields=[],
        out_of_range_fields=out_of_range,
        warnings=warnings,
        validated_data=validated_clean
    )
