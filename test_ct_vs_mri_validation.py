import os
import pytest
from fastapi.testclient import TestClient
from stage_05_generative.api.main import app
from stage_05_generative.api.services.image_validator import validate_and_save_medical_image

client = TestClient(app)

def test_ct_image_fails_when_validated_as_mri():
    user_image_path = os.path.join("uploads", "med_ebdaaf21a834_8766.jpeg")
    assert os.path.exists(user_image_path), "User test image not found"

    with open(user_image_path, "rb") as f:
        file_bytes = f.read()

    # 1. Test unit function with MRI type
    mri_result = validate_and_save_medical_image(
        file_bytes=file_bytes,
        original_filename="WhatsApp Image 2026-10-09 at 08.45.12.jpeg",
        image_type="MRI"
    )

    assert mri_result["valid"] is False
    assert mri_result["detected_modality"] == "CT Scan"
    assert "CT scan" in mri_result["error"] or "CT Scan" in mri_result["error"]

    # 2. Test unit function with CT Scan type
    ct_result = validate_and_save_medical_image(
        file_bytes=file_bytes,
        original_filename="WhatsApp Image 2026-10-09 at 08.45.12.jpeg",
        image_type="CT Scan"
    )

    assert ct_result["valid"] is True
    assert ct_result["image_type"] == "CT Scan"

def test_api_endpoint_rejects_ct_in_mri_slot():
    user_image_path = os.path.join("uploads", "med_ebdaaf21a834_8766.jpeg")
    with open(user_image_path, "rb") as f:
        file_bytes = f.read()

    # MRI endpoint call
    resp_mri = client.post(
        "/api/pipeline/validate-image",
        data={"image_type": "MRI"},
        files={"file": ("WhatsApp Image 2026-10-09 at 08.45.12.jpeg", file_bytes, "image/jpeg")}
    )
    data_mri = resp_mri.json()
    assert data_mri["valid"] is False
    assert data_mri.get("detected_modality") == "CT Scan"

    # CT Scan endpoint call
    resp_ct = client.post(
        "/api/pipeline/validate-image",
        data={"image_type": "CT Scan"},
        files={"file": ("WhatsApp Image 2026-10-09 at 08.45.12.jpeg", file_bytes, "image/jpeg")}
    )
    data_ct = resp_ct.json()
    assert data_ct["valid"] is True
