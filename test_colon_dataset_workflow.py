import os
import requests
import pytest
BASE_URL = "http://127.0.0.1:8000"
DATASET_IMG_PATH = "Stage2/data/mri images/colon_aca/colonca4273.jpeg" if os.path.exists("Stage2/data/mri images/colon_aca/colonca4273.jpeg") else "Stage2/data/Test Set/colon_aca/colonca4273.jpeg"



def test_colon_dataset_image_exists():
    """Verify the new Colon Adenocarcinoma dataset image exists on disk."""
    assert os.path.exists(DATASET_IMG_PATH), f"Dataset image not found at {DATASET_IMG_PATH}"
    assert os.path.getsize(DATASET_IMG_PATH) > 10000


def test_colon_image_validation():
    """Verify colonca4273.jpeg validates as valid Pathology image."""
    with open(DATASET_IMG_PATH, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/api/pipeline/validate-image",
            files={"file": ("colonca4273.jpeg", f, "image/jpeg")},
            data={"image_type": "Pathology"}
        )
    assert resp.status_code == 200
    data = resp.json()
    assert data["valid"] is True
    assert data["model_supported"] is True
    assert "Colon Adenocarcinoma" in data["message"]


def test_colon_image_streaming():
    """Verify colon test set images are served with image/jpeg Content-Type."""
    resp1 = requests.get(f"{BASE_URL}/api/pipeline/image/colon_aca/colonca4273.jpeg")
    assert resp1.status_code == 200
    assert "image/jpeg" in resp1.headers.get("content-type", "")
    assert len(resp1.content) > 10000

    resp2 = requests.get(f"{BASE_URL}/api/pipeline/image/coad/colonca4273.jpeg")
    assert resp2.status_code == 200
    assert "image/jpeg" in resp2.headers.get("content-type", "")


def test_colon_preset_available():
    """Verify the COLON_TEST_SET preset is exposed in /api/pipeline/presets."""
    resp = requests.get(f"{BASE_URL}/api/pipeline/presets")
    assert resp.status_code == 200
    presets = resp.json()
    assert "COLON_TEST_SET" in presets
    colon_preset = presets["COLON_TEST_SET"]
    assert colon_preset["data"]["cancer_type"] == "Colon (COAD)"
    assert "colonca4273.jpeg" in colon_preset["data"]["image_file"]


def test_colon_stage2_dl_inference():
    """Verify Stage 2 DL model runs trajectory inference with colon test set image."""
    payload = {
        "patient_id": "PAT-COAD-4273",
        "cancer_type": "Colon (COAD)",
        "cancer_stage": "Stage III",
        "tumor_marker_level": 18.4,
        "ctDNA_level": 0.82,
        "pathology_image_file": "colon_aca/colonca4273.jpeg",
        "image_type": "Pathology"
    }
    resp = requests.post(f"{BASE_URL}/api/pipeline/stage/2", json=payload)
    assert resp.status_code == 200
    res = resp.json()
    assert res["stage"] == "DL"
    assert res["status"] == "Completed"
    assert "MultimodalLSTM" in res["model_architecture"]
    assert len(res["trajectory_timepoints"]) == 5


def test_colon_full_6stage_pipeline():
    """Verify complete 6-stage clinical CDSS analysis runs on the new colon dataset case."""
    presets_resp = requests.get(f"{BASE_URL}/api/pipeline/presets")
    colon_data = presets_resp.json()["COLON_TEST_SET"]["data"]

    resp = requests.post(f"{BASE_URL}/api/pipeline/run-all", json=colon_data)
    assert resp.status_code == 200
    result = resp.json()
    assert result["status"] == "Completed"

    # All 6 stages executed
    stages = result["pipeline_stages"]
    assert "stage1_ml" in stages
    assert "stage2_dl" in stages
    assert "stage3_nlp" in stages
    assert "stage4_slm" in stages
    assert "stage5_genai" in stages
    assert "stage6_agentic" in stages

    fa = result["final_assessment"]
    assert fa["patient_id"] == "PAT-COAD-4273"
    assert fa["cancer_type"] == "Colon (COAD)"
    assert fa["risk_probability_pct"] > 0
    assert len(fa["genai_summary"]) > 50
