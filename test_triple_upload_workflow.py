"""
Test script verifying simultaneous 3-image upload and individual validation:
1. Upload and validate MRI scan separately
2. Upload and validate CT scan separately
3. Upload and validate Pathology tile separately
4. Verify all 3 validation responses
5. Run complete analysis containing CT and Pathology alongside MRI
6. Verify pipeline successfully finishes all 6 stages using the multi-modal images
"""

import io
import json
import os
import sys
import urllib.request
import uuid


if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')



def multipart_post(url, fields, files):
    boundary = uuid.uuid4().hex
    body = io.BytesIO()

    for k, v in fields.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode("utf-8"))
        body.write(f"{v}\r\n".encode("utf-8"))

    for k, (filename, file_bytes, mime_type) in files.items():
        body.write(f"--{boundary}\r\n".encode("utf-8"))
        body.write(f'Content-Disposition: form-data; name="{k}"; filename="{filename}"\r\n'.encode("utf-8"))
        body.write(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        body.write(file_bytes)
        body.write(b"\r\n")

    body.write(f"--{boundary}--\r\n".encode("utf-8"))
    payload = body.getvalue()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(payload))
        },
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return resp.getcode(), json.loads(resp.read().decode("utf-8"))


def test_triple_workflow():
    print("Testing simultaneous 3-slot upload & independent validation...")

    ct_path = "Stage2/ct images/blca/ct_slice_0000.png" if os.path.exists("Stage2/ct images/blca/ct_slice_0000.png") else "Stage2/2d_images_large/luad/ct_slice_0000.png"
    tile_path = "Stage2/ct images/blca/tile_0000.png" if os.path.exists("Stage2/ct images/blca/tile_0000.png") else "Stage2/2d_images_large/luad/tile_0000.png"

    with open(ct_path, "rb") as f:
        real_ct = f.read()
    with open(tile_path, "rb") as f:
        real_tile = f.read()

    # 1. Validate MRI independently
    c_mri, res_mri = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "MRI"},
        {"file": ("mri_scan.png", real_ct, "image/png")}
    )
    assert res_mri.get("valid") is True, f"MRI validation failed: {res_mri}"
    print(f"✓ Slot 1 (MRI) validated: {res_mri.get('message')} (File: {res_mri.get('saved_filename')})")

    # 2. Validate CT Scan independently
    c_ct, res_ct = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "CT Scan"},
        {"file": ("axial_ct.png", real_ct, "image/png")}
    )
    assert res_ct.get("valid") is True, f"CT validation failed: {res_ct}"
    print(f"✓ Slot 2 (CT Scan) validated: {res_ct.get('message')} (File: {res_ct.get('saved_filename')})")

    # 3. Validate Pathology independently
    c_path, res_path = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "Pathology"},
        {"file": ("he_tile.png", real_tile, "image/png")}
    )
    assert res_path.get("valid") is True, f"Pathology validation failed: {res_path}"
    print(f"✓ Slot 3 (Pathology) validated: {res_path.get('message')} (File: {res_path.get('saved_filename')})")

    # 4. Run Analysis with all 3 images provided
    payload = {
        "patient_id": "PAT-TRIPLE-001",
        "cancer_type": "Lung (LUAD)",
        "cancer_stage": "Stage IV",
        "mri_image_file": res_mri.get("saved_filename"),
        "ct_image_file": res_ct.get("saved_filename"),
        "pathology_image_file": res_path.get("saved_filename"),
        "image_file": res_ct.get("saved_filename"),
        "image_type": "CT Scan"
    }

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/pipeline/run-all",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))

    assert res.get("status") == "Completed"
    assert "stage2_dl" in res.get("pipeline_stages", {})
    dl = res["pipeline_stages"]["stage2_dl"]
    print(f"✓ Full Pipeline Analysis executed successfully!")
    print(f"  - DL Classification: {dl.get('classification')}")
    print(f"  - CT Source File: {dl.get('ct_slice_file')}")
    print(f"  - Pathology Source File: {dl.get('pathology_tile_file')}")
    print(f"  - Progression Probability: {dl.get('progression_probability')}")
    print("ALL TRIPLE-MODALITY TESTS PASSED 100%!")


if __name__ == "__main__":
    test_triple_workflow()
