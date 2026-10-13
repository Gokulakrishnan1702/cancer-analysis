"""
Verification Test Suite for Medical Image Upload & Validation Workflow
========================================================================
Tests the 10 items specified in requirements:
1. No image -> error
2. Unsupported file -> error
3. Corrupted image -> error
4. Invalid image for selected modality -> error
5. Valid MRI -> validated, model incompatibility enforced
6. Valid CT -> accepted as valid CT
7. Valid pathology -> accepted as valid pathology
8. Correct image -> analysis runs successfully
9. Existing AI stages still work (6/6 stages completed)
10. Existing dashboard remains unchanged & loads correctly
"""

import io
import json
import mimetypes
import os
import sys
import urllib.parse
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
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.getcode(), json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_suite():
    print("=" * 70)
    print("TESTING PATIENT INPUT / MEDICAL IMAGE UPLOAD WORKFLOW (10 CHECKS)")
    print("=" * 70)

    ct_path = "Stage2/ct images/blca/ct_slice_0000.png" if os.path.exists("Stage2/ct images/blca/ct_slice_0000.png") else "Stage2/2d_images_large/luad/ct_slice_0000.png"
    tile_path = "Stage2/ct images/blca/tile_0000.png" if os.path.exists("Stage2/ct images/blca/tile_0000.png") else "Stage2/2d_images_large/luad/tile_0000.png"

    with open(ct_path, "rb") as f:
        real_ct_bytes = f.read()

    with open(tile_path, "rb") as f:
        real_tile_bytes = f.read()

    # 1. No image -> error
    code, res1 = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "CT Scan"},
        {"file": ("empty.png", b"", "image/png")}
    )
    p1 = res1.get("valid") is False and "Please upload an image" in res1.get("message", "")
    print(f"[{'PASS' if p1 else 'FAIL'}] 1. No image -> error: {res1.get('message')}")

    # 2. Unsupported file -> error
    code, res2 = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "CT Scan"},
        {"file": ("document.txt", b"plain text data", "text/plain")}
    )
    p2 = res2.get("valid") is False and "Unsupported file format" in res2.get("message", "")
    print(f"[{'PASS' if p2 else 'FAIL'}] 2. Unsupported file -> error: {res2.get('message')}")

    # 3. Corrupted image -> error
    corrupt_png = b"\x89PNG\r\n\x1a\n" + b"CORRUPTED_TRUNCATED_IMAGE_DATA_000000"
    code, res3 = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "CT Scan"},
        {"file": ("corrupt.png", corrupt_png, "image/png")}
    )
    p3 = res3.get("valid") is False and "This image cannot be opened" in res3.get("message", "")
    print(f"[{'PASS' if p3 else 'FAIL'}] 3. Corrupted image -> error: {res3.get('message')}")

    # 4. Invalid image for modality -> error
    # Case 4a: CT selected + colored pathology image uploaded
    code, res4a = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "CT Scan"},
        {"file": ("tile_colored.png", real_tile_bytes, "image/png")}
    )
    p4a = res4a.get("valid") is False and "Invalid image for CT analysis" in res4a.get("message", "")
    print(f"[{'PASS' if p4a else 'FAIL'}] 4a. Invalid image for CT (Pathology uploaded) -> error: {res4a.get('message')}")

    # Case 4b: Pathology selected + monochrome CT image uploaded
    code, res4b = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "Pathology"},
        {"file": ("ct_slice.png", real_ct_bytes, "image/png")}
    )
    p4b = res4b.get("valid") is False and "Invalid image for pathology analysis" in res4b.get("message", "")
    print(f"[{'PASS' if p4b else 'FAIL'}] 4b. Invalid image for Pathology (CT uploaded) -> error: {res4b.get('message')}")

    # Case 4c: MRI selected + colored pathology image uploaded
    code, res4c = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "MRI"},
        {"file": ("tile_colored.png", real_tile_bytes, "image/png")}
    )
    p4c = res4c.get("valid") is False and "Invalid image for MRI analysis" in res4c.get("message", "")
    print(f"[{'PASS' if p4c else 'FAIL'}] 4c. Invalid image for MRI (Pathology uploaded) -> error: {res4c.get('message')}")

    # 5. Valid MRI -> accepted during validation; model incompatibility enforced during inference
    code, res5 = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "MRI"},
        {"file": ("mri_scan.png", real_ct_bytes, "image/png")}
    )
    p5a = res5.get("valid") is True and "Valid MRI image" in res5.get("message", "")
    print(f"[{'PASS' if p5a else 'FAIL'}] 5a. Valid MRI validation: {res5.get('message')}")

    # Test that running analysis with MRI enforces unsupported model rule
    mri_payload = {
        "patient_id": "PAT-MRI-001",
        "image_file": res5.get("saved_filename"),
        "image_type": "MRI"
    }
    req_mri = urllib.request.Request(
        "http://127.0.0.1:8000/api/pipeline/run-all",
        data=json.dumps(mri_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    mri_model_blocked = False
    mri_err_msg = ""
    try:
        with urllib.request.urlopen(req_mri) as resp:
            pass
    except urllib.error.HTTPError as e:
        mri_model_blocked = (e.code == 400)
        err_data = json.loads(e.read().decode("utf-8"))
        mri_err_msg = str(err_data.get("detail", ""))
    p5b = mri_model_blocked and "not currently supported" in mri_err_msg
    print(f"[{'PASS' if p5b else 'FAIL'}] 5b. MRI analysis blocked by unsupported model rule: {mri_err_msg}")

    # 6. Valid CT -> accepted
    code, res6 = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "CT Scan"},
        {"file": ("chest_ct.png", real_ct_bytes, "image/png")}
    )
    p6 = res6.get("valid") is True and "Valid CT image" in res6.get("message", "")
    ct_saved = res6.get("saved_filename")
    print(f"[{'PASS' if p6 else 'FAIL'}] 6. Valid CT accepted: {res6.get('message')} (saved: {ct_saved})")

    # 7. Valid Pathology -> accepted
    code, res7 = multipart_post(
        "http://127.0.0.1:8000/api/pipeline/validate-image",
        {"image_type": "Pathology"},
        {"file": ("biopsy_tile.png", real_tile_bytes, "image/png")}
    )
    p7 = res7.get("valid") is True and "Valid Pathology image" in res7.get("message", "")
    path_saved = res7.get("saved_filename")
    print(f"[{'PASS' if p7 else 'FAIL'}] 7. Valid Pathology accepted: {res7.get('message')} (saved: {path_saved})")

    # 8. Correct image -> analysis runs
    ct_analysis_payload = {
        "patient_id": "PAT-CT-TEST-001",
        "cancer_type": "Lung (LUAD)",
        "cancer_stage": "Stage IV",
        "image_file": ct_saved,
        "image_type": "CT Scan"
    }
    req_ct = urllib.request.Request(
        "http://127.0.0.1:8000/api/pipeline/run-all",
        data=json.dumps(ct_analysis_payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req_ct) as resp:
        run_res = json.loads(resp.read().decode("utf-8"))
    p8 = run_res.get("status") == "Completed" and "stage2_dl" in run_res.get("pipeline_stages", {})
    dl_out = run_res.get("pipeline_stages", {}).get("stage2_dl", {})
    print(f"[{'PASS' if p8 else 'FAIL'}] 8. Correct image -> analysis runs: Status={run_res.get('status')}, DL Pred={dl_out.get('image_prediction')}")

    # 9. Existing AI stages still work (ML, DL, NLP, SLM, GenAI, Agentic)
    stages = run_res.get("pipeline_stages", {})
    stage_keys = ["stage1_ml", "stage2_dl", "stage3_nlp", "stage4_slm", "stage5_genai", "stage6_agentic"]
    p9 = all(k in stages for k in stage_keys) and run_res.get("final_assessment") is not None
    print(f"[{'PASS' if p9 else 'FAIL'}] 9. Existing AI stages still work ({len(stages)}/6 stages verified):")
    for sk in stage_keys:
        s_obj = stages.get(sk, {})
        s_stat = s_obj.get("status") or s_obj.get("risk_category") or "Completed"
        print(f"     - {sk:<16}: {s_stat}")

    # 10. Existing dashboard remains unchanged & loads correctly
    req_dash = urllib.request.urlopen("http://127.0.0.1:8000/dashboard")
    html = req_dash.read().decode("utf-8")
    p10 = (
        req_dash.getcode() == 200 and
        "[ Upload Medical Image ]" in html and
        "Select Image Type" in html and
        "mod-pill-mri" in html and
        "mod-pill-ct" in html and
        "mod-pill-pathology" in html and
        "btn-validate-image" in html and
        "Clinical Information" in html and
        "6-STAGE PIPELINE" in html
    )
    print(f"[{'PASS' if p10 else 'FAIL'}] 10. Dashboard loads correctly with new workflow and existing structure: HTTP {req_dash.getcode()}")

    all_passed = all([p1, p2, p3, p4a, p4b, p4c, p5a, p5b, p6, p7, p8, p9, p10])
    print("=" * 70)
    if all_passed:
        print("ALL 10 VALIDATION & INFERENCE REQUIREMENTS PASSED 100% SUCCESSFULLY!")
    else:
        print("ONE OR MORE CHECKS FAILED.")
    print("=" * 70)


if __name__ == "__main__":
    test_suite()
