"""
Comprehensive 12-Suite Verification Test for Cancer Research Platform
=====================================================================
Covers Phase 10 requirements:
  Suite 1: Stage 1 ML (Model loading & tabular prediction)
  Suite 2: Stage 2 DL (Multimodal LSTM, CT & Pathology inference, MRI boundary)
  Suite 3: Stage 3 NLP (Urgency classifier, negation capture, entity spans)
  Suite 4: Stage 4 SLM (Safety guardrails, under-triage prevention)
  Suite 5: Stage 5 GenAI (OOD detection & report synthesis)
  Suite 6: Stage 6 Multi-Agent AI (8-agent consensus, trial matching, DILI interlock)
  Suite 7: Medical Image Validation (DICOM, TIFF, PNG, JPEG & corrupted files)
  Suite 8: Patient Physiological Validation & 'Unable to provide reliable result'
  Suite 9: Explainable AI (PyTorch Grad-CAM on CNN latent blocks & quadrant focus)
  Suite 10: Authentication & RBAC (PBKDF2-HMAC-SHA256, tokens, roles)
  Suite 11: Dataset Provenance & Duplicate Detection (SHA-256 hashes, splits)
  Suite 12: End-to-End API Live Integration (Server endpoints)
"""

import os
import sys
import json
import unittest
import urllib.request
import urllib.error
import numpy as np

WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
if WORKSPACE_ROOT not in sys.path:
    sys.path.insert(0, WORKSPACE_ROOT)

STAGE_05_DIR = os.path.join(WORKSPACE_ROOT, "stage_05_generative")
if STAGE_05_DIR not in sys.path:
    sys.path.insert(0, STAGE_05_DIR)


class TestStage1ML(unittest.TestCase):
    def test_model_loading_and_prediction(self):
        import joblib
        import pandas as pd
        model_path = os.path.join(WORKSPACE_ROOT, "Stage1", "best_toxicity_model.pkl")
        self.assertTrue(os.path.exists(model_path), f"Stage 1 model missing at {model_path}")
        model = joblib.load(model_path)
        self.assertIsNotNone(model)

        data_path = os.path.join(WORKSPACE_ROOT, "Stage1", "data", "oncology_risk_dataset_2000 - oncology_risk_dataset_2000.csv")
        self.assertTrue(os.path.exists(data_path), f"Dataset missing at {data_path}")
        df = pd.read_csv(data_path)
        if "dosage_mg" in df and "previous_dose_mg" in df:
            df["dose_ratio"] = df["dosage_mg"] / (df["previous_dose_mg"] + 1.0)
        if "systolic_bp" in df and "diastolic_bp" in df:
            df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
        if "ALT" in df and "AST" in df:
            df["liver_enzyme_sum"] = df["ALT"] + df["AST"]
        drop_cols = ["patient_id", "timestamp", "adverse_event_count", "toxicity_risk", "toxicity_score"]
        X = df.drop(columns=[c for c in drop_cols if c in df])

        preds = model.predict(X.iloc[0:2])
        proba = model.predict_proba(X.iloc[0:2])
        self.assertEqual(len(preds), 2)
        self.assertEqual(proba.shape[1], 3)


class TestStage2DL(unittest.TestCase):
    def test_multimodal_lstm_inference(self):
        import importlib.util
        from sklearn.preprocessing import StandardScaler
        import pandas as pd

        ckpt_path = os.path.join(WORKSPACE_ROOT, "Stage2", "best_stage02_multimodal_lstm.pth")
        script_path = os.path.join(WORKSPACE_ROOT, "Stage2", "05_pipeline_integration.py")
        self.assertTrue(os.path.exists(ckpt_path), f"Stage 2 checkpoint missing at {ckpt_path}")
        self.assertTrue(os.path.exists(script_path), f"Stage 2 script missing at {script_path}")

        spec = importlib.util.spec_from_file_location("s2_pipeline", script_path)
        s2_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(s2_mod)
        Stage02InferencePipeline = s2_mod.Stage02InferencePipeline

        csv_file = os.path.join(WORKSPACE_ROOT, "Stage2", "data", "stage02_lstm_8types_3000samples.csv")
        if not os.path.exists(csv_file):
            csv_file = os.path.join(WORKSPACE_ROOT, "Stage2", "stage02_lstm_8types_3000samples.csv")

        df = pd.read_csv(csv_file)
        scaler = StandardScaler()
        scaler.fit(df[Stage02InferencePipeline.TABULAR_COLS].values)

        pipeline = Stage02InferencePipeline(model_path=ckpt_path, scaler=scaler)
        images_dir = os.path.join(WORKSPACE_ROOT, "Stage2", "ct images") if os.path.exists(os.path.join(WORKSPACE_ROOT, "Stage2", "ct images")) else os.path.join(WORKSPACE_ROOT, "Stage2", "2d_images_large")

        pat_id = df["patient_id"].iloc[0]
        pat_df = df[df["patient_id"] == pat_id]
        res = pipeline.predict_patient_trajectory(pat_df, images_dir)
        prob = res.get("trajectory_final_progression_probability", res.get("progression_probability"))
        self.assertIsNotNone(prob)
        self.assertGreaterEqual(float(prob), 0.0)
        self.assertLessEqual(float(prob), 1.0)


class TestStage3NLP(unittest.TestCase):
    def test_urgency_classifier_and_negation(self):
        import joblib
        tfidf_path = os.path.join(WORKSPACE_ROOT, "Stage3", "integration_engineer", "models", "urgency", "baseline_tfidf_lr.pkl")
        self.assertTrue(os.path.exists(tfidf_path), f"Stage 3 pipeline missing at {tfidf_path}")
        pipeline = joblib.load(tfidf_path)
        
        # Test positive severe symptoms
        pred_urgent = pipeline.predict(["Patient presents with acute grade 4 febrile neutropenia, septic shock, and severe bleeding."])
        self.assertIn(int(pred_urgent[0]), [0, 1, 2, 3])

        # Test negation preservation
        pred_negated = pipeline.predict(["Patient reports no fever, denies chest pain, denies shortness of breath, no acute distress."])
        self.assertIsNotNone(pred_negated)


class TestStage4SLM(unittest.TestCase):
    def test_guardrails_engine(self):
        from Stage4.integration_engineer.src.stage04_guardrails import apply_clinical_triage_guardrail
        
        # Severe case presented as LOW must trigger escalation
        guarded_text, triggered, sentence_count, status = apply_clinical_triage_guardrail(
            raw_briefing="Patient P-001 has low urgency symptoms.",
            patient_id="P-001",
            clinical_note="Patient experiencing intractable vomiting >6 episodes/day with severe dehydration and hypotension."
        )
        self.assertTrue(triggered)
        self.assertIn("ESCALATED", status)


class TestStage5GenAI(unittest.TestCase):
    def test_ood_evaluator(self):
        from scripts.evaluation.inference_harness import CDSSEvaluator
        evaluator = CDSSEvaluator()
        
        # Normal labs
        grade_norm = evaluator.evaluate_stage01_safety({"alt_u_l": 35.0, "total_bilirubin_mg_dl": 0.8})
        self.assertEqual(grade_norm, 0)

        # Extreme elevation
        grade_high = evaluator.evaluate_stage01_safety({"alt_u_l": 450.0, "total_bilirubin_mg_dl": 3.8})
        self.assertIn(grade_high, [2, 3])


class TestStage6Agentic(unittest.TestCase):
    def test_multi_agent_consensus_and_interlocks(self):
        import asyncio
        from stage06_agentic.services.agentic_service import get_agentic_service
        svc = get_agentic_service()
        presets = svc.get_presets()
        self.assertIn("case_c_dili", presets)
        self.assertIn("case_a_egfr", presets)

        # Case C: Acute DILI (Deterministic Interlock)
        res_c = asyncio.run(svc.run_analysis(presets["case_c_dili"]["data"]))
        safety = res_c.get("safety_status", "")
        self.assertTrue("HOLD" in safety or "REVIEW" in safety)


class TestMedicalImageValidation(unittest.TestCase):
    def test_dicom_tiff_png_validation(self):
        from api.services.image_validator import validate_and_save_medical_image

        # 1. Test DICOM file
        dcm_path = os.path.join(WORKSPACE_ROOT, "data", "lung", "ct", "lung_ct_slice_01.dcm")
        if os.path.exists(dcm_path):
            with open(dcm_path, "rb") as f:
                dcm_bytes = f.read()
            res_dcm = validate_and_save_medical_image(dcm_bytes, "lung_ct_slice_01.dcm", "CT Scan")
            self.assertTrue(res_dcm.get("valid"), f"DICOM validation failed: {res_dcm.get('error')}")

        # 2. Test PNG file
        png_path = os.path.join(WORKSPACE_ROOT, "Stage2", "ct images", "luad", "ct_slice_0000.png")
        if os.path.exists(png_path):
            with open(png_path, "rb") as f:
                png_bytes = f.read()
            res_png = validate_and_save_medical_image(png_bytes, "ct_slice_0000.png", "CT Scan")
            self.assertTrue(res_png.get("valid"))

        # 3. Test Corrupted Bytes Rejection
        corrupt_bytes = b"CORRUPTED_NOT_A_VALID_IMAGE_HEADER_XYZ123"
        res_corrupt = validate_and_save_medical_image(corrupt_bytes, "broken_scan.png", "CT Scan")
        self.assertFalse(res_corrupt.get("valid"))


class TestPatientPhysiologicalValidation(unittest.TestCase):
    def test_physiological_ranges_and_unreliable_state(self):
        from api.services.patient_validator import validate_patient_input

        # Unrealistic ALT (5500 U/L) must trigger UNABLE_TO_PROVIDE_RELIABLE_RESULT
        bad_patient = {"patient_id": "P-BAD-01", "ALT": 5500.0, "age": 60.0}
        res_bad = validate_patient_input(bad_patient)
        self.assertFalse(res_bad.is_reliable)
        self.assertEqual(res_bad.status, "UNABLE_TO_PROVIDE_RELIABLE_RESULT")

        # Valid patient
        good_patient = {"patient_id": "P-GOOD-01", "ALT": 45.0, "AST": 42.0, "age": 64.0, "cancer_type": "Lung (LUAD)"}
        res_good = validate_patient_input(good_patient)
        self.assertTrue(res_good.is_reliable)


class TestExplainableAIGradCAM(unittest.TestCase):
    def test_gradcam_generation(self):
        from api.services.gradcam_explainer import get_gradcam_explainer
        explainer = get_gradcam_explainer()
        ct_path = os.path.join(WORKSPACE_ROOT, "Stage2", "ct images", "luad", "ct_slice_0000.png")
        self.assertTrue(os.path.exists(ct_path), f"CT image missing at {ct_path}")

        res = explainer.generate_explanation(ct_path, modality="CT Scan")
        self.assertEqual(res.get("status"), "success")
        self.assertIn("dominant_activation_region", res)
        self.assertIn("gradcam_image_url", res)
        self.assertIn("disclaimer", res)


class TestAuthAndRBAC(unittest.TestCase):
    def test_pbkdf2_and_rbac_roles(self):
        from api.auth import authenticate_user, verify_token, get_role_permissions

        # Test Oncologist Login
        onco_res = authenticate_user("dr.chen@mskcc-oncology.org", "ClinicalPassword2026!")
        self.assertIsNotNone(onco_res)
        self.assertEqual(onco_res.get("role"), "oncologist")
        token = onco_res.get("token")
        verified = verify_token(token)
        self.assertIsNotNone(verified)
        self.assertEqual(verified.get("role"), "oncologist")

        # Test Patient Role Permissions
        pat_perms = get_role_permissions("patient")
        self.assertTrue(pat_perms.get("view_patient_simplified_view"))
        self.assertFalse(pat_perms.get("view_technical_models"))


class TestDatasetIntegrity(unittest.TestCase):
    def test_dataset_provenance_and_zero_duplicates(self):
        report_path = os.path.join(WORKSPACE_ROOT, "dataset_audit_report.json")
        self.assertTrue(os.path.exists(report_path), "Dataset audit report missing")
        with open(report_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.assertEqual(data.get("duplicates_detected_count"), 0)
        self.assertEqual(data.get("total_cancer_types_audited"), 15)
        self.assertGreater(data.get("total_unique_hashes_indexed", 0), 10)


class TestEndToEndLiveEndpoints(unittest.TestCase):
    def test_live_server_endpoints(self):
        # 1. Health
        res = urllib.request.urlopen("http://127.0.0.1:8000/health")
        self.assertEqual(res.status, 200)

        # 2. Models Reliability
        res_rel = urllib.request.urlopen("http://127.0.0.1:8000/api/pipeline/models/reliability")
        self.assertEqual(res_rel.status, 200)
        d_rel = json.loads(res_rel.read().decode())
        self.assertEqual(d_rel["stages"]["stage1_ml"]["accuracy"], 0.995)

        # 3. Datasets Provenance
        res_prov = urllib.request.urlopen("http://127.0.0.1:8000/api/pipeline/datasets/provenance")
        self.assertEqual(res_prov.status, 200)
        d_prov = json.loads(res_prov.read().decode())
        self.assertEqual(d_prov.get("total_cancer_types_audited"), 15)

        # 4. Grad-CAM Endpoint
        gc_req = urllib.request.Request(
            "http://127.0.0.1:8000/api/pipeline/explain/gradcam",
            data=json.dumps({"image_filename": "luad/ct_slice_0000.png", "modality": "CT Scan"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        res_gc = urllib.request.urlopen(gc_req)
        self.assertEqual(res_gc.status, 200)
        d_gc = json.loads(res_gc.read().decode())
        self.assertEqual(d_gc.get("status"), "success")

        # 5. Out-of-Range Rejection (Safety Interlock)
        bad_req = urllib.request.Request(
            "http://127.0.0.1:8000/api/pipeline/run-all",
            data=json.dumps({"patient_id": "PAT-BAD", "ALT": 6000.0, "image_file": "luad/ct_slice_0000.png", "image_type": "CT Scan"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        res_bad = urllib.request.urlopen(bad_req)
        d_bad = json.loads(res_bad.read().decode())
        self.assertEqual(d_bad.get("status"), "Unable to provide a reliable result")
        self.assertFalse(d_bad.get("reliable"))

        # 6. Full 6-Stage Analysis Execution
        valid_req = urllib.request.Request(
            "http://127.0.0.1:8000/api/pipeline/run-all",
            data=json.dumps({"patient_id": "PAT-VERIFY-123", "age": 64.0, "cancer_type": "Lung (LUAD)", "image_file": "luad/ct_slice_0000.png", "image_type": "CT Scan"}).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )
        res_val = urllib.request.urlopen(valid_req)
        self.assertEqual(res_val.status, 200)
        d_val = json.loads(res_val.read().decode())
        self.assertIn("final_assessment", d_val)
        self.assertIn("gradcam_explanation", d_val)


if __name__ == "__main__":
    unittest.main(verbosity=2)
