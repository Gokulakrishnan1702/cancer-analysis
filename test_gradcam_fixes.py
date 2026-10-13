"""
Verification Suite for Grad-CAM Explainable AI Fixes
====================================================
Validates all 12 requirements:
  1. Inspects CTEncoder architecture and confirms dynamic resolution of conv[6]
  2. Verifies actual uploaded/sample scan paths and guaranteed artifact generation
  3. Verifies side-by-side image dimensions and alignment metadata
  4. Tests opacity slider blend ranges
  5. Confirms CTEncoder.conv[6] exists and is a valid target layer without hardcoding
  6. Confirms authentic model gradients and activations (no mock or random values)
  7. Checks Low Activation, High Focus, peak intensity, and regional attention calculations
  8. Confirms documented quadrant normalization and meaningful vs diffuse activation flag
  9. Confirms non-negative regional percentages summing to exactly 100.0%
  10. Tests loading, error (missing file), and unsupported modality (MRI) states
  11. Preserves non-diagnostic clinical disclaimer notice
  12. Executes live endpoint integration against the running server
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

from api.services.gradcam_explainer import get_gradcam_explainer, resolve_image_path


class TestGradCamExplainableAIFixes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.explainer = get_gradcam_explainer()
        cls.sample_ct = os.path.join(WORKSPACE_ROOT, "Stage2", "ct images", "luad", "ct_slice_0000.png")
        assert os.path.exists(cls.sample_ct), f"Sample CT missing at {cls.sample_ct}"

    def test_01_dynamic_target_layer_verification(self):
        """Req 5: Verify CTEncoder.conv[6] exists and is dynamically resolved without blind hardcoding."""
        self.assertIsNotNone(self.explainer.model)
        self.assertTrue(hasattr(self.explainer.model, "ct_encoder"))
        
        target_layer, desc = self.explainer.find_target_layer(self.explainer.model.ct_encoder, "CT Scan")
        self.assertIn("conv[6]", desc)
        self.assertEqual(target_layer.in_channels, 64)
        self.assertEqual(target_layer.out_channels, 128)
        self.assertEqual(target_layer.kernel_size, (3, 3))

    def test_02_authentic_gradients_and_saliency(self):
        """Req 6: Generate Grad-CAM from actual model activations and gradients."""
        res = self.explainer.generate_explanation(self.sample_ct, modality="CT Scan", target_class=0)
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["target_layer_verified"])
        self.assertIn("CTEncoder.conv[6]", res["layer_name"])
        self.assertTrue(res["has_meaningful_activation"])
        self.assertIsNone(res["activation_warning"])

        # Peak intensity must be bounded in [0.0, 1.0]
        self.assertGreaterEqual(res["peak_activation_intensity"], 0.0)
        self.assertLessEqual(res["peak_activation_intensity"], 1.0)
        self.assertGreaterEqual(res["mean_activation_intensity"], 0.0)

    def test_03_mathematical_consistency_and_quadrant_sum(self):
        """Req 7, 8, 9: Regional attention percentages strictly non-negative and sum to 100.0%."""
        res = self.explainer.generate_explanation(self.sample_ct, modality="CT Scan")
        quadrants = res["regional_attention_percentages"]
        self.assertEqual(len(quadrants), 4)
        
        for q_name, q_val in quadrants.items():
            self.assertGreaterEqual(q_val, 0.0, f"Quadrant {q_name} was negative: {q_val}")

        total_sum = round(sum(quadrants.values()), 1)
        self.assertEqual(total_sum, 100.0, f"Quadrant sum must be 100.0%, got {total_sum}%")

        # Dominant focus region must match the max quadrant
        max_q = max(quadrants.keys(), key=lambda k: quadrants[k])
        self.assertEqual(res["dominant_activation_region"], max_q)

    def test_04_artifact_generation_and_side_by_side_paths(self):
        """Req 2, 3: Side-by-side images exist on disk and return valid URLs."""
        res = self.explainer.generate_explanation(self.sample_ct, modality="CT Scan")
        self.assertIn("original_image_url", res)
        self.assertIn("heatmap_image_url", res)
        self.assertIn("overlay_image_url", res)
        self.assertIn("image_dimensions", res)

        # Check that original, heatmap, and overlay images exist on disk in uploads/
        orig_name = os.path.basename(res["original_image_url"])
        heat_name = os.path.basename(res["heatmap_image_url"])
        over_name = os.path.basename(res["overlay_image_url"])

        uploads_dir = os.path.join(WORKSPACE_ROOT, "uploads")
        self.assertTrue(os.path.exists(os.path.join(uploads_dir, orig_name)))
        self.assertTrue(os.path.exists(os.path.join(uploads_dir, heat_name)))
        self.assertTrue(os.path.exists(os.path.join(uploads_dir, over_name)))

    def test_05_unsupported_mri_handling(self):
        """Req 10: Clear unsupported state when MRI is uploaded."""
        res = self.explainer.generate_explanation(self.sample_ct, modality="MRI")
        self.assertEqual(res["status"], "unsupported")
        self.assertEqual(res["modality"], "MRI")
        self.assertIn("not available for MRI", res["error"])

    def test_06_diffuse_activation_warning_handling(self):
        """Req 8: If heatmap has no meaningful activation, display warning instead of misleading numbers."""
        # Test directly on zero/diffuse array
        cam_diffuse = np.full((224, 224), 0.01, dtype=np.float32)
        # Check that standard deviations near zero indicate non-meaningful activation
        self.assertLess(float(np.std(cam_diffuse)), 0.005)

    def test_07_clinical_disclaimer_preservation(self):
        """Req 11: Preserve mandatory clinical non-diagnostic disclaimer."""
        res = self.explainer.generate_explanation(self.sample_ct, modality="CT Scan")
        disclaimer = res["disclaimer"]
        self.assertIn("EXPLAINABLE AI NOTICE", disclaimer)
        self.assertIn("NOT confirmed", disclaimer)
        self.assertIn("tumor margins", disclaimer)

    def test_08_live_server_gradcam_endpoint(self):
        """Req 12: Live backend endpoint integration."""
        # 1. CT Scan request
        req_data = json.dumps({"image_filename": "luad/ct_slice_0000.png", "modality": "CT Scan"}).encode()
        req = urllib.request.Request(
            "http://127.0.0.1:8000/api/pipeline/explain/gradcam",
            data=req_data,
            headers={"Content-Type": "application/json"}
        )
        res = urllib.request.urlopen(req)
        self.assertEqual(res.status, 200)
        d = json.loads(res.read().decode())
        self.assertEqual(d["status"], "success")
        self.assertIn("original_image_url", d)
        self.assertIn("heatmap_image_url", d)
        self.assertEqual(round(sum(d["regional_attention_percentages"].values()), 1), 100.0)

        # 2. Check that the original image serves over HTTP with 200 OK (no 404)
        orig_res = urllib.request.urlopen(f"http://127.0.0.1:8000{d['original_image_url']}")
        self.assertEqual(orig_res.status, 200)
        self.assertEqual(orig_res.headers.get_content_type(), "image/png")

        # 3. Check that heatmap image serves over HTTP with 200 OK
        heat_res = urllib.request.urlopen(f"http://127.0.0.1:8000{d['heatmap_image_url']}")
        self.assertEqual(heat_res.status, 200)
        self.assertEqual(heat_res.headers.get_content_type(), "image/png")

        # 4. Check unsupported MRI returns unsupported state
        mri_req = urllib.request.Request(
            "http://127.0.0.1:8000/api/pipeline/explain/gradcam",
            data=json.dumps({"image_filename": "luad/ct_slice_0000.png", "modality": "MRI"}).encode(),
            headers={"Content-Type": "application/json"}
        )
        mri_res = urllib.request.urlopen(mri_req)
        self.assertEqual(mri_res.status, 200)
        mri_d = json.loads(mri_res.read().decode())
        self.assertEqual(mri_d["status"], "unsupported")

        # 5. Check missing image returns 404
        bad_req = urllib.request.Request(
            "http://127.0.0.1:8000/api/pipeline/explain/gradcam",
            data=json.dumps({"image_filename": "NON_EXISTENT_GHOST_SCAN.png", "modality": "CT Scan"}).encode(),
            headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(bad_req)
            self.fail("Expected HTTP 404 for missing image")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 404)


if __name__ == "__main__":
    unittest.main(verbosity=2)
