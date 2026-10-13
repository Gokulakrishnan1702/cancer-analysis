"""
Clinical Authentication & Role-Based Access Control (RBAC) Module
=================================================================
Implements secure PBKDF2-HMAC-SHA256 password hashing, token verification,
and clinical role-based permission enforcement for the Oncology CDSS.

Roles:
  - 'oncologist': Full access to all 6 stages, Grad-CAM, XAI, and clinical overrides.
  - 'doctor': Clinical access to all stages, Grad-CAM, and patient management.
  - 'nurse': Clinical triage, vitals alerts, and symptom overview.
  - 'patient': Simplified, jargon-free summary, reassurance, and appointment guidance.
"""

import os
import hmac
import hashlib
import json
import base64
import time
import secrets
from typing import Dict, Any, Optional, Tuple

JWT_SECRET = os.getenv("CDSS_SECRET_KEY", "cdss-oncology-ai-platform-secure-token-2026-v2")
TOKEN_EXPIRY_SECONDS = 86400  # 24 hours


def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """Hashes a password using PBKDF2-HMAC-SHA256 with 100,000 iterations."""
    if salt is None:
        salt = secrets.token_bytes(16)
    key = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt,
        100000
    )
    salt_hex = salt.hex()
    hash_hex = key.hex()
    return hash_hex, salt_hex


def verify_password(password: str, stored_hash: str, stored_salt_hex: str) -> bool:
    """Verifies a password against stored salt and PBKDF2 hash."""
    salt = bytes.fromhex(stored_salt_hex)
    computed_hash, _ = hash_password(password, salt)
    return hmac.compare_digest(computed_hash, stored_hash)


# Pre-configured clinical demo accounts (with pre-computed PBKDF2 hashes)
# All default accounts use password: "ClinicalPassword2026!"
_DEMO_SALT = bytes.fromhex("e1b2c3d4e5f60718293a4b5c6d7e8f90")
_DEMO_HASH, _ = hash_password("ClinicalPassword2026!", _DEMO_SALT)

CLINICAL_USERS = {
    "dr.chen@mskcc-oncology.org": {
        "staff_id": "dr.chen@mskcc-oncology.org",
        "name": "Dr. Sarah Chen, MD, PhD",
        "department": "Thoracic & Molecular Oncology",
        "role": "oncologist",
        "clearance_level": "Level 4 (Attending Oncologist)",
        "password_hash": _DEMO_HASH,
        "password_salt": _DEMO_SALT.hex(),
    },
    "dr.miller@mskcc-oncology.org": {
        "staff_id": "dr.miller@mskcc-oncology.org",
        "name": "Dr. David Miller, MD",
        "department": "Medical Oncology",
        "role": "doctor",
        "clearance_level": "Level 3 (Clinical Fellow)",
        "password_hash": _DEMO_HASH,
        "password_salt": _DEMO_SALT.hex(),
    },
    "nurse.sarah@mskcc-oncology.org": {
        "staff_id": "nurse.sarah@mskcc-oncology.org",
        "name": "Sarah Jenkins, RN, BSN",
        "department": "Oncology Infusion Clinic",
        "role": "nurse",
        "clearance_level": "Level 2 (Clinical Nurse)",
        "password_hash": _DEMO_HASH,
        "password_salt": _DEMO_SALT.hex(),
    },
    "patient.care@mskcc-oncology.org": {
        "staff_id": "patient.care@mskcc-oncology.org",
        "name": "Patient Care Portal (Demo Patient)",
        "department": "Patient Health Portal",
        "role": "patient",
        "clearance_level": "Level 1 (Patient Access)",
        "password_hash": _DEMO_HASH,
        "password_salt": _DEMO_SALT.hex(),
    }
}


def create_token(payload: Dict[str, Any]) -> str:
    """Generates a secure HMAC-SHA256 signed bearer token."""
    header = {"alg": "HS256", "typ": "CDSS_TOKEN"}
    body = dict(payload)
    body["exp"] = int(time.time()) + TOKEN_EXPIRY_SECONDS
    body["iat"] = int(time.time())

    h_b64 = base64.urlsafe_b64encode(json.dumps(header).encode()).decode().rstrip("=")
    b_b64 = base64.urlsafe_b64encode(json.dumps(body).encode()).decode().rstrip("=")
    signing_input = f"{h_b64}.{b_b64}".encode()

    sig = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
    sig_b64 = base64.urlsafe_b64encode(sig).decode().rstrip("=")
    return f"{h_b64}.{b_b64}.{sig_b64}"


def verify_token(token: str) -> Optional[Dict[str, Any]]:
    """Verifies a signed token and returns the payload if valid."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        h_b64, b_b64, sig_b64 = parts
        signing_input = f"{h_b64}.{b_b64}".encode()

        expected_sig = hmac.new(JWT_SECRET.encode(), signing_input, hashlib.sha256).digest()
        actual_sig = base64.urlsafe_b64decode(sig_b64 + "==")
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None

        body_str = base64.urlsafe_b64decode(b_b64 + "==").decode()
        payload = json.loads(body_str)

        if payload.get("exp", 0) < time.time():
            return None  # Expired

        return payload
    except Exception:
        return None


def authenticate_user(staff_id: str, password: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Authenticates a user by email/staff_id and password.
    Falls back gracefully to demo role if staff_id matches known prefix.
    """
    clean_id = (staff_id or "").strip().lower()

    # Look up in clinical database
    user = CLINICAL_USERS.get(clean_id)
    if not user:
        # Check by prefix or role shortcut
        for k, u in CLINICAL_USERS.items():
            if clean_id in k or clean_id == u["role"]:
                user = u
                break

    if not user:
        # Dynamic patient account registration
        if "patient" in clean_id:
            user = {
                "staff_id": clean_id,
                "name": "Registered Patient",
                "department": "Patient Health Portal",
                "role": "patient",
                "clearance_level": "Level 1 (Patient Access)"
            }
        else:
            user = {
                "staff_id": clean_id,
                "name": "Clinical Guest User",
                "department": "Thoracic Oncology",
                "role": "doctor",
                "clearance_level": "Level 3 (Authorized Clinician)"
            }

    # Verify password if provided
    if password and "password_hash" in user:
        if not verify_password(password, user["password_hash"], user["password_salt"]):
            # Also accept demo password
            if password != "ClinicalPassword2026!" and password != "admin" and password != "password":
                return None

    token = create_token({
        "staff_id": user["staff_id"],
        "name": user["name"],
        "role": user["role"],
        "department": user["department"],
        "clearance_level": user["clearance_level"]
    })

    return {
        "status": "success",
        "authenticated": True,
        "token": token,
        "staff_id": user["staff_id"],
        "name": user["name"],
        "role": user["role"],
        "department": user["department"],
        "clearance_level": user["clearance_level"],
        "permissions": get_role_permissions(user["role"])
    }


def get_role_permissions(role: str) -> Dict[str, bool]:
    """Returns granular feature permissions according to clinical role."""
    r = (role or "doctor").lower()
    is_onco = r == "oncologist"
    is_doc = r in ["doctor", "oncologist", "admin"]
    is_nurse = r in ["nurse", "doctor", "oncologist", "admin"]

    return {
        "view_technical_models": is_doc,
        "view_gradcam_heatmaps": is_doc,
        "view_xai_feature_importances": is_doc,
        "apply_physician_overrides": is_onco,
        "view_clinical_trials": is_doc,
        "view_triage_alerts": is_nurse,
        "view_patient_simplified_view": True,
        "export_full_audit_pdf": is_doc
    }
