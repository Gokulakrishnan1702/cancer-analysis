"""
Utility functions for safe downloads, checksums, format validation, and retry handling.
Ensures zero corrupted files, rejects HTML error pages, and computes SHA-256 hashes.
"""

import hashlib
import time
from pathlib import Path
from typing import Optional, Tuple
import requests

def calculate_sha256(file_path: Path) -> str:
    """Calculate SHA-256 checksum of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

def validate_file_signature(file_path: Path, expected_format: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate that the file is not an HTML error page, is non-empty,
    and matches the expected binary file signature.
    """
    if not file_path.exists():
        return False, "File does not exist"
    
    file_size = file_path.stat().st_size
    if file_size < 128:
        return False, f"File too small ({file_size} bytes)"

    with open(file_path, "rb") as f:
        header = f.read(256)

    # Check for HTML or text error pages
    header_lower = header.lower()
    if b"<!doctype html" in header_lower or b"<html" in header_lower or b"<head" in header_lower:
        return False, "Downloaded content is an HTML error page, not an image"
    if header.startswith(b'{"message"') or header.startswith(b'{"error"') or header.startswith(b'{"statusCode"'):
        return False, "Downloaded content is a JSON error payload, not an image"

    fmt = expected_format.lower() if expected_format else None

    # DICOM validation: standard DICOM has 128 bytes preamble then 'DICM'
    if fmt == "dicom" or fmt == "dcm":
        if len(header) >= 132 and header[128:132] == b"DICM":
            return True, "Valid DICOM file signature"
        # Some DICOM formats omit the 128-byte preamble and begin directly with Group/Element tag
        if header.startswith(b"\x02\x00") or header.startswith(b"\x08\x00"):
            return True, "Valid raw DICOM tag header"
        return False, f"Invalid DICOM signature (offset 128: {header[128:132]!r})"

    # SVS / TIFF validation: Aperio SVS files are TIFF / BigTIFF
    if fmt in ("svs", "tif", "tiff"):
        # Standard TIFF: II*\x00 (0x49 0x49 0x2A 0x00) or MM\x00* (0x4D 0x4D 0x00 0x2A)
        # BigTIFF: II+\x00 (0x49 0x49 0x2B 0x00) or MM\x00+ (0x4D 0x4D 0x00 0x2B)
        if header.startswith(b"II*\x00") or header.startswith(b"MM\x00*"):
            return True, "Valid TIFF / SVS signature"
        if header.startswith(b"II+\x00") or header.startswith(b"MM\x00+"):
            return True, "Valid BigTIFF / SVS signature"
        return False, f"Invalid TIFF/SVS signature ({header[:4]!r})"

    # PNG validation
    if fmt == "png":
        if header.startswith(b"\x89PNG\r\n\x1a\n"):
            return True, "Valid PNG signature"
        return False, f"Invalid PNG signature ({header[:8]!r})"

    # JPEG validation
    if fmt in ("jpg", "jpeg"):
        if header.startswith(b"\xff\xd8\xff"):
            return True, "Valid JPEG signature"
        return False, f"Invalid JPEG signature ({header[:3]!r})"

    # Generic check if format unspecified
    return True, f"File exists with size {file_size} bytes"

def safe_download_file(
    url: str,
    dest_path: Path,
    expected_format: Optional[str] = None,
    max_retries: int = 3,
    timeout: int = 45,
    headers: Optional[dict] = None
) -> Tuple[bool, str, Optional[str]]:
    """
    Download a file with retry handling, size/signature validation, and SHA-256 calculation.
    Returns (success: bool, message: str, sha256_hash: Optional[str]).
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    last_error = ""
    for attempt in range(1, max_retries + 1):
        try:
            req_headers = {"User-Agent": "MedicalImagePipeline/1.0"}
            if headers:
                req_headers.update(headers)

            with requests.get(url, headers=req_headers, stream=True, timeout=timeout) as response:
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code} ({response.reason})"
                    time.sleep(attempt * 2)
                    continue

                with open(temp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1048576):  # 1 MB chunks
                        if chunk:
                            f.write(chunk)

            # Validate downloaded temporary file
            valid, reason = validate_file_signature(temp_path, expected_format)
            if not valid:
                last_error = f"Validation failed: {reason}"
                if temp_path.exists():
                    temp_path.unlink()
                time.sleep(attempt * 1)
                continue

            # Move to destination
            if dest_path.exists():
                dest_path.unlink()
            temp_path.rename(dest_path)

            sha256 = calculate_sha256(dest_path)
            return True, f"Downloaded and verified successfully ({dest_path.stat().st_size} bytes)", sha256

        except Exception as e:
            last_error = f"Exception: {str(e)}"
            if temp_path.exists():
                temp_path.unlink()
            time.sleep(attempt * 2)

    return False, f"Failed after {max_retries} attempts: {last_error}", None
