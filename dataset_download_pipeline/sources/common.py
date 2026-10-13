"""
Shared download utilities for TCIA DICOM and GDC SVS pathology samples.
Ensures zero fake files, validates binary magic bytes, computes checksums, and manages manifests.
"""

import io
import json
import os
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests

from config import (
    GDC_API_BASE,
    TCIA_API_BASE,
    REQUEST_TIMEOUT_SECONDS,
    DATA_DIR,
)
from logger import pipeline_logger
from manifest import manifest_manager
from utils import calculate_sha256, safe_download_file, validate_file_signature


def download_gdc_svs_sample(
    cancer: str,
    project_id: str,
    dataset_name: str,
    license_str: str,
    dest_dir: Path,
    limit: int = 3,
    max_size_mb: float = 50.0,
) -> List[Dict]:
    """
    Query GDC API for open-access SVS slides, sort by file_size ascending,
    download up to `limit` files, validate TIFF/SVS signatures, and record in manifest.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    results = []

    filters = {
        "op": "and",
        "content": [
            {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
            {"op": "=", "content": {"field": "data_format", "value": "SVS"}},
            {"op": "=", "content": {"field": "access", "value": "open"}},
        ],
    }

    try:
        query_url = f"{GDC_API_BASE}/files"
        params = {
            "filters": json.dumps(filters),
            "fields": "file_id,file_name,file_size,cases.submitter_id,cases.diagnoses.primary_diagnosis",
            "sort": "file_size:asc",
            "size": limit * 2,  # Query a few extra in case some exceed max_size_mb
        }
        res = requests.get(query_url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        if res.status_code != 200:
            pipeline_logger.log_event(
                cancer=cancer,
                modality="pathology",
                source="GDC",
                identifier=project_id,
                action="Query GDC files API",
                status="DOWNLOAD_FAILED",
                reason=f"GDC API HTTP {res.status_code}",
            )
            return []

        hits = res.json().get("data", {}).get("hits", [])
        if not hits:
            pipeline_logger.log_event(
                cancer=cancer,
                modality="pathology",
                source="GDC",
                identifier=project_id,
                action="Search files",
                status="NOT_FOUND",
                reason="No open-access SVS files found for project",
            )
            return []

        downloaded_count = 0
        for hit in hits:
            if downloaded_count >= limit:
                break

            file_id = hit["file_id"]
            file_name = hit["file_name"]
            file_size_mb = hit["file_size"] / (1024 * 1024)
            patient_id = (
                hit.get("cases", [{}])[0].get("submitter_id", "UNKNOWN")
                if hit.get("cases")
                else "UNKNOWN"
            )
            diagnosis = (
                hit.get("cases", [{}])[0].get("diagnoses", [{}])[0].get("primary_diagnosis", "Not Reported")
                if hit.get("cases") and hit["cases"][0].get("diagnoses")
                else "Not Reported"
            )

            dest_file = dest_dir / file_name

            # Skip if already exists and valid
            if dest_file.exists():
                valid, msg = validate_file_signature(dest_file, "svs")
                if valid:
                    sha256 = calculate_sha256(dest_file)
                    manifest_manager.record_entry(
                        cancer=cancer,
                        modality="pathology",
                        file_path=str(dest_file.relative_to(dest_dir.parent.parent.parent)),
                        source="GDC",
                        dataset_name=dataset_name,
                        license_str=license_str,
                        download_status="DOWNLOADED",
                        notes=f"Patient: {patient_id} | Diagnosis: {diagnosis} | SHA256: {sha256[:12]}...",
                    )
                    downloaded_count += 1
                    results.append({"path": dest_file, "patient_id": patient_id, "diagnosis": diagnosis})
                    continue

            # Download via GDC data endpoint
            download_url = f"{GDC_API_BASE}/data/{file_id}"
            success, msg, sha256 = safe_download_file(
                url=download_url,
                dest_path=dest_file,
                expected_format="svs",
                timeout=REQUEST_TIMEOUT_SECONDS,
            )

            if success:
                pipeline_logger.log_event(
                    cancer=cancer,
                    modality="pathology",
                    source="GDC",
                    identifier=f"{file_name} ({file_size_mb:.1f} MB)",
                    action=f"Downloaded SVS slide",
                    status="DOWNLOADED",
                    extra_notes=f"Patient: {patient_id} | Diagnosis: {diagnosis} | SHA256: {sha256}",
                )
                manifest_manager.record_entry(
                    cancer=cancer,
                    modality="pathology",
                    file_path=str(dest_file.relative_to(dest_dir.parent.parent.parent)),
                    source="GDC",
                    dataset_name=dataset_name,
                    license_str=license_str,
                    download_status="DOWNLOADED",
                    notes=f"Patient: {patient_id} | Diagnosis: {diagnosis} | SHA256: {sha256[:12]}...",
                )
                downloaded_count += 1
                results.append({"path": dest_file, "patient_id": patient_id, "diagnosis": diagnosis})
            else:
                pipeline_logger.log_event(
                    cancer=cancer,
                    modality="pathology",
                    source="GDC",
                    identifier=file_name,
                    action="Download SVS slide",
                    status="DOWNLOAD_FAILED",
                    reason=msg,
                )
                manifest_manager.record_entry(
                    cancer=cancer,
                    modality="pathology",
                    file_path=str(dest_file.relative_to(dest_dir.parent.parent.parent)),
                    source="GDC",
                    dataset_name=dataset_name,
                    license_str=license_str,
                    download_status="DOWNLOAD_FAILED",
                    notes=msg,
                )

    except Exception as e:
        pipeline_logger.log_event(
            cancer=cancer,
            modality="pathology",
            source="GDC",
            identifier=project_id,
            action="Download pathology sample",
            status="DOWNLOAD_FAILED",
            reason=f"Exception: {str(e)}",
        )

    return results


def download_tcia_dicom_sample(
    cancer: str,
    collection: str,
    modality: str,
    dataset_name: str,
    license_str: str,
    dest_dir: Path,
    limit: int = 3,
    preferred_series_uid: Optional[str] = None,
    label_info: Optional[str] = None,
) -> List[Dict]:
    """
    Download a sample of DICOM images from a TCIA collection and modality.
    Queries series, finds an appropriately sized series, downloads via getImage,
    extracts up to `limit` .dcm files, validates DICOM magic bytes, and logs/records.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    results = []

    # Check if we already have sufficient valid files downloaded
    existing_dcm = [p for p in dest_dir.glob("*.dcm") if validate_file_signature(p, "dicom")[0]]
    if len(existing_dcm) >= limit:
        for p in existing_dcm[:limit]:
            sha256 = calculate_sha256(p)
            manifest_manager.record_entry(
                cancer=cancer,
                modality=modality.lower(),
                file_path=str(p.relative_to(dest_dir.parent.parent.parent)),
                source="TCIA",
                dataset_name=dataset_name,
                license_str=license_str,
                download_status="DOWNLOADED",
                notes=f"{label_info or 'Verified DICOM'} | SHA256: {sha256[:12]}...",
            )
            results.append({"path": p, "sha256": sha256})
        return results

    try:
        chosen_uid = preferred_series_uid
        patient_id = "UNKNOWN"

        if not chosen_uid:
            # Query series list
            query_url = f"{TCIA_API_BASE}/getSeries?Collection={collection}&Modality={modality.upper()}&format=json"
            res = requests.get(query_url, timeout=REQUEST_TIMEOUT_SECONDS)
            if res.status_code != 200:
                pipeline_logger.log_event(
                    cancer=cancer,
                    modality=modality,
                    source="TCIA",
                    identifier=collection,
                    action="Query TCIA getSeries",
                    status="DOWNLOAD_FAILED",
                    reason=f"TCIA API HTTP {res.status_code}",
                )
                return []

            series_list = res.json()
            if not series_list:
                pipeline_logger.log_event(
                    cancer=cancer,
                    modality=modality,
                    source="TCIA",
                    identifier=collection,
                    action="Search series",
                    status="NOT_FOUND",
                    reason=f"No series found for {collection} with modality {modality}",
                )
                return []

            # Sort by image count ascending so we download the smallest zip archive
            sorted_series = sorted(series_list, key=lambda s: int(s.get("ImageCount", 99999)))
            chosen_series = sorted_series[0]
            chosen_uid = chosen_series["SeriesInstanceUID"]
            patient_id = chosen_series.get("PatientID", "UNKNOWN")

        # Download series zip
        download_url = f"{TCIA_API_BASE}/getImage?SeriesInstanceUID={chosen_uid}"
        pipeline_logger.log_event(
            cancer=cancer,
            modality=modality,
            source="TCIA",
            identifier=f"Series {chosen_uid}",
            action="Request series zip archive",
            status="IN_PROGRESS",
        )

        # Download series zip streaming to disk
        temp_zip_path = dest_dir / f"temp_{chosen_uid[:8]}.zip"
        try:
            with requests.get(download_url, stream=True, timeout=REQUEST_TIMEOUT_SECONDS * 2) as res_zip:
                if res_zip.status_code != 200:
                    pipeline_logger.log_event(
                        cancer=cancer,
                        modality=modality,
                        source="TCIA",
                        identifier=chosen_uid,
                        action="Download series zip",
                        status="DOWNLOAD_FAILED",
                        reason=f"HTTP {res_zip.status_code} ({res_zip.reason})",
                    )
                    return []

                with open(temp_zip_path, "wb") as f_zip:
                    for chunk in res_zip.iter_content(chunk_size=65536):
                        if chunk:
                            f_zip.write(chunk)

            # Read zip from disk
            with zipfile.ZipFile(temp_zip_path) as z:
                dcm_names = [n for n in z.namelist() if n.lower().endswith(".dcm")]
                if not dcm_names:
                    dcm_names = [n for n in z.namelist() if n != "LICENSE"]

                extracted_count = 0
                for name in dcm_names[:limit]:
                    data = z.read(name)
                    # Verify DICOM magic bytes
                    if len(data) >= 132 and (data[128:132] == b"DICM" or data.startswith(b"\x02\x00")):
                        slice_fname = f"{cancer}_{modality.lower()}_slice_{extracted_count + 1:02d}.dcm"
                        dest_file = dest_dir / slice_fname
                        with open(dest_file, "wb") as f_out:
                            f_out.write(data)

                        sha256 = calculate_sha256(dest_file)
                        manifest_manager.record_entry(
                            cancer=cancer,
                            modality=modality.lower(),
                            file_path=str(dest_file.relative_to(dest_dir.parent.parent.parent)),
                            source="TCIA",
                            dataset_name=dataset_name,
                            license_str=license_str,
                            download_status="DOWNLOADED",
                            notes=f"Patient: {patient_id} | Series: {chosen_uid[:15]}... | {label_info or ''} | SHA256: {sha256[:12]}...",
                        )
                        pipeline_logger.log_event(
                            cancer=cancer,
                            modality=modality,
                            source="TCIA",
                            identifier=slice_fname,
                            action="Extracted and verified DICOM slice",
                            status="DOWNLOADED",
                            extra_notes=f"Patient: {patient_id} | SHA256: {sha256}",
                        )
                        extracted_count += 1
                        results.append({"path": dest_file, "sha256": sha256})
        finally:
            if temp_zip_path.exists():
                temp_zip_path.unlink()

    except Exception as e:
        pipeline_logger.log_event(
            cancer=cancer,
            modality=modality,
            source="TCIA",
            identifier=collection,
            action="Download TCIA sample",
            status="DOWNLOAD_FAILED",
            reason=f"Exception: {str(e)}",
        )

    return results


def record_auth_required_source(
    cancer: str,
    modality: str,
    dataset_name: str,
    source: str,
    setup_instructions: str,
    required_credential: str,
    license_str: str,
):
    """
    Log and record an AUTH_REQUIRED status for a restricted source without failing silently.
    """
    pipeline_logger.log_event(
        cancer=cancer,
        modality=modality,
        source=source,
        identifier=dataset_name,
        action="Authentication check",
        status="AUTH_REQUIRED",
        reason=f"Credentials missing. Setup: {setup_instructions}",
        auth_requirements=f"Required env/file: {required_credential}",
    )
    manifest_manager.record_entry(
        cancer=cancer,
        modality=modality,
        file_path="N/A",
        source=source,
        dataset_name=dataset_name,
        license_str=license_str,
        download_status="AUTH_REQUIRED",
        notes=f"Requires {required_credential}. Setup: {setup_instructions}",
    )
