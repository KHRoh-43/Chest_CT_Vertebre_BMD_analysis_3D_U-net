import os
import subprocess
import sys
import json
from datetime import datetime

# ============================================================
# 설정
# ============================================================
DATA_ROOT = r"C:\Users\comds\Desktop\졸업작품\dataset\NSCLC LUNG CANCER\NSCLC-Radiomics"
OUTPUT_DIR = r"C:\Users\comds\Desktop\졸업작품\dataset\NSCLC LUNG CANCER\nifti_output"
LOG_PATH = os.path.join(OUTPUT_DIR, "conversion_log.json")

COMPRESS = True
ANONYMIZE = True
MIN_DICOM_COUNT = 10  # DICOM 파일이 이 수 이상인 폴더만 CT 볼륨으로 인정

# ============================================================
# dcm2niix 확인
# ============================================================
def check_dcm2niix():
    try:
        result = subprocess.run(["dcm2niix", "--version"], capture_output=True, text=True)
        print(f"dcm2niix 확인: {result.stdout.strip()}")
        return True
    except FileNotFoundError:
        print("dcm2niix 미설치. pip install dcm2niix")
        return False

# ============================================================
# 환자 폴더에서 실제 CT 시리즈 폴더 찾기
# ============================================================
def find_ct_series(patient_path):
    """
    환자 폴더 내에서:
    - 300.000000-Segmentation 제외
    - DICOM 파일 수가 MIN_DICOM_COUNT 이상인 폴더만 선택
    - 여러 개면 DICOM 수가 가장 많은 것 선택
    """
    best_folder = None
    best_count = 0

    for study_dir in os.listdir(patient_path):
        study_path = os.path.join(patient_path, study_dir)
        if not os.path.isdir(study_path):
            continue

        for series_dir in os.listdir(study_path):
            series_path = os.path.join(study_path, series_dir)
            if not os.path.isdir(series_path):
                continue

            # Segmentation 폴더 제외
            if "segmentation" in series_dir.lower():
                continue

            # DICOM 파일 수 카운트
            dcm_count = len([f for f in os.listdir(series_path)
                           if f.endswith(('.dcm', '.DCM'))
                           or (not os.path.splitext(f)[1])])

            if dcm_count >= MIN_DICOM_COUNT and dcm_count > best_count:
                best_count = dcm_count
                best_folder = series_path

    return best_folder, best_count

# ============================================================
# 일괄 변환
# ============================================================
def batch_convert():
    if not check_dcm2niix():
        sys.exit(1)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 환자 폴더 수집 (LUNG1-001 ~ LUNG1-422)
    patient_dirs = sorted([
        d for d in os.listdir(DATA_ROOT)
        if os.path.isdir(os.path.join(DATA_ROOT, d)) and d.startswith("LUNG1-")
    ])

    print(f"환자 수: {len(patient_dirs)}명")
    print(f"출력 폴더: {OUTPUT_DIR}\n")

    log = []
    success = 0
    fail = 0
    skip = 0

    for idx, patient_id in enumerate(patient_dirs, 1):
        patient_path = os.path.join(DATA_ROOT, patient_id)
        ct_folder, dcm_count = find_ct_series(patient_path)

        if ct_folder is None:
            print(f"[{idx:3d}/422] {patient_id} → ⚠️ CT 시리즈 없음 (SKIP)")
            log.append({"patient_id": patient_id, "status": "skip", "reason": "no CT series"})
            skip += 1
            continue

        filename = patient_id
        ext = ".nii.gz" if COMPRESS else ".nii"
        output_file = os.path.join(OUTPUT_DIR, filename + ext)

        cmd = [
            "dcm2niix",
            "-z", "y" if COMPRESS else "n",
            "-m", "y",
            "-f", filename,
            "-o", OUTPUT_DIR,
        ]
        if ANONYMIZE:
            cmd.extend(["-ba", "y"])
        cmd.append(ct_folder)

        result = subprocess.run(cmd, capture_output=True, text=True)
        file_exists = os.path.exists(output_file)

        if result.returncode == 0 and file_exists:
            size_mb = os.path.getsize(output_file) / (1024 * 1024)
            print(f"[{idx:3d}/422] {patient_id} → ✅ ({dcm_count} slices, {size_mb:.1f} MB)")
            log.append({
                "patient_id": patient_id,
                "status": "success",
                "dicom_folder": ct_folder,
                "dicom_count": dcm_count,
                "output_file": output_file,
                "size_mb": round(size_mb, 1),
            })
            success += 1
        else:
            print(f"[{idx:3d}/422] {patient_id} → ❌ {result.stderr[:80]}")
            log.append({
                "patient_id": patient_id,
                "status": "fail",
                "dicom_folder": ct_folder,
                "error": result.stderr[:200],
            })
            fail += 1

    # 로그 저장
    log_data = {
        "timestamp": datetime.now().isoformat(),
        "data_root": DATA_ROOT,
        "output_dir": OUTPUT_DIR,
        "summary": {
            "total": len(patient_dirs),
            "success": success,
            "failed": fail,
            "skipped": skip,
        },
        "results": log,
    }

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"변환 완료")
    print(f"{'=' * 60}")
    print(f"  성공 : {success}개")
    print(f"  실패 : {fail}개")
    print(f"  스킵 : {skip}개")
    print(f"  출력 : {OUTPUT_DIR}")
    print(f"  로그 : {LOG_PATH}")

if __name__ == "__main__":
    batch_convert()
