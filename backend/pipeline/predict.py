"""
예측 파이프라인:
1. 업로드된 .nii 파일을 임시 디렉터리에 저장
2. seg 파일을 GLI/MEN/MET 폴더에 복사 (GT seg proxy)
3. feature_extractor로 21차원 벡터 생성
4. RF 모델로 예측
"""
import sys
import shutil
import pickle
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from loguru import logger

from core.config import settings

# ML 파이프라인 경로를 sys.path에 추가
ml_src = settings.ml_path / "src"
if str(ml_src) not in sys.path:
    sys.path.insert(0, str(settings.ml_path))

from src.classifier.feature_extractor import build_feature_vector  # noqa: E402

CLASSES = ["GLI", "MEN", "MET"]

# 뇌 영역 추정 (좌표 기반 단순 매핑)
def estimate_brain_region(seg_path: Path, tumor_type: str) -> str:
    """세그멘테이션 중심 좌표 → 뇌 영역 추정"""
    try:
        import nibabel as nib
        import numpy as np

        img = nib.load(str(seg_path))
        data = img.get_fdata()
        affine = img.affine

        # 종양 복셀 (배경 제외) 좌표
        vox = np.argwhere(data > 0)
        if len(vox) == 0:
            return "불명확"

        # 복셀 중심 → MNI 근사 좌표 (mm)
        centroid_vox = vox.mean(axis=0)
        centroid_mm = affine[:3, :3] @ centroid_vox + affine[:3, 3]
        x, y, z = centroid_mm  # L-R, A-P, I-S

        # 좌/우 반구
        side = "우측" if x > 0 else "좌측"

        # 전후 (y축 기준 대략)
        if y > 30:
            region = "전두엽"
        elif y > 0:
            region = "두정엽"
        elif y > -30:
            region = "측두엽"
        else:
            region = "후두엽"

        # 깊이 (z축 기준)
        if z > 50:
            depth = "상부"
        elif z > 10:
            depth = "중부"
        else:
            depth = "하부"

        return f"{side} {region} {depth}"
    except Exception:
        return "불명확"
FEAT_COLS = [
    f"{c.lower()}_{f}"
    for c in CLASSES
    for f in [
        "total_voxels", "total_volume_mm3", "et_ratio", "edema_ratio",
        "core_ratio", "lesion_count", "has_tumor"
    ]
]


def load_classifier():
    """RF 모델 로드 (싱글턴 패턴)"""
    if not hasattr(load_classifier, "_model"):
        path = settings.model_path
        if not path.exists():
            raise FileNotFoundError(f"모델 파일 없음: {path}")
        # Windows에서 저장된 pickle을 Linux에서 로드할 때 WindowsPath 패치
        import pathlib
        temp = pathlib.WindowsPath
        pathlib.WindowsPath = pathlib.PosixPath
        try:
            with open(path, "rb") as f:
                load_classifier._model = pickle.load(f)
        finally:
            pathlib.WindowsPath = temp
        logger.info(f"RF 모델 로드 완료: {path}")
    return load_classifier._model


def run_prediction(
    subject_id: str,
    seg_nii_path: Path,
) -> dict:
    """
    Args:
        subject_id: 고유 식별자 (UUID)
        seg_nii_path: 업로드된 seg .nii 파일 경로

    Returns:
        {
            subject_id, prediction, confidence,
            gli_prob, men_prob, met_prob, features
        }
    """
    clf = load_classifier()

    # seg를 GLI/MEN/MET 폴더에 복사 (feature_extractor 입력 구조 맞춤)
    seg_paths = {}
    for cls in CLASSES:
        out_dir = settings.seg_output_path / cls
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{subject_id}.nii"
        shutil.copy2(seg_nii_path, dst)
        seg_paths[cls] = dst

    # 특징 추출
    feats = build_feature_vector(seg_paths)
    X = pd.DataFrame([feats])[FEAT_COLS]

    # 예측
    y_pred_enc = clf.model.predict(X.values)[0]
    proba = clf.model.predict_proba(X.values)[0]

    label_map = {0: "GLI", 1: "MEN", 2: "MET"}
    prediction = label_map[y_pred_enc]
    confidence = float(proba[y_pred_enc])

    brain_region = estimate_brain_region(seg_nii_path, prediction)

    result = {
        "subject_id": subject_id,
        "prediction": prediction,
        "confidence": confidence,
        "gli_prob": float(proba[0]),
        "men_prob": float(proba[1]),
        "met_prob": float(proba[2]),
        "features": feats,
        "brain_region": brain_region,
    }
    logger.info(f"[{subject_id}] → {prediction} (conf={confidence:.3f}) @ {brain_region}")
    return result
