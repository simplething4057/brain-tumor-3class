"""
Segmentation 출력 → Feature Vector 변환

3종 모델의 출력 레이블 구조:
  GLI: {0: BG, 1: NCR(괴사핵), 2: ED(부종), 3: ET(조영증강)}
  MEN: {0: BG, 1: ET,          2: NE-T,     3: SNFH}
  MET: {0: BG, 1: NETC,        2: SNFH,     3: ET}
"""
from pathlib import Path
import numpy as np
from loguru import logger

from src.utils.nii_utils import load_nii, get_voxel_volume_mm3, count_lesions


LABEL_MAPS = {
    "GLI": {0: "BG", 1: "NCR", 2: "ED",   3: "ET"},
    "MEN": {0: "BG", 1: "ET",  2: "NE_T", 3: "SNFH"},
    "MET": {0: "BG", 1: "NETC",2: "SNFH", 3: "ET"},
}

# 각 타입에서 ET에 해당하는 레이블 번호
ET_LABEL = {"GLI": 3, "MEN": 1, "MET": 3}
# 부종/SNFH 레이블
EDEMA_LABEL = {"GLI": 2, "MEN": 3, "MET": 2}
# 핵/코어 레이블
CORE_LABEL = {"GLI": 1, "MEN": 2, "MET": 1}


def extract_features_from_seg(
    seg_path: Path,
    tumor_type: str,
    affine: np.ndarray = None,
) -> dict:
    """
    단일 segmentation NIfTI → feature dict

    Features:
      - {type}_total_voxels       : 전체 종양 복셀 수
      - {type}_total_volume_mm3   : 전체 종양 부피 (mm³)
      - {type}_et_ratio           : ET 비율
      - {type}_edema_ratio        : 부종/SNFH 비율
      - {type}_core_ratio         : 핵/코어 비율
      - {type}_lesion_count       : 병변 개수 (연결 요소)
      - {type}_has_tumor          : 종양 검출 여부 (0/1)
    """
    seg, seg_affine = load_nii(seg_path)
    seg = np.round(seg).astype(int)

    if affine is None:
        affine = seg_affine

    vox_vol = get_voxel_volume_mm3(affine)
    total_vox = np.sum(seg > 0)
    total_vol = total_vox * vox_vol

    et_lbl = ET_LABEL[tumor_type]
    ed_lbl = EDEMA_LABEL[tumor_type]
    core_lbl = CORE_LABEL[tumor_type]

    et_vox = np.sum(seg == et_lbl)
    ed_vox = np.sum(seg == ed_lbl)
    core_vox = np.sum(seg == core_lbl)

    lesion_count = count_lesions(seg, label=et_lbl)

    prefix = tumor_type.lower()
    return {
        f"{prefix}_total_voxels":     int(total_vox),
        f"{prefix}_total_volume_mm3": float(total_vol),
        f"{prefix}_et_ratio":         float(et_vox / (total_vox + 1e-8)),
        f"{prefix}_edema_ratio":      float(ed_vox / (total_vox + 1e-8)),
        f"{prefix}_core_ratio":       float(core_vox / (total_vox + 1e-8)),
        f"{prefix}_lesion_count":     int(lesion_count),
        f"{prefix}_has_tumor":        int(total_vox > 0),
    }


def build_feature_vector(
    seg_paths: dict[str, Path],
) -> dict:
    """
    GLI / MEN / MET 세 모델의 출력을 합쳐 하나의 feature vector 생성.

    Args:
        seg_paths: {"GLI": Path, "MEN": Path, "MET": Path}

    Returns:
        통합 feature dict (총 21개 특징)
    """
    combined = {}
    for tumor_type, path in seg_paths.items():
        if path is None or not Path(path).exists():
            logger.warning(f"[{tumor_type}] segmentation 없음 → 0으로 채움")
            prefix = tumor_type.lower()
            combined.update({
                f"{prefix}_total_voxels": 0,
                f"{prefix}_total_volume_mm3": 0.0,
                f"{prefix}_et_ratio": 0.0,
                f"{prefix}_edema_ratio": 0.0,
                f"{prefix}_core_ratio": 0.0,
                f"{prefix}_lesion_count": 0,
                f"{prefix}_has_tumor": 0,
            })
        else:
            feats = extract_features_from_seg(Path(path), tumor_type)
            combined.update(feats)

    return combined
