"""NIfTI 파일 처리 유틸리티"""
from pathlib import Path
import numpy as np
import nibabel as nib


def load_nii(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """NIfTI 로드 → (data array, affine)"""
    img = nib.load(str(path))
    return img.get_fdata(), img.affine


def save_nii(data: np.ndarray, affine: np.ndarray, out_path: str | Path) -> None:
    """numpy array → NIfTI 저장"""
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data.astype(np.float32), affine)
    nib.save(img, str(out_path))


def get_voxel_volume_mm3(affine: np.ndarray) -> float:
    """affine에서 복셀 부피(mm³) 계산"""
    voxel_size = np.abs(np.diag(affine)[:3])
    return float(np.prod(voxel_size))


def count_lesions(seg: np.ndarray, label: int) -> int:
    """특정 레이블의 연결된 병변 개수 반환 (MET 특징 추출용)"""
    from scipy.ndimage import label as ndlabel
    binary = (seg == label).astype(int)
    _, num = ndlabel(binary)
    return num
