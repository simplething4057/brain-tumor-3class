"""
전처리 파이프라인
- BraTS 표준 포맷(t1c, t1n, t2f, t2w)으로 정규화
- UCSF MET 데이터의 모달리티 매핑 처리
- 1mm isotropic 리샘플링
"""
from pathlib import Path
from typing import Optional
import numpy as np
import SimpleITK as sitk
from loguru import logger


TUMOR_TYPE = ["GLI", "MEN", "MET"]


def resample_to_spacing(
    image: sitk.Image,
    target_spacing: list[float] = [1.0, 1.0, 1.0],
    interpolator=sitk.sitkLinear,
) -> sitk.Image:
    """이미지를 target_spacing(mm)으로 리샘플링"""
    orig_spacing = list(image.GetSpacing())
    orig_size = list(image.GetSize())

    new_size = [
        int(round(orig_size[i] * orig_spacing[i] / target_spacing[i]))
        for i in range(3)
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)


def normalize_intensity(arr: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    """비-뇌 영역 제외 후 퍼센타일 기반 정규화"""
    brain_mask = arr > 0
    if brain_mask.sum() == 0:
        return arr
    p_high = np.percentile(arr[brain_mask], percentile)
    arr = np.clip(arr, 0, p_high)
    arr = arr / (p_high + 1e-8)
    return arr.astype(np.float32)


def prepare_subject_brats(
    subject_dir: Path,
    tumor_type: str,
    out_dir: Path,
    file_patterns: dict,
    target_spacing: list[float] = [1.0, 1.0, 1.0],
) -> Optional[Path]:
    """
    단일 피험자를 BraTS 표준 포맷으로 전처리 후 저장.

    UCSF MET의 경우:
      T1c → t1c, T1 → t1n, FLAIR → t2f, t2w = t1c 복사(대체)
    """
    subject_id = subject_dir.name
    patterns = file_patterns[tumor_type]
    out_subject_dir = out_dir / tumor_type / subject_id
    out_subject_dir.mkdir(parents=True, exist_ok=True)

    modalities = ["t1c", "t1n", "t2f", "t2w"]
    processed = {}

    for mod in modalities:
        pat = patterns.get(mod)

        # t2w가 없는 경우(UCSF MET) → t1c로 대체
        if pat is None:
            if mod == "t2w" and "t1c" in processed:
                logger.warning(f"[{subject_id}] t2w 없음 → t1c 복사로 대체")
                processed[mod] = processed["t1c"]
                continue
            else:
                logger.error(f"[{subject_id}] {mod} 파일 패턴 없음")
                return None

        src_path = subject_dir / pat.format(id=subject_id)
        if not src_path.exists():
            logger.warning(f"[{subject_id}] {mod} 파일 없음: {src_path}")
            return None

        img = sitk.ReadImage(str(src_path))
        img = resample_to_spacing(img, target_spacing)
        arr = sitk.GetArrayFromImage(img)
        arr = normalize_intensity(arr)
        processed[mod] = (arr, img)

    # BraTS 표준 파일명으로 저장
    for mod, val in processed.items():
        if isinstance(val, tuple):
            arr, ref_img = val
        else:
            arr, ref_img = val  # 대체된 경우
        out_path = out_subject_dir / f"{subject_id}-{mod}.nii.gz"
        out_img = sitk.GetImageFromArray(arr)
        out_img.CopyInformation(ref_img)
        sitk.WriteImage(out_img, str(out_path))

    logger.info(f"[{subject_id}] 전처리 완료 → {out_subject_dir}")
    return out_subject_dir


def preprocess_all(config: dict) -> None:
    """config 기반으로 전체 데이터 전처리 실행"""
    raw_cfg = config["data"]["raw"]
    processed_dir = Path(config["data"]["processed"])
    file_patterns = config["data"]["file_patterns"]
    spacing = config["preprocessing"]["target_spacing"]

    for tumor_type in TUMOR_TYPE:
        raw_dir = Path(raw_cfg[tumor_type])
        if not raw_dir.exists():
            logger.warning(f"[{tumor_type}] 데이터 경로 없음: {raw_dir}")
            continue

        subjects = [d for d in raw_dir.iterdir() if d.is_dir()]
        logger.info(f"[{tumor_type}] {len(subjects)}개 피험자 전처리 시작")

        for subject_dir in subjects:
            prepare_subject_brats(
                subject_dir=subject_dir,
                tumor_type=tumor_type,
                out_dir=processed_dir,
                file_patterns=file_patterns,
                target_spacing=spacing,
            )
