"""
GPU 없이 GT segmentation을 inference 결과 대신 사용
각 피험자의 GT seg를 GLI/MEN/MET 3개 폴더 모두에 복사

전략:
  - 각 피험자의 GT seg를 outputs/segmentation/{GLI,MEN,MET}/{sid}.nii.gz 로 복사
  - feature extractor가 각 폴더별 레이블 체계로 해석 → 교차 특징 생성
  - GLI/MEN/MET 레이블 번호가 다르게 매핑되어 종양 유형별 패턴 구분 가능

사용법:
  python scripts/copy_gt_segs.py
"""
import shutil
from pathlib import Path
from loguru import logger

BASE_DIR = Path(__file__).parent.parent

# raw 데이터 경로 (seg 파일은 전처리 시 복사되지 않으므로 raw에서 직접 사용)
PROCESSED = BASE_DIR / "data" / "raw"

# 출력 경로
SEG_OUT = BASE_DIR / "outputs" / "segmentation"


def copy_gt_for_tumor_type(tumor_type: str):
    src_dir = PROCESSED / tumor_type
    if not src_dir.exists():
        logger.warning(f"{tumor_type} 전처리 폴더 없음: {src_dir}")
        return 0

    count = 0
    for subj_dir in sorted(src_dir.iterdir()):
        if not subj_dir.is_dir():
            continue
        sid = subj_dir.name
        seg_src = subj_dir / f"{sid}-seg.nii.gz"

        if not seg_src.exists():
            logger.warning(f"  GT seg 없음: {seg_src}")
            continue

        # 3개 폴더 모두에 복사
        for folder in ["GLI", "MEN", "MET"]:
            out_dir = SEG_OUT / folder
            out_dir.mkdir(parents=True, exist_ok=True)
            dst = out_dir / f"{sid}.nii.gz"
            shutil.copy2(seg_src, dst)

        logger.info(f"  ✓ {sid} → GLI/MEN/MET")
        count += 1

    return count


def main():
    total = 0
    for tumor_type in ["GLI", "MEN", "MET"]:
        logger.info(f"=== {tumor_type} GT seg 복사 ===")
        n = copy_gt_for_tumor_type(tumor_type)
        logger.info(f"  {n}명 완료")
        total += n

    logger.info(f"=== 전체 완료: {total}명 ===")

    # 결과 확인
    for folder in ["GLI", "MEN", "MET"]:
        files = list((SEG_OUT / folder).glob("*.nii.gz"))
        logger.info(f"outputs/segmentation/{folder}: {len(files)}개 파일")


if __name__ == "__main__":
    main()
