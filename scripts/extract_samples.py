"""
data/raw/ 에 각 클래스별 샘플 추출
- GLI: ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip → data/raw/GLI/
- MEN: ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData.zip → data/raw/MEN/
- MET: UCSF_BrainMetastases_v1.3.zip → data/raw/MET/ (BraTS 표준 명명으로 변환)

사용법:
  python scripts/extract_samples.py --n 10
  python scripts/extract_samples.py --n 50 --all
"""
import argparse
import zipfile
import shutil
from pathlib import Path
from loguru import logger

# ── 경로 설정 ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent.parent  # brain-tumor-3class/
LLM_DIR  = BASE_DIR.parent.parent / "LLM"  # Claude/Projects/LLM/

GLI_ZIP = LLM_DIR / "Brats2023" / "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData.zip"
MEN_ZIP = LLM_DIR / "Brats2023" / "ASNR-MICCAI-BraTS2023-MEN-Challenge-TrainingData.zip"
MET_ZIP = LLM_DIR / "UCSF_BrainMetastases_v1.3.zip"

OUT_GLI = BASE_DIR / "data" / "raw" / "GLI"
OUT_MEN = BASE_DIR / "data" / "raw" / "MEN"
OUT_MET = BASE_DIR / "data" / "raw" / "MET"

# UCSF 파일명 → BraTS 모달리티 매핑
UCSF_MAP = {
    "_T1post.nii.gz": "-t1c.nii.gz",
    "_T1pre.nii.gz":  "-t1n.nii.gz",
    "_FLAIR.nii.gz":  "-t2f.nii.gz",
    "_T2Synth.nii.gz":"-t2w.nii.gz",
    "_BraTS-seg.nii.gz": "-seg.nii.gz",
}


def extract_brats(zip_path: Path, out_dir: Path, n: int, prefix_in: str):
    """GLI / MEN 공통 추출 (BraTS 표준 포맷)"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # 피험자 폴더 목록 수집
        subjects = sorted(set(
            p.split("/")[1]
            for p in zf.namelist()
            if p.count("/") >= 2 and not p.endswith("/")
        ))
        selected = subjects[:n]
        logger.info(f"추출 대상: {len(selected)}명 / 전체 {len(subjects)}명")

        for sid in selected:
            sid_dir = out_dir / sid
            sid_dir.mkdir(exist_ok=True)
            for member in zf.namelist():
                # prefix_in: "ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData" or "BraTS-MEN-Train"
                if member.startswith(f"{prefix_in}/{sid}/") and not member.endswith("/"):
                    fname = Path(member).name
                    target = sid_dir / fname
                    if target.exists():
                        continue
                    with zf.open(member) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst)
            logger.info(f"  ✓ {sid}")


def extract_ucsf(zip_path: Path, out_dir: Path, n: int):
    """UCSF MET 추출 + BraTS 표준 명명으로 변환"""
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        # 피험자 ID 수집 (UCSF_BrainMetastases_TRAIN/{id}/ 형태)
        subjects = sorted(set(
            p.split("/")[1]
            for p in zf.namelist()
            if p.startswith("UCSF_BrainMetastases_TRAIN/")
            and p.count("/") >= 2
            and not p.endswith("/")
        ))
        selected = subjects[:n]
        logger.info(f"UCSF MET 추출 대상: {len(selected)}명 / 전체 {len(subjects)}명")

        for i, uid in enumerate(selected):
            # BraTS-MET-{순번:05d}-000 형식으로 변환
            brats_id = f"BraTS-MET-{i+1:05d}-000"
            sid_dir = out_dir / brats_id
            sid_dir.mkdir(exist_ok=True)

            for member in zf.namelist():
                if not member.startswith(f"UCSF_BrainMetastases_TRAIN/{uid}/"):
                    continue
                if member.endswith("/"):
                    continue
                fname = Path(member).name  # 예: 100101A_T1post.nii.gz

                # 매핑에 해당하는 파일만 추출
                matched = None
                for ucsf_suffix, brats_suffix in UCSF_MAP.items():
                    if fname.endswith(ucsf_suffix):
                        matched = brats_id + brats_suffix
                        break
                if matched is None:
                    continue  # subtraction 등 불필요 파일 스킵

                target = sid_dir / matched
                if target.exists():
                    continue
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)

            logger.info(f"  ✓ {uid} → {brats_id}")

        # ID 매핑 저장
        mapping_file = out_dir / "ucsf_id_mapping.txt"
        with open(mapping_file, "w") as f:
            for i, uid in enumerate(selected):
                f.write(f"BraTS-MET-{i+1:05d}-000\t{uid}\n")
        logger.info(f"ID 매핑 저장: {mapping_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=10, help="클래스당 피험자 수 (기본: 10)")
    parser.add_argument("--gli", action="store_true", help="GLI만 추출")
    parser.add_argument("--men", action="store_true", help="MEN만 추출")
    parser.add_argument("--met", action="store_true", help="MET만 추출")
    args = parser.parse_args()

    # 플래그 없으면 전체 추출
    do_all = not (args.gli or args.men or args.met)

    if args.gli or do_all:
        logger.info(f"=== GLI 추출 ({args.n}명) ===")
        extract_brats(GLI_ZIP, OUT_GLI, args.n,
                      prefix_in="ASNR-MICCAI-BraTS2023-GLI-Challenge-TrainingData")

    if args.men or do_all:
        logger.info(f"=== MEN 추출 ({args.n}명) ===")
        extract_brats(MEN_ZIP, OUT_MEN, args.n,
                      prefix_in="BraTS-MEN-Train")

    if args.met or do_all:
        logger.info(f"=== MET 추출 ({args.n}명) ===")
        extract_ucsf(MET_ZIP, OUT_MET, args.n)

    logger.info("=== 추출 완료 ===")
    logger.info(f"GLI: {OUT_GLI}")
    logger.info(f"MEN: {OUT_MEN}")
    logger.info(f"MET: {OUT_MET}")


if __name__ == "__main__":
    main()
