"""
GT seg 기반 새 피험자 예측 스크립트

사용법:
  # 단일 피험자 (폴더 경로 지정)
  python scripts/predict_new.py --subject data/raw/GLI/BraTS-GLI-00000-000

  # 여러 피험자 일괄 처리
  python scripts/predict_new.py --dir data/raw/GLI

  # 정답 레이블 포함해서 정확도 함께 출력
  python scripts/predict_new.py --dir data/raw/GLI --true_label GLI

입력 조건:
  - 피험자 폴더 안에 {id}-seg.nii.gz 파일이 존재해야 함
  - 모델 파일: models/weights/meta_classifier.pkl
"""
import sys
import shutil
import argparse
import pickle
from pathlib import Path

_ROOT = Path(__file__).parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
from loguru import logger

from src.classifier.feature_extractor import build_feature_vector

SEG_OUT   = _ROOT / "outputs" / "segmentation"
MODEL_PATH = _ROOT / "models" / "weights" / "meta_classifier.pkl"
CLASSES   = ["GLI", "MEN", "MET"]


# ── 모델 로드 ──────────────────────────────────────────────────────────────────
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}\n먼저 python main.py --step train 실행 필요")
    with open(MODEL_PATH, "rb") as f:
        clf = pickle.load(f)
    logger.info(f"모델 로드: {MODEL_PATH}")
    return clf


# ── GT seg → outputs/segmentation/{GLI,MEN,MET}/ 복사 ────────────────────────
def copy_seg_to_outputs(subj_dir: Path) -> tuple[str, dict]:
    """
    subj_dir 안의 {id}-seg.nii.gz 를 GLI/MEN/MET 폴더에 복사.
    Returns: (subject_id, seg_paths dict)
    """
    sid = subj_dir.name
    seg_src = subj_dir / f"{sid}-seg.nii.gz"

    if not seg_src.exists():
        raise FileNotFoundError(f"seg 파일 없음: {seg_src}")

    seg_paths = {}
    for folder in CLASSES:
        out_dir = SEG_OUT / folder
        out_dir.mkdir(parents=True, exist_ok=True)
        dst = out_dir / f"{sid}.nii.gz"
        shutil.copy2(seg_src, dst)
        seg_paths[folder] = dst

    logger.debug(f"  {sid} → GLI/MEN/MET 복사 완료")
    return sid, seg_paths


# ── 단일 피험자 예측 ───────────────────────────────────────────────────────────
def predict_one(clf, subj_dir: Path, true_label: str = None) -> dict:
    sid, seg_paths = copy_seg_to_outputs(subj_dir)

    feats = build_feature_vector(seg_paths)
    feat_cols = [c for c in feats if c not in ["subject_id", "true_label"]]
    X = pd.DataFrame([feats])[feat_cols]

    pred   = clf.predict(X)[0]
    proba  = clf.predict_proba(X)
    conf   = proba[pred].values[0]

    result = {
        "subject_id": sid,
        "prediction": pred,
        "confidence": round(conf, 4),
        "GLI_prob":   round(proba["GLI"].values[0], 4),
        "MEN_prob":   round(proba["MEN"].values[0], 4),
        "MET_prob":   round(proba["MET"].values[0], 4),
    }
    if true_label:
        result["true_label"] = true_label
        result["correct"]    = (pred == true_label)

    return result


# ── 결과 출력 포매터 ──────────────────────────────────────────────────────────
def print_result(r: dict):
    correct_mark = ""
    if "correct" in r:
        correct_mark = "  ✓" if r["correct"] else f"  ✗ (정답: {r['true_label']})"

    print(f"\n{'='*52}")
    print(f"  피험자:   {r['subject_id']}")
    print(f"  예측:     {r['prediction']}  (신뢰도 {r['confidence']*100:.1f}%){correct_mark}")
    print(f"  확률:     GLI={r['GLI_prob']:.3f}  MEN={r['MEN_prob']:.3f}  MET={r['MET_prob']:.3f}")
    print(f"{'='*52}")


def print_summary(results: list[dict]):
    if not results:
        return
    total = len(results)
    correct_list = [r for r in results if r.get("correct") is True]
    wrong_list   = [r for r in results if r.get("correct") is False]

    print(f"\n{'─'*52}")
    print(f"  총 {total}명  |  정답 {len(correct_list)}명  |  오답 {len(wrong_list)}명")
    if total > 0 and "correct" in results[0]:
        acc = len(correct_list) / total
        print(f"  Accuracy: {acc:.3f} ({len(correct_list)}/{total})")
    if wrong_list:
        print(f"\n  오분류 목록:")
        for r in wrong_list:
            print(f"    {r['subject_id']}  예측={r['prediction']}  정답={r['true_label']}  "
                  f"신뢰도={r['confidence']:.3f}")
    print(f"{'─'*52}\n")


# ── 메인 ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="GT seg 기반 새 피험자 예측")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--subject", type=str,
                       help="단일 피험자 폴더 경로 (예: data/raw/GLI/BraTS-GLI-00000-000)")
    group.add_argument("--dir", type=str,
                       help="피험자 폴더들이 담긴 부모 디렉터리 (예: data/raw/GLI)")
    parser.add_argument("--true_label", type=str, choices=CLASSES, default=None,
                        help="정답 레이블 (정확도 계산 시 사용)")
    args = parser.parse_args()

    clf = load_model()

    if args.subject:
        subj_dir = _ROOT / args.subject if not Path(args.subject).is_absolute() else Path(args.subject)
        result   = predict_one(clf, subj_dir, args.true_label)
        print_result(result)

    else:
        parent = _ROOT / args.dir if not Path(args.dir).is_absolute() else Path(args.dir)
        if not parent.exists():
            logger.error(f"폴더 없음: {parent}")
            return

        subj_dirs = sorted([d for d in parent.iterdir() if d.is_dir()])
        if not subj_dirs:
            logger.error(f"피험자 폴더 없음: {parent}")
            return

        logger.info(f"{len(subj_dirs)}명 처리 시작: {parent}")
        results = []
        for i, subj_dir in enumerate(subj_dirs, 1):
            try:
                r = predict_one(clf, subj_dir, args.true_label)
                results.append(r)
                status = f"✓ {r['prediction']} ({r['confidence']*100:.0f}%)"
                if "correct" in r:
                    status += "  정답" if r["correct"] else f"  오답(정답:{r['true_label']})"
                logger.info(f"[{i:3d}/{len(subj_dirs)}] {subj_dir.name}  →  {status}")
            except FileNotFoundError as e:
                logger.warning(f"  스킵: {e}")

        print_summary(results)

        # CSV 저장
        if results:
            out_csv = _ROOT / "outputs" / "predictions" / "predict_new_results.csv"
            out_csv.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(results).to_csv(out_csv, index=False)
            logger.info(f"결과 저장: {out_csv}")


if __name__ == "__main__":
    main()
