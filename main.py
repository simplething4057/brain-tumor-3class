"""
Brain Tumor 3-Class Discrimination System
실행 진입점

사용법:
  # 1. 전처리
  python main.py --step preprocess

  # 2. 3종 추론 (Docker 필요)
  python main.py --step infer --subject BraTS-GLI-00000-000

  # 3. 특징 추출
  python main.py --step extract

  # 4. 분류기 학습
  python main.py --step train

  # 5. 단일 피험자 예측
  python main.py --step predict --subject BraTS-GLI-00000-000
"""
import argparse
from pathlib import Path
from loguru import logger

from src.utils.config import load_config


def step_preprocess(config: dict):
    from src.preprocessing.preprocess import preprocess_all
    logger.info("=== STEP: 전처리 시작 ===")
    preprocess_all(config)
    logger.info("=== 전처리 완료 ===")


def step_infer(config: dict, subject_id: str | None = None):
    from src.inference.brats_infer import run_all_types, InferenceInput
    logger.info("=== STEP: 추론 시작 ===")

    processed_dir = Path(config["data"]["processed"])
    output_dir = Path(config["inference"]["output_dir"])

    # 피험자 목록 결정
    if subject_id:
        subjects = []
        for tumor_type in ["GLI", "MEN", "MET"]:
            candidate = processed_dir / tumor_type / subject_id
            if candidate.exists():
                subjects.append((tumor_type, candidate))
    else:
        subjects = [
            (d.parent.name, d)
            for tumor_type in ["GLI", "MEN", "MET"]
            for d in (processed_dir / tumor_type).iterdir()
            if d.is_dir()
        ]

    for tumor_type, subj_dir in subjects:
        sid = subj_dir.name
        inputs = InferenceInput(
            t1c=subj_dir / f"{sid}-t1c.nii.gz",
            t1n=subj_dir / f"{sid}-t1n.nii.gz",
            t2f=subj_dir / f"{sid}-t2f.nii.gz",
            t2w=subj_dir / f"{sid}-t2w.nii.gz",
        )
        run_all_types(
            subject_id=sid,
            inputs=inputs,
            config=config,
            output_base=output_dir,
        )

    logger.info("=== 추론 완료 ===")


def step_extract(config: dict):
    import pandas as pd
    from src.classifier.feature_extractor import build_feature_vector
    logger.info("=== STEP: 특징 추출 시작 ===")

    seg_dir = Path(config["inference"]["output_dir"])
    out_dir = Path(config["features"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    # outputs/segmentation/GLI/ 에서 subject 목록 수집 후 ID prefix로 레이블 결정
    gli_seg_dir = seg_dir / "GLI"
    if not gli_seg_dir.exists():
        logger.error("outputs/segmentation/GLI/ 폴더 없음")
        return

    label_prefix = {"GLI": "BraTS-GLI", "MEN": "BraTS-MEN", "MET": "BraTS-MET"}

    for seg_file in sorted(gli_seg_dir.glob("*.nii.gz")):
        sid = seg_file.stem.replace(".nii", "")

        # subject ID prefix로 true_label 결정
        true_label = None
        for label, prefix in label_prefix.items():
            if sid.startswith(prefix):
                true_label = label
                break
        if true_label is None:
            logger.warning(f"레이블 결정 불가 (prefix 불일치): {sid}")
            continue

        seg_paths = {
            "GLI": seg_dir / "GLI" / f"{sid}.nii.gz",
            "MEN": seg_dir / "MEN" / f"{sid}.nii.gz",
            "MET": seg_dir / "MET" / f"{sid}.nii.gz",
        }
        feats = build_feature_vector(seg_paths)
        feats["subject_id"] = sid
        feats["true_label"] = true_label
        records.append(feats)

    df = pd.DataFrame(records)
    out_path = out_dir / "features.csv"
    df.to_csv(out_path, index=False)
    logger.info(f"특징 저장 완료: {out_path} ({len(df)}개 샘플)")


def step_train(config: dict):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from src.classifier.meta_classifier import TumorTypeClassifier
    logger.info("=== STEP: 분류기 학습 시작 ===")

    feat_path = Path(config["features"]["output_dir"]) / "features.csv"
    df = pd.read_csv(feat_path)

    feature_cols = [c for c in df.columns if c not in ["subject_id", "true_label"]]
    X = df[feature_cols]
    y = df["true_label"].tolist()

    clf_cfg = config["classifier"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=clf_cfg["test_split"] + clf_cfg["val_split"],
        stratify=y,
        random_state=42,
    )

    clf = TumorTypeClassifier(config)
    clf.fit(X_train, y_train)
    clf.evaluate(X_test, y_test)
    clf.save()

    imp = clf.feature_importance()
    logger.info(f"\n특징 중요도 Top 10:\n{imp.head(10)}")


def step_predict(config: dict, subject_id: str):
    import pandas as pd
    from src.classifier.meta_classifier import TumorTypeClassifier
    from src.classifier.feature_extractor import build_feature_vector
    logger.info(f"=== STEP: 예측 ({subject_id}) ===")

    seg_dir = Path(config["inference"]["output_dir"])
    seg_paths = {
        t: seg_dir / t / f"{subject_id}.nii.gz"
        for t in ["GLI", "MEN", "MET"]
    }

    feats = build_feature_vector(seg_paths)
    X = pd.DataFrame([feats])

    clf = TumorTypeClassifier.load(config["classifier"]["model_save_path"])
    pred = clf.predict(X)[0]
    proba = clf.predict_proba(X)

    logger.info(f"예측 결과: {pred}")
    logger.info(f"확률:\n{proba.to_string()}")
    return pred, proba


def main():
    parser = argparse.ArgumentParser(description="Brain Tumor 3-Class Discrimination")
    parser.add_argument("--step", required=True,
                        choices=["preprocess", "infer", "extract", "train", "predict"])
    parser.add_argument("--subject", default=None, help="단일 피험자 ID")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.step == "preprocess":
        step_preprocess(config)
    elif args.step == "infer":
        step_infer(config, args.subject)
    elif args.step == "extract":
        step_extract(config)
    elif args.step == "train":
        step_train(config)
    elif args.step == "predict":
        if not args.subject:
            raise ValueError("--subject 필요")
        step_predict(config, args.subject)


if __name__ == "__main__":
    main()
