"""
Meta-Classifier: feature vector → {GLI, MEN, MET} 분류
1단계: RandomForest (빠른 프로토타입)
2단계: MLP (정확도 향상)
"""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from loguru import logger


LABEL_MAP = {"GLI": 0, "MEN": 1, "MET": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}


class TumorTypeClassifier:
    """GLI / MEN / MET 3종 분류기"""

    def __init__(self, config: dict):
        clf_cfg = config["classifier"]
        self.model_type = clf_cfg["type"]
        self.save_path = Path(clf_cfg["model_save_path"])

        rf_cfg = clf_cfg.get("random_forest", {})
        self.model = RandomForestClassifier(
            n_estimators=rf_cfg.get("n_estimators", 200),
            max_depth=rf_cfg.get("max_depth", 10),
            class_weight=rf_cfg.get("class_weight", "balanced"),
            random_state=42,
            n_jobs=-1,
        )
        self.feature_names: list[str] = []
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: list[str]) -> None:
        """
        Args:
            X: feature DataFrame (행=피험자, 열=특징)
            y: 레이블 리스트 ["GLI", "MEN", "MET", ...]
        """
        self.feature_names = list(X.columns)
        y_enc = [LABEL_MAP[label] for label in y]

        logger.info(f"분류기 학습 시작: {len(y_enc)}개 샘플")
        self.model.fit(X.values, y_enc)
        self.is_fitted = True
        logger.info("학습 완료")

    def predict(self, X: pd.DataFrame) -> list[str]:
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다. fit() 먼저 실행.")
        y_pred = self.model.predict(X[self.feature_names].values)
        return [INV_LABEL_MAP[y] for y in y_pred]

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        proba = self.model.predict_proba(X[self.feature_names].values)
        return pd.DataFrame(proba, columns=["GLI", "MEN", "MET"])

    def evaluate(self, X: pd.DataFrame, y_true: list[str]) -> dict:
        y_pred = self.predict(X)
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=["GLI", "MEN", "MET"])

        logger.info("\n" + classification_report(y_true, y_pred))
        logger.info(f"Confusion Matrix:\n{cm}")

        return {"report": report, "confusion_matrix": cm.tolist()}

    def feature_importance(self) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("모델이 학습되지 않았습니다.")
        imp = self.model.feature_importances_
        return (
            pd.DataFrame({"feature": self.feature_names, "importance": imp})
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self) -> None:
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.save_path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"모델 저장: {self.save_path}")

    @classmethod
    def load(cls, path: str | Path) -> "TumorTypeClassifier":
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"모델 로드: {path}")
        return obj
