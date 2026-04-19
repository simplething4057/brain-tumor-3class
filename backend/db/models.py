from datetime import datetime
from sqlalchemy import String, Float, DateTime, Text, JSON
from sqlalchemy.orm import Mapped, mapped_column
from db.database import Base


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    subject_id: Mapped[str] = mapped_column(String(100), index=True)
    prediction: Mapped[str] = mapped_column(String(10))  # GLI | MEN | MET
    confidence: Mapped[float] = mapped_column(Float)
    gli_prob: Mapped[float] = mapped_column(Float)
    men_prob: Mapped[float] = mapped_column(Float)
    met_prob: Mapped[float] = mapped_column(Float)
    features: Mapped[dict] = mapped_column(JSON, nullable=True)
    brain_region: Mapped[str | None] = mapped_column(String(100), nullable=True)
    report: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subject_id": self.subject_id,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "gli_prob": self.gli_prob,
            "men_prob": self.men_prob,
            "met_prob": self.met_prob,
            "brain_region": self.brain_region,
            "report": self.report,
            "created_at": self.created_at.isoformat(),
        }
