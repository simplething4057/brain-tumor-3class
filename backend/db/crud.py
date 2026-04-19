import asyncio
from sqlalchemy import select, desc
from db.models import PredictionRecord
from db.database import SessionLocal


def _create_prediction_sync(data: dict) -> PredictionRecord:
    with SessionLocal() as s:
        record = PredictionRecord(**data)
        s.add(record)
        s.commit()
        s.refresh(record)
        return record


def _get_history_sync(skip: int, limit: int) -> list[PredictionRecord]:
    with SessionLocal() as s:
        return s.execute(
            select(PredictionRecord)
            .order_by(desc(PredictionRecord.created_at))
            .offset(skip).limit(limit)
        ).scalars().all()


def _get_by_id_sync(record_id: int) -> PredictionRecord | None:
    with SessionLocal() as s:
        return s.get(PredictionRecord, record_id)


def _update_report_sync(record_id: int, report: str) -> PredictionRecord | None:
    with SessionLocal() as s:
        record = s.get(PredictionRecord, record_id)
        if record:
            record.report = report
            s.commit()
            s.refresh(record)
        return record


# ── async 인터페이스 (FastAPI 라우터에서 await 가능) ─────────────────────────

async def create_prediction(db, data: dict) -> PredictionRecord:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _create_prediction_sync, data)


async def get_history(db, skip: int = 0, limit: int = 50) -> list[PredictionRecord]:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_history_sync, skip, limit)


async def get_prediction_by_id(db, record_id: int) -> PredictionRecord | None:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _get_by_id_sync, record_id)


async def update_report(db, record_id: int, report: str) -> PredictionRecord | None:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None, _update_report_sync, record_id, report
    )
