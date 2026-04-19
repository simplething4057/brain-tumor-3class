"""
FastAPI 라우터:
POST /api/predict        — .nii 파일 업로드 → 예측
GET  /api/files/{id}/{modality} — .nii 파일 서빙 (NiiVue용)
POST /api/report/{id}    — RAG 보고서 생성
GET  /api/history        — 예측 이력 조회
GET  /api/history/{id}   — 단건 조회
"""
import uuid
import shutil
import traceback
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Form
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from loguru import logger

from core.config import settings
from db.database import get_db
from db import crud
from pipeline.predict import run_prediction
from rag.report_generator import generate_report

router = APIRouter(prefix="/api")

ALLOWED_MODALITIES = {"t1n", "t1c", "t2f", "t2w", "seg"}


# ─── 예측 ────────────────────────────────────────────────────────────────────

@router.post("/predict")
async def predict(
    seg: UploadFile = File(..., description="seg .nii 파일"),
    t1c: UploadFile = File(None, description="T1C .nii 파일 (3D 뷰어용, 선택)"),
    t1n: UploadFile = File(None, description="T1N .nii 파일 (선택)"),
    t2f: UploadFile = File(None, description="T2F .nii 파일 (선택)"),
    t2w: UploadFile = File(None, description="T2W .nii 파일 (선택)"),
    db: AsyncSession = Depends(get_db),
):
    subject_id = str(uuid.uuid4())
    subj_dir = settings.upload_path / subject_id
    subj_dir.mkdir(parents=True, exist_ok=True)

    # seg 저장 (필수)
    seg_path = subj_dir / "seg.nii"
    with open(seg_path, "wb") as f:
        shutil.copyfileobj(seg.file, f)

    # 선택 모달리티 저장
    for name, upload in [("t1c", t1c), ("t1n", t1n), ("t2f", t2f), ("t2w", t2w)]:
        if upload is not None:
            out = subj_dir / f"{name}.nii"
            with open(out, "wb") as f:
                shutil.copyfileobj(upload.file, f)

    # 예측
    try:
        result = run_prediction(subject_id, seg_path)
    except Exception as e:
        logger.error(f"예측 실패 [{subject_id}]: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"예측 실패: {str(e)}")

    # DB 저장
    record = await crud.create_prediction(db, {
        "subject_id": subject_id,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "gli_prob": result["gli_prob"],
        "men_prob": result["men_prob"],
        "met_prob": result["met_prob"],
        "features": result["features"],
        "brain_region": result.get("brain_region"),
    })

    return {
        "record_id": record.id,
        "subject_id": subject_id,
        "prediction": result["prediction"],
        "confidence": result["confidence"],
        "brain_region": result.get("brain_region"),
        "probabilities": {
            "GLI": result["gli_prob"],
            "MEN": result["men_prob"],
            "MET": result["met_prob"],
        },
    }


# ─── 파일 서빙 (NiiVue용) ────────────────────────────────────────────────────

@router.get("/files/{subject_id}/{modality}")
async def get_file(subject_id: str, modality: str):
    if modality not in ALLOWED_MODALITIES:
        raise HTTPException(status_code=400, detail=f"허용 모달리티: {ALLOWED_MODALITIES}")

    # 업로드 디렉터리에서 먼저 탐색
    nii_path = settings.upload_path / subject_id / f"{modality}.nii"
    if not nii_path.exists():
        raise HTTPException(status_code=404, detail=f"파일 없음: {nii_path}")

    return FileResponse(
        path=str(nii_path),
        media_type="application/octet-stream",
        filename=f"{subject_id}-{modality}.nii",
        headers={"Access-Control-Allow-Origin": "*"},
    )


# ─── RAG 보고서 생성 ─────────────────────────────────────────────────────────

@router.post("/report/{record_id}")
async def create_report(record_id: int, db: AsyncSession = Depends(get_db)):
    record = await crud.get_prediction_by_id(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="예측 기록 없음")

    # 이미 보고서 있으면 반환
    if record.report:
        return {"record_id": record_id, "report": record.report}

    # 업로드된 모달리티 자동 감지
    subj_dir = settings.upload_path / record.subject_id
    detected_mods = [m for m in ["t1c", "t1n", "t2f", "t2w"] if (subj_dir / f"{m}.nii").exists()]
    if (subj_dir / "seg.nii").exists():
        detected_mods.append("seg")

    report = await generate_report(
        prediction=record.prediction,
        confidence=record.confidence,
        gli_prob=record.gli_prob,
        men_prob=record.men_prob,
        met_prob=record.met_prob,
        features=record.features,
        brain_region=record.brain_region,
        modalities=detected_mods or None,
    )

    updated = await crud.update_report(db, record_id, report)
    return {"record_id": record_id, "report": updated.report}


# ─── 이력 조회 ───────────────────────────────────────────────────────────────

@router.get("/history")
async def get_history(
    skip: int = 0,
    limit: int = 50,
    db: AsyncSession = Depends(get_db),
):
    records = await crud.get_history(db, skip=skip, limit=limit)
    return [r.to_dict() for r in records]


@router.get("/history/{record_id}")
async def get_history_item(record_id: int, db: AsyncSession = Depends(get_db)):
    record = await crud.get_prediction_by_id(db, record_id)
    if not record:
        raise HTTPException(status_code=404, detail="기록 없음")
    return record.to_dict()
