from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from db.database import init_db
from api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 시작 시 DB 테이블 생성
    await init_db()
    yield


app = FastAPI(
    title="Brain Tumor Diagnosis API",
    description="GLI / MEN / MET 3종 뇌종양 분류 및 방사선 소견 생성 API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용 — 배포 시 프론트 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router)


@app.get("/health")
async def health():
    return {"status": "ok"}
