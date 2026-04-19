import asyncio
from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker
from loguru import logger
from core.config import settings

# URL 정규화 → psycopg2 (로컬 개발 / Windows asyncpg 호환성 문제 우회)
_url = settings.database_url
_url = _url.replace("postgresql+asyncpg", "postgresql+psycopg2")
_url = _url.replace("?ssl=disable", "")

engine = create_engine(_url, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    """FastAPI dependency — sync session을 async로 래핑"""
    loop = asyncio.get_event_loop()
    session = SessionLocal()
    try:
        yield session
        await loop.run_in_executor(None, session.commit)
    except Exception:
        await loop.run_in_executor(None, session.rollback)
        raise
    finally:
        await loop.run_in_executor(None, session.close)


async def init_db():
    try:
        Base.metadata.create_all(engine)
        logger.info("DB 테이블 초기화 완료")
    except Exception as e:
        logger.warning(f"DB 연결 실패 (서버는 계속 실행): {e}")
