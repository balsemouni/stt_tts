import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_user = os.getenv("DATABASE_USER", "postgres")
_pass = os.getenv("DATABASE_PASSWORD", "")
_host = os.getenv("DATABASE_HOST", "localhost")
_port = os.getenv("DATABASE_PORT", "5432")
DATABASE_URL = f"postgresql+asyncpg://{_user}:{_pass}@{_host}:{_port}/users"
engine = create_async_engine(DATABASE_URL, echo=True)
AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise