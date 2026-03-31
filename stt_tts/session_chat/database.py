import os
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

_user = os.getenv("DATABASE_USER", "postgres")
_pass = os.getenv("DATABASE_PASSWORD", "")
_host = os.getenv("DATABASE_HOST", "localhost")
_port = os.getenv("DATABASE_PORT", "5432")
DATABASE_URL = f"postgresql+psycopg2://{_user}:{_pass}@{_host}:{_port}/session_chat"

engine = create_engine(DATABASE_URL, echo=False, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class Base(DeclarativeBase):
    pass

def init_db():
    from models import Base as ModelBase
    ModelBase.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()