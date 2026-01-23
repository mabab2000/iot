import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    Boolean,
    TIMESTAMP,
    func,
    select,
)
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL not set in environment")

# Ensure asyncpg dialect is used
if DATABASE_URL.startswith("postgresql://"):
    ASYNC_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
else:
    ASYNC_DATABASE_URL = DATABASE_URL

app = FastAPI()

# Allow dashboard access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SQLAlchemy async engine + table definition
metadata = MetaData()

alcohol_table = Table(
    "alcohol_data",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("alcohol", Integer, nullable=False),
    Column("detected", Boolean, nullable=False, default=False),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
)

engine: Optional[AsyncEngine] = None


class AlcoholData(BaseModel):
    alcohol: int
    detected: bool


@app.on_event("startup")
async def startup():
    global engine
    # Disable asyncpg prepared statement cache for pgbouncer compatibility
    engine = create_async_engine(
        ASYNC_DATABASE_URL,
        echo=False,
        connect_args={"statement_cache_size": 0},
    )
    # create tables if not exists
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)


@app.on_event("shutdown")
async def shutdown():
    global engine
    if engine:
        await engine.dispose()


@app.post("/data")
async def receive_data(data: AlcoholData):
    """Save received alcohol reading into Postgres."""
    if engine is None:
        raise HTTPException(status_code=500, detail="Database engine not initialized")

    insert_stmt = alcohol_table.insert().values(alcohol=data.alcohol, detected=data.detected)
    async with engine.begin() as conn:
        await conn.execute(insert_stmt)
    return {"status": "success"}


@app.get("/data")
async def get_data():
    """Return all alcohol readings from the database (most recent first)."""
    if engine is None:
        raise HTTPException(status_code=500, detail="Database engine not initialized")

    sel = select(
        alcohol_table.c.id,
        alcohol_table.c.alcohol,
        alcohol_table.c.detected,
        alcohol_table.c.created_at,
    ).order_by(alcohol_table.c.created_at.desc())

    async with engine.connect() as conn:
        result = await conn.execute(sel)
        rows = result.fetchall()

    if not rows:
        return []

    return [
        {
            "id": r.id,
            "alcohol": r.alcohol,
            "detected": r.detected,
            "created_at": r.created_at,
        }
        for r in rows
    ]
