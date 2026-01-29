import os
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
import re
import httpx

from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    Boolean,
    TIMESTAMP,
    func,
    select,
    Text,
    String,
)
import sqlalchemy
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

# Store analysis requests/results
analysis_table = Table(
    "analysis",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("text", Text, nullable=False),
    Column("result", String(16), nullable=False),
    Column("created_at", TIMESTAMP(timezone=True), server_default=func.now(), nullable=False),
)

engine: Optional[AsyncEngine] = None


class AlcoholData(BaseModel):
    alcohol: int
    detected: bool



class TextRequest(BaseModel):
    text: str


@app.on_event("startup")
async def startup():
    global engine
    # Disable asyncpg prepared statement cache for pgbouncer compatibility
    engine = create_async_engine(
        ASYNC_DATABASE_URL,
        echo=False,
        connect_args={"statement_cache_size": 0},
    )
    # Drop and recreate analysis table if it exists with wrong column types
    async with engine.begin() as conn:
        # Drop analysis table if it exists (to fix column type mismatch)
        await conn.execute(sqlalchemy.text("DROP TABLE IF EXISTS analysis CASCADE;"))
        # create tables if not exists
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


@app.post("/analyze")
async def analyze_text(payload: TextRequest):
    """Analyze the provided text via OpenAI and return 'ON' or 'OFF'."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")

    system_prompt = (
        "You are a classifier. Given an instruction or text, respond with exactly one of these four tokens and nothing else:\n"
        "FAN ON\nFAN OFF\nLED ON\nLED OFF"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": payload.text},
    ]

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"model": "gpt-3.5-turbo", "messages": messages, "max_tokens": 6, "temperature": 0.0},
            )
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"OpenAI request failed: {e}")

    if resp.status_code != 200:
        raise HTTPException(status_code=502, detail=f"OpenAI API error: {resp.status_code}")

    data = resp.json()
    try:
        text = data["choices"][0]["message"]["content"]
    except Exception:
        raise HTTPException(status_code=502, detail="Unexpected response from OpenAI")

    text = text.strip().upper()
    # Prefer explicit full-token matches: "FAN ON", "FAN OFF", "LED ON", "LED OFF"
    m = re.search(r"\b(FAN ON|FAN OFF|LED ON|LED OFF)\b", text)
    if m:
        result = m.group(1)
    else:
        # If model returned an option-string like "1.FAN ON;FAN OFF" or "2.LED ON ;LED OFF",
        # pick the ON or OFF option based on the user's intent heuristic.
        opt_match = re.search(r"(FAN\s+ON\s*;\s*FAN\s+OFF|LED\s+ON\s*;\s*LED\s+OFF)", text)
        lower = payload.text.lower()
        prefer_on = any(k in lower for k in ("turn on", "please on", "switch on", "enable", "start"))
        if opt_match:
            opts = re.split(r"\s*;\s*", opt_match.group(0))
            # choose the option containing ON when user intent prefers ON, else OFF
            if prefer_on:
                pick = next((p for p in opts if "ON" in p), opts[0])
            else:
                pick = next((p for p in opts if "OFF" in p), opts[0])
            result = pick.strip()
        else:
            # Final fallback: infer device and state from user text
            device = None
            if "fan" in lower:
                device = "FAN"
            elif any(k in lower for k in ("led", "light", "bulb", "lamp")):
                device = "LED"

            if device:
                if prefer_on:
                    result = f"{device} ON"
                else:
                    result = f"{device} OFF"
            else:
                # As a last resort, default to "FAN OFF"
                result = "FAN OFF"

    # persist analysis to database (update existing row or insert if none exists)
    if engine is None:
        raise HTTPException(status_code=500, detail="Database engine not initialized")

    async with engine.begin() as conn:
        # Check if any row exists
        check_stmt = select(analysis_table.c.id).limit(1)
        existing = await conn.execute(check_stmt)
        row = existing.first()
        
        if row:
            # Update existing row
            update_stmt = analysis_table.update().values(
                text=payload.text, 
                result=result,
                created_at=func.now()
            ).where(analysis_table.c.id == row.id)
            await conn.execute(update_stmt)
        else:
            # Insert new row if none exists
            insert_stmt = analysis_table.insert().values(text=payload.text, result=result)
            await conn.execute(insert_stmt)

    return {"status": "success", "result": result}


@app.get("/analysis")
async def get_analysis():
    """Return all analysis records (most recent first)."""
    if engine is None:
        raise HTTPException(status_code=500, detail="Database engine not initialized")

    sel = select(
        analysis_table.c.id,
        analysis_table.c.text,
        analysis_table.c.result,
        analysis_table.c.created_at,
    ).order_by(analysis_table.c.created_at.desc())

    async with engine.connect() as conn:
        result = await conn.execute(sel)
        rows = result.fetchall()

    if not rows:
        return []

    return [
        {"id": r.id, "text": r.text, "result": r.result, "created_at": r.created_at}
        for r in rows
    ]



