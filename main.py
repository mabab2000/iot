import os
from typing import Optional, Any
import asyncio

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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


class TextRequest(BaseModel):
    text: str


# WebSocket connection management
_ws_connections: set[WebSocket] = set()
_ws_lock = asyncio.Lock()


async def broadcast(message: Any) -> None:
    """Send `message` (JSON-serializable) to all connected websockets.

    Removes connections that are closed or raise on send.
    """
    to_remove = []
    async with _ws_lock:
        conns = list(_ws_connections)

    for ws in conns:
        try:
            await ws.send_json(message)
        except Exception:
            to_remove.append(ws)

    if to_remove:
        async with _ws_lock:
            for ws in to_remove:
                _ws_connections.discard(ws)



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


@app.post("/analyze")
async def analyze_text(payload: TextRequest):
    """Analyze the provided text via OpenAI and return 'ON' or 'OFF'."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment")

    system_prompt = (
        "You are a classifier. Given an instruction or text, respond with exactly one word: ON or OFF. "
        "Return ON when the text requests or implies turning something on (e.g., 'turn on the LED', 'please enable the light'). "
        "Return OFF otherwise. Respond with only the single word ON or OFF and nothing else."
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
                json={"model": "gpt-3.5-turbo", "messages": messages, "max_tokens": 3, "temperature": 0.0},
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
    # Extract ON or OFF from the response
    m = re.search(r"\b(ON|OFF)\b", text)
    if m:
        result = m.group(1)
    else:
        # fallback heuristic based on user text
        lower = payload.text.lower()
        if any(k in lower for k in ("turn on", "please on", "switch on", "enable", "start")):
            result = "ON"
        else:
            result = "OFF"

    # notify websocket clients about the analysis result as JSON {"result":"OFF"}
    await broadcast({"result": result})

    return {"status": "success", "result": result}


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint that keeps client connections open and receives optional pings.

    Clients should connect to `ws://<host>/ws`. When an analysis happens, server will
    push JSON messages like {"type":"analyze","result":"ON","text":"..."}.
    """
    await websocket.accept()
    async with _ws_lock:
        _ws_connections.add(websocket)

    try:
        while True:
            # keep connection alive by waiting for incoming messages (clients may send pings)
            try:
                await websocket.receive_text()
            except WebSocketDisconnect:
                break
            except Exception:
                # ignore other receive errors and continue listening
                await asyncio.sleep(0.1)
    finally:
        async with _ws_lock:
            _ws_connections.discard(websocket)
