"""
FastAPI backend for the AI-Driven Adaptive Traffic Control &
Emergency Green Corridor System.

Exposes:
- POST /api/junction/update      -> ingest AI-engine detection results
- POST /api/emergency/trigger    -> activate a Green Corridor override
- POST /api/emergency/clear      -> deactivate the active Green Corridor
- GET  /api/junctions            -> current snapshot of all junctions
- WS   /ws/dashboard             -> live push feed for the control-room UI

Run with:
    uvicorn main:app --reload --port 8000
"""

import json
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from models import LaneUpdate, GreenCorridorRequest, GreenCorridorStop
from junction_manager import JunctionManager
from websocket_manager import ConnectionManager

manager = JunctionManager()
ws_manager = ConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    manager.set_broadcast_callback(ws_manager.broadcast)
    loop_task = asyncio.create_task(manager.run_signal_loop(interval_seconds=1.0))
    yield
    loop_task.cancel()


app = FastAPI(
    title="Raipur Adaptive Traffic Control System",
    description="Backend for AI-driven adaptive signal timing and emergency green corridors.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"status": "ok", "service": "Raipur Adaptive Traffic Control System"}


@app.get("/api/junctions")
async def get_junctions():
    return manager.get_summary()


@app.post("/api/junction/update")
async def update_junction(update: LaneUpdate):
    """Ingests one reading from a SINGLE approach-road camera (north/east/
    south/west) at a junction. A full junction receives 4 of these per
    detection cycle, one per approach."""
    manager.apply_lane_update(update)

    # Auto-trigger a single-junction corridor if the AI engine flags an
    # emergency on this approach and no route-level corridor is active yet.
    if update.emergency.is_emergency and manager.active_corridor_route is None:
        manager.trigger_green_corridor(
            [GreenCorridorStop(junction_id=update.junction_id, approach=update.approach)],
            update.emergency.vehicle_type or "ambulance",
        )

    await ws_manager.broadcast(manager.get_summary())
    return {"status": "accepted", "junction_id": update.junction_id, "approach": update.approach}


@app.post("/api/emergency/trigger")
async def trigger_emergency(request: GreenCorridorRequest):
    manager.trigger_green_corridor(request.route, request.vehicle_type)
    await ws_manager.broadcast(manager.get_summary())
    return {"status": "green_corridor_active", "route": request.route}


@app.post("/api/emergency/clear")
async def clear_emergency():
    manager.clear_green_corridor()
    await ws_manager.broadcast(manager.get_summary())
    return {"status": "green_corridor_cleared"}


@app.websocket("/ws/dashboard")
async def websocket_dashboard(websocket: WebSocket):
    await ws_manager.connect(websocket)
    try:
        await websocket.send_text(json.dumps(manager.get_summary()))
        while True:
            await websocket.receive_text()  # keep-alive / ignore inbound pings
    except WebSocketDisconnect:
        ws_manager.disconnect(websocket)
