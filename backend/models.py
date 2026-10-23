"""
Pydantic models for request/response validation across the backend API.

Each junction is modelled as a real 4-way intersection: 4 independent
approach roads (north / east / south / west), each with its own camera
feed, its own vehicle counts, and its own red/yellow/green signal head.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict

APPROACHES = ["north", "east", "south", "west"]
ROUTES = {
    "route_ns": ["north", "south"],
    "route_ew": ["east", "west"],
}


class VehicleCounts(BaseModel):
    bike: int = 0
    car: int = 0
    heavy_vehicle: int = 0


class EmergencyInfo(BaseModel):
    is_emergency: bool = False
    vehicle_type: Optional[str] = None
    confidence: float = 0.0


class LaneUpdate(BaseModel):
    """One AI-engine reading from a SINGLE approach-road camera at a
    junction (one of the 4 cameras that make up a full junction)."""
    junction_id: str
    junction_name: str
    approach: str = Field(..., description="One of: north, east, south, west")
    road_label: str = ""
    counts: VehicleCounts
    total_vehicles: int
    queue_length: int
    recommended_green_seconds: int
    emergency: EmergencyInfo


class LaneState(BaseModel):
    approach: str
    road_label: str
    counts: VehicleCounts
    total_vehicles: int
    queue_length: int
    current_phase: str  # "red", "yellow", "green"
    phase_remaining_seconds: int
    recommended_green_seconds: int
    is_corridor_lane: bool = False


class RouteState(BaseModel):
    route_id: str  # "route_ns" or "route_ew"
    name: str      # "North-South Route" or "East-West Route"
    current_phase: str  # "red", "yellow", "green"
    phase_remaining_seconds: int
    dynamic_green_seconds: int
    approaches: List[str]
    total_vehicles: int = 0
    total_queue: int = 0


class JunctionState(BaseModel):
    junction_id: str
    name: str
    lanes: Dict[str, LaneState]  # keyed by approach: north / east / south / west
    routes: Optional[Dict[str, RouteState]] = None
    active_route: str = "route_ns"
    active_green_approach: Optional[str] = None
    green_corridor_active: bool = False
    corridor_approach: Optional[str] = None
    corridor_route_id: Optional[str] = None
    last_updated: str


class GreenCorridorStop(BaseModel):
    """One hop of the emergency vehicle's route: which junction, and
    which approach road it is arriving on (that approach gets forced
    green while the other three at that junction are forced red)."""
    junction_id: str
    approach: str = Field(..., description="Approach road the emergency vehicle needs green on: north/east/south/west")


class GreenCorridorRequest(BaseModel):
    route: List[GreenCorridorStop] = Field(
        ..., description="Ordered list of {junction_id, approach} stops the emergency vehicle will pass through"
    )
    vehicle_type: str = "ambulance"
