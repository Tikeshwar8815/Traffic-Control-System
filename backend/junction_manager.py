"""
Core stateful traffic-signal manager.

Every junction models a REAL 4-way intersection: 4 independent approach
roads (north / east / south / west), each with its own camera-derived
vehicle counts and its own red/yellow/green signal head. During normal
operation only ONE approach is green at a time and the system round-robins
through north -> east -> south -> west, exactly like a real controller
(never more than one green to avoid conflicting flows).

During an emergency Green Corridor, the specific approach the ambulance is
arriving on is forced green while the OTHER THREE approaches at that same
junction are forced red. This repeats junction-by-junction along the
ambulance's route, so cross-traffic always stops for the ambulance's road
while the rest of the junction is held safely red.
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List, Optional
from datetime import datetime, timezone
import asyncio

from models import (
    LaneUpdate, LaneState, JunctionState, RouteState,
    VehicleCounts, EmergencyInfo, GreenCorridorStop, APPROACHES,
)

DEFAULT_GREEN_SECONDS = 20
DEFAULT_YELLOW_SECONDS = 4
GREEN_CORRIDOR_HOLD_SECONDS = 25

ROUTES_CONFIG = {
    "route_ns": {
        "name": "Route 1: North - South Corridor",
        "approaches": ["north", "south"],
    },
    "route_ew": {
        "name": "Route 2: East - West Corridor",
        "approaches": ["east", "west"],
    },
}
ROUTE_KEYS = ["route_ns", "route_ew"]

# Descriptive road names for each junction's 4 approaches across 6 junctions
APPROACH_ROAD_LABELS = {
    "J1": {
        "north": "GE Road (from Telibandha)",
        "south": "GE Road (towards Jail Road)",
        "east": "Fafadih Chowk Road",
        "west": "Vivekanand Nagar Link Road",
    },
    "J2": {
        "north": "Jail Road (from GE Road)",
        "south": "Jail Road (towards Pandri)",
        "east": "Malviya Road",
        "west": "Shastri Chowk Link",
    },
    "J3": {
        "north": "Pandri Road (from Jail Road)",
        "south": "Devendra Nagar Road",
        "east": "Kutchery Road",
        "west": "Pandri Market Road",
    },
    "J4": {
        "north": "Telibandha Road (from Pandri)",
        "south": "VIP Road (towards Airport)",
        "east": "Ring Road No.1 (to Bilaspur)",
        "west": "Marine Drive / Lake Road",
    },
    "J5": {
        "north": "MG Road (from Railway Station)",
        "south": "Sadar Bazar Main Road",
        "east": "GE Road (from Sharda Chowk)",
        "west": "GE Road (towards Tatibandh)",
    },
    "J6": {
        "north": "Collectorate Parisar Road",
        "south": "Raj Bhavan - Civil Lines Road",
        "east": "Shastri Chowk Link Road",
        "west": "Gandhi Chowk Road",
    },
}


class Lane:
    """One of the 4 approach roads at a junction."""

    def __init__(self, approach: str, road_label: str):
        self.approach = approach
        self.road_label = road_label
        self.counts = VehicleCounts()
        self.total_vehicles = 0
        self.queue_length = 0
        self.recommended_green_seconds = DEFAULT_GREEN_SECONDS
        self.current_phase = "red"
        self.phase_remaining_seconds = DEFAULT_GREEN_SECONDS
        self.is_corridor_lane = False

    def to_state(self) -> LaneState:
        return LaneState(
            approach=self.approach,
            road_label=self.road_label,
            counts=self.counts,
            total_vehicles=self.total_vehicles,
            queue_length=self.queue_length,
            current_phase=self.current_phase,
            phase_remaining_seconds=self.phase_remaining_seconds,
            recommended_green_seconds=self.recommended_green_seconds,
            is_corridor_lane=self.is_corridor_lane,
        )

    def apply_update(self, update: LaneUpdate):
        if update.road_label:
            self.road_label = update.road_label
        self.counts = update.counts
        self.total_vehicles = update.total_vehicles
        self.queue_length = update.queue_length
        self.recommended_green_seconds = update.recommended_green_seconds


class Junction:
    """
    Models a real 4-way intersection controlled by a 2-route phase machine:
    - Route 1 (route_ns): North & South operate synchronously
    - Route 2 (route_ew): East & West operate synchronously
    Green times are computed dynamically based on real-time vehicle density.
    """

    def __init__(self, junction_id: str, name: str):
        self.junction_id = junction_id
        self.name = name
        labels = APPROACH_ROAD_LABELS.get(junction_id, {a: a.title() + " Approach" for a in APPROACHES})
        self.lanes: Dict[str, Lane] = {a: Lane(a, labels.get(a, a.title())) for a in APPROACHES}

        # 2-Route cycle setup: route_ns starts green, route_ew starts red
        self.active_route_index = 0
        self.route_phase = "green"  # "green" or "yellow"
        self.phase_remaining_seconds = DEFAULT_GREEN_SECONDS
        self.dynamic_green_seconds = DEFAULT_GREEN_SECONDS

        self.green_corridor_active = False
        self.corridor_approach: Optional[str] = None
        self.corridor_route_id: Optional[str] = None
        self.emergency = EmergencyInfo()
        self.last_updated = datetime.now(timezone.utc).isoformat()

        self._sync_lane_phases()

    def _sync_lane_phases(self):
        """Updates all 4 lanes according to the current active route and phase."""
        if self.green_corridor_active:
            return

        active_route_key = ROUTE_KEYS[self.active_route_index]
        active_approaches = ROUTES_CONFIG[active_route_key]["approaches"]
        inactive_route_key = ROUTE_KEYS[(self.active_route_index + 1) % len(ROUTE_KEYS)]
        inactive_approaches = ROUTES_CONFIG[inactive_route_key]["approaches"]

        # Active route lanes
        for a in active_approaches:
            self.lanes[a].current_phase = self.route_phase
            self.lanes[a].phase_remaining_seconds = self.phase_remaining_seconds

        # Inactive cross-route lanes (Red, remaining time until green)
        red_countdown = self.phase_remaining_seconds + (DEFAULT_YELLOW_SECONDS if self.route_phase == "green" else 0)
        for a in inactive_approaches:
            self.lanes[a].current_phase = "red"
            self.lanes[a].phase_remaining_seconds = red_countdown

    def get_dynamic_green_for_route(self, route_key: str) -> int:
        """Calculates dynamic green duration based on real-time vehicle counts & queue length."""
        approaches = ROUTES_CONFIG[route_key]["approaches"]
        rec_times = [self.lanes[a].recommended_green_seconds for a in approaches]
        queues = [self.lanes[a].queue_length for a in approaches]
        totals = [self.lanes[a].total_vehicles for a in approaches]

        base_rec = max(rec_times) if rec_times else DEFAULT_GREEN_SECONDS
        queue_boost = int(max(queues) * 1.5) if queues else 0
        density_boost = int(max(totals) * 0.5) if totals else 0

        dynamic_green = max(base_rec, DEFAULT_GREEN_SECONDS + queue_boost + density_boost)
        # Clamped within IRC standard bounds (15s to 60s)
        return max(15, min(60, dynamic_green))

    def to_state(self) -> JunctionState:
        active_route_key = (
            self.corridor_route_id if self.green_corridor_active
            else ROUTE_KEYS[self.active_route_index]
        )

        routes_dict = {}
        for r_key in ROUTE_KEYS:
            apprs = ROUTES_CONFIG[r_key]["approaches"]
            is_active = (r_key == active_route_key)
            if self.green_corridor_active:
                phase = "green" if is_active else "red"
            else:
                phase = self.route_phase if is_active else "red"

            tot_v = sum(self.lanes[a].total_vehicles for a in apprs)
            tot_q = sum(self.lanes[a].queue_length for a in apprs)

            routes_dict[r_key] = RouteState(
                route_id=r_key,
                name=ROUTES_CONFIG[r_key]["name"],
                current_phase=phase,
                phase_remaining_seconds=self.lanes[apprs[0]].phase_remaining_seconds,
                dynamic_green_seconds=self.get_dynamic_green_for_route(r_key),
                approaches=apprs,
                total_vehicles=tot_v,
                total_queue=tot_q,
            )

        active_lane = self.corridor_approach if self.green_corridor_active else ROUTES_CONFIG[active_route_key]["approaches"][0]

        return JunctionState(
            junction_id=self.junction_id,
            name=self.name,
            lanes={a: lane.to_state() for a, lane in self.lanes.items()},
            routes=routes_dict,
            active_route=active_route_key,
            active_green_approach=active_lane,
            green_corridor_active=self.green_corridor_active,
            corridor_approach=self.corridor_approach,
            corridor_route_id=self.corridor_route_id,
            last_updated=self.last_updated,
        )

    def tick(self):
        """Advances the synchronized 2-route state machine by one second."""
        if self.green_corridor_active:
            # During emergency override, corridor countdown ticks
            if self.phase_remaining_seconds > 0:
                self.phase_remaining_seconds -= 1
                for lane in self.lanes.values():
                    lane.phase_remaining_seconds = self.phase_remaining_seconds
            return

        self.phase_remaining_seconds -= 1
        if self.phase_remaining_seconds <= 0:
            if self.route_phase == "green":
                # Green ended -> transition active route to Yellow
                self.route_phase = "yellow"
                self.phase_remaining_seconds = DEFAULT_YELLOW_SECONDS
            else:
                # Yellow ended -> switch to other Route with dynamically calculated green duration
                self.active_route_index = (self.active_route_index + 1) % len(ROUTE_KEYS)
                self.route_phase = "green"
                active_route_key = ROUTE_KEYS[self.active_route_index]
                self.dynamic_green_seconds = self.get_dynamic_green_for_route(active_route_key)
                self.phase_remaining_seconds = self.dynamic_green_seconds

        self._sync_lane_phases()

    def apply_lane_update(self, update: LaneUpdate):
        lane = self.lanes.get(update.approach)
        if lane is None:
            return
        lane.apply_update(update)
        self.emergency = update.emergency
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def force_green_corridor(self, approach: str, seconds: int = GREEN_CORRIDOR_HOLD_SECONDS):
        """
        Emergency Green Corridor:
        - The corridor approach and its paired opposing route direction stay GREEN
          (cross route is forced RED without stopping the same axis flow).
        - The other route (perpendicular traffic) is forced RED.
        """
        if approach not in self.lanes:
            return

        corridor_route_key = "route_ns" if approach in ["north", "south"] else "route_ew"
        cross_route_key = "route_ew" if corridor_route_key == "route_ns" else "route_ns"

        self.green_corridor_active = True
        self.corridor_approach = approach
        self.corridor_route_id = corridor_route_key
        self.phase_remaining_seconds = seconds

        # Emergency corridor route: Both sides green, arrival lane flagged as corridor
        for a in ROUTES_CONFIG[corridor_route_key]["approaches"]:
            self.lanes[a].current_phase = "green"
            self.lanes[a].phase_remaining_seconds = seconds
            self.lanes[a].is_corridor_lane = (a == approach)

        # Cross route: Forced RED
        for a in ROUTES_CONFIG[cross_route_key]["approaches"]:
            self.lanes[a].current_phase = "red"
            self.lanes[a].phase_remaining_seconds = seconds
            self.lanes[a].is_corridor_lane = False

    def release_corridor(self):
        for lane in self.lanes.values():
            lane.is_corridor_lane = False
        self.green_corridor_active = False
        self.corridor_approach = None
        self.corridor_route_id = None

        # Return to normal cycling: active route transitions to yellow clearance
        self.route_phase = "yellow"
        self.phase_remaining_seconds = DEFAULT_YELLOW_SECONDS
        self._sync_lane_phases()


class JunctionManager:
    """Owns all 6 Junction instances plus active Green Corridor routes."""

    KNOWN_JUNCTIONS = {
        "J1": "GE Road - Fafadih Chowk",
        "J2": "Jail Road - Malviya Road Cross",
        "J3": "Pandri - Devendra Nagar Chowk",
        "J4": "Telibandha Chowk",
        "J5": "Jaistambh Chowk",
        "J6": "Kutchery Chowk",
    }

    def __init__(self):
        self.junctions: Dict[str, Junction] = {
            jid: Junction(jid, name) for jid, name in self.KNOWN_JUNCTIONS.items()
        }
        self.active_corridor_route: Optional[List[GreenCorridorStop]] = None
        self.active_corridor_vehicle_type: Optional[str] = None
        self._broadcast_callback = None

    def set_broadcast_callback(self, callback):
        self._broadcast_callback = callback

    def get_or_create(self, junction_id: str, name: str) -> Junction:
        if junction_id not in self.junctions:
            self.junctions[junction_id] = Junction(junction_id, name)
        return self.junctions[junction_id]

    def apply_lane_update(self, update: LaneUpdate):
        junction = self.get_or_create(update.junction_id, update.junction_name)
        junction.apply_lane_update(update)

    def trigger_green_corridor(self, route: List[GreenCorridorStop], vehicle_type: str = "ambulance"):
        self.active_corridor_route = route
        self.active_corridor_vehicle_type = vehicle_type
        for stop in route:
            junction = self.get_or_create(stop.junction_id, self.KNOWN_JUNCTIONS.get(stop.junction_id, stop.junction_id))
            junction.force_green_corridor(stop.approach)

    def clear_green_corridor(self):
        if self.active_corridor_route:
            for stop in self.active_corridor_route:
                if stop.junction_id in self.junctions:
                    self.junctions[stop.junction_id].release_corridor()
        self.active_corridor_route = None
        self.active_corridor_vehicle_type = None

    def get_all_states(self) -> List[JunctionState]:
        return [j.to_state() for j in self.junctions.values()]

    def get_summary(self) -> dict:
        return {
            "junctions": [s.dict() for s in self.get_all_states()],
            "green_corridor_active": self.active_corridor_route is not None,
            "green_corridor_route": (
                [stop.dict() for stop in self.active_corridor_route]
                if self.active_corridor_route else None
            ),
            "green_corridor_vehicle_type": self.active_corridor_vehicle_type,
        }

    async def run_signal_loop(self, interval_seconds: float = 1.0):
        while True:
            for junction in self.junctions.values():
                junction.tick()
            if self._broadcast_callback:
                await self._broadcast_callback(self.get_summary())
            await asyncio.sleep(interval_seconds)
