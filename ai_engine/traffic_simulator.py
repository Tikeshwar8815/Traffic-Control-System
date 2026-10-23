"""
AI & Computer Vision Engine - Main Simulation / Orchestration Script
-----------------------------------------------------------------------
Simulates a REAL 4-way junction: 4 separate approach-road cameras
(north / east / south / west) per junction, each running its own
vehicle detector + emergency detector + signal-timer pipeline every
cycle, and pushing per-approach results to the FastAPI backend over
REST. Also demonstrates an ambulance travelling a multi-junction route,
where at EACH junction only the approach the ambulance is arriving on
is requested green (the backend forces the other 3 approaches red).

Run this after starting the backend server:
    python traffic_simulator.py
"""

import time
import random
import argparse
import requests

from vehicle_detector import VehicleDetector
from emergency_detector import EmergencyVehicleDetector, SirenAudioAnalyzer, VisualEmergencyClassifier
from signal_timer import SignalTimerCalculator

BACKEND_URL = "http://localhost:8000"

APPROACHES = ["north", "east", "south", "west"]

JUNCTIONS = [
    {"id": "J1", "name": "GE Road - Fafadih Chowk"},
    {"id": "J2", "name": "Jail Road - Malviya Road Cross"},
    {"id": "J3", "name": "Pandri - Devendra Nagar Chowk"},
    {"id": "J4", "name": "Telibandha Chowk"},
    {"id": "J5", "name": "Jaistambh Chowk"},
    {"id": "J6", "name": "Kutchery Chowk"},
]

# Descriptive road names for each junction's 4 approaches
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

# The ambulance's medical route across 6 junctions in Raipur:
# Connects Mekahara / Central Hospital to Ramkrishna Care / VIP Road
EMERGENCY_ROUTE = [
    {"junction_id": "J1", "approach": "south"},
    {"junction_id": "J5", "approach": "south"},
    {"junction_id": "J6", "approach": "south"},
    {"junction_id": "J2", "approach": "south"},
    {"junction_id": "J3", "approach": "south"},
    {"junction_id": "J4", "approach": "south"},
]


class ApproachSimulator:
    """Wraps the full AI pipeline (detection -> classification -> timing)
    for ONE approach-road camera at ONE junction. A full 4-way junction
    is 4 of these running independently, just like 4 real CCTV feeds."""

    def __init__(self, junction_id: str, junction_name: str, approach: str, road_label: str):
        self.junction_id = junction_id
        self.junction_name = junction_name
        self.approach = approach
        self.road_label = road_label
        self.vehicle_detector = VehicleDetector()
        self.emergency_detector = EmergencyVehicleDetector()
        self.timer_calculator = SignalTimerCalculator()
        self._warm_up_background_model()

    def _warm_up_background_model(self, frames: int = 6):
        """
        MOG2 background subtraction needs a few empty-road frames to learn
        the static background before it can reliably flag moving/foreign
        objects (vehicles) as foreground. Feeding a handful of empty frames
        at startup avoids an artificial "0 vehicles" reading on cycle 1.
        """
        for _ in range(frames):
            empty_frame = VehicleDetector.generate_synthetic_frame(num_vehicles=0)
            self.vehicle_detector.analyze_frame(empty_frame)

    def run_cycle(self, force_emergency: bool = False) -> dict:
        num_vehicles = random.randint(2, 20)
        frame = VehicleDetector.generate_synthetic_frame(num_vehicles=num_vehicles)
        analysis = self.vehicle_detector.analyze_frame(frame)

        audio_buffer = SirenAudioAnalyzer.generate_synthetic_siren(is_siren=force_emergency)
        emergency_frame = VisualEmergencyClassifier.generate_synthetic_emergency_frame(
            with_lights=force_emergency
        )
        emergency_result = self.emergency_detector.detect(emergency_frame, audio_buffer)

        green_time = self.timer_calculator.calculate_green_time(
            analysis.counts, analysis.queue_length
        )

        is_emergency = emergency_result.is_emergency or force_emergency

        return {
            "junction_id": self.junction_id,
            "junction_name": self.junction_name,
            "approach": self.approach,
            "road_label": self.road_label,
            "counts": analysis.counts,
            "total_vehicles": analysis.total_vehicles,
            "queue_length": analysis.queue_length,
            "recommended_green_seconds": green_time,
            "emergency": {
                "is_emergency": is_emergency,
                "vehicle_type": (emergency_result.vehicle_type if is_emergency else None),
                "confidence": emergency_result.combined_confidence,
            },
        }


def post_lane_update(payload: dict):
    try:
        resp = requests.post(f"{BACKEND_URL}/api/junction/update", json=payload, timeout=3)
        resp.raise_for_status()
        tag = "EMERGENCY" if payload["emergency"]["is_emergency"] else "normal"
        print(f"[OK] {payload['junction_id']}/{payload['approach']:<5} "
              f"({payload['road_label']}) queue={payload['queue_length']} "
              f"green={payload['recommended_green_seconds']}s [{tag}]")
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] Could not reach backend: {exc}")


def trigger_green_corridor(route: list, vehicle_type: str = "ambulance"):
    try:
        resp = requests.post(
            f"{BACKEND_URL}/api/emergency/trigger",
            json={"route": route, "vehicle_type": vehicle_type},
            timeout=3,
        )
        resp.raise_for_status()
        route_str = " -> ".join(f"{s['junction_id']}({s['approach']})" for s in route)
        print(f"[GREEN CORRIDOR TRIGGERED] route={route_str}")
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] Could not trigger green corridor: {exc}")


def clear_green_corridor():
    try:
        requests.post(f"{BACKEND_URL}/api/emergency/clear", timeout=3)
        print("[GREEN CORRIDOR CLEARED]")
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR] Could not clear green corridor: {exc}")


def main(cycle_seconds: float, simulate_emergency_every: int):
    # Build 4 independent approach-camera simulators per junction (16 total).
    simulators = []
    for j in JUNCTIONS:
        for approach in APPROACHES:
            label = APPROACH_ROAD_LABELS[j["id"]][approach]
            simulators.append(ApproachSimulator(j["id"], j["name"], approach, label))

    route_lookup = {(stop["junction_id"], stop["approach"]) for stop in EMERGENCY_ROUTE}

    cycle_index = 0
    while True:
        cycle_index += 1
        force_emergency_this_cycle = (
            simulate_emergency_every > 0 and cycle_index % simulate_emergency_every == 0
        )

        if force_emergency_this_cycle:
            route_str = " -> ".join(f"{s['junction_id']}({s['approach']})" for s in EMERGENCY_ROUTE)
            print(f"\n=== Cycle {cycle_index}: Simulating ambulance on route {route_str} ===")
            trigger_green_corridor(EMERGENCY_ROUTE)

        for sim in simulators:
            force = force_emergency_this_cycle and (sim.junction_id, sim.approach) in route_lookup
            payload = sim.run_cycle(force_emergency=force)
            post_lane_update(payload)

        if force_emergency_this_cycle:
            time.sleep(cycle_seconds * 2)
            clear_green_corridor()

        time.sleep(cycle_seconds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Traffic Engine Simulator")
    parser.add_argument("--cycle-seconds", type=float, default=5.0,
                         help="Seconds between each simulated detection cycle")
    parser.add_argument("--emergency-every", type=int, default=6,
                         help="Trigger a simulated ambulance every N cycles (0 to disable)")
    args = parser.parse_args()

    print("Starting AI Traffic Engine Simulator...")
    print(f"Junctions: {[j['id'] for j in JUNCTIONS]} x 4 approaches each = "
          f"{len(JUNCTIONS) * len(APPROACHES)} camera feeds")
    print(f"Backend: {BACKEND_URL}")
    main(args.cycle_seconds, args.emergency_every)
