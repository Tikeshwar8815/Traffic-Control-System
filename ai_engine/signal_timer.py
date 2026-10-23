"""
AI & Computer Vision Engine - Dynamic Signal Timer Calculator
-----------------------------------------------------------------
Computes adaptive green-light durations based on real-time vehicle
queue length and composition, replacing static fixed-time signal
plans with demand-responsive timing.
"""

from dataclasses import dataclass


@dataclass
class TimerConfig:
    base_green_seconds: int = 20
    min_green_seconds: int = 15
    max_green_seconds: int = 90
    seconds_per_vehicle: float = 2.2
    heavy_vehicle_weight: float = 1.8
    bike_weight: float = 0.6
    car_weight: float = 1.0
    yellow_seconds: int = 4
    all_red_seconds: int = 2


class SignalTimerCalculator:
    """Calculates dynamic green-phase duration for a single approach/lane group."""

    def __init__(self, config: TimerConfig = None):
        self.config = config or TimerConfig()

    def calculate_green_time(self, counts: dict, queue_length: int) -> int:
        """
        counts: {"bike": int, "car": int, "heavy_vehicle": int}
        queue_length: number of vehicles physically waiting at the stop line

        Weighted demand = sum(count_i * weight_i), blended with the raw
        queue length so a long backlog always pushes the timer up even
        if classification undercounts.
        """
        cfg = self.config
        weighted_demand = (
            counts.get("bike", 0) * cfg.bike_weight
            + counts.get("car", 0) * cfg.car_weight
            + counts.get("heavy_vehicle", 0) * cfg.heavy_vehicle_weight
        )
        demand_score = max(weighted_demand, queue_length)

        extra_time = demand_score * cfg.seconds_per_vehicle
        green_time = cfg.base_green_seconds + extra_time

        green_time = max(cfg.min_green_seconds, min(cfg.max_green_seconds, green_time))
        return int(round(green_time))

    def calculate_full_cycle(self, junction_lane_data: dict) -> dict:
        """
        junction_lane_data: {lane_name: {"counts": {...}, "queue_length": int}}
        Returns per-lane green times plus fixed yellow/all-red phases,
        and a total cycle length for the junction.
        """
        result = {}
        total_cycle = 0
        for lane_name, data in junction_lane_data.items():
            green = self.calculate_green_time(data["counts"], data["queue_length"])
            result[lane_name] = {
                "green_seconds": green,
                "yellow_seconds": self.config.yellow_seconds,
                "all_red_seconds": self.config.all_red_seconds,
            }
            total_cycle += green + self.config.yellow_seconds + self.config.all_red_seconds

        result["total_cycle_seconds"] = total_cycle
        return result
