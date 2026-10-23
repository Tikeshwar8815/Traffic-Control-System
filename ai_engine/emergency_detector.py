"""
AI & Computer Vision Engine - Emergency Vehicle Detector
----------------------------------------------------------
Combines two independent signals to decide if an emergency vehicle
(ambulance / fire truck / police) is present at a junction:

1. Siren audio-frequency analysis - simulates a microphone feed and
   uses FFT to detect the characteristic warbling frequency band of
   emergency sirens (roughly 600 Hz - 1600 Hz, alternating every
   0.3 - 0.6s).
2. Visual flashing-light classification - scans a video frame for the
   red/blue strobe pattern typical of emergency light bars.

Either signal crossing its confidence threshold marks the frame as
an emergency event; using both together raises overall confidence.
"""

import numpy as np
import cv2
from dataclasses import dataclass


SIREN_FREQ_BAND = (600, 1600)      # Hz - typical siren fundamental band
SIREN_CONFIDENCE_THRESHOLD = 0.6
VISUAL_CONFIDENCE_THRESHOLD = 0.55


@dataclass
class EmergencyDetectionResult:
    is_emergency: bool
    audio_confidence: float
    visual_confidence: float
    combined_confidence: float
    vehicle_type: str = "unknown"


class SirenAudioAnalyzer:
    """Analyzes an audio buffer using FFT to detect siren-like frequency patterns."""

    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate

    def analyze(self, audio_buffer: np.ndarray) -> float:
        """
        Returns a confidence score [0, 1] that the buffer contains a siren.
        `audio_buffer` is a 1D numpy array of float samples in [-1, 1].
        """
        if audio_buffer.size == 0:
            return 0.0

        windowed = audio_buffer * np.hanning(len(audio_buffer))
        spectrum = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(len(windowed), d=1.0 / self.sample_rate)

        band_mask = (freqs >= SIREN_FREQ_BAND[0]) & (freqs <= SIREN_FREQ_BAND[1])
        band_energy = float(np.sum(spectrum[band_mask] ** 2))
        total_energy = float(np.sum(spectrum ** 2)) + 1e-9
        energy_ratio = band_energy / total_energy

        # Detect the "warble" - amplitude modulation between 1-4 Hz typical of sirens
        envelope = np.abs(audio_buffer)
        env_spectrum = np.abs(np.fft.rfft(envelope * np.hanning(len(envelope))))
        env_freqs = np.fft.rfftfreq(len(envelope), d=1.0 / self.sample_rate)
        warble_mask = (env_freqs >= 1.0) & (env_freqs <= 4.0)
        warble_strength = float(np.sum(env_spectrum[warble_mask])) / (float(np.sum(env_spectrum)) + 1e-9)

        confidence = min(1.0, 0.6 * energy_ratio * 4 + 0.4 * warble_strength * 3)
        return round(confidence, 3)

    @staticmethod
    def generate_synthetic_siren(duration: float = 1.0, sample_rate: int = 22050,
                                  is_siren: bool = True) -> np.ndarray:
        """Generates a synthetic audio buffer for demo/testing purposes."""
        t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
        if is_siren:
            warble_hz = 2.0
            base_freq = 900 + 400 * np.sin(2 * np.pi * warble_hz * t)
            phase = 2 * np.pi * np.cumsum(base_freq) / sample_rate
            signal = 0.8 * np.sin(phase)
            signal += 0.05 * np.random.default_rng().normal(size=t.shape)
        else:
            rng = np.random.default_rng()
            signal = 0.3 * rng.normal(size=t.shape)  # generic traffic/road noise
        return np.clip(signal, -1.0, 1.0)


class VisualEmergencyClassifier:
    """Detects red/blue emergency-light strobe patterns in a video frame."""

    def __init__(self):
        # HSV ranges for emergency red and blue light bars
        self.red_ranges = [((0, 120, 150), (10, 255, 255)), ((170, 120, 150), (180, 255, 255))]
        self.blue_range = ((100, 120, 150), (130, 255, 255))

    def analyze(self, frame: np.ndarray) -> float:
        """Returns a confidence score [0, 1] based on red+blue pixel coverage."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h, w = frame.shape[:2]
        total_pixels = h * w

        red_mask = np.zeros((h, w), dtype=np.uint8)
        for lower, upper in self.red_ranges:
            red_mask |= cv2.inRange(hsv, np.array(lower), np.array(upper))

        blue_mask = cv2.inRange(hsv, np.array(self.blue_range[0]), np.array(self.blue_range[1]))

        red_ratio = float(np.count_nonzero(red_mask)) / total_pixels
        blue_ratio = float(np.count_nonzero(blue_mask)) / total_pixels

        # Both colors present together is a strong emergency light-bar signal
        combined_ratio = red_ratio + blue_ratio
        co_presence_bonus = 0.3 if (red_ratio > 0.001 and blue_ratio > 0.001) else 0.0

        confidence = min(1.0, combined_ratio * 40 + co_presence_bonus)
        return round(confidence, 3)

    @staticmethod
    def generate_synthetic_emergency_frame(width: int = 640, height: int = 480,
                                            with_lights: bool = True) -> np.ndarray:
        """Generates a synthetic frame, optionally containing a red/blue light bar."""
        frame = np.full((height, width, 3), 40, dtype=np.uint8)
        if with_lights:
            cv2.rectangle(frame, (280, 180), (320, 195), (0, 0, 255), -1)   # red (BGR)
            cv2.rectangle(frame, (320, 180), (360, 195), (255, 0, 0), -1)   # blue (BGR)
        return frame


class EmergencyVehicleDetector:
    """Fuses audio and visual signals to flag emergency vehicles."""

    def __init__(self):
        self.audio_analyzer = SirenAudioAnalyzer()
        self.visual_classifier = VisualEmergencyClassifier()

    def detect(self, frame: np.ndarray, audio_buffer: np.ndarray) -> EmergencyDetectionResult:
        audio_conf = self.audio_analyzer.analyze(audio_buffer)
        visual_conf = self.visual_classifier.analyze(frame)

        combined = round(0.55 * audio_conf + 0.45 * visual_conf, 3)
        is_emergency = (
            audio_conf >= SIREN_CONFIDENCE_THRESHOLD
            or visual_conf >= VISUAL_CONFIDENCE_THRESHOLD
            or combined >= 0.5
        )

        vehicle_type = "ambulance" if is_emergency else "unknown"

        return EmergencyDetectionResult(
            is_emergency=is_emergency,
            audio_confidence=audio_conf,
            visual_confidence=visual_conf,
            combined_confidence=combined,
            vehicle_type=vehicle_type,
        )
