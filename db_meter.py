#!/usr/bin/env python3
"""
DB Meter Module for Audio Level Visualization

This module provides audio level metering functionality with simple terminal output.
It includes utilities for:
- Converting audio amplitudes to dB levels
- Normalizing dB values for display
- Drawing visual meter bars
- Managing participant audio level tracking

The module uses a simple terminal display with single-line compact output.
"""

import sys
import time
import threading
import numpy as np
from typing import Dict, Any, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    # Avoid circular imports while maintaining type hints
    from stream_audio import AudioStreamer

# DB meter configuration constants
MAX_AUDIO_BAR = 20      # Maximum width of audio meter bars in characters
INPUT_DB_MIN = -70.0    # Minimum dB level for meter scaling
INPUT_DB_MAX = 0.0      # Maximum dB level for meter scaling
FPS = 16               # Refresh rate for meter updates


def _esc(*codes: int) -> str:
    """
    Generate ANSI escape sequence for terminal colors and formatting.
    
    Args:
        *codes: Variable number of ANSI escape codes
        
    Returns:
        ANSI escape sequence string
    """
    return "\033[" + ";".join(str(c) for c in codes) + "m"


def normalize_db(amplitude_db: float, db_min: float = INPUT_DB_MIN, db_max: float = INPUT_DB_MAX) -> float:
    """
    Normalize a dB value to a 0-1 range for meter display.
    
    Args:
        amplitude_db: The dB level to normalize
        db_min: Minimum dB value for the range (default: INPUT_DB_MIN)
        db_max: Maximum dB value for the range (default: INPUT_DB_MAX)
        
    Returns:
        Normalized value between 0.0 and 1.0
    """
    # Clamp the amplitude to the specified range
    amplitude_db = max(db_min, min(amplitude_db, db_max))
    # Normalize to 0-1 range
    return (amplitude_db - db_min) / (db_max - db_min)


def calculate_db_from_samples(audio_samples: np.ndarray) -> float:
    """
    Calculate dB level from audio samples.
    
    Args:
        audio_samples: Audio samples as numpy array (int16 format expected)
        
    Returns:
        dB level as float
    """
    if len(audio_samples) == 0:
        return INPUT_DB_MIN
    
    # Calculate RMS (Root Mean Square) amplitude
    rms = np.sqrt(np.mean(audio_samples.astype(np.float32) ** 2))
    
    # Convert to dB relative to maximum int16 value
    max_int16 = np.iinfo(np.int16).max
    db_level = 20.0 * np.log10(rms / max_int16 + 1e-6)  # Add small epsilon to avoid log(0)
    
    return db_level


class SimpleTerminalMeter:
    """
    Simple terminal-based audio meter display.
    
    Provides a compact, single-line display suitable for terminals.
    """
    
    def __init__(self):
        self.stdout_lock = threading.Lock()
        
    def print_meter(self, streamer: 'AudioStreamer'):
        """
        Print a compact single-line audio meter with participant information.
        
        Args:
            streamer: AudioStreamer instance containing current state
        """
        if not streamer.meter_running:
            return
        
        # Build status information (compact format)
        status_info = (f"I:{streamer.input_callback_count} "
                      f"O:{streamer.output_callback_count} "
                      f"Q:{streamer.audio_input_queue.qsize()} "
                      f"P:{len(streamer.participants)} "
                      f"M:{streamer.mixer_frames_received} "
                      f"A:{streamer.output_frames_with_audio} ")
        
        meter_parts = []
        
        # === Local Microphone Meter ===
        amplitude_db = normalize_db(streamer.micro_db, db_min=INPUT_DB_MIN, db_max=INPUT_DB_MAX)
        nb_bar = round(amplitude_db * MAX_AUDIO_BAR)
        
        # Color based on level
        color_code = 31 if amplitude_db > 0.75 else 33 if amplitude_db > 0.5 else 32
        bar = "#" * nb_bar + "-" * (MAX_AUDIO_BAR - nb_bar)
        
        # Live/mute indicator
        with streamer.mute_lock:
            is_muted = streamer.is_muted
        
        if is_muted:
            live_indicator = f"{_esc(90)}●{_esc(0)} "  # Gray dot for muted
        else:
            live_indicator = f"{_esc(1, 38, 2, 255, 0, 0)}●{_esc(0)} "   # Bright red dot for live
        
        # Combine local mic components
        local_part = (f"{live_indicator}Mic[{streamer.micro_db:6.1f}]"
                     f"{_esc(color_code)}[{bar}]{_esc(0)}")
        meter_parts.append(local_part)
        
        # === Participant Meters ===
        current_time = time.time()
        with streamer.participants_lock:
            # Apply timeout logic to mark inactive participants
            for participant_id, info in list(streamer.participants.items()):
                time_since_update = current_time - info['last_update']
                if info.get('has_audio', False) and time_since_update > 2.0:
                    info['has_audio'] = False
                    info['db_level'] = INPUT_DB_MIN
            
            # Create meter for each participant
            for participant_id, info in streamer.participants.items():
                # Calculate participant meter (smaller than local mic)
                participant_amplitude_db = normalize_db(info['db_level'], 
                                                      db_min=INPUT_DB_MIN, 
                                                      db_max=INPUT_DB_MAX)
                participant_nb_bar = round(participant_amplitude_db * (MAX_AUDIO_BAR // 2))
                
                participant_color_code = (31 if participant_amplitude_db > 0.75 
                                        else 33 if participant_amplitude_db > 0.5 
                                        else 32)
                participant_bar = "#" * participant_nb_bar + "-" * ((MAX_AUDIO_BAR // 2) - participant_nb_bar)
                
                # Status indicator based on audio activity
                if info.get('has_audio', False):
                    participant_indicator = f"{_esc(94)}●{_esc(0)} "  # Blue dot for active
                else:
                    participant_indicator = f"{_esc(90)}●{_esc(0)} "  # Gray dot for inactive
                
                # Combine participant components
                participant_part = (f"{participant_indicator}{info['name'][:6]}"
                                  f"[{info['db_level']:6.1f}]"
                                  f"{_esc(participant_color_code)}[{participant_bar}]{_esc(0)}")
                meter_parts.append(participant_part)
        
        # Combine all parts into single line
        meter_text = status_info + " ".join(meter_parts)
        
        # Output with proper terminal control
        with self.stdout_lock:
            # Clear line, position cursor at start, hide cursor, print meter
            sys.stdout.write(f"\033[2K\r\033[?25l{meter_text}")
            sys.stdout.flush()
    
    def cleanup(self):
        """Restore terminal cursor and clear the meter line."""
        with self.stdout_lock:
            # Clear the meter line and show cursor
            sys.stdout.write("\033[2K\r\033[?25h")
            sys.stdout.flush()


def create_meter_ui() -> SimpleTerminalMeter:
    """
    Factory function to create the simple terminal meter UI.
    
    Returns:
        SimpleTerminalMeter instance
    """
    return SimpleTerminalMeter() 