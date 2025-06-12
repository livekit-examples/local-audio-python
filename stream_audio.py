#!/usr/bin/env -S uv run --script
# /// script
# dependencies = [
#   "livekit",
#   "livekit_api",
#   "sounddevice",
#   "python-dotenv",
#   "asyncio",
#   "numpy",
# ]
# ///
import os
import logging
import asyncio
import argparse
import sys
import time
import threading
import select
import termios
import tty
import curses
from dotenv import load_dotenv
from signal import SIGINT, SIGTERM
from livekit import rtc
from livekit.rtc import apm
import sounddevice as sd
import numpy as np
from auth import generate_token

load_dotenv()
# ensure LIVEKIT_URL, LIVEKIT_API_KEY, and LIVEKIT_API_SECRET are set in your .env file
LIVEKIT_URL = os.environ.get("LIVEKIT_URL")
ROOM_NAME = os.environ.get("ROOM_NAME")

# using exact values from example.py
SAMPLE_RATE = 24000  # 48kHz to match DC Microphone native rate
NUM_CHANNELS = 1
FRAME_SAMPLES = 240  # 10ms at 48kHz - required for APM
BLOCKSIZE = 2400  # 100ms buffer

# original
# SAMPLE_RATE = 48000  # 48kHz to match DC Microphone native rate
# NUM_CHANNELS = 1
# FRAME_SAMPLES = 480  # 10ms at 48kHz - required for APM
# BLOCKSIZE = 4800  # 100ms buffer


# dB meter settings
MAX_AUDIO_BAR = 20  # Reduced from 30 to make display more compact
INPUT_DB_MIN = -70.0
INPUT_DB_MAX = 0.0
FPS = 16

def _esc(*codes: int) -> str:
    return "\033[" + ";".join(str(c) for c in codes) + "m"

def _normalize_db(amplitude_db: float, db_min: float, db_max: float) -> float:
    amplitude_db = max(db_min, min(amplitude_db, db_max))
    return (amplitude_db - db_min) / (db_max - db_min)

def list_audio_devices():
    """List all available audio devices for debugging"""
    print("\n=== AUDIO DEVICES DEBUG ===")
    try:
        devices = sd.query_devices()
        print(f"Total devices found: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"Device {i}: {device['name']}")
            print(f"  Channels: in={device['max_input_channels']}, out={device['max_output_channels']}")
            print(f"  Sample rates: {device['default_samplerate']}")
            print(f"  Hostapi: {device['hostapi']}")
        
        default_in, default_out = sd.default.device
        print(f"\nDefault input device: {default_in}")
        print(f"Default output device: {default_out}")
        
        if default_in is not None:
            in_info = sd.query_devices(default_in)
            print(f"Default input info: {in_info['name']} - {in_info['max_input_channels']} channels")
        
        if default_out is not None:
            out_info = sd.query_devices(default_out)
            print(f"Default output info: {out_info['name']} - {out_info['max_output_channels']} channels")
            
    except Exception as e:
        print(f"Error listing audio devices: {e}")
    print("=== END AUDIO DEVICES ===\n")

class AudioStreamer:
    def __init__(self, enable_aec: bool = True, loop: asyncio.AbstractEventLoop = None):
        self.enable_aec = enable_aec
        self.running = True
        self.logger = logging.getLogger(__name__)
        self.loop = loop  # Store the event loop reference
        
        # Mute state
        self.is_muted = False
        self.mute_lock = threading.Lock()
        
        # Debug counters
        self.input_callback_count = 0
        self.output_callback_count = 0
        self.frames_processed = 0
        self.frames_sent_to_livekit = 0
        self.last_debug_time = time.time()
        
        # Audio I/O streams
        self.input_stream: sd.InputStream | None = None
        self.output_stream: sd.OutputStream | None = None
        
        # LiveKit components
        self.source = rtc.AudioSource(SAMPLE_RATE, NUM_CHANNELS)
        self.room: rtc.Room | None = None
        
        # Audio processing
        self.audio_processor: apm.AudioProcessingModule | None = None
        if enable_aec:
            self.logger.info("Initializing Audio Processing Module with Echo Cancellation")
            self.audio_processor = apm.AudioProcessingModule(
                echo_cancellation=True,
                noise_suppression=True,
                high_pass_filter=True,
                auto_gain_control=True
            )
        
        # Audio buffers and synchronization
        self.output_buffer = bytearray()
        self.output_lock = threading.Lock()
        self.audio_input_queue = asyncio.Queue(maxsize=100)  # Prevent memory buildup
        
        # Timing and delay tracking for AEC
        self.output_delay = 0.0
        self.input_delay = 0.0
        
        # dB meter
        self.micro_db = INPUT_DB_MIN
        self.input_device_name = "Microphone"
        
        # Participant tracking for dB meters
        self.participants = {}  # participant_id -> {'name': str, 'db_level': float, 'last_update': float}
        self.participants_lock = threading.Lock()
        
        # UI
        self.ui: CursesUI | None = None
        
    def start_audio_devices(self):
        """Initialize and start audio input/output devices"""
        try:
            self.logger.info("Starting audio devices...")
            
            # List all devices for debugging
            list_audio_devices()
            
            # Get device info - but override input device to use working microphone
            input_device, output_device = sd.default.device
            
            # Override to use DC Microphone (device 1) which is working
            #input_device = 1  # DC Microphone
            
            self.logger.info(f"Using input device: {input_device}, output device: {output_device}")
            
            if input_device is not None:
                device_info = sd.query_devices(input_device)
                if isinstance(device_info, dict):
                    self.input_device_name = device_info.get("name", "Microphone")
                    self.logger.info(f"Input device info: {device_info}")
                    
                    # Check if device supports our requirements
                    if device_info['max_input_channels'] < NUM_CHANNELS:
                        self.logger.warning(f"Input device only has {device_info['max_input_channels']} channels, need {NUM_CHANNELS}")
            
            self.logger.info(f"Creating input stream: rate={SAMPLE_RATE}, channels={NUM_CHANNELS}, blocksize={BLOCKSIZE}")
            
            # Start input stream
            self.input_stream = sd.InputStream(
                callback=self._input_callback,
                dtype="int16",
                channels=NUM_CHANNELS,
                device=input_device,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
            )
            self.input_stream.start()
            self.logger.info(f"Started audio input: {self.input_device_name}")
            
            # Start output stream  
            self.output_stream = sd.OutputStream(
                callback=self._output_callback,
                dtype="int16",
                channels=NUM_CHANNELS,
                device=output_device,
                samplerate=SAMPLE_RATE,
                blocksize=BLOCKSIZE,
            )
            self.output_stream.start()
            self.logger.info("Started audio output")
            
            # Test if streams are active
            time.sleep(0.1)  # Give streams time to start
            self.logger.info(f"Input stream active: {self.input_stream.active}")
            self.logger.info(f"Output stream active: {self.output_stream.active}")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio devices: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            raise
    
    def stop_audio_devices(self):
        """Stop and cleanup audio devices"""
        self.logger.info("Stopping audio devices...")
        self.running = False
        
        if self.input_stream:
            self.input_stream.stop()
            self.input_stream.close()
            self.input_stream = None
            self.logger.info("Stopped input stream")
            
        if self.output_stream:
            self.output_stream.stop()
            self.output_stream.close()
            self.output_stream = None
            self.logger.info("Stopped output stream")
            
        self.logger.info("Audio devices stopped")
    
    def toggle_mute(self):
        """Toggle microphone mute state"""
        with self.mute_lock:
            self.is_muted = not self.is_muted
            status = "MUTED" if self.is_muted else "LIVE"
            self.logger.info(f"Microphone {status}")

    def _input_callback(self, indata: np.ndarray, frame_count: int, time_info, status) -> None:
        """Sounddevice input callback - processes microphone audio"""
        self.input_callback_count += 1
        
        # Debug logging every few seconds
        current_time = time.time()
        if current_time - self.last_debug_time > 5.0:
            self.logger.info(f"Input callback stats: called {self.input_callback_count} times, "
                           f"processed {self.frames_processed} frames, "
                           f"sent {self.frames_sent_to_livekit} to LiveKit")
            self.last_debug_time = current_time
        
        if status:
            self.logger.warning(f"Input callback status: {status}")
            
        if not self.running:
            self.logger.debug("Input callback: not running, returning")
            return
            
        # Log first few callbacks for debugging
        if self.input_callback_count <= 5:
            self.logger.info(f"Input callback #{self.input_callback_count}: "
                           f"frame_count={frame_count}, "
                           f"indata.shape={indata.shape}, "
                           f"indata.dtype={indata.dtype}")
            self.logger.info(f"Audio level check - max: {np.max(np.abs(indata))}, "
                           f"mean: {np.mean(np.abs(indata)):.2f}")
            
        # Check mute state and apply if needed
        with self.mute_lock:
            is_muted = self.is_muted
        
        # If muted, replace audio data with silence but continue processing for meter
        processed_indata = indata.copy()
        if is_muted:
            processed_indata.fill(0)
            
        # Calculate delays for AEC
        self.input_delay = time_info.currentTime - time_info.inputBufferAdcTime
        total_delay = self.output_delay + self.input_delay
        
        if self.audio_processor:
            self.audio_processor.set_stream_delay_ms(int(total_delay * 1000))
        
        # Process audio in 10ms frames for AEC
        num_frames = frame_count // FRAME_SAMPLES
        
        if self.input_callback_count <= 3:
            self.logger.info(f"Processing {num_frames} frames of {FRAME_SAMPLES} samples each")
        
        for i in range(num_frames):
            start = i * FRAME_SAMPLES
            end = start + FRAME_SAMPLES
            if end > frame_count:
                break
                
            # Use original data for meter calculation, processed data for transmission
            original_chunk = indata[start:end, 0]  # For meter calculation
            capture_chunk = processed_indata[start:end, 0]  # For transmission (may be muted)
            
            # Create audio frame for AEC processing
            capture_frame = rtc.AudioFrame(
                data=capture_chunk.tobytes(),
                samples_per_channel=FRAME_SAMPLES,
                sample_rate=SAMPLE_RATE,
                num_channels=NUM_CHANNELS,
            )
            
            self.frames_processed += 1
            
            # Apply AEC if enabled
            if self.audio_processor:
                try:
                    self.audio_processor.process_stream(capture_frame)
                    if self.frames_processed <= 5:
                        self.logger.debug(f"Applied AEC to frame {self.frames_processed}")
                except Exception as e:
                    self.logger.warning(f"Error processing audio stream: {e}")
            
            # Calculate dB level for meter using original (unmuted) audio
            rms = np.sqrt(np.mean(original_chunk.astype(np.float32) ** 2))
            max_int16 = np.iinfo(np.int16).max
            self.micro_db = 20.0 * np.log10(rms / max_int16 + 1e-6)
            
            # Send to LiveKit using the stored event loop reference
            if self.loop and not self.loop.is_closed():
                try:
                    # Check queue size
                    queue_size = self.audio_input_queue.qsize()
                    if queue_size > 50:
                        self.logger.warning(f"Audio input queue getting full: {queue_size} items")
                    
                    # Use the stored loop reference instead of trying to get current loop
                    self.loop.call_soon_threadsafe(
                        self.audio_input_queue.put_nowait, capture_frame
                    )
                    self.frames_sent_to_livekit += 1
                    
                    if self.frames_sent_to_livekit <= 5:
                        self.logger.info(f"Sent frame {self.frames_sent_to_livekit} to LiveKit queue")
                        
                except Exception as e:
                    # Queue might be full or event loop might be closed
                    if self.frames_processed <= 10:
                        self.logger.warning(f"Failed to queue audio frame: {e}")
            else:
                if self.frames_processed <= 5:
                    self.logger.error("No valid event loop available for queuing audio frame")
    
    def _output_callback(self, outdata: np.ndarray, frame_count: int, time_info, status) -> None:
        """Sounddevice output callback - plays received audio"""
        self.output_callback_count += 1
        
        if status:
            self.logger.warning(f"Output callback status: {status}")
            
        # Log first few callbacks
        if self.output_callback_count <= 3:
            self.logger.info(f"Output callback #{self.output_callback_count}: "
                           f"frame_count={frame_count}, buffer_size={len(self.output_buffer)}")
        
        if not self.running:
            outdata.fill(0)
            return
            
        # Update output delay for AEC
        self.output_delay = time_info.outputBufferDacTime - time_info.currentTime
        
        # Fill output buffer from received audio
        with self.output_lock:
            bytes_needed = frame_count * 2  # 2 bytes per int16 sample
            if len(self.output_buffer) < bytes_needed:
                # Not enough data, fill what we have and zero the rest
                available_bytes = len(self.output_buffer)
                if available_bytes > 0:
                    outdata[:available_bytes // 2, 0] = np.frombuffer(
                        self.output_buffer[:available_bytes],
                        dtype=np.int16,
                        count=available_bytes // 2,
                    )
                    outdata[available_bytes // 2:, 0] = 0
                    del self.output_buffer[:available_bytes]
                else:
                    outdata.fill(0)
            else:
                # Enough data available
                chunk = self.output_buffer[:bytes_needed]
                outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16, count=frame_count)
                del self.output_buffer[:bytes_needed]
        
        # Process output through AEC reverse stream
        if self.audio_processor:
            num_chunks = frame_count // FRAME_SAMPLES
            for i in range(num_chunks):
                start = i * FRAME_SAMPLES
                end = start + FRAME_SAMPLES
                if end > frame_count:
                    break
                    
                render_chunk = outdata[start:end, 0]
                render_frame = rtc.AudioFrame(
                    data=render_chunk.tobytes(),
                    samples_per_channel=FRAME_SAMPLES,
                    sample_rate=SAMPLE_RATE,
                    num_channels=NUM_CHANNELS,
                )
                try:
                    self.audio_processor.process_reverse_stream(render_frame)
                except Exception as e:
                    if self.output_callback_count <= 10:
                        self.logger.warning(f"Error processing reverse stream: {e}")
    
    def print_audio_meter(self):
        """Deprecated - replaced by CursesUI"""
        pass

    def init_terminal(self):
        """Deprecated - replaced by CursesUI"""
        pass
            
    def restore_terminal(self):
        """Deprecated - replaced by CursesUI"""
        pass

class CursesUI:
    """Full-screen ncurses UI for the audio streaming application"""
    
    def __init__(self, streamer):
        self.streamer = streamer
        self.stdscr = None
        self.running = True
        self.ui_lock = threading.Lock()
        self.last_update = 0
        self.update_interval = 1.0 / FPS  # Update at FPS rate
        
        # Color pairs
        self.COLOR_NORMAL = 1
        self.COLOR_MUTED = 2
        self.COLOR_LIVE = 3
        self.COLOR_REMOTE = 4
        self.COLOR_GREEN_BAR = 5
        self.COLOR_YELLOW_BAR = 6
        self.COLOR_RED_BAR = 7
        self.COLOR_HEADER = 8
        
    def init_colors(self):
        """Initialize color pairs for the UI"""
        curses.start_color()
        curses.use_default_colors()
        
        # Define color pairs
        curses.init_pair(self.COLOR_NORMAL, curses.COLOR_WHITE, -1)
        curses.init_pair(self.COLOR_MUTED, curses.COLOR_BLACK, curses.COLOR_WHITE)
        curses.init_pair(self.COLOR_LIVE, curses.COLOR_RED, -1)
        curses.init_pair(self.COLOR_REMOTE, curses.COLOR_BLUE, -1)
        curses.init_pair(self.COLOR_GREEN_BAR, curses.COLOR_GREEN, -1)
        curses.init_pair(self.COLOR_YELLOW_BAR, curses.COLOR_YELLOW, -1)
        curses.init_pair(self.COLOR_RED_BAR, curses.COLOR_RED, -1)
        curses.init_pair(self.COLOR_HEADER, curses.COLOR_CYAN, -1)
    
    def init_screen(self, stdscr):
        """Initialize the curses screen"""
        self.stdscr = stdscr
        curses.curs_set(0)  # Hide cursor
        stdscr.nodelay(1)   # Non-blocking getch()
        stdscr.timeout(50)  # 50ms timeout for getch
        
        self.init_colors()
        
        # Clear screen
        stdscr.clear()
        stdscr.refresh()
    
    def draw_header(self, y_offset=0):
        """Draw the application header"""
        if not self.stdscr:
            return y_offset
            
        height, width = self.stdscr.getmaxyx()
        
        # Title
        title = "LiveKit Audio Streamer"
        self.stdscr.addstr(y_offset, (width - len(title)) // 2, title, 
                          curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD)
        y_offset += 1
        
        # Room info
        room_info = f"Room: {ROOM_NAME or 'N/A'}"
        self.stdscr.addstr(y_offset, (width - len(room_info)) // 2, room_info,
                          curses.color_pair(self.COLOR_NORMAL))
        y_offset += 1
        
        # Controls help
        controls = "Controls: [M]ute, [Q]uit"
        self.stdscr.addstr(y_offset, (width - len(controls)) // 2, controls,
                          curses.color_pair(self.COLOR_NORMAL))
        y_offset += 2
        
        return y_offset
    
    def draw_audio_bar(self, x, y, width, db_level, is_muted=False):
        """Draw an audio level bar"""
        if not self.stdscr:
            return
            
        # Normalize dB level
        normalized = _normalize_db(db_level, INPUT_DB_MIN, INPUT_DB_MAX)
        bar_fill = int(normalized * width)
        
        # Choose color based on level
        if normalized > 0.75:
            color = self.COLOR_RED_BAR
        elif normalized > 0.5:
            color = self.COLOR_YELLOW_BAR
        else:
            color = self.COLOR_GREEN_BAR
            
        if is_muted:
            color = self.COLOR_MUTED
        
        # Draw the bar
        try:
            # Filled portion
            if bar_fill > 0:
                self.stdscr.addstr(y, x, "█" * bar_fill, curses.color_pair(color))
            
            # Empty portion
            if bar_fill < width:
                self.stdscr.addstr(y, x + bar_fill, "░" * (width - bar_fill),
                                  curses.color_pair(self.COLOR_NORMAL))
        except curses.error:
            pass  # Ignore if we're at screen edge
    
    def draw_local_participant(self, y_offset):
        """Draw the local participant (microphone) section"""
        if not self.stdscr:
            return y_offset
            
        height, width = self.stdscr.getmaxyx()
        
        # Section header
        self.stdscr.addstr(y_offset, 2, "Local Microphone", 
                          curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD)
        y_offset += 1
        
        # Status indicator
        with self.streamer.mute_lock:
            is_muted = self.streamer.is_muted
        
        status_text = "MUTED" if is_muted else "LIVE"
        status_color = self.COLOR_MUTED if is_muted else self.COLOR_LIVE
        
        self.stdscr.addstr(y_offset, 4, "Status: ", curses.color_pair(self.COLOR_NORMAL))
        self.stdscr.addstr(y_offset, 12, status_text, 
                          curses.color_pair(status_color) | curses.A_BOLD)
        y_offset += 1
        
        # Device name
        device_text = f"Device: {self.streamer.input_device_name}"
        self.stdscr.addstr(y_offset, 4, device_text, curses.color_pair(self.COLOR_NORMAL))
        y_offset += 1
        
        # Audio level
        db_text = f"Level:  {self.streamer.micro_db:5.1f} dB"
        self.stdscr.addstr(y_offset, 4, db_text, curses.color_pair(self.COLOR_NORMAL))
        
        # Audio bar
        bar_width = min(40, width - 25)
        self.draw_audio_bar(25, y_offset, bar_width, self.streamer.micro_db, is_muted)
        y_offset += 2
        
        return y_offset
    
    def draw_remote_participants(self, y_offset):
        """Draw the remote participants section"""
        if not self.stdscr:
            return y_offset
            
        height, width = self.stdscr.getmaxyx()
        
        # Section header
        participant_count = len(self.streamer.participants)
        header = f"Remote Participants ({participant_count})"
        self.stdscr.addstr(y_offset, 2, header, 
                          curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD)
        y_offset += 1
        
        if participant_count == 0:
            self.stdscr.addstr(y_offset, 4, "No remote participants connected",
                              curses.color_pair(self.COLOR_NORMAL))
            y_offset += 2
            return y_offset
        
        # List participants
        current_time = time.time()
        with self.streamer.participants_lock:
            participants_list = list(self.streamer.participants.items())
        
        max_participants_to_show = min(8, height - y_offset - 5)  # Reserve space for bottom
        shown = 0
        
        for participant_id, info in participants_list:
            if shown >= max_participants_to_show:
                break
                
            # Skip stale participants
            if current_time - info['last_update'] > 5.0:
                continue
            
            name = info['name'][:20]  # Truncate long names
            db_level = info['db_level']
            
            # Participant name and level
            participant_text = f"  {name:<20} {db_level:5.1f} dB"
            self.stdscr.addstr(y_offset, 4, participant_text, 
                              curses.color_pair(self.COLOR_REMOTE))
            
            # Audio bar
            bar_width = min(30, width - 40)
            if bar_width > 0:
                self.draw_audio_bar(width - bar_width - 2, y_offset, bar_width, db_level)
            
            y_offset += 1
            shown += 1
        
        if shown < participant_count:
            self.stdscr.addstr(y_offset, 4, f"... and {participant_count - shown} more",
                              curses.color_pair(self.COLOR_NORMAL))
            y_offset += 1
            
        y_offset += 1
        return y_offset
    
    def draw_statistics(self, y_offset):
        """Draw system statistics"""
        if not self.stdscr:
            return y_offset
            
        height, width = self.stdscr.getmaxyx()
        
        # Section header
        self.stdscr.addstr(y_offset, 2, "Statistics", 
                          curses.color_pair(self.COLOR_HEADER) | curses.A_BOLD)
        y_offset += 1
        
        # Audio statistics
        stats = [
            f"Input callbacks:  {self.streamer.input_callback_count}",
            f"Output callbacks: {self.streamer.output_callback_count}",
            f"Frames processed: {self.streamer.frames_processed}",
            f"Frames to LiveKit: {self.streamer.frames_sent_to_livekit}",
            f"Audio queue size: {self.streamer.audio_input_queue.qsize()}",
        ]
        
        for i, stat in enumerate(stats):
            if y_offset + i >= height - 1:
                break
            self.stdscr.addstr(y_offset + i, 4, stat, curses.color_pair(self.COLOR_NORMAL))
        
        y_offset += len(stats) + 1
        return y_offset
    
    def update_display(self):
        """Update the entire display"""
        if not self.stdscr:
            return
            
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
            
        with self.ui_lock:
            try:
                self.stdscr.clear()
                
                y_offset = 0
                y_offset = self.draw_header(y_offset)
                y_offset = self.draw_local_participant(y_offset)
                y_offset = self.draw_remote_participants(y_offset)
                y_offset = self.draw_statistics(y_offset)
                
                self.stdscr.refresh()
                self.last_update = current_time
                
            except curses.error:
                pass  # Screen size issues, ignore
    
    def handle_input(self):
        """Handle keyboard input"""
        if not self.stdscr:
            return True
            
        try:
            key = self.stdscr.getch()
            if key == -1:  # No input
                return True
            elif key == ord('q') or key == ord('Q'):
                return False  # Quit
            elif key == ord('m') or key == ord('M'):
                self.streamer.toggle_mute()
            elif key == 27:  # Escape
                return False
            elif key == ord('\x03'):  # Ctrl+C
                return False
        except curses.error:
            pass
            
        return True
    
    def run_ui_loop(self):
        """Main UI loop - should be called from main thread"""
        while self.running and self.streamer.running:
            if not self.handle_input():
                self.streamer.running = False
                break
                
            self.update_display()
            time.sleep(0.02)  # Small delay to prevent excessive CPU usage
    
    def stop(self):
        """Stop the UI"""
        self.running = False

async def main(participant_name: str, enable_aec: bool = True):
    logger = logging.getLogger(__name__)
    logger.info("=== STARTING AUDIO STREAMER ===")
    
    # Get the running event loop
    loop = asyncio.get_running_loop()
    
    # Verify environment
    logger.info(f"LIVEKIT_URL: {LIVEKIT_URL}")
    logger.info(f"ROOM_NAME: {ROOM_NAME}")
    
    if not LIVEKIT_URL or not ROOM_NAME:
        logger.error("Missing LIVEKIT_URL or ROOM_NAME environment variables")
        return
    
    # Create audio streamer with loop reference
    streamer = AudioStreamer(enable_aec, loop=loop)
    
    # Create room
    room = rtc.Room(loop=loop)
    streamer.room = room
    
    # Audio processing task
    async def audio_processing_task():
        """Process audio frames from input queue and send to LiveKit"""
        frames_sent = 0
        logger.info("Audio processing task started")
        
        while streamer.running:
            try:
                # Get audio frame from input callback
                frame = await asyncio.wait_for(streamer.audio_input_queue.get(), timeout=1.0)
                await streamer.source.capture_frame(frame)
                frames_sent += 1
                
                if frames_sent <= 5:
                    logger.info(f"Sent frame {frames_sent} to LiveKit source")
                elif frames_sent % 100 == 0:
                    logger.info(f"Sent {frames_sent} frames total to LiveKit")
                    
            except asyncio.TimeoutError:
                logger.debug("No audio frames in queue (timeout)")
                continue
            except Exception as e:
                logger.error(f"Error in audio processing: {e}")
                break
        
        logger.info(f"Audio processing task ended. Total frames sent: {frames_sent}")
    
    # UI wrapper function for curses
    def run_with_ui(stdscr):
        """Run the application with curses UI"""
        # Create and initialize UI
        ui = CursesUI(streamer)
        streamer.ui = ui
        ui.init_screen(stdscr)
        
        # UI will handle keyboard input and display updates
        ui.run_ui_loop()
    
    # Function to handle received audio frames
    async def receive_audio_frames(stream: rtc.AudioStream, participant: rtc.RemoteParticipant):
        frames_received = 0
        logger.info("Audio receive task started")
        
        # Use participant info passed from event handler
        participant_id = participant.sid
        participant_name = participant.identity or f"User_{participant.sid[:8]}"
        
        logger.info(f"Receiving audio from participant: {participant_name} ({participant_id})")
        
        async for frame_event in stream:
            if not streamer.running:
                break
                
            frames_received += 1
            if frames_received <= 5:
                logger.info(f"Received audio frame {frames_received} from {participant_name}")
            elif frames_received % 100 == 0:
                logger.info(f"Received {frames_received} frames total from {participant_name}")
                
            # Calculate dB level for this participant
            frame_data = frame_event.frame.data
            if len(frame_data) > 0:
                # Convert to numpy array for dB calculation
                audio_samples = np.frombuffer(frame_data, dtype=np.int16)
                if len(audio_samples) > 0:
                    rms = np.sqrt(np.mean(audio_samples.astype(np.float32) ** 2))
                    max_int16 = np.iinfo(np.int16).max
                    participant_db = 20.0 * np.log10(rms / max_int16 + 1e-6)
                    
                    # Update participant info
                    with streamer.participants_lock:
                        streamer.participants[participant_id] = {
                            'name': participant_name,
                            'db_level': participant_db,
                            'last_update': time.time()
                        }
                
            # Add received audio to output buffer
            audio_data = frame_event.frame.data.tobytes()
            with streamer.output_lock:
                streamer.output_buffer.extend(audio_data)
        
        logger.info(f"Audio receive task ended for {participant_name}. Total frames received: {frames_received}")
        
        # Clean up participant when stream ends
        with streamer.participants_lock:
            if participant_id in streamer.participants:
                del streamer.participants[participant_id]

    # Event handlers
    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.RemoteTrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        logger.info("track subscribed: %s from participant %s (%s)", publication.sid, participant.sid, participant.identity)
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Starting audio stream for participant: {participant.identity}")
            audio_stream = rtc.AudioStream(track, sample_rate=SAMPLE_RATE, num_channels=NUM_CHANNELS)
            asyncio.ensure_future(receive_audio_frames(audio_stream, participant))

    @room.on("track_published")
    def on_track_published(
        publication: rtc.RemoteTrackPublication, participant: rtc.RemoteParticipant
    ):
        logger.info(
            "track published: %s from participant %s (%s)",
            publication.sid,
            participant.sid,
            participant.identity,
        )

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info("participant connected: %s %s", participant.sid, participant.identity)
        # Initialize participant in our tracking
        with streamer.participants_lock:
            streamer.participants[participant.sid] = {
                'name': participant.identity or f"User_{participant.sid[:8]}",
                'db_level': INPUT_DB_MIN,
                'last_update': time.time()
            }
        logger.info(f"Added participant to tracking: {participant.identity}")

    @room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        logger.info("participant disconnected: %s %s", participant.sid, participant.identity)
        # Remove participant from our tracking
        with streamer.participants_lock:
            if participant.sid in streamer.participants:
                del streamer.participants[participant.sid]
                logger.info(f"Removed participant from tracking: {participant.identity}")

    @room.on("connected")
    def on_connected():
        logger.info("Successfully connected to LiveKit room")

    @room.on("disconnected")
    def on_disconnected(reason):
        logger.info(f"Disconnected from LiveKit room: {reason}")

    try:
        # Start audio devices
        logger.info("Starting audio devices...")
        streamer.start_audio_devices()
        
        # Connect to LiveKit room
        logger.info("Connecting to LiveKit room...")
        token = generate_token(ROOM_NAME, participant_name, participant_name)
        logger.info(f"Generated token for participant: {participant_name}")
        
        await room.connect(LIVEKIT_URL, token)
        logger.info("connected to room %s", room.name)
        
        # Publish microphone track
        logger.info("Publishing microphone track...")
        track = rtc.LocalAudioTrack.create_audio_track("mic", streamer.source)
        options = rtc.TrackPublishOptions()
        options.source = rtc.TrackSource.SOURCE_MICROPHONE
        publication = await room.local_participant.publish_track(track, options)
        logger.info("published track %s", publication.sid)
        
        if enable_aec:
            logger.info("Echo cancellation is enabled")
        else:
            logger.info("Echo cancellation is disabled")
        
        # Start background tasks
        logger.info("Starting background tasks...")
        audio_task = asyncio.create_task(audio_processing_task())
        
        logger.info("=== Audio streaming started. Starting UI... ===")
        
        # Run the UI in a separate thread
        ui_thread = threading.Thread(target=lambda: curses.wrapper(run_with_ui), daemon=True)
        ui_thread.start()
        
        # Keep running until interrupted
        try:
            while streamer.running:
                await asyncio.sleep(0.1)
        except KeyboardInterrupt:
            logger.info("Stopping audio streaming...")
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    finally:
        # Cleanup
        logger.info("Starting cleanup...")
        streamer.running = False
        
        if 'audio_task' in locals():
            audio_task.cancel()
            try:
                await audio_task
            except asyncio.CancelledError:
                pass
        
        if streamer.ui:
            streamer.ui.stop()
        
        streamer.stop_audio_devices()
        await room.disconnect()
        
        logger.info("=== CLEANUP COMPLETE ===")

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="LiveKit bidirectional audio streaming with AEC")
    parser.add_argument(
        "--name", 
        "-n",
        type=str,
        default="audio-streamer",
        help="Participant name to use when connecting to the room (default: audio-streamer)"
    )
    parser.add_argument(
        "--disable-aec",
        action="store_true",
        help="Disable acoustic echo cancellation (AEC)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    args = parser.parse_args()
    
    # Set up logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("stream_audio.log"),
            # Only log to console in debug mode - otherwise interferes with meter
            *([logging.StreamHandler()] if args.debug else []),
        ],
    )
    
    # Also log to console with colors for easier debugging (only in debug mode)
    if args.debug:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(formatter)
    
    # Fix deprecation warning by using asyncio.run() instead of get_event_loop()
    async def cleanup():
        task = asyncio.current_task()
        tasks = [t for t in asyncio.all_tasks() if t is not task]
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)

    def signal_handler():
        asyncio.create_task(cleanup())

    # Use asyncio.run() to properly handle the event loop
    try:
        # For signal handling, we need to use the lower-level approach
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        main_task = asyncio.ensure_future(main(args.name, enable_aec=not args.disable_aec))
        for signal in [SIGINT, SIGTERM]:
            loop.add_signal_handler(signal, signal_handler)

        try:
            loop.run_until_complete(main_task)
        except KeyboardInterrupt:
            pass
        finally:
            loop.close()
    except KeyboardInterrupt:
        pass 