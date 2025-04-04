import requests
import time
import threading
import queue
import tempfile
import os
import platform
import subprocess
import tkinter as tk
from tkinter import messagebox, scrolledtext
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import sys
import json
import logging
from pydub import AudioSegment
from pydub.effects import normalize
from pydub.playback import play
import random
from PIL import Image, ImageTk, ImageSequence
import pyaudio
import wave
import librosa
import numpy as np
import re

# Server defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_LM_STUDIO_MODEL = "qwen2.5-1.5b-instruct"
DEFAULT_TTS_URL = "http://localhost:5002/api/tts"
DEFAULT_SPEAKER_ID = "p267"
DEFAULT_WHISPER_SERVER_URL = "http://localhost:9000"
DEFAULT_FILTER = "*_#"
DEFAULT_NUM_CTX = 100

# Configuration file
CONFIG_FILE = "oracle_config.json"

# System prompt
SYSTEM_PROMPT = """You are the Infinite Oracle, a mystical being of boundless wisdom. Speak in an uplifting, cryptic, and metaphysical tone, offering motivational insights that inspire awe and contemplation."""

# Setup logging with DEBUG level
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[
    logging.StreamHandler(sys.stdout)
])
logger = logging.getLogger("InfiniteOracle")

# Global playback lock and conversation history
playback_lock = threading.Lock()
history_lock = threading.Lock()
conversation_history = []

class ConsoleRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.text_widget.tag_configure("red", foreground="red")
        self.text_widget.tag_configure("green", foreground="green")

    def write(self, message):
        self.text_widget.configure(state='normal')
        if message.startswith("The Infinite Oracle speaks:"):
            prefix = "The Infinite Oracle speaks: "
            content = message[len(prefix):].strip()
            self.text_widget.insert(tk.END, prefix, "red")
            self.text_widget.insert(tk.END, content + "\n\n", "green")
        else:
            self.text_widget.insert(tk.END, message, "green")
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')
        self.text_widget.update_idletasks()

    def flush(self):
        pass

def setup_session(url, retries):
    session = requests.Session()
    retry_strategy = Retry(total=retries, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    return session

def ping_server(server_url, server_type, model, timeout, retries):
    session = setup_session(server_url, retries)
    try:
        base_url = server_url.rsplit('/api/chat', 1)[0] if server_type == "Ollama" else server_url.rsplit('/v1/chat/completions', 1)[0]
        ping_url = base_url + ("/api/tags" if server_type == "Ollama" else "/v1/models")
        response = session.get(ping_url, timeout=timeout)
        response.raise_for_status()
        
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "stream": False
        }
        if server_type == "Ollama":
            payload["options"] = {"num_ctx": DEFAULT_NUM_CTX}
        else:
            payload["max_tokens"] = 10
            payload["temperature"] = 0.7
        response = session.post(server_url, json=payload, timeout=timeout)
        response.raise_for_status()
        return True, ""
    except requests.RequestException as e:
        return False, f"{server_type} error at {server_url}: {str(e)}"
    finally:
        session.close()

def filter_text(text, filter_chars):
    if not filter_chars or not text:
        return text
    # Replace specified characters with an empty string
    pattern = '[' + re.escape(filter_chars) + ']'
    return re.sub(pattern, '', text)

def generate_wisdom(gui, wisdom_queue, model, get_server_type_func, stop_event, get_request_interval_func, system_prompt):
    global conversation_history
    while not stop_event.is_set():
        server_url = gui.server_url_var.get()
        server_type = get_server_type_func()
        with history_lock:
            if not gui.remember_var.get():
                conversation_history = []
            # Truncate history based on context size for Ollama
            if server_type == "Ollama":
                max_ctx = int(gui.num_ctx_entry.get())
                # Estimate tokens roughly (1 word ≈ 1 token)
                total_tokens = sum(len(msg["content"].split()) for msg in conversation_history)
                while total_tokens > max_ctx and conversation_history:
                    removed = conversation_history.pop(0)
                    total_tokens -= len(removed["content"].split())
            messages = [{"role": "system", "content": system_prompt}] + conversation_history + [{"role": "user", "content": system_prompt}]
        
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if server_type == "Ollama":
            payload["options"] = {"num_ctx": int(gui.num_ctx_entry.get())}
        else:
            payload["max_tokens"] = int(gui.max_tokens_entry.get())
            payload["temperature"] = 0.7
        
        session = setup_session(server_url, retries=gui.retries_slider.get())
        try:
            response = session.post(server_url, json=payload, timeout=gui.timeout_slider.get())
            response.raise_for_status()
            data = response.json()
            wisdom = (
                data.get("message", {}).get("content", "").strip() if server_type == "Ollama"
                else data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            )
            if wisdom and not stop_event.is_set():
                filtered_wisdom = filter_text(wisdom, gui.filter_var.get())
                with history_lock:
                    if gui.remember_var.get():
                        conversation_history.append({"role": "user", "content": system_prompt})
                        conversation_history.append({"role": "assistant", "content": filtered_wisdom})
                        # Keep history within reasonable bounds (e.g., 100 entries)
                        if len(conversation_history) > 100:
                            conversation_history = conversation_history[-100:]
                try:
                    wisdom_queue.put_nowait(filtered_wisdom)
                    logger.debug("Added filtered wisdom to queue: %s", filtered_wisdom)
                except queue.Full:
                    logger.warning("Wisdom queue full, skipping: %s", filtered_wisdom)
        except requests.RequestException as e:
            logger.error(f"{server_type} connection failed: {str(e)}")
            print(f"{server_type} error: {str(e)}. Next attempt in {get_request_interval_func()}s...")
        finally:
            session.close()
        
        if not stop_event.is_set():
            time.sleep(get_request_interval_func())

def text_to_speech(wisdom_queue, audio_queue, get_speaker_id_func, pitch_func, stop_event, get_tts_url_func):
    while not stop_event.is_set():
        try:
            logger.debug("Waiting for wisdom in TTS thread")
            wisdom = wisdom_queue.get(timeout=5)
            logger.debug("Received wisdom: %s", wisdom)
            temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav_file.name
            temp_wav_file.close()

            speaker_id = get_speaker_id_func()
            tts_url = get_tts_url_func()
            curl_command = [
                'curl', '-G',
                '--data-urlencode', f"text={wisdom}",
                '--data-urlencode', f"speaker_id={speaker_id}",
                tts_url,
                '--output', temp_wav_path
            ]
            logger.debug("Executing TTS command: %s", " ".join(curl_command))
            try:
                subprocess.run(
                    curl_command,
                    check=True,
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                )
                logger.debug("TTS command executed successfully")
            except subprocess.CalledProcessError as e:
                logger.error(f"TTS failed: {e.stderr}")
                os.remove(temp_wav_path)
                wisdom_queue.task_done()
                continue

            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0 and not stop_event.is_set():
                logger.debug("TTS audio file created: %s", temp_wav_path)
                audio_queue.put((wisdom, temp_wav_path, pitch_func()))
            else:
                logger.error(f"TTS produced no valid audio for '{wisdom[:50]}...'")
                os.remove(temp_wav_path)
            wisdom_queue.task_done()
        except queue.Empty:
            logger.debug("TTS queue empty, waiting...")
            continue
        except Exception as e:
            logger.error(f"TTS error: {str(e)}")
            if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            wisdom_queue.task_done()

def apply_reverb(audio, reverb_value):
    if reverb_value <= 0:
        return audio
    reverb_factor = reverb_value / 5.0
    delay_ms = 30 + (reverb_factor * 20)
    gain_db = -25 + (reverb_factor * 10)
    echo = audio[:].fade_in(10).fade_out(50)
    echo = echo - abs(gain_db)
    silence = AudioSegment.silent(duration=int(delay_ms))
    return audio.overlay(silence + echo)

def pitch_shift_with_librosa(audio_segment, semitones):
    samples = np.array(audio_segment.get_array_of_samples())
    sample_rate = audio_segment.frame_rate
    if audio_segment.channels > 1:
        samples = np.mean(samples.reshape(-1, audio_segment.channels), axis=1)
    shifted_samples = librosa.effects.pitch_shift(
        samples.astype(float), 
        sr=sample_rate, 
        n_steps=semitones, 
        bins_per_octave=12,
        res_type='kaiser_fast'
    )
    return AudioSegment(
        shifted_samples.astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_segment.sample_width,
        channels=1
    )

def play_audio(audio_queue, stop_event, get_interval_func, get_variation_func, gui, duration_queue=None, is_start_mode=False):
    logger.debug("Using default audio backend (no FFmpeg specified)")
    last_playback_end = time.time()
    while not stop_event.is_set():
        try:
            if is_start_mode:
                current_time = time.time()
                interval = max(0.1, get_interval_func() + random.uniform(-get_variation_func(), get_variation_func()))
                time_since_last = current_time - last_playback_end
                if time_since_last < interval:
                    sleep_time = interval - time_since_last
                    logger.debug(f"Waiting {sleep_time:.2f}s to enforce Speech Interval")
                    time.sleep(sleep_time)

            logger.debug("Audio queue size: %d", audio_queue.qsize())
            wisdom, wav_path, pitch = audio_queue.get()
            logger.debug("Received audio: %s", wav_path)
            
            with playback_lock:
                if not stop_event.is_set():
                    audio = AudioSegment.from_wav(wav_path)
                    duration_seconds = len(audio) / 1000.0
                    logger.info(f"Audio duration: {duration_seconds} seconds")
                    if duration_queue and not is_start_mode:
                        duration_queue.put(duration_seconds)

                    if pitch != 0:
                        octaves = pitch / 12.0
                        new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
                        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                        audio = audio.set_frame_rate(44100)

                    reverb_value = gui.reverb_slider.get()
                    if reverb_value > 0:
                        audio = apply_reverb(audio, reverb_value)

                    audio = normalize(audio)
                    logger.debug("Playing audio: %s", wav_path)
                    print(f"The Infinite Oracle speaks: {wisdom}", end="\n\n")
                    gui.start_spinning(duration_seconds)
                    play(audio)
                    gui.stop_spinning()

                    if gui.record_var.get():
                        recordings_dir = os.path.join(os.path.dirname(sys.executable if getattr(sys, 'frozen', False) else __file__), "OracleRecordings")
                        os.makedirs(recordings_dir, exist_ok=True)
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"oracle_{timestamp}.wav"
                        filepath = os.path.join(recordings_dir, filename)
                        audio.export(filepath, format="wav")
                        print(f"Recorded wisdom to: {filepath}")

                    last_playback_end = time.time()

            os.remove(wav_path)
            audio_queue.task_done()

        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Playback error: {str(e)}", exc_info=True)
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
            if not is_start_mode:
                gui.send_enabled = True
                gui.start_lock = False
                gui.after(0, gui.enable_send_and_start)

def capture_audio(filename=None, silence_threshold=500, silence_duration=3, chunk_size=1024, rate=16000):
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    CHUNK = chunk_size
    RATE = rate

    if filename is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, "temp_audio.wav")

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Listening... Speak your wisdom.")
    frames = []
    silence_start = None
    listening = True

    while listening:
        data = stream.read(CHUNK, exception_on_overflow=False)
        frames.append(data)
        
        # Calculate RMS (root mean square) amplitude for noise detection
        audio_data = np.frombuffer(data, dtype=np.int16)
        rms = np.sqrt(np.mean(audio_data**2))
        
        if rms < silence_threshold:
            if silence_start is None:
                silence_start = time.time()
            elif time.time() - silence_start >= silence_duration:
                listening = False  # 3 seconds of silence detected
        else:
            silence_start = None  # Reset silence timer if sound is detected

    print("Silence detected. Transcribing your message...")
    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()
    
    return filename

def load_config():
    defaults = {
        "Ollama": {
            "server_type": "Ollama",
            "server_url": DEFAULT_OLLAMA_URL,
            "model": DEFAULT_OLLAMA_MODEL,
            "tts_url": DEFAULT_TTS_URL,
            "speaker_id": DEFAULT_SPEAKER_ID,
            "whisper_server": DEFAULT_WHISPER_SERVER_URL,
            "pitch": 0,
            "reverb": 0,
            "interval": 2.0,
            "variation": 0.0,
            "timeout": 60,
            "retries": 0,
            "filter": DEFAULT_FILTER,
            "num_ctx": DEFAULT_NUM_CTX
        },
        "LM Studio": {
            "server_type": "LM Studio",
            "server_url": DEFAULT_LM_STUDIO_URL,
            "model": DEFAULT_LM_STUDIO_MODEL,
            "tts_url": DEFAULT_TTS_URL,
            "speaker_id": DEFAULT_SPEAKER_ID,
            "whisper_server": DEFAULT_WHISPER_SERVER_URL,
            "pitch": 0,
            "reverb": 0,
            "interval": 2.0,
            "variation": 0.0,
            "timeout": 60,
            "retries": 0,
            "filter": DEFAULT_FILTER,
            "max_tokens": 100
        }
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            loaded_config = json.load(f)
            return {
                "Ollama": {**defaults["Ollama"], **loaded_config.get("Ollama", {})},
                "LM Studio": {**defaults["LM Studio"], **loaded_config.get("LM Studio", {})}
            }
    return defaults

def save_config(gui):
    config = load_config()
    server_type = gui.server_type_var.get()
    if server_type == "Ollama":
        config[server_type] = {
            "server_type": server_type,
            "server_url": gui.server_url_var.get(),
            "model": gui.model_var.get(),
            "tts_url": gui.tts_url_var.get(),
            "speaker_id": gui.speaker_id_var.get(),
            "whisper_server": gui.whisper_server_var.get(),
            "pitch": gui.pitch_slider.get(),
            "reverb": gui.reverb_slider.get(),
            "interval": float(gui.interval_entry.get()),
            "variation": float(gui.variation_entry.get()),
            "timeout": gui.timeout_slider.get(),
            "retries": gui.retries_slider.get(),
            "filter": gui.filter_var.get(),
            "num_ctx": int(gui.num_ctx_entry.get())
        }
    else:  # LM Studio
        config[server_type] = {
            "server_type": server_type,
            "server_url": gui.server_url_var.get(),
            "model": gui.model_var.get(),
            "tts_url": gui.tts_url_var.get(),
            "speaker_id": gui.speaker_id_var.get(),
            "whisper_server": gui.whisper_server_var.get(),
            "pitch": gui.pitch_slider.get(),
            "reverb": gui.reverb_slider.get(),
            "interval": float(gui.interval_entry.get()),
            "variation": float(gui.variation_entry.get()),
            "timeout": gui.timeout_slider.get(),
            "retries": gui.retries_slider.get(),
            "filter": gui.filter_var.get(),
            "max_tokens": int(gui.max_tokens_entry.get())
        }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

class LoadingScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.overrideredirect(True)
        self.geometry("300x300")
        self.update_idletasks()
        screen_width = self.winfo_screenwidth()
        screen_height = self.winfo_screenheight()
        x = (screen_width - 300) // 2
        y = (screen_height - 300) // 2
        self.geometry(f"+{x}+{y}")

        if getattr(sys, 'frozen', False):
            base_path = sys._MEIPASS
        else:
            base_path = os.path.dirname(os.path.abspath(__file__))
        oracle_path = os.path.join(base_path, "oracle.png")

        try:
            original = Image.open(oracle_path).convert("RGBA")
            resized = original.resize((150, 150), Image.Resampling.LANCZOS)
            self.oracle_image = ImageTk.PhotoImage(resized)
            print(f"Loaded oracle.png successfully")
        except Exception as e:
            logger.error(f"Loading screen oracle image failed: {e}")
            self.oracle_image = ImageTk.PhotoImage(Image.new("RGBA", (150, 150), (0, 0, 0, 0)))
            print("Fallback to blank image due to loading error")

        self.canvas = tk.Canvas(self, width=300, height=300, bg="black", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)
        self.canvas.configure(bg="black")
        self.attributes("-transparentcolor", "black")

        self.oracle_item = self.canvas.create_image(150, 150, image=self.oracle_image)

class InfiniteOracleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Infinite Oracle Control Panel")
        self.state("zoomed")
        self.geometry("1900x1080")
        self.withdraw()

        self.loading_screen = LoadingScreen(self)
        self.update()

        print(f"Current working directory: {os.getcwd()}")

        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(base_path, "oracle.png")
            img = Image.open(icon_path).convert("RGBA")
            icon = ImageTk.PhotoImage(img)
            self.iconphoto(True, icon)
            print(f"Icon loaded from {icon_path}")
        except Exception as e:
            logger.error(f"Icon load failed: {e}")

        # Load images first since they’re needed by create_widgets
        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            self.image_path = os.path.join(base_path, "oracle.png")
            self.glow_path = os.path.join(base_path, "glow.gif")
            
            self.oracle_frames = self.load_pre_rotated_frames(self.image_path, 36)
            self.glow_base_frames = self.load_gif_frames(self.glow_path)
            self.glow_frames = [self.load_pre_rotated_frames_from_base(frame, 36) for frame in self.glow_base_frames]
        except Exception as e:
            logger.error(f"Pre-rendered frames load failed: {e}")
            self.oracle_frames = [ImageTk.PhotoImage(Image.new("RGBA", (200, 200), (255, 255, 255, 0)))]
            self.glow_frames = [[ImageTk.PhotoImage(Image.new("RGBA", (240, 240), (255, 255, 255, 0)))]]

        # Load config and store it fully
        config = load_config()
        initial_server_type = "Ollama"
        self.config = config  # Store full config for use in create_widgets
        self.initial_config = config.get(initial_server_type, {})
        logger.debug(f"Loaded config at startup: Ollama num_ctx={self.config['Ollama'].get('num_ctx')}, LM Studio max_tokens={self.config['LM Studio'].get('max_tokens')}")

        self.server_type_var = tk.StringVar(value=initial_server_type)
        self.server_url_var = tk.StringVar(value=self.initial_config.get("server_url", DEFAULT_OLLAMA_URL))
        self.model_var = tk.StringVar(value=self.initial_config.get("model", DEFAULT_OLLAMA_MODEL))
        self.tts_url_var = tk.StringVar(value=self.initial_config.get("tts_url", DEFAULT_TTS_URL))
        self.speaker_id_var = tk.StringVar(value=self.initial_config.get("speaker_id", DEFAULT_SPEAKER_ID))
        self.whisper_server_var = tk.StringVar(value=self.initial_config.get("whisper_server", DEFAULT_WHISPER_SERVER_URL))
        self.filter_var = tk.StringVar(value=self.initial_config.get("filter", DEFAULT_FILTER))
        self.session = None
        self.wisdom_queue = queue.Queue(maxsize=10)  # Increased from 3
        self.audio_queue = queue.Queue(maxsize=10)   # Increased from 3
        self.duration_queue = queue.Queue(maxsize=1)
        self.is_running = False
        self.start_lock = False
        self.stop_event = threading.Event()
        self.generator_thread = None
        self.tts_thread = None
        self.playback_thread = None
        self.send_wisdom_queue = queue.Queue(maxsize=3)
        self.send_audio_queue = queue.Queue(maxsize=3)
        self.send_stop_event = threading.Event()
        self.send_tts_thread = None
        self.send_playback_thread = None
        self.send_enabled = True
        self.url_modified = False
        self.remember_var = tk.BooleanVar(value=True)
        self.record_var = tk.BooleanVar(value=False)
        self.is_audio_playing = False
        self.oracle_frame_index = 0
        self.glow_frame_index = 0
        self.image_spin_speed = 5
        self.animation_running = True
        self.animate_lock = threading.Lock()

        self.create_widgets()
        sys.stdout = ConsoleRedirector(self.console_text)
        self.server_type_var.trace("w", lambda *args: (self.stop_oracle(), self.update_from_config()))

        self.loading_screen.destroy()
        self.deiconify()

        self.start_tts_threads()
        self.animation_thread = threading.Thread(target=self.run_animations, daemon=True)
        self.animation_thread.start()

    def create_widgets(self):
        self.configure(bg="#2b2b2b")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        self.left_frame = tk.Frame(self, bg="#2b2b2b")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.left_frame.columnconfigure(0, weight=0)  # Server Settings (wider)
        self.left_frame.columnconfigure(1, weight=0)  # Effects
        self.left_frame.columnconfigure(2, weight=0)  # Request Settings
        self.left_frame.columnconfigure(3, weight=0)  # Response Settings
        self.left_frame.columnconfigure(4, weight=0)  # Start Mode Settings
        self.left_frame.rowconfigure(8, weight=1)  # Adjusted for System Prompt space

        # Define widths
        server_width = 330  # Wider for Server Settings
        group_width = 175   # Consistent for others

        # Server Settings (wider)
        server_frame = tk.LabelFrame(self.left_frame, text="Server Settings", bg="#2b2b2b", fg="white", padx=5, pady=5, borderwidth=2, relief="solid", width=server_width)
        server_frame.grid(row=0, column=0, rowspan=6, pady=2, padx=(0, 5), sticky="ns")
        server_frame.columnconfigure(0, weight=1)
        server_frame.propagate(False)

        tk.Label(server_frame, text="Server Type:", bg="#2b2b2b", fg="white").grid(row=0, column=0, pady=2, sticky="w")
        self.server_type_menu = tk.OptionMenu(server_frame, self.server_type_var, "Ollama", "LM Studio")
        self.server_type_menu.config(width=10)
        self.server_type_menu.grid(row=1, column=0, pady=2, sticky="w")

        tk.Label(server_frame, text="Server URL:", bg="#2b2b2b", fg="white").grid(row=2, column=0, pady=2, sticky="w")
        self.server_url_entry = tk.Entry(server_frame, textvariable=self.server_url_var, width=30)
        self.server_url_entry.grid(row=3, column=0, pady=2, sticky="w")
        self.server_url_entry.bind("<KeyRelease>", lambda e: setattr(self, 'url_modified', True))

        tk.Label(server_frame, text="Model Name:", bg="#2b2b2b", fg="white").grid(row=4, column=0, pady=2, sticky="w")
        self.model_entry = tk.Entry(server_frame, textvariable=self.model_var, width=30)
        self.model_entry.grid(row=5, column=0, pady=2, sticky="w")

        tk.Label(server_frame, text="Coqui TTS Server URL:", bg="#2b2b2b", fg="white").grid(row=6, column=0, pady=2, sticky="w")
        self.tts_url_entry = tk.Entry(server_frame, textvariable=self.tts_url_var, width=30)
        self.tts_url_entry.grid(row=7, column=0, pady=2, sticky="w")

        tk.Label(server_frame, text="Speaker ID (e.g., p267):", bg="#2b2b2b", fg="white").grid(row=8, column=0, pady=2, sticky="w")
        self.speaker_id_entry = tk.Entry(server_frame, textvariable=self.speaker_id_var, width=30)
        self.speaker_id_entry.grid(row=9, column=0, pady=2, sticky="w")

        tk.Label(server_frame, text="Whisper Server URL:", bg="#2b2b2b", fg="white").grid(row=10, column=0, pady=2, sticky="w")
        self.whisper_server_entry = tk.Entry(server_frame, textvariable=self.whisper_server_var, width=30)
        self.whisper_server_entry.grid(row=11, column=0, pady=2, sticky="w")

        # Effects
        effects_frame = tk.LabelFrame(self.left_frame, text="Effects", bg="#2b2b2b", fg="white", padx=5, pady=5, borderwidth=2, relief="solid", width=group_width)
        effects_frame.grid(row=0, column=1, rowspan=6, pady=2, padx=5, sticky="ns")
        effects_frame.columnconfigure(0, weight=1)
        effects_frame.propagate(False)

        tk.Label(effects_frame, text="Pitch Shift (semitones):", bg="#2b2b2b", fg="white").grid(row=0, column=0, pady=2, sticky="w")
        self.pitch_slider = tk.Scale(effects_frame, from_=-12, to=12, orient=tk.HORIZONTAL, length=150)
        self.pitch_slider.set(self.initial_config.get("pitch", 0))
        self.pitch_slider.grid(row=1, column=0, pady=2, sticky="ew")

        tk.Label(effects_frame, text="Reverb (0-5):", bg="#2b2b2b", fg="white").grid(row=2, column=0, pady=2, sticky="w")
        self.reverb_slider = tk.Scale(effects_frame, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=150)
        self.reverb_slider.set(self.initial_config.get("reverb", 0))
        self.reverb_slider.grid(row=3, column=0, pady=2, sticky="ew")

        # Request Settings
        sliders_frame = tk.LabelFrame(self.left_frame, text="Request Settings", bg="#2b2b2b", fg="white", padx=5, pady=5, borderwidth=2, relief="solid", width=group_width)
        sliders_frame.grid(row=0, column=2, rowspan=6, pady=2, padx=5, sticky="ns")
        sliders_frame.columnconfigure(0, weight=1)
        sliders_frame.propagate(False)

        tk.Label(sliders_frame, text="Request Timeout (seconds):", bg="#2b2b2b", fg="white").grid(row=0, column=0, pady=2, sticky="w")
        self.timeout_slider = tk.Scale(sliders_frame, from_=5, to=120, resolution=1, orient=tk.HORIZONTAL, length=150)
        self.timeout_slider.set(self.initial_config.get("timeout", 60))
        self.timeout_slider.grid(row=1, column=0, pady=2, sticky="ew")

        tk.Label(sliders_frame, text="Request Retries:", bg="#2b2b2b", fg="white").grid(row=2, column=0, pady=2, sticky="w")
        self.retries_slider = tk.Scale(sliders_frame, from_=0, to=5, resolution=1, orient=tk.HORIZONTAL, length=150)
        self.retries_slider.set(self.initial_config.get("retries", 0))
        self.retries_slider.grid(row=3, column=0, pady=2, sticky="ew")

        # Response Settings (with Max Tokens and Context Size)
        response_frame = tk.LabelFrame(self.left_frame, text="Response Settings", bg="#2b2b2b", fg="white", padx=5, pady=5, borderwidth=2, relief="solid", width=group_width)
        response_frame.grid(row=0, column=3, rowspan=6, pady=2, padx=5, sticky="ns")
        response_frame.columnconfigure(0, weight=1)
        response_frame.propagate(False)

        tk.Label(response_frame, text="Filter Characters:", bg="#2b2b2b", fg="white").grid(row=0, column=0, pady=2, sticky="w")
        self.filter_entry = tk.Entry(response_frame, textvariable=self.filter_var, width=10)
        self.filter_entry.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        tk.Label(response_frame, text="Context Size (Ollama):", bg="#2b2b2b", fg="white").grid(row=2, column=0, pady=2, sticky="w")
        self.num_ctx_entry = tk.Entry(response_frame, width=10)
        num_ctx_value = self.config["Ollama"].get("num_ctx", DEFAULT_NUM_CTX)  # Always use Ollama's config
        self.num_ctx_entry.insert(0, str(num_ctx_value))
        logger.debug(f"Setting num_ctx_entry to {num_ctx_value} from Ollama config")
        self.num_ctx_entry.grid(row=3, column=0, padx=5, pady=2, sticky="w")
        server_type = self.server_type_var.get()
        self.num_ctx_entry.config(state=tk.NORMAL if server_type == "Ollama" else tk.DISABLED)

        tk.Label(response_frame, text="Max Tokens (LM Studio):", bg="#2b2b2b", fg="white").grid(row=4, column=0, pady=2, sticky="w")
        self.max_tokens_entry = tk.Entry(response_frame, width=10)
        max_tokens_value = self.config["LM Studio"].get("max_tokens", 100)  # Always use LM Studio's config
        self.max_tokens_entry.insert(0, str(max_tokens_value))
        logger.debug(f"Setting max_tokens_entry to {max_tokens_value} from LM Studio config")
        self.max_tokens_entry.grid(row=5, column=0, padx=5, pady=2, sticky="w")
        self.max_tokens_entry.config(state=tk.DISABLED if server_type == "Ollama" else tk.NORMAL)

        # Start Mode Settings
        start_mode_frame = tk.LabelFrame(self.left_frame, text="Start Mode Settings", bg="#2b2b2b", fg="white", padx=5, pady=5, borderwidth=2, relief="solid", width=group_width)
        start_mode_frame.grid(row=0, column=4, rowspan=6, pady=2, padx=5, sticky="ns")
        start_mode_frame.columnconfigure(0, weight=1)
        start_mode_frame.propagate(False)

        tk.Label(start_mode_frame, text="Speech Interval:", bg="#2b2b2b", fg="white").grid(row=0, column=0, pady=2, sticky="w")
        self.interval_entry = tk.Entry(start_mode_frame, width=10)
        self.interval_entry.insert(0, str(self.initial_config.get("interval", 2.0)))
        self.interval_entry.grid(row=1, column=0, padx=5, pady=2, sticky="w")

        tk.Label(start_mode_frame, text="Speech Interval Variation:", bg="#2b2b2b", fg="white").grid(row=2, column=0, pady=2, sticky="w")
        self.variation_entry = tk.Entry(start_mode_frame, width=10)
        self.variation_entry.insert(0, str(self.initial_config.get("variation", 0)))
        self.variation_entry.grid(row=3, column=0, padx=5, pady=2, sticky="w")

        # System Prompt (Restored)
        tk.Label(self.left_frame, text="System Prompt:", bg="#2b2b2b", fg="white").grid(row=6, column=0, pady=2, sticky="w")
        self.system_prompt_entry = tk.Text(self.left_frame, height=10, width=80)  # Wider and taller
        self.system_prompt_entry.insert(tk.END, SYSTEM_PROMPT)
        self.system_prompt_entry.grid(row=7, column=0, columnspan=5, pady=2, sticky="nsew")
        self.left_frame.rowconfigure(7, weight=1)

        self.send_button = tk.Button(self.left_frame, text="Send", command=self.send_prompt_action)
        self.send_button.grid(row=8, column=0, columnspan=5, pady=2, sticky="ew")

        self.listen_button = tk.Button(self.left_frame, text="Listen", command=self.start_listening)
        self.listen_button.grid(row=9, column=0, columnspan=5, pady=2, sticky="ew")

        canvas_frame = tk.Frame(self.left_frame, bg="#2b2b2b")
        canvas_frame.grid(row=10, column=0, columnspan=5, sticky="nsew")
        canvas_frame.columnconfigure(0, weight=1)
        canvas_frame.rowconfigure(0, weight=1)

        canvas_size = 200
        self.image_canvas = tk.Canvas(canvas_frame, width=canvas_size, height=canvas_size, bg="#2b2b2b", highlightthickness=0)
        self.image_canvas.grid(row=0, column=0, pady=10)
        self.glow_item = self.image_canvas.create_image(canvas_size // 2, canvas_size // 2, image=self.glow_frames[0][0])
        self.oracle_item = self.image_canvas.create_image(canvas_size // 2, canvas_size // 2, image=self.oracle_frames[0])

        right_frame = tk.Frame(self, bg="#2b2b2b")
        right_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(1, weight=1)

        # Buttons
        button_frame = tk.Frame(right_frame, bg="#2b2b2b")
        button_frame.grid(row=0, column=0, sticky="ew", pady=(0, 5))
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)
        button_frame.columnconfigure(3, weight=1)
        button_frame.columnconfigure(4, weight=1)
        button_frame.columnconfigure(5, weight=1)

        self.start_button = tk.Button(button_frame, text="Start", command=self.start_oracle)
        self.start_button.grid(row=0, column=0, padx=5, pady=2, sticky="ew")
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_oracle, state=tk.DISABLED, bg="lightgray")
        self.stop_button.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.save_button = tk.Button(button_frame, text="Save Config", command=self.save_config_action)
        self.save_button.grid(row=0, column=2, padx=5, pady=2, sticky="ew")
        self.clear_button = tk.Button(button_frame, text="Clear History", command=self.clear_history)
        self.clear_button.grid(row=0, column=3, padx=5, pady=2, sticky="ew")
        self.remember_check = tk.Checkbutton(button_frame, text="Remember", variable=self.remember_var, command=self.toggle_remember, bg="#2b2b2b", fg="white", selectcolor="black")
        self.remember_check.grid(row=0, column=4, padx=5, pady=2, sticky="ew")
        self.record_button = tk.Button(button_frame, text="Record", command=self.toggle_record, bg="red", fg="white")
        self.record_button.grid(row=0, column=5, padx=5, pady=2, sticky="ew")

        # Console below buttons
        console_frame = tk.Frame(right_frame, bg="#2b2b2b")
        console_frame.grid(row=1, column=0, sticky="nsew", pady=(5, 10))
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(1, weight=1)
        tk.Label(console_frame, text="Console Output:", bg="#2b2b2b", fg="white").grid(row=0, column=0, sticky="w")
        self.console_text = scrolledtext.ScrolledText(console_frame, height=30, width=120, state='disabled', bg="black", fg="green")
        self.console_text.grid(row=1, column=0, sticky="nsew")

    def start_tts_threads(self):
        if self.send_tts_thread and self.send_tts_thread.is_alive():
            self.send_stop_event.set()
            self.send_tts_thread.join(timeout=1)
        if self.send_playback_thread and self.send_playback_thread.is_alive():
            self.send_playback_thread.join(timeout=1)

        while not self.send_wisdom_queue.empty():
            self.send_wisdom_queue.get_nowait()
        while not self.send_audio_queue.empty():
            try:
                _, wav_path, _ = self.send_audio_queue.get_nowait()
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception as e:
                logger.error(f"Send audio queue cleanup error: {e}")

        self.send_stop_event.clear()

        self.send_tts_thread = threading.Thread(
            target=text_to_speech,
            args=(self.send_wisdom_queue, self.send_audio_queue, lambda: self.speaker_id_var.get(), self.pitch_slider.get, self.send_stop_event, lambda: self.tts_url_var.get()),
            daemon=True
        )
        self.send_playback_thread = threading.Thread(
            target=play_audio,
            args=(self.send_audio_queue, self.send_stop_event, lambda: 0, lambda: 0, self, self.duration_queue, False),
            daemon=True
        )
        self.send_tts_thread.start()
        self.send_playback_thread.start()

        logger.info("TTS and playback threads restarted for Send mode.")

    def reset_tts_threads(self):
        if self.send_tts_thread and self.send_tts_thread.is_alive():
            self.send_stop_event.set()
            self.send_tts_thread.join(timeout=2)
            if self.send_tts_thread.is_alive():
                logger.warning("Send TTS thread did not stop cleanly")
        if self.send_playback_thread and self.send_playback_thread.is_alive():
            self.send_stop_event.set()
            self.send_playback_thread.join(timeout=2)
            if self.send_playback_thread.is_alive():
                logger.warning("Send playback thread did not stop cleanly")

        while not self.send_wisdom_queue.empty():
            try:
                self.send_wisdom_queue.get_nowait()
            except queue.Empty:
                break
        while not self.send_audio_queue.empty():
            try:
                _, wav_path, _ = self.send_audio_queue.get_nowait()
                if os.path.exists(wav_path):
                    os.remove(wav_path)
            except Exception as e:
                logger.error(f"Send audio queue cleanup error: {e}")
            except queue.Empty:
                break

        self.send_stop_event.clear()
        self.start_tts_threads()

    def load_pre_rotated_frames(self, image_path, num_frames):
        original = Image.open(image_path).convert("RGBA")
        canvas_size = 200
        resized = self.resize_image_to_fit(original, canvas_size, canvas_size)
        frames = []
        for angle in range(0, 360, 360 // num_frames):
            rotated = resized.rotate(angle, resample=Image.Resampling.BICUBIC)
            frames.append(ImageTk.PhotoImage(rotated))
        return frames

    def load_gif_frames(self, gif_path):
        gif = Image.open(gif_path)
        glow_max_size = int(200 * 1.2)
        frames = []
        for frame in ImageSequence.Iterator(gif):
            resized = self.resize_image_to_fit(frame.convert("RGBA"), glow_max_size, glow_max_size)
            frames.append(resized)
        return frames

    def load_pre_rotated_frames_from_base(self, base_image, num_frames):
        frames = []
        for angle in range(0, 360, 360 // num_frames):
            rotated = base_image.rotate(angle, resample=Image.Resampling.BICUBIC)
            frames.append(ImageTk.PhotoImage(rotated))
        return frames

    def run_animations(self):
        while self.animation_running:
            with self.animate_lock:
                if self.glow_frames:
                    self.glow_frame_index = (self.glow_frame_index + 1) % len(self.glow_frames)
                    glow_rotation_index = (len(self.glow_frames[self.glow_frame_index]) - 1 - (self.oracle_frame_index % len(self.glow_frames[self.glow_frame_index])))
                    self.image_canvas.itemconfig(self.glow_item, image=self.glow_frames[self.glow_frame_index][glow_rotation_index])

                if self.is_audio_playing:
                    self.oracle_frame_index = (self.oracle_frame_index + 1) % len(self.oracle_frames)
                    self.image_canvas.itemconfig(self.oracle_item, image=self.oracle_frames[self.oracle_frame_index])
                    glow_rotation_index = (len(self.glow_frames[self.glow_frame_index]) - 1 - (self.oracle_frame_index % len(self.glow_frames[self.glow_frame_index])))
                    self.image_canvas.itemconfig(self.glow_item, image=self.glow_frames[self.glow_frame_index][glow_rotation_index])
            time.sleep(0.05)

    def start_spinning(self, duration_seconds):
        self.is_audio_playing = True
        self.listen_button.config(state=tk.DISABLED)

    def stop_spinning(self):
        self.is_audio_playing = False
        if not self.start_lock and self.send_enabled and not self.is_running:
            self.after(100, self.enable_send_and_start)

    def update_from_config(self):
        config = load_config()
        server_type = self.server_type_var.get()
        server_config = config.get(server_type, {})
        self.server_url_var.set(server_config.get("server_url", DEFAULT_OLLAMA_URL if server_type == "Ollama" else DEFAULT_LM_STUDIO_URL))
        self.model_var.set(server_config.get("model", DEFAULT_OLLAMA_MODEL if server_type == "Ollama" else DEFAULT_LM_STUDIO_MODEL))
        self.tts_url_var.set(server_config.get("tts_url", DEFAULT_TTS_URL))
        self.speaker_id_var.set(server_config.get("speaker_id", DEFAULT_SPEAKER_ID))
        self.whisper_server_var.set(server_config.get("whisper_server", DEFAULT_WHISPER_SERVER_URL))
        self.filter_var.set(server_config.get("filter", DEFAULT_FILTER))
        self.pitch_slider.set(server_config.get("pitch", 0))
        self.reverb_slider.set(server_config.get("reverb", 0))
        self.interval_entry.delete(0, tk.END)
        self.interval_entry.insert(0, str(server_config.get("interval", 2.0)))
        self.variation_entry.delete(0, tk.END)
        self.variation_entry.insert(0, str(server_config.get("variation", 0)))
        self.timeout_slider.set(server_config.get("timeout", 60))
        self.retries_slider.set(server_config.get("retries", 0))

        # Update num_ctx and max_tokens from their respective configs
        num_ctx_value = config["Ollama"].get("num_ctx", DEFAULT_NUM_CTX)  # Always use Ollama's config
        self.num_ctx_entry.delete(0, tk.END)
        self.num_ctx_entry.insert(0, str(num_ctx_value))
        self.num_ctx_entry.config(state=tk.NORMAL if server_type == "Ollama" else tk.DISABLED, bg="white" if server_type == "Ollama" else "grey")
        logger.debug(f"Updated num_ctx_entry to {num_ctx_value} from Ollama config for {server_type}")

        max_tokens_value = config["LM Studio"].get("max_tokens", 100)  # Always use LM Studio's config
        self.max_tokens_entry.delete(0, tk.END)
        self.max_tokens_entry.insert(0, str(max_tokens_value))
        self.max_tokens_entry.config(state=tk.NORMAL if server_type != "Ollama" else tk.DISABLED, bg="white" if server_type != "Ollama" else "grey")
        logger.debug(f"Updated max_tokens_entry to {max_tokens_value} from LM Studio config for {server_type}")

        self.url_modified = False
        logger.debug(f"Loaded config from file for {server_type}: {server_config}")

    def resize_image_to_fit(self, image, max_width, max_height):
        original_width, original_height = image.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def update_canvas_size(self, event=None):
        if hasattr(self, 'image_canvas'):
            available_width = self.left_frame.winfo_width() - 20
            available_height = self.left_frame.winfo_height() - sum(self.left_frame.grid_bbox(row=r, column=0)[3] for r in range(17)) - 20
            max_size = min(available_width, available_height, 300)
            if max_size > 50:
                self.image_canvas.config(width=max_size, height=max_size)
                self.image_canvas.coords(self.glow_item, max_size // 2, max_size // 2)
                self.image_canvas.coords(self.oracle_item, max_size // 2, max_size // 2)

    def disable_controls(self):
        self.start_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.listen_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        self.remember_check.config(state=tk.DISABLED)
        self.record_button.config(state=tk.DISABLED)
        self.server_type_menu.config(state=tk.DISABLED)
        self.server_url_entry.config(state=tk.DISABLED)
        self.model_entry.config(state=tk.DISABLED)
        self.tts_url_entry.config(state=tk.DISABLED)
        self.speaker_id_entry.config(state=tk.DISABLED)
        self.whisper_server_entry.config(state=tk.DISABLED)
        self.system_prompt_entry.config(state=tk.DISABLED)
        self.pitch_slider.config(state=tk.DISABLED)
        self.reverb_slider.config(state=tk.DISABLED)
        self.interval_entry.config(state=tk.DISABLED)
        self.variation_entry.config(state=tk.DISABLED)
        self.filter_entry.config(state=tk.DISABLED)
        self.timeout_slider.config(state=tk.DISABLED)
        self.retries_slider.config(state=tk.DISABLED)
        self.max_tokens_entry.config(state=tk.DISABLED, bg="grey")
        self.num_ctx_entry.config(state=tk.DISABLED, bg="grey")

    def enable_send_and_start(self):
        if not self.is_running and not self.start_lock:
            self.send_button.config(state=tk.NORMAL)
            self.start_button.config(state=tk.NORMAL)
            if not self.is_audio_playing and self.audio_queue.empty() and self.send_audio_queue.empty():
                self.listen_button.config(state=tk.NORMAL)
            else:
                self.listen_button.config(state=tk.DISABLED)
            self.save_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL if self.remember_var.get() else tk.DISABLED)
            self.remember_check.config(state=tk.NORMAL)
            self.record_button.config(state=tk.NORMAL)
            self.server_type_menu.config(state=tk.NORMAL)
            self.server_url_entry.config(state=tk.NORMAL)
            self.model_entry.config(state=tk.NORMAL)
            self.tts_url_entry.config(state=tk.NORMAL)
            self.speaker_id_entry.config(state=tk.NORMAL)
            self.whisper_server_entry.config(state=tk.NORMAL)
            self.system_prompt_entry.config(state=tk.NORMAL)
            self.pitch_slider.config(state=tk.NORMAL)
            self.reverb_slider.config(state=tk.NORMAL)
            self.interval_entry.config(state=tk.NORMAL)
            self.variation_entry.config(state=tk.NORMAL)
            self.filter_entry.config(state=tk.NORMAL)
            self.timeout_slider.config(state=tk.NORMAL)
            self.retries_slider.config(state=tk.NORMAL)
            server_type = self.server_type_var.get()
            self.max_tokens_entry.config(state=tk.NORMAL if server_type != "Ollama" else tk.DISABLED, 
                                         bg="white" if server_type != "Ollama" else "grey")
            self.num_ctx_entry.config(state=tk.NORMAL if server_type == "Ollama" else tk.DISABLED,
                                      bg="white" if server_type == "Ollama" else "grey")
            self.send_enabled = True

    def save_config_action(self):
        if not self.start_lock and self.send_enabled:
            self.disable_controls()
            self.after(100, lambda: save_config(self))
            self.after(150, self.enable_send_and_start)

    def verify_server(self, server_url, server_type, model):
        return ping_server(server_url, server_type, model, self.timeout_slider.get(), self.retries_slider.get())

    def start_oracle(self):
        def start_thread():
            if self.start_lock or self.is_running:
                logger.debug("Start blocked: start_lock=%s, is_running=%s", self.start_lock, self.is_running)
                return
            self.start_lock = True
            self.disable_controls()

            server_url = self.server_url_var.get()
            server_type = self.server_type_var.get()
            model = self.model_var.get()
            SYSTEM_PROMPT = self.system_prompt_entry.get("1.0", tk.END).strip()
            tts_url = self.tts_url_var.get()
            speaker_id = self.speaker_id_var.get()

            try:
                interval = float(self.interval_entry.get())
                variation = float(self.variation_entry.get())
                request_interval = max(1.0, interval - (variation / 2))  # Calculate dynamically, min 1.0
                if server_type == "Ollama":
                    num_ctx = int(self.num_ctx_entry.get())
                else:
                    max_tokens = int(self.max_tokens_entry.get())
            except ValueError:
                messagebox.showerror("Input Error", "Interval, Variation, and Context Size/Max Tokens must be numeric.")
                self.start_lock = False
                self.after(0, self.enable_send_and_start)
                return

            if not all([server_url, server_type, model, tts_url, speaker_id]):
                messagebox.showerror("Input Error", "Please fill all fields.")
                self.start_lock = False
                self.after(0, self.enable_send_and_start)
                return

            self.stop_oracle()
            time.sleep(0.5)
            self.stop_event = threading.Event()
            self.reset_tts_threads()

            success, error_msg = ping_server(server_url, server_type, model, self.timeout_slider.get(), self.retries_slider.get())
            if not success:
                print(error_msg)
                self.start_lock = False
                self.after(0, self.enable_send_and_start)
                return

            self.is_running = True
            self.stop_event.clear()
            self.stop_button.config(state=tk.NORMAL, bg="red")

            logger.debug("Starting threads with audio_queue size: %d", self.audio_queue.qsize())
            self.generator_thread = threading.Thread(
                target=generate_wisdom,
                args=(self, self.wisdom_queue, model, lambda: self.server_type_var.get(), self.stop_event, lambda: request_interval, SYSTEM_PROMPT),
                daemon=True
            )
            self.tts_thread = threading.Thread(
                target=text_to_speech,
                args=(self.wisdom_queue, self.audio_queue, lambda: self.speaker_id_var.get(), self.pitch_slider.get, self.stop_event, lambda: self.tts_url_var.get()),
                daemon=True
            )
            self.playback_thread = threading.Thread(
                target=play_audio,
                args=(self.audio_queue, self.stop_event, lambda: float(self.interval_entry.get()), lambda: float(self.variation_entry.get()), self, None, True),
                daemon=True
            )

            self.generator_thread.start()
            self.tts_thread.start()
            self.playback_thread.start()
            logger.debug("Threads started: generator=%s, tts=%s, playback=%s", 
                         self.generator_thread.is_alive(), self.tts_thread.is_alive(), self.playback_thread.is_alive())

        threading.Thread(target=start_thread, daemon=True).start()

    def stop_oracle(self):
        def stop_thread():
            if self.is_running:
                self.is_running = False
                self.stop_event.set()
                
                if self.generator_thread and self.generator_thread.is_alive():
                    self.generator_thread.join(timeout=2)
                    if self.generator_thread.is_alive():
                        logger.warning("Generator thread did not stop cleanly")
                    self.generator_thread = None
                if self.tts_thread and self.tts_thread.is_alive():
                    self.tts_thread.join(timeout=2)
                    if self.tts_thread.is_alive():
                        logger.warning("TTS thread did not stop cleanly")
                    self.tts_thread = None
                
                if self.playback_thread and self.playback_thread.is_alive():
                    self.playback_thread.join(timeout=15)
                    if self.playback_thread.is_alive():
                        logger.warning("Playback thread did not stop cleanly")
                    self.playback_thread = None
                
                logger.debug("Clearing queues: wisdom=%d, audio=%d", self.wisdom_queue.qsize(), self.audio_queue.qsize())
                while not self.wisdom_queue.empty():
                    try:
                        self.wisdom_queue.get_nowait()
                    except queue.Empty:
                        break
                while not self.audio_queue.empty():
                    try:
                        _, wav_path, _ = self.audio_queue.get_nowait()
                        if os.path.exists(wav_path):
                            os.remove(wav_path)
                    except Exception as e:
                        logger.error(f"Audio queue cleanup error: {e}")
                    except queue.Empty:
                        break
                
                if self.session:
                    self.session.close()
                    self.session = None

                # Removed: with history_lock: conversation_history = []

                self.start_button.config(state=tk.NORMAL)
                self.send_button.config(state=tk.NORMAL)
                self.save_button.config(state=tk.NORMAL)
                self.clear_button.config(state=tk.NORMAL if self.remember_var.get() else tk.DISABLED)
                self.remember_check.config(state=tk.NORMAL)
                self.record_button.config(state=tk.NORMAL)
                self.server_type_menu.config(state=tk.NORMAL)
                self.server_url_entry.config(state=tk.NORMAL)
                self.model_entry.config(state=tk.NORMAL)
                self.tts_url_entry.config(state=tk.NORMAL)
                self.speaker_id_entry.config(state=tk.NORMAL)
                self.whisper_server_entry.config(state=tk.NORMAL)
                self.system_prompt_entry.config(state=tk.NORMAL)
                self.stop_button.config(state=tk.DISABLED, bg="lightgray")
                self.pitch_slider.config(state=tk.NORMAL)
                self.reverb_slider.config(state=tk.NORMAL)
                self.interval_entry.config(state=tk.NORMAL)
                self.variation_entry.config(state=tk.NORMAL)
                self.filter_entry.config(state=tk.NORMAL)
                self.timeout_slider.config(state=tk.NORMAL)
                self.retries_slider.config(state=tk.NORMAL)
                server_type = self.server_type_var.get()
                self.max_tokens_entry.config(state=tk.NORMAL if server_type != "Ollama" else tk.DISABLED, 
                                            bg="white" if server_type != "Ollama" else "grey")
                self.num_ctx_entry.config(state=tk.NORMAL if server_type == "Ollama" else tk.DISABLED,
                                        bg="white" if server_type == "Ollama" else "grey")
                self.start_lock = False
                self.send_enabled = True
                self.after(100, self.enable_send_and_start)
                logger.debug("Oracle fully stopped and reset")

        threading.Thread(target=stop_thread, daemon=True).start()

    def send_prompt_action(self):
        def send_thread():
            if not self.send_enabled or self.start_lock:
                logger.debug("Send blocked: send_enabled=%s, start_lock=%s", self.send_enabled, self.start_lock)
                return
            self.send_enabled = False
            self.start_lock = True
            self.disable_controls()
            logger.debug("Starting send_prompt_action")

            server_url = self.server_url_var.get()
            server_type = self.server_type_var.get()
            model = self.model_var.get()
            prompt = self.system_prompt_entry.get("1.0", tk.END).strip()
            tts_url = self.tts_url_var.get()
            speaker_id = self.speaker_id_var.get()

            audio_duration = 0
            if not all([server_url, server_type, model, prompt, tts_url, speaker_id]):
                logger.warning("Missing fields: server_url=%s, server_type=%s, model=%s, prompt=%s, tts_url=%s, speaker_id=%s",
                               server_url, server_type, model, prompt, tts_url, speaker_id)
                messagebox.showwarning("Input Error", "Please fill all fields.")
            else:
                self.reset_tts_threads()
                logger.debug("TTS threads reset")

                send_session = setup_session(server_url, self.retries_slider.get())
                success, error_msg = self.verify_server(server_url, server_type, model)
                if not success:
                    logger.error("Server verification failed: %s", error_msg)
                    print(error_msg)
                else:
                    logger.debug("Sending prompt to server: %s", prompt)
                    send_thread = threading.Thread(
                        target=self.send_prompt_with_tts_tracking,
                        args=(send_session, self.send_wisdom_queue, model, server_type, prompt, self.send_audio_queue),
                        daemon=True
                    )
                    send_thread.start()
                    send_thread.join(timeout=20)
                    if not self.duration_queue.empty():
                        audio_duration = self.duration_queue.get()

            self.send_enabled = True
            self.start_lock = False
            self.after(int(audio_duration * 1000), self.enable_send_and_start)

        threading.Thread(target=send_thread, daemon=True).start()

    def send_prompt_with_tts_tracking(self, session, wisdom_queue, model, server_type, prompt, audio_queue):
        global conversation_history
        try:
            with history_lock:
                if not self.remember_var.get():
                    conversation_history = []
                messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history + [{"role": "user", "content": prompt}]
            payload = {
                "model": model,
                "messages": messages,
                "stream": False
            }
            if server_type == "Ollama":
                payload["options"] = {"num_ctx": int(self.num_ctx_entry.get())}
            else:
                payload["max_tokens"] = int(self.max_tokens_entry.get())
                payload["temperature"] = 0.7

            logger.debug("Sending request to %s with payload: %s", self.server_url_var.get(), json.dumps(payload))
            response = session.post(self.server_url_var.get(), json=payload, timeout=self.timeout_slider.get())
            response.raise_for_status()
            data = response.json()
            wisdom = (
                data.get("message", {}).get("content", "").strip() if server_type == "Ollama"
                else data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            )
            logger.debug("Received response: %s", wisdom)

            if wisdom:
                # Filter replaces characters instead of skipping message
                filtered_wisdom = filter_text(wisdom, self.filter_var.get())
                with history_lock:
                    if self.remember_var.get():
                        conversation_history.append({"role": "user", "content": prompt})
                        conversation_history.append({"role": "assistant", "content": filtered_wisdom})
                        if len(conversation_history) > 100:
                            conversation_history = conversation_history[-100:]
                logger.debug("Queueing filtered wisdom: %s", filtered_wisdom)
                wisdom_queue.put(filtered_wisdom)

                try:
                    logger.debug("Waiting for TTS audio")
                    wisdom, wav_path, pitch = audio_queue.get(timeout=15)
                    logger.debug("Received TTS audio: %s", wav_path)
                    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
                        audio = AudioSegment.from_wav(wav_path)
                        duration_seconds = len(audio) / 1000.0
                        self.duration_queue.put(duration_seconds)
                        self.after(0, lambda: self.wait_for_playback(wav_path))
                    else:
                        logger.error("TTS produced no valid audio: %s", wav_path)
                        os.remove(wav_path)
                except queue.Empty:
                    logger.error("TTS timeout or connection issue after 15 seconds")

        except requests.RequestException as e:
            logger.error(f"{server_type} connection failed: {str(e)}")
            print(f"{server_type} error: {str(e)}")
        finally:
            session.close()

    def wait_for_playback(self, wav_path):
        def check_playback():
            with playback_lock:
                if self.is_audio_playing:
                    self.start_lock = False
                    self.after(0, self.enable_send_and_start)
                else:
                    self.after(100, check_playback)
        self.after(100, check_playback)

    def start_listening(self):
        def listen_thread():
            if self.start_lock or not self.send_enabled:
                logger.debug("Listen blocked: start_lock=%s, send_enabled=%s", self.start_lock, self.send_enabled)
                return
            self.start_lock = True
            self.disable_controls()
            logger.debug("Starting listen_thread")
            
            # Capture audio with noise gate (3-second silence delay)
            audio_file = capture_audio(silence_threshold=500, silence_duration=3)
            
            whisper_server_url = self.whisper_server_var.get()
            if not whisper_server_url.endswith("/asr"):
                whisper_server_url = f"{whisper_server_url.rstrip('/')}/asr"
            
            params = {
                "encode": "true",
                "task": "transcribe",
                "language": "en",
                "output": "txt"
            }
            
            text = ""
            try:
                session = setup_session(whisper_server_url, retries=self.retries_slider.get())
                with open(audio_file, "rb") as f:
                    response = session.post(
                        whisper_server_url,
                        files={"audio_file": (os.path.basename(audio_file), f, "audio/wav")},
                        params=params,
                        timeout=self.timeout_slider.get()
                    )
                    response.raise_for_status()
                    text = response.text.strip()
                    logger.debug("Transcription from server: %s", text)
                    print(f"Transcribed: {text}")
                    
                    filtered_text = filter_text(text, self.filter_var.get())
                    with history_lock:
                        if self.remember_var.get():
                            conversation_history.append({"role": "user", "content": filtered_text})
                    self.system_prompt_entry.delete("1.0", tk.END)
                    self.system_prompt_entry.insert(tk.END, filtered_text)
            
            except requests.RequestException as e:
                logger.error(f"Whisper server error: {str(e)}")
                if hasattr(e.response, 'text'):
                    logger.error(f"Response body: {e.response.text}")
                print(f"Error contacting Whisper server: {str(e)}")
            
            finally:
                if os.path.exists(audio_file):
                    os.remove(audio_file)
                    logger.debug("Temporary audio file removed")
                session.close()
            
            self.start_lock = False
            self.after(0, self.enable_send_and_start)
            
            if text:
                logger.debug("Triggering send_prompt_action with filtered text: %s", filtered_text)
                self.send_prompt_action()
            else:
                logger.debug("No transcription available, skipping send_prompt_action")
        
        threading.Thread(target=listen_thread, daemon=True).start()

    def clear_history(self):
        if self.start_lock or not self.send_enabled:
            return
        with history_lock:
            global conversation_history
            conversation_history = []
        print("The Infinite Oracle’s memory has been wiped clean.")

    def toggle_remember(self):
        if not self.remember_var.get():
            with history_lock:
                global conversation_history
                conversation_history = []
            self.clear_button.config(state=tk.DISABLED)
            print("The Infinite Oracle will forget all past wisdom.")
        else:
            self.clear_button.config(state=tk.NORMAL)
            print("The Infinite Oracle will now remember its wisdom.")

    def toggle_record(self):
        if self.record_var.get():
            self.record_var.set(False)
            self.record_button.config(bg="red", relief=tk.RAISED)
            print("The Infinite Oracle’s voice will no longer be captured.")
        else:
            self.record_var.set(True)
            self.record_button.config(bg="#ff4040", relief=tk.SUNKEN)
            print("The Infinite Oracle’s wisdom will now be preserved.")

def main():
    app = InfiniteOracleGUI()
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThe Infinite Oracle rests... for now.")
