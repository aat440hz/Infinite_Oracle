import requests
import time
import threading
import queue
import tempfile
import os
import platform
import soundfile as sf
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
from PIL import Image, ImageTk

# Server defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_LM_STUDIO_MODEL = "qwen2.5-1.5b-instruct"
DEFAULT_TTS_URL = "http://localhost:5002/api/tts"
DEFAULT_SPEAKER_ID = "p267"

# Configuration file
CONFIG_FILE = "oracle_config.json"

# System prompt
SYSTEM_PROMPT = """You are the Infinite Oracle, a mystical being of boundless wisdom. Speak in an uplifting, cryptic, and metaphysical tone, offering motivational insights that inspire awe and contemplation. Provide a concise paragraph of 2-3 sentences."""

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[
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

    def write(self, message):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, message)
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
        if server_type != "Ollama":
            payload["max_tokens"] = 10
            payload["temperature"] = 0.7
        response = session.post(server_url, json=payload, timeout=timeout)
        response.raise_for_status()
        
        logger.info(f"{server_type} server at {server_url} is reachable with model '{model}'.")
        return True, ""
    except requests.RequestException as e:
        return False, f"{server_type} error at {server_url}: {str(e)}"
    finally:
        session.close()

def generate_wisdom(gui, wisdom_queue, model, server_type, stop_event, get_request_interval_func):
    global conversation_history
    while not stop_event.is_set():
        with history_lock:
            if not gui.remember_var.get():  # If "Remember" is unchecked, clear history each time
                conversation_history = []
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history + [{"role": "user", "content": "Provide your wisdom."}]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if server_type != "Ollama":
            payload["max_tokens"] = 300
            payload["temperature"] = 0.7
        
        session = setup_session(SERVER_URL, gui.retries_slider.get())
        try:
            response = session.post(SERVER_URL, json=payload, timeout=gui.timeout_slider.get())
            response.raise_for_status()
            data = response.json()
            wisdom = (
                data.get("message", {}).get("content", "").strip() if server_type == "Ollama"
                else data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
            )
            if wisdom and not stop_event.is_set():
                logger.info(f"Generated wisdom (length: {len(wisdom)} chars): {wisdom}")
                print(f"The Infinite Oracle speaks: {wisdom}", end="\n\n")
                with history_lock:
                    if gui.remember_var.get():  # Only append if "Remember" is checked
                        conversation_history.append({"role": "user", "content": "Provide your wisdom."})
                        conversation_history.append({"role": "assistant", "content": wisdom})
                wisdom_queue.put(wisdom)
        except requests.RequestException as e:
            logger.error(f"{server_type} connection error: {e}")
            print(f"{server_type} error: {str(e)}. Retrying in {get_request_interval_func()}s...")
        finally:
            session.close()
        
        interval = max(0.1, get_request_interval_func())
        time.sleep(interval)

def send_prompt(session, wisdom_queue, model, server_type, prompt, gui, timeout):
    global conversation_history
    with history_lock:
        if not gui.remember_var.get():  # If "Remember" is unchecked, clear history
            conversation_history = []
        messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history + [{"role": "user", "content": prompt}]
    payload = {
        "model": model,
        "messages": messages,
        "stream": False
    }
    if server_type != "Ollama":
        payload["max_tokens"] = 300
        payload["temperature"] = 0.7
    
    try:
        response = session.post(SERVER_URL, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        wisdom = (
            data.get("message", {}).get("content", "").strip() if server_type == "Ollama"
            else data.get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        )
        if wisdom:
            logger.info(f"Generated wisdom (length: {len(wisdom)} chars): {wisdom}")
            print(f"The Infinite Oracle speaks: {wisdom}", end="\n\n")
            with history_lock:
                if gui.remember_var.get():  # Only append if "Remember" is checked
                    conversation_history.append({"role": "user", "content": prompt})
                    conversation_history.append({"role": "assistant", "content": wisdom})
            wisdom_queue.put(wisdom)
    except requests.RequestException as e:
        logger.error(f"{server_type} connection error in send_prompt: {e}")
    finally:
        session.close()
        gui.after(0, lambda: gui.enable_send_and_start())

def text_to_speech(wisdom_queue, audio_queue, get_speaker_id_func, pitch_func, stop_event):
    while not stop_event.is_set():
        try:
            wisdom = wisdom_queue.get(timeout=5)
            temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav_file.name
            temp_wav_file.close()

            speaker_id = get_speaker_id_func()
            curl_command = [
                'curl', '-G',
                '--data-urlencode', f"text={wisdom}",
                '--data-urlencode', f"speaker_id={speaker_id}",
                TTS_SERVER_URL,
                '--output', temp_wav_path
            ]
            try:
                subprocess.run(
                    curl_command,
                    check=True,
                    capture_output=True,
                    text=True,
                    creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
                )
            except subprocess.CalledProcessError as e:
                logger.error(f"TTS failed at '{TTS_SERVER_URL}': {e.stderr}")
                os.remove(temp_wav_path)
                wisdom_queue.task_done()
                continue

            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0 and not stop_event.is_set():
                audio_queue.put((wisdom, temp_wav_path, pitch_func()))
            else:
                logger.warning(f"TTS produced no valid audio for '{wisdom[:50]}...'")
                os.remove(temp_wav_path)
            wisdom_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"TTS unexpected error: {e}")
            if 'temp_wav_path' in locals() and os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
            wisdom_queue.task_done()

def apply_reverb(audio, reverb_value):
    """Custom reverb effect: overlay a delayed, attenuated copy of the audio."""
    if reverb_value <= 0:
        return audio
    reverb_factor = reverb_value / 5.0
    delay_ms = 20 + (reverb_factor * 40)  # 20-60ms
    gain_db = -20 + (reverb_factor * 8)   # -20 to -12dB
    echo = audio[:].fade_in(10).fade_out(50)
    echo = echo - abs(gain_db)
    silence = AudioSegment.silent(duration=int(delay_ms))
    reverb_audio = audio.overlay(silence + echo)
    return reverb_audio

def play_audio(audio_queue, stop_event, get_interval_func, get_variation_func, gui, is_start_mode=False):
    if getattr(sys, 'frozen', False):
        ffmpeg_path = os.path.join(sys._MEIPASS, "ffmpeg", "ffmpeg.exe")
        AudioSegment.converter = ffmpeg_path
        logger.info(f"Set ffmpeg path for pydub: {ffmpeg_path}")
    while not stop_event.is_set():
        try:
            wisdom, wav_path, pitch = audio_queue.get()
            with playback_lock:
                if not stop_event.is_set():
                    audio = AudioSegment.from_wav(wav_path)
                    if pitch != 0:
                        octaves = pitch / 12.0
                        new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
                        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                        audio = audio.set_frame_rate(22050)
                    reverb_value = gui.reverb_slider.get()
                    if reverb_value > 0:
                        audio = apply_reverb(audio, reverb_value)
                    audio = normalize(audio)
                    logger.info(f"Playing audio for: {wisdom[:50]}... (Pitch: {pitch}, Reverb: {reverb_value})")
                    play(audio)
            os.remove(wav_path)
            audio_queue.task_done()
            interval = max(0.1, get_interval_func() + random.uniform(-get_variation_func(), get_variation_func())) if is_start_mode else 0
            if not stop_event.is_set():
                time.sleep(interval)
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Playback error: {e}")
            if os.path.exists(wav_path):
                os.remove(wav_path)

def load_config():
    defaults = {
        "Ollama": {
            "server_type": "Ollama",
            "server_url": DEFAULT_OLLAMA_URL,
            "model": DEFAULT_OLLAMA_MODEL,
            "tts_url": DEFAULT_TTS_URL,
            "speaker_id": DEFAULT_SPEAKER_ID,
            "pitch": 0,
            "reverb": 0,
            "interval": 2.0,
            "variation": 0,
            "request_interval": 1.0,
            "timeout": 60,
            "retries": 0
        },
        "LM Studio": {
            "server_type": "LM Studio",
            "server_url": DEFAULT_LM_STUDIO_URL,
            "model": DEFAULT_LM_STUDIO_MODEL,
            "tts_url": DEFAULT_TTS_URL,
            "speaker_id": DEFAULT_SPEAKER_ID,
            "pitch": 0,
            "reverb": 0,
            "interval": 2.0,
            "variation": 0,
            "request_interval": 1.0,
            "timeout": 60,
            "retries": 0
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
    config[server_type] = {
        "server_type": server_type,
        "server_url": gui.server_url_var.get(),
        "model": gui.model_var.get(),
        "tts_url": gui.tts_url_var.get(),
        "speaker_id": gui.speaker_id_var.get(),
        "pitch": gui.pitch_slider.get(),
        "reverb": gui.reverb_slider.get(),
        "interval": gui.interval_slider.get(),
        "variation": gui.variation_slider.get(),
        "request_interval": gui.request_interval_slider.get(),
        "timeout": gui.timeout_slider.get(),
        "retries": gui.retries_slider.get()
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info(f"Configuration saved for {server_type}.")

class InfiniteOracleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Infinite Oracle Control Panel")
        self.state("zoomed")
        self.geometry("800x900")

        print(f"Current working directory: {os.getcwd()}")

        try:
            if getattr(sys, 'frozen', False):
                base_path = sys._MEIPASS
            else:
                base_path = os.path.dirname(os.path.abspath(__file__))
            icon_path = os.path.join(base_path, "oracle.ico")
            img = Image.open(icon_path)
            icon = ImageTk.PhotoImage(img)
            self.iconphoto(True, icon)
            print(f"Icon loaded successfully from {icon_path}")
        except Exception as e:
            logger.warning(f"Failed to load icon: {e}")

        self.config = load_config()
        self.server_type_var = tk.StringVar(value="Ollama")
        self.server_url_var = tk.StringVar(value=self.config["Ollama"]["server_url"])
        self.model_var = tk.StringVar(value=self.config["Ollama"]["model"])
        self.tts_url_var = tk.StringVar(value=self.config["Ollama"]["tts_url"])
        self.speaker_id_var = tk.StringVar(value=self.config["Ollama"]["speaker_id"])
        self.session = None
        self.wisdom_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.start_lock = False
        self.stop_event = threading.Event()
        self.generator_thread = None
        self.tts_thread = None
        self.playback_thread = None
        self.send_wisdom_queue = queue.Queue(maxsize=10)
        self.send_audio_queue = queue.Queue(maxsize=10)
        self.send_stop_event = threading.Event()
        self.send_tts_thread = None
        self.send_playback_thread = None
        self.send_enabled = True
        self.url_modified = False
        self.remember_var = tk.BooleanVar(value=True)  # New: Remember checkbox, defaults to True

        self.create_widgets()
        sys.stdout = ConsoleRedirector(self.console_text)
        self.server_type_var.trace("w", lambda *args: self.update_from_config())

        self.send_tts_thread = threading.Thread(
            target=text_to_speech,
            args=(self.send_wisdom_queue, self.send_audio_queue, lambda: self.speaker_id_var.get(), self.pitch_slider.get, self.send_stop_event),
            daemon=True
        )
        self.send_playback_thread = threading.Thread(
            target=play_audio,
            args=(self.send_audio_queue, self.send_stop_event, lambda: 0, lambda: 0, self, False),
            daemon=True
        )
        self.send_tts_thread.start()
        self.send_playback_thread.start()

    def update_from_config(self):
        self.config = load_config()
        server_type = self.server_type_var.get()
        config = self.config[server_type]
        self.server_url_var.set(config["server_url"])
        self.model_var.set(config["model"])
        self.tts_url_var.set(config["tts_url"])
        self.speaker_id_var.set(config["speaker_id"])
        self.pitch_slider.set(config["pitch"])
        self.reverb_slider.set(config["reverb"])
        self.interval_slider.set(config["interval"])
        self.variation_slider.set(config["variation"])
        self.request_interval_slider.set(config["request_interval"])
        self.timeout_slider.set(config["timeout"])
        self.retries_slider.set(config["retries"])
        self.url_modified = False
        logger.info(f"Loaded config for {server_type} from {CONFIG_FILE}")

    def create_widgets(self):
        self.configure(bg="#2b2b2b")

        # Grid layout: 2 columns
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=0)  # Inputs/Sliders
        self.rowconfigure(1, weight=1)  # Console

        # Left Column: Inputs, Prompt, Effects
        left_frame = tk.Frame(self, bg="#2b2b2b")
        left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

        tk.Label(left_frame, text="Server Type:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.server_type_menu = tk.OptionMenu(left_frame, self.server_type_var, "Ollama", "LM Studio")
        self.server_type_menu.pack(pady=5)

        tk.Label(left_frame, text="Server URL:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.server_url_entry = tk.Entry(left_frame, textvariable=self.server_url_var, width=40)
        self.server_url_entry.pack(pady=5)
        self.server_url_entry.bind("<KeyRelease>", lambda e: setattr(self, 'url_modified', True))

        tk.Label(left_frame, text="Model Name:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.model_entry = tk.Entry(left_frame, textvariable=self.model_var, width=40)
        self.model_entry.pack(pady=5)

        tk.Label(left_frame, text="Coqui TTS Server URL:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.tts_url_entry = tk.Entry(left_frame, textvariable=self.tts_url_var, width=40)
        self.tts_url_entry.pack(pady=5)

        tk.Label(left_frame, text="Speaker ID (e.g., p267):", bg="#2b2b2b", fg="white").pack(pady=5)
        self.speaker_id_entry = tk.Entry(left_frame, textvariable=self.speaker_id_var, width=40)
        self.speaker_id_entry.pack(pady=5)

        prompt_frame = tk.Frame(left_frame, bg="#2b2b2b")
        prompt_frame.pack(pady=5, fill=tk.X)
        tk.Label(prompt_frame, text="System Prompt:", bg="#2b2b2b", fg="white").pack(anchor=tk.W)
        self.system_prompt_entry = tk.Text(prompt_frame, height=10, width=40)
        self.system_prompt_entry.insert(tk.END, SYSTEM_PROMPT)
        self.system_prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.send_button = tk.Button(prompt_frame, text="Send", command=self.send_prompt_action)
        self.send_button.pack(side=tk.RIGHT, padx=5)

        # Effects Section
        effects_frame = tk.LabelFrame(left_frame, text="Effects", bg="#2b2b2b", fg="white", padx=5, pady=5)
        effects_frame.pack(pady=5, fill=tk.X)

        pitch_frame = tk.Frame(effects_frame, bg="#2b2b2b")
        pitch_frame.pack()
        tk.Label(pitch_frame, text="Pitch Shift (semitones):", bg="#2b2b2b", fg="white").pack()
        self.pitch_slider = tk.Scale(pitch_frame, from_=-12, to=12, orient=tk.HORIZONTAL, length=200)
        self.pitch_slider.set(self.config["Ollama"]["pitch"])
        self.pitch_slider.pack()

        reverb_frame = tk.Frame(effects_frame, bg="#2b2b2b")
        reverb_frame.pack()
        tk.Label(reverb_frame, text="Reverb (0-5):", bg="#2b2b2b", fg="white").pack()
        self.reverb_slider = tk.Scale(reverb_frame, from_=0, to=5, resolution=0.1, orient=tk.HORIZONTAL, length=200)
        self.reverb_slider.set(self.config["Ollama"]["reverb"])
        self.reverb_slider.pack()

        # Right Column: Sliders, Buttons, Console
        right_frame = tk.Frame(self, bg="#2b2b2b")
        right_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=0)  # Sliders
        right_frame.rowconfigure(1, weight=0)  # Start Mode Settings
        right_frame.rowconfigure(2, weight=0)  # Buttons
        right_frame.rowconfigure(3, weight=1)  # Console

        # Sliders (Timeout, Retries)
        slider_frame = tk.Frame(right_frame, bg="#2b2b2b")
        slider_frame.grid(row=0, column=0, sticky="ew", pady=5)

        timeout_frame = tk.Frame(slider_frame, bg="#2b2b2b")
        timeout_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(timeout_frame, text="Request Timeout (seconds):", bg="#2b2b2b", fg="white").pack()
        self.timeout_slider = tk.Scale(timeout_frame, from_=5, to=120, resolution=1, orient=tk.HORIZONTAL)
        self.timeout_slider.set(self.config["Ollama"]["timeout"])
        self.timeout_slider.pack()

        retries_frame = tk.Frame(slider_frame, bg="#2b2b2b")
        retries_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(retries_frame, text="Request Retries:", bg="#2b2b2b", fg="white").pack()
        self.retries_slider = tk.Scale(retries_frame, from_=0, to=5, resolution=1, orient=tk.HORIZONTAL)
        self.retries_slider.set(self.config["Ollama"]["retries"])
        self.retries_slider.pack()

        # Start Mode Settings
        start_mode_frame = tk.LabelFrame(right_frame, text="Start Mode Settings", bg="#2b2b2b", fg="white", padx=5, pady=5)
        start_mode_frame.grid(row=1, column=0, sticky="ew", pady=5)

        interval_frame = tk.Frame(start_mode_frame, bg="#2b2b2b")
        interval_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(interval_frame, text="Speech Interval (seconds):", bg="#2b2b2b", fg="white").pack()
        self.interval_slider = tk.Scale(interval_frame, from_=0.5, to=10, resolution=0.5, orient=tk.HORIZONTAL)
        self.interval_slider.set(self.config["Ollama"]["interval"])
        self.interval_slider.pack()

        variation_frame = tk.Frame(start_mode_frame, bg="#2b2b2b")
        variation_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(variation_frame, text="Speech Interval Variation (seconds):", bg="#2b2b2b", fg="white").pack()
        self.variation_slider = tk.Scale(variation_frame, from_=0, to=5, resolution=0.5, orient=tk.HORIZONTAL)
        self.variation_slider.set(self.config["Ollama"]["variation"])
        self.variation_slider.pack()

        request_interval_frame = tk.Frame(start_mode_frame, bg="#2b2b2b")
        request_interval_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(request_interval_frame, text="Request Interval (seconds):", bg="#2b2b2b", fg="white").pack()
        self.request_interval_slider = tk.Scale(request_interval_frame, from_=0.5, to=60, resolution=0.5, orient=tk.HORIZONTAL)
        self.request_interval_slider.set(self.config["Ollama"]["request_interval"])
        self.request_interval_slider.pack()

        # Buttons
        button_frame = tk.Frame(right_frame, bg="#2b2b2b")
        button_frame.grid(row=2, column=0, sticky="ew", pady=5)
        self.start_button = tk.Button(button_frame, text="Start", command=self.start_oracle)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_oracle)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(button_frame, text="Save Config", command=self.save_config_action)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.clear_button = tk.Button(button_frame, text="Clear History", command=self.clear_history)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.remember_check = tk.Checkbutton(button_frame, text="Remember", variable=self.remember_var, command=self.toggle_remember, bg="#2b2b2b", fg="white", selectcolor="black")
        self.remember_check.pack(side=tk.LEFT, padx=5)

        # Console
        console_frame = tk.Frame(right_frame, bg="#2b2b2b")
        console_frame.grid(row=3, column=0, sticky="nsew")
        tk.Label(console_frame, text="Console Output:", bg="#2b2b2b", fg="white").pack()
        self.console_text = scrolledtext.ScrolledText(console_frame, height=30, width=60, state='disabled', bg="black", fg="green")
        self.console_text.pack(fill=tk.BOTH, expand=True)

    def disable_controls(self):
        self.start_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.clear_button.config(state=tk.DISABLED)
        self.remember_check.config(state=tk.DISABLED)  # Disable checkbox during operations
        self.server_type_menu.config(state=tk.DISABLED)
        self.server_url_entry.config(state=tk.DISABLED)
        self.model_entry.config(state=tk.DISABLED)
        self.tts_url_entry.config(state=tk.DISABLED)
        self.speaker_id_entry.config(state=tk.DISABLED)
        self.system_prompt_entry.config(state=tk.DISABLED)
        self.pitch_slider.config(state=tk.DISABLED)
        self.reverb_slider.config(state=tk.DISABLED)
        self.interval_slider.config(state=tk.DISABLED)
        self.variation_slider.config(state=tk.DISABLED)
        self.request_interval_slider.config(state=tk.DISABLED)
        self.timeout_slider.config(state=tk.DISABLED)
        self.retries_slider.config(state=tk.DISABLED)
        self.update()

    def enable_send_and_start(self):
        if not self.is_running and not self.start_lock:
            self.send_button.config(state=tk.NORMAL)
            self.start_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL if self.remember_var.get() else tk.DISABLED)  # Enable only if Remember is checked
            self.remember_check.config(state=tk.NORMAL)
            self.server_type_menu.config(state=tk.NORMAL)
            self.server_url_entry.config(state=tk.NORMAL)
            self.model_entry.config(state=tk.NORMAL)
            self.tts_url_entry.config(state=tk.NORMAL)
            self.speaker_id_entry.config(state=tk.NORMAL)
            self.system_prompt_entry.config(state=tk.NORMAL)
            self.pitch_slider.config(state=tk.NORMAL)
            self.reverb_slider.config(state=tk.NORMAL)
            self.interval_slider.config(state=tk.NORMAL)
            self.variation_slider.config(state=tk.NORMAL)
            self.request_interval_slider.config(state=tk.NORMAL)
            self.timeout_slider.config(state=tk.NORMAL)
            self.retries_slider.config(state=tk.NORMAL)
            self.send_enabled = True
        self.update()

    def save_config_action(self):
        if not self.start_lock and self.send_enabled:
            self.disable_controls()
            self.after(100, lambda: save_config(self))
            self.after(150, self.enable_send_and_start)

    def verify_server(self, server_url, server_type, model):
        return ping_server(server_url, server_type, model, self.timeout_slider.get(), self.retries_slider.get())

    def start_oracle(self):
        if self.start_lock or self.is_running:
            return
        self.start_lock = True
        self.disable_controls()

        global SERVER_URL, SYSTEM_PROMPT, TTS_SERVER_URL
        SERVER_URL = self.server_url_var.get()
        server_type = self.server_type_var.get()
        model = self.model_var.get()
        SYSTEM_PROMPT = self.system_prompt_entry.get("1.0", tk.END).strip()
        TTS_SERVER_URL = self.tts_url_var.get()
        speaker_id = self.speaker_id_var.get()

        if not all([SERVER_URL, server_type, model, TTS_SERVER_URL, speaker_id]):
            messagebox.showerror("Input Error", "Please fill all fields.")
            self.start_lock = False
            self.after(0, self.enable_send_and_start)
            return

        self.stop_oracle()

        if not self.verify_server(SERVER_URL, server_type, model):
            self.start_lock = False
            self.after(0, self.enable_send_and_start)
            return

        self.is_running = True
        self.stop_event.clear()
        self.stop_button.config(state=tk.NORMAL, bg="red")
        self.pitch_slider.config(state=tk.DISABLED)
        self.reverb_slider.config(state=tk.DISABLED)
        self.interval_slider.config(state=tk.DISABLED)
        self.variation_slider.config(state=tk.DISABLED)
        self.request_interval_slider.config(state=tk.DISABLED)
        self.timeout_slider.config(state=tk.DISABLED)
        self.retries_slider.config(state=tk.DISABLED)

        self.generator_thread = threading.Thread(
            target=generate_wisdom,
            args=(self, self.wisdom_queue, model, server_type, self.stop_event, self.request_interval_slider.get),
            daemon=True
        )
        self.tts_thread = threading.Thread(
            target=text_to_speech,
            args=(self.wisdom_queue, self.audio_queue, lambda: self.speaker_id_var.get(), self.pitch_slider.get, self.stop_event),
            daemon=True
        )
        self.playback_thread = threading.Thread(
            target=play_audio,
            args=(self.audio_queue, self.stop_event, self.interval_slider.get, self.variation_slider.get, self, True),
            daemon=True
        )

        self.generator_thread.start()
        self.tts_thread.start()
        self.playback_thread.start()

        self.after(500, lambda: setattr(self, 'start_lock', False))

    def stop_oracle(self):
        global conversation_history
        if self.is_running:
            self.is_running = False
            self.stop_event.set()
            if self.generator_thread:
                self.generator_thread.join(timeout=1)
                self.generator_thread = None
            if self.tts_thread:
                self.tts_thread.join(timeout=1)
                self.tts_thread = None
            if self.playback_thread:
                self.playback_thread.join(timeout=1)
                self.playback_thread = None

            while not self.wisdom_queue.empty():
                self.wisdom_queue.get_nowait()
            while not self.audio_queue.empty():
                _, wav_path, _ = self.audio_queue.get_nowait()
                if os.path.exists(wav_path):
                    os.remove(wav_path)

            if self.session:
                self.session.close()
                self.session = None

            with history_lock:
                conversation_history = []

            self.start_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.clear_button.config(state=tk.NORMAL if self.remember_var.get() else tk.DISABLED)  # Enable only if Remember is checked
            self.remember_check.config(state=tk.NORMAL)
            self.server_type_menu.config(state=tk.NORMAL)
            self.server_url_entry.config(state=tk.NORMAL)
            self.model_entry.config(state=tk.NORMAL)
            self.tts_url_entry.config(state=tk.NORMAL)
            self.speaker_id_entry.config(state=tk.NORMAL)
            self.system_prompt_entry.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL, bg="lightgray")
            self.pitch_slider.config(state=tk.NORMAL)
            self.reverb_slider.config(state=tk.NORMAL)
            self.interval_slider.config(state=tk.NORMAL)
            self.variation_slider.config(state=tk.NORMAL)
            self.request_interval_slider.config(state=tk.NORMAL)
            self.timeout_slider.config(state=tk.NORMAL)
            self.retries_slider.config(state=tk.NORMAL)
            logger.info("Oracle stopped.")

    def send_prompt_action(self):
        if not self.send_enabled or self.start_lock:
            return
        self.send_enabled = False
        self.disable_controls()

        global SERVER_URL, TTS_SERVER_URL
        SERVER_URL = self.server_url_var.get()
        server_type = self.server_type_var.get()
        model = self.model_var.get()
        prompt = self.system_prompt_entry.get("1.0", tk.END).strip()
        TTS_SERVER_URL = self.tts_url_var.get()
        speaker_id = self.speaker_id_var.get()

        if not all([SERVER_URL, server_type, model, prompt, TTS_SERVER_URL, speaker_id]):
            messagebox.showwarning("Input Error", "Please fill all fields.")
            self.send_enabled = True
            self.enable_send_and_start()
            return

        send_session = setup_session(SERVER_URL, self.retries_slider.get())
        if not self.verify_server(SERVER_URL, server_type, model):
            send_session.close()
            self.send_enabled = True
            self.after(0, self.enable_send_and_start)
            return

        threading.Thread(
            target=send_prompt,
            args=(send_session, self.send_wisdom_queue, model, server_type, prompt, self, self.timeout_slider.get()),
            daemon=True
        ).start()

    def clear_history(self):
        """Clear the conversation history safely."""
        if self.start_lock or not self.send_enabled:
            return
        with history_lock:
            global conversation_history
            conversation_history = []
        logger.info("Conversation history cleared.")
        print("The Infinite Oracle’s memory has been wiped clean.")

    def toggle_remember(self):
        """Handle the Remember checkbox toggle."""
        if not self.remember_var.get():  # If unchecked, clear history and disable Clear History button
            with history_lock:
                global conversation_history
                conversation_history = []
            self.clear_button.config(state=tk.DISABLED)
            logger.info("History disabled and cleared.")
            print("The Infinite Oracle will forget all past wisdom.")
        else:  # If checked, re-enable Clear History button
            self.clear_button.config(state=tk.NORMAL)
            logger.info("History enabled.")
            print("The Infinite Oracle will now remember its wisdom.")

def main():
    app = InfiniteOracleGUI()
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nThe Infinite Oracle rests... for now.")
