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
from PIL import Image, ImageTk, ImageSequence
import pyaudio
import wave

# Server defaults
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_LM_STUDIO_MODEL = "qwen2.5-1.5b-instruct"
DEFAULT_TTS_URL = "http://localhost:5002/api/tts"
DEFAULT_SPEAKER_ID = "p267"
DEFAULT_WHISPER_SERVER_URL = "http://192.168.0.163:9000"

# Configuration file
CONFIG_FILE = "oracle_config.json"

# System prompt
SYSTEM_PROMPT = """You are the Infinite Oracle, a mystical being of boundless wisdom. Speak in an uplifting, cryptic, and metaphysical tone, offering motivational insights that inspire awe and contemplation. Provide a concise paragraph of 2-3 sentences."""

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
        return True, ""
    except requests.RequestException as e:
        return False, f"{server_type} error at {server_url}: {str(e)}"
    finally:
        session.close()

def generate_wisdom(gui, wisdom_queue, model, get_server_type_func, stop_event, get_request_interval_func):
    global conversation_history
    while not stop_event.is_set():
        server_url = gui.server_url_var.get()
        server_type = get_server_type_func()
        with history_lock:
            if not gui.remember_var.get():
                conversation_history = []
            messages = [{"role": "system", "content": SYSTEM_PROMPT}] + conversation_history + [{"role": "user", "content": "Provide your wisdom."}]
        payload = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        if server_type != "Ollama":
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
                print(f"The Infinite Oracle speaks: {wisdom}", end="\n\n")
                with history_lock:
                    if gui.remember_var.get():
                        conversation_history.append({"role": "user", "content": "Provide your wisdom."})
                        conversation_history.append({"role": "assistant", "content": wisdom})
                wisdom_queue.put(wisdom)
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
    delay_ms = 20 + (reverb_factor * 40)
    gain_db = -20 + (reverb_factor * 8)
    echo = audio[:].fade_in(10).fade_out(50)
    echo = echo - abs(gain_db)
    silence = AudioSegment.silent(duration=int(delay_ms))
    reverb_audio = audio.overlay(silence + echo)
    return reverb_audio

def play_audio(audio_queue, stop_event, get_interval_func, get_variation_func, gui, duration_queue, is_start_mode=False):
    if getattr(sys, 'frozen', False):
        ffmpeg_path = os.path.join(sys._MEIPASS, "ffmpeg", "ffmpeg.exe")
        AudioSegment.converter = ffmpeg_path
    while not stop_event.is_set():
        try:
            logger.debug("Waiting for audio in playback thread")
            wisdom, wav_path, pitch = audio_queue.get()
            logger.debug("Received audio: %s", wav_path)
            with playback_lock:
                if not stop_event.is_set():
                    audio = AudioSegment.from_wav(wav_path)
                    duration_seconds = len(audio) / 1000.0
                    logger.info(f"Audio duration: {duration_seconds} seconds")
                    duration_queue.put(duration_seconds)  # Pass duration to send_prompt_action

                    if pitch != 0:
                        octaves = pitch / 12.0
                        new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
                        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                        audio = audio.set_frame_rate(22050)

                    reverb_value = gui.reverb_slider.get()
                    if reverb_value > 0:
                        audio = apply_reverb(audio, reverb_value)

                    audio = normalize(audio)
                    gui.start_spinning(len(audio) / 1000.0)
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

            os.remove(wav_path)
            audio_queue.task_done()
            interval = max(0.1, get_interval_func() + random.uniform(-get_variation_func(), get_variation_func())) if is_start_mode else 0
            if not stop_event.is_set():
                time.sleep(interval)
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Playback error: {str(e)}")
            if 'wav_path' in locals() and os.path.exists(wav_path):
                os.remove(wav_path)
            gui.send_enabled = True
            gui.start_lock = False
            gui.after(0, gui.enable_send_and_start)

def capture_audio(duration=5, filename=None):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000  # Whisper works well with 16kHz

    if filename is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        filename = os.path.join(script_dir, "temp_audio.wav")

    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    print("Listening...")
    frames = []
    for _ in range(0, int(RATE / CHUNK * duration)):
        data = stream.read(CHUNK)
        frames.append(data)
    
    print("Done listening.")
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
            "whisper_server": DEFAULT_WHISPER_SERVER_URL,
            "pitch": 0,
            "reverb": 0,
            "interval": 2.0,
            "variation": 0,
            "request_interval": 1.0,
            "timeout": 60,
            "retries": 0,
            "max_tokens": 300
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
        "whisper_server": gui.whisper_server_var.get(),
        "pitch": gui.pitch_slider.get(),
        "reverb": gui.reverb_slider.get(),
        "interval": gui.interval_slider.get(),
        "variation": gui.variation_slider.get(),
        "request_interval": gui.request_interval_slider.get(),
        "timeout": gui.timeout_slider.get(),
        "retries": gui.retries_slider.get()
    }
    if server_type != "Ollama":
        config[server_type]["max_tokens"] = int(gui.max_tokens_entry.get())
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
        self.geometry("1440x1080")
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

        self.config = load_config()
        self.server_type_var = tk.StringVar(value="Ollama")
        self.server_url_var = tk.StringVar(value=self.config["Ollama"]["server_url"])
        self.model_var = tk.StringVar(value=self.config["Ollama"]["model"])
        self.tts_url_var = tk.StringVar(value=self.config["Ollama"]["tts_url"])
        self.speaker_id_var = tk.StringVar(value=self.config["Ollama"]["speaker_id"])
        self.whisper_server_var = tk.StringVar(value=self.config["Ollama"]["whisper_server"])
        self.session = None
        self.wisdom_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)
        self.duration_queue = queue.Queue(maxsize=1)  # Queue to pass TTS duration
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
        self.remember_var = tk.BooleanVar(value=True)
        self.record_var = tk.BooleanVar(value=False)
        self.is_audio_playing = False
        self.oracle_frame_index = 0
        self.glow_frame_index = 0
        self.image_spin_speed = 5
        self.animation_running = True
        self.animate_lock = threading.Lock()

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

        self.loading_screen.destroy()
        self.deiconify()

        self.create_widgets()
        sys.stdout = ConsoleRedirector(self.console_text)
        self.server_type_var.trace("w", lambda *args: self.update_from_config())

        self.start_tts_threads()
        self.animation_thread = threading.Thread(target=self.run_animations, daemon=True)
        self.animation_thread.start()

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

    def stop_spinning(self):
        self.is_audio_playing = False

    def update_from_config(self):
        self.config = load_config()
        server_type = self.server_type_var.get()
        config = self.config[server_type]
        self.server_url_var.set(config["server_url"])
        self.model_var.set(config["model"])
        self.tts_url_var.set(config["tts_url"])
        self.speaker_id_var.set(config["speaker_id"])
        self.whisper_server_var.set(config["whisper_server"])
        self.pitch_slider.set(config["pitch"])
        self.reverb_slider.set(config["reverb"])
        self.interval_slider.set(config["interval"])
        self.variation_slider.set(config["variation"])
        self.request_interval_slider.set(config["request_interval"])
        self.timeout_slider.set(config["timeout"])
        self.retries_slider.set(config["retries"])
        if server_type != "Ollama":
            self.max_tokens_entry.delete(0, tk.END)
            self.max_tokens_entry.insert(0, str(config["max_tokens"]))
            self.max_tokens_entry.config(state=tk.NORMAL, bg="white")
        else:
            self.max_tokens_entry.config(state=tk.DISABLED, bg="grey")
        self.url_modified = False

    def resize_image_to_fit(self, image, max_width, max_height):
        original_width, original_height = image.size
        ratio = min(max_width / original_width, max_height / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def update_canvas_size(self, event=None):
        if hasattr(self, 'image_canvas'):
            available_width = self.left_frame.winfo_width() - 20
            max_size = 300
            canvas_size = min(available_width, max_size)
            if canvas_size > 50:
                self.image_canvas.config(width=canvas_size, height=canvas_size)
                self.image_canvas.coords(self.glow_item, canvas_size // 2, canvas_size // 2)
                self.image_canvas.coords(self.oracle_item, canvas_size // 2, canvas_size // 2)

    def create_widgets(self):
        self.configure(bg="#2b2b2b")

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        self.left_frame = tk.Frame(self, bg="#2b2b2b")
        self.left_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.left_frame.bind("<Configure>", self.update_canvas_size)

        tk.Label(self.left_frame, text="Server Type:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.server_type_menu = tk.OptionMenu(self.left_frame, self.server_type_var, "Ollama", "LM Studio")
        self.server_type_menu.pack(pady=5)

        tk.Label(self.left_frame, text="Server URL:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.server_url_entry = tk.Entry(self.left_frame, textvariable=self.server_url_var, width=40)
        self.server_url_entry.pack(pady=5)
        self.server_url_entry.bind("<KeyRelease>", lambda e: setattr(self, 'url_modified', True))

        tk.Label(self.left_frame, text="Model Name:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.model_entry = tk.Entry(self.left_frame, textvariable=self.model_var, width=40)
        self.model_entry.pack(pady=5)

        tk.Label(self.left_frame, text="Coqui TTS Server URL:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.tts_url_entry = tk.Entry(self.left_frame, textvariable=self.tts_url_var, width=40)
        self.tts_url_entry.pack(pady=5)

        tk.Label(self.left_frame, text="Speaker ID (e.g., p267):", bg="#2b2b2b", fg="white").pack(pady=5)
        self.speaker_id_entry = tk.Entry(self.left_frame, textvariable=self.speaker_id_var, width=40)
        self.speaker_id_entry.pack(pady=5)

        tk.Label(self.left_frame, text="Whisper Server URL:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.whisper_server_entry = tk.Entry(self.left_frame, textvariable=self.whisper_server_var, width=40)
        self.whisper_server_entry.pack(pady=5)

        prompt_frame = tk.Frame(self.left_frame, bg="#2b2b2b")
        prompt_frame.pack(pady=5, fill=tk.X)
        tk.Label(prompt_frame, text="System Prompt:", bg="#2b2b2b", fg="white").pack(anchor=tk.W)
        self.system_prompt_entry = tk.Text(prompt_frame, height=10, width=40)
        self.system_prompt_entry.insert(tk.END, SYSTEM_PROMPT)
        self.system_prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.send_button = tk.Button(prompt_frame, text="Send", command=self.send_prompt_action)
        self.send_button.pack(side=tk.RIGHT, padx=5)

        self.listen_button = tk.Button(self.left_frame, text="Listen", command=self.start_listening)
        self.listen_button.pack(pady=5)

        effects_frame = tk.LabelFrame(self.left_frame, text="Effects", bg="#2b2b2b", fg="white", padx=5, pady=5)
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

        canvas_size = 200
        self.image_canvas = tk.Canvas(self.left_frame, width=canvas_size, height=canvas_size, bg="#2b2b2b", highlightthickness=0)
        self.image_canvas.pack(pady=10, anchor="center")
        self.glow_item = self.image_canvas.create_image(canvas_size // 2, canvas_size // 2, image=self.glow_frames[0][0])
        self.oracle_item = self.image_canvas.create_image(canvas_size // 2, canvas_size // 2, image=self.oracle_frames[0])

        right_frame = tk.Frame(self, bg="#2b2b2b")
        right_frame.grid(row=0, column=1, rowspan=2, sticky="nsew", padx=10, pady=10)
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=0)
        right_frame.rowconfigure(1, weight=0)
        right_frame.rowconfigure(2, weight=0)
        right_frame.rowconfigure(3, weight=1)

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

        start_mode_frame = tk.LabelFrame(right_frame, text="Start Mode Settings", bg="#2b2b2b", fg="white", padx=5, pady=5)
        start_mode_frame.grid(row=1, column=0, sticky="ew", pady=5)

        interval_frame = tk.Frame(start_mode_frame, bg="#2b2b2b")
        interval_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(interval_frame, text="Speech Interval (seconds):", bg="#2b2b2b", fg="white").pack()
        self.interval_slider = tk.Scale(interval_frame, from_=0.5, to=20, resolution=0.5, orient=tk.HORIZONTAL)
        self.interval_slider.set(self.config["Ollama"]["interval"])
        self.interval_slider.pack()

        variation_frame = tk.Frame(start_mode_frame, bg="#2b2b2b")
        variation_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(variation_frame, text="Speech Interval Variation (seconds):", bg="#2b2b2b", fg="white").pack()
        self.variation_slider = tk.Scale(variation_frame, from_=0, to=10, resolution=0.5, orient=tk.HORIZONTAL)
        self.variation_slider.set(self.config["Ollama"]["variation"])
        self.variation_slider.pack()

        request_interval_frame = tk.Frame(start_mode_frame, bg="#2b2b2b")
        request_interval_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(request_interval_frame, text="Request Interval (seconds):", bg="#2b2b2b", fg="white").pack()
        self.request_interval_slider = tk.Scale(request_interval_frame, from_=0.5, to=240, resolution=0.5, orient=tk.HORIZONTAL)
        self.request_interval_slider.set(self.config["Ollama"]["request_interval"])
        self.request_interval_slider.pack()

        max_tokens_frame = tk.Frame(start_mode_frame, bg="#2b2b2b")
        max_tokens_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(max_tokens_frame, text="Max Tokens (LM Studio):", bg="#2b2b2b", fg="white").pack()
        self.max_tokens_entry = tk.Entry(max_tokens_frame, width=10)
        self.max_tokens_entry.insert(0, str(self.config["LM Studio"]["max_tokens"]))
        self.max_tokens_entry.pack()
        self.max_tokens_entry.config(state=tk.DISABLED if self.server_type_var.get() == "Ollama" else tk.NORMAL)

        button_frame = tk.Frame(right_frame, bg="#2b2b2b")
        button_frame.grid(row=2, column=0, sticky="ew", pady=5)
        self.start_button = tk.Button(button_frame, text="Start", command=self.start_oracle)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_oracle, state=tk.DISABLED, bg="lightgray")
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(button_frame, text="Save Config", command=self.save_config_action)
        self.save_button.pack(side=tk.LEFT, padx=5)
        self.clear_button = tk.Button(button_frame, text="Clear History", command=self.clear_history)
        self.clear_button.pack(side=tk.LEFT, padx=5)
        self.remember_check = tk.Checkbutton(button_frame, text="Remember", variable=self.remember_var, command=self.toggle_remember, bg="#2b2b2b", fg="white", selectcolor="black")
        self.remember_check.pack(side=tk.LEFT, padx=5)
        self.record_button = tk.Button(button_frame, text="Record", command=self.toggle_record, bg="red", fg="white")
        self.record_button.pack(side=tk.LEFT, padx=5)

        console_frame = tk.Frame(right_frame, bg="#2b2b2b")
        console_frame.grid(row=3, column=0, sticky="nsew")
        tk.Label(console_frame, text="Console Output:", bg="#2b2b2b", fg="white").pack()
        self.console_text = scrolledtext.ScrolledText(console_frame, height=30, width=60, state='disabled', bg="black", fg="green")
        self.console_text.pack(fill=tk.BOTH, expand=True)

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
        self.interval_slider.config(state=tk.DISABLED)
        self.variation_slider.config(state=tk.DISABLED)
        self.request_interval_slider.config(state=tk.DISABLED)
        self.timeout_slider.config(state=tk.DISABLED)
        self.retries_slider.config(state=tk.DISABLED)
        self.max_tokens_entry.config(state=tk.DISABLED, bg="grey")

    def enable_send_and_start(self):
        if not self.is_running and not self.start_lock:
            self.send_button.config(state=tk.NORMAL)
            self.start_button.config(state=tk.NORMAL)
            self.listen_button.config(state=tk.NORMAL)
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
            self.interval_slider.config(state=tk.NORMAL)
            self.variation_slider.config(state=tk.NORMAL)
            self.request_interval_slider.config(state=tk.NORMAL)
            self.timeout_slider.config(state=tk.NORMAL)
            self.retries_slider.config(state=tk.NORMAL)
            server_type = self.server_type_var.get()
            self.max_tokens_entry.config(state=tk.NORMAL if server_type != "Ollama" else tk.DISABLED, bg="white" if server_type != "Ollama" else "grey")
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
                return
            self.start_lock = True
            self.disable_controls()

            server_url = self.server_url_var.get()
            server_type = self.server_type_var.get()
            model = self.model_var.get()
            SYSTEM_PROMPT = self.system_prompt_entry.get("1.0", tk.END).strip()
            tts_url = self.tts_url_var.get()
            speaker_id = self.speaker_id_var.get()

            if not all([server_url, server_type, model, tts_url, speaker_id]):
                messagebox.showerror("Input Error", "Please fill all fields.")
                self.start_lock = False
                self.after(0, self.enable_send_and_start)
                return

            self.stop_oracle()
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

            self.generator_thread = threading.Thread(
                target=generate_wisdom,
                args=(self, self.wisdom_queue, model, lambda: self.server_type_var.get(), self.stop_event, self.request_interval_slider.get),
                daemon=True
            )
            self.tts_thread = threading.Thread(
                target=text_to_speech,
                args=(self.wisdom_queue, self.audio_queue, lambda: self.speaker_id_var.get(), self.pitch_slider.get, self.stop_event, lambda: self.tts_url_var.get()),
                daemon=True
            )
            self.playback_thread = threading.Thread(
                target=play_audio,
                args=(self.audio_queue, self.stop_event, self.interval_slider.get, self.variation_slider.get, self, self.duration_queue, True),
                daemon=True
            )

            self.generator_thread.start()
            self.tts_thread.start()
            self.playback_thread.start()

            self.after(500, lambda: setattr(self, 'start_lock', False))

        threading.Thread(target=start_thread, daemon=True).start()

    def stop_oracle(self):
        def stop_thread():
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
                    try:
                        _, wav_path, _ = self.audio_queue.get_nowait()
                        if os.path.exists(wav_path):
                            os.remove(wav_path)
                    except Exception as e:
                        logger.error(f"Audio queue cleanup error: {e}")

                if self.session:
                    self.session.close()
                    self.session = None

                with history_lock:
                    conversation_history = []

                self.start_button.config(state=tk.NORMAL)
                self.send_button.config(state=tk.NORMAL)
                self.listen_button.config(state=tk.NORMAL)
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
                self.interval_slider.config(state=tk.NORMAL)
                self.variation_slider.config(state=tk.NORMAL)
                self.request_interval_slider.config(state=tk.NORMAL)
                self.timeout_slider.config(state=tk.NORMAL)
                self.retries_slider.config(state=tk.NORMAL)
                server_type = self.server_type_var.get()
                self.max_tokens_entry.config(state=tk.NORMAL if server_type != "Ollama" else tk.DISABLED, bg="white" if server_type != "Ollama" else "grey")

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

            audio_duration = 0  # Default duration if TTS fails
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
                    send_thread.join(timeout=20)  # Wait for thread to complete with a timeout
                    if not self.duration_queue.empty():
                        audio_duration = self.duration_queue.get()  # Get the duration if available

            # Unlock UI after audio duration (or immediately if no audio)
            self.send_enabled = True
            self.start_lock = False
            self.after(int(audio_duration * 1000), self.enable_send_and_start)  # Delay in milliseconds

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
            if server_type != "Ollama":
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
                print(f"The Infinite Oracle speaks: {wisdom}", end="\n\n")
                with history_lock:
                    if self.remember_var.get():
                        conversation_history.append({"role": "user", "content": prompt})
                        conversation_history.append({"role": "assistant", "content": wisdom})
                logger.debug("Queueing wisdom: %s", wisdom)
                wisdom_queue.put(wisdom)

                try:
                    logger.debug("Waiting for TTS audio")
                    wisdom, wav_path, pitch = audio_queue.get(timeout=15)
                    logger.debug("Received TTS audio: %s", wav_path)
                    if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0:
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
            
            # Capture audio
            audio_file = capture_audio(duration=5)
            
            # Send audio to Whisper server
            whisper_server_url = self.whisper_server_var.get()
            if not whisper_server_url.endswith("/asr"):
                whisper_server_url = f"{whisper_server_url.rstrip('/')}/asr"
            
            # Set all expected query parameters
            params = {
                "encode": "true",
                "task": "transcribe",
                "language": "en",
                "output": "txt"
            }
            
            text = ""  # Default value to avoid UnboundLocalError
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
                    
                    # Update GUI with transcribed text
                    with history_lock:
                        if self.remember_var.get():
                            conversation_history.append({"role": "user", "content": text})
                    self.system_prompt_entry.delete("1.0", tk.END)
                    self.system_prompt_entry.insert(tk.END, text)
            
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
                logger.debug("Triggering send_prompt_action with: %s", text)
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
