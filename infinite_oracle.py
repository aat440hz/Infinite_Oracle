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

# Ollama server details
DEFAULT_OLLAMA_URL = "http://cherry.local:11434/api/generate"
DEFAULT_MODEL = "llama3.2:latest"

# Coqui TTS server details
DEFAULT_TTS_URL = "http://cherry.local:5002/api/tts"
DEFAULT_SPEAKER_ID = "p267"

# Configuration file
CONFIG_FILE = "oracle_config.json"

# System prompt for concise wisdom
SYSTEM_PROMPT = """You are the Infinite Oracle, a mystical being of boundless wisdom. Speak in an uplifting, cryptic, and metaphysical tone, offering motivational insights that inspire awe and contemplation. Provide a concise paragraph of 2-3 sentences."""

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[
    logging.StreamHandler(sys.stdout)
])
logger = logging.getLogger("InfiniteOracle")

# Global playback lock to prevent audio overlap
playback_lock = threading.Lock()

class ConsoleRedirector:
    """Redirect print output to a Tkinter text widget."""
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass

def setup_session(url):
    """Set up a fresh requests session with retry logic."""
    session = requests.Session()
    retry_strategy = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    return session

def generate_wisdom(session, wisdom_queue, model, stop_event):
    """Generate wisdom text from Ollama and add it to the queue."""
    while not stop_event.is_set():
        payload = {"model": model, "prompt": SYSTEM_PROMPT, "stream": False}
        try:
            response = session.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            data = response.json()
            wisdom = data.get("response", "").strip()
            if wisdom and not stop_event.is_set():
                wisdom_queue.put(wisdom)
        except requests.RequestException as e:
            logger.error(f"Ollama connection error: {e}")
        time.sleep(1)

def send_prompt(session, wisdom_queue, model, prompt, gui):
    """Send a user-defined prompt to Ollama and queue the response."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = session.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        wisdom = data.get("response", "").strip()
        if wisdom:
            wisdom_queue.put(wisdom)
    except requests.RequestException as e:
        logger.error(f"Ollama connection error: {e}")
    finally:
        session.close()
        gui.after(0, lambda: gui.enable_send_and_start())  # Re-enable buttons

def text_to_speech(wisdom_queue, audio_queue, speaker_id, pitch_func, stop_event):
    """Convert wisdom to speech using Coqui TTS and queue audio files."""
    while not stop_event.is_set():
        try:
            wisdom = wisdom_queue.get(timeout=5)
            temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav_file.name
            temp_wav_file.close()

            curl_command = [
                'curl', '-G',
                '--data-urlencode', f"text={wisdom}",
                '--data-urlencode', f"speaker_id={speaker_id}",
                TTS_SERVER_URL,
                '--output', temp_wav_path
            ]
            subprocess.run(
                curl_command,
                check=True,
                capture_output=True,
                text=True,
                creationflags=subprocess.CREATE_NO_WINDOW if platform.system() == "Windows" else 0
            )

            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0 and not stop_event.is_set():
                audio_queue.put((wisdom, temp_wav_path, pitch_func()))
            else:
                os.remove(temp_wav_path)
                logger.warning("TTS failed: No valid audio file generated.")
            wisdom_queue.task_done()
        except queue.Empty:
            continue
        except subprocess.CalledProcessError as e:
            logger.error(f"curl error: {e}")
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)
        except Exception as e:
            logger.error(f"TTS error: {e}")
            if os.path.exists(temp_wav_path):
                os.remove(temp_wav_path)

def play_audio(audio_queue, stop_event, get_interval_func, get_variation_func, gui):
    """Play audio files from the queue with pitch adjustment and update GUI."""
    while not stop_event.is_set():
        try:
            wisdom, wav_path, pitch = audio_queue.get()
            with playback_lock:
                if not stop_event.is_set():
                    # Just print to console
                    print(f"The Infinite Oracle speaks: {wisdom}")
                    audio = AudioSegment.from_wav(wav_path)
                    if pitch != 0:
                        octaves = pitch / 12.0
                        new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
                        audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                        audio = audio.set_frame_rate(22050)
                    audio = normalize(audio)
                    play(audio)
            os.remove(wav_path)
            audio_queue.task_done()
            interval = max(0.1, get_interval_func() + random.uniform(-get_variation_func(), get_variation_func()))
            if not stop_event.is_set():
                time.sleep(interval)
        except queue.Empty:
            time.sleep(0.1)
        except Exception as e:
            logger.error(f"Playback error: {e}")
            if os.path.exists(wav_path):
                os.remove(wav_path)

def load_config():
    """Load configuration from file or return defaults."""
    defaults = {
        "ollama_url": DEFAULT_OLLAMA_URL,
        "model": DEFAULT_MODEL,
        "tts_url": DEFAULT_TTS_URL,
        "speaker_id": DEFAULT_SPEAKER_ID,
        "pitch": 0,
        "interval": 2.0,
        "variation": 0
    }
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return {**defaults, **json.load(f)}
    return defaults

def save_config(gui):
    """Save current settings to configuration file."""
    config = {
        "ollama_url": gui.ollama_url_var.get(),
        "model": gui.model_var.get(),
        "tts_url": gui.tts_url_var.get(),
        "speaker_id": gui.speaker_id_var.get(),
        "pitch": gui.pitch_slider.get(),
        "interval": gui.interval_slider.get(),
        "variation": gui.variation_slider.get()
    }
    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)
    logger.info("Configuration saved.")

class InfiniteOracleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Infinite Oracle Control Panel")
        self.state("zoomed")
        self.geometry("800x900")

        # Debug current working directory
        print(f"Current working directory: {os.getcwd()}")

        # Set custom icon with PIL, using bundled resource
        try:
            if getattr(sys, 'frozen', False):  # Running as .exe
                base_path = sys._MEIPASS
            else:  # Running as .py
                base_path = os.path.dirname(__file__)
            icon_path = os.path.join(base_path, "oracle.ico")
            img = Image.open(icon_path)
            icon = ImageTk.PhotoImage(img)
            self.iconphoto(True, icon)
            print(f"Icon loaded successfully from {icon_path}")
        except Exception as e:
            logger.warning(f"Failed to load icon: {e}")

        self.config = load_config()
        self.ollama_url_var = tk.StringVar(value=self.config["ollama_url"])
        self.model_var = tk.StringVar(value=self.config["model"])
        self.tts_url_var = tk.StringVar(value=self.config["tts_url"])
        self.speaker_id_var = tk.StringVar(value=self.config["speaker_id"])
        self.session = None
        self.wisdom_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.start_lock = False  # Lock to prevent Start spam
        self.stop_event = threading.Event()
        self.generator_thread = None
        self.tts_thread = None
        self.playback_thread = None
        self.send_wisdom_queue = queue.Queue(maxsize=10)
        self.send_audio_queue = queue.Queue(maxsize=10)
        self.send_stop_event = threading.Event()
        self.send_tts_thread = None
        self.send_playback_thread = None
        self.send_enabled = True  # Flag to throttle Send button

        self.create_widgets()
        sys.stdout = ConsoleRedirector(self.console_text)

        # Start persistent Send threads
        self.send_tts_thread = threading.Thread(
            target=text_to_speech,
            args=(self.send_wisdom_queue, self.send_audio_queue, self.speaker_id_var.get(), self.pitch_slider.get, self.send_stop_event),
            daemon=True
        )
        self.send_playback_thread = threading.Thread(
            target=play_audio,
            args=(self.send_audio_queue, self.send_stop_event, lambda: 0, lambda: 0, self),
            daemon=True
        )
        self.send_tts_thread.start()
        self.send_playback_thread.start()

    def create_widgets(self):
        self.configure(bg="#2b2b2b")
        tk.Label(self, text="Ollama Server URL:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.ollama_url_entry = tk.Entry(self, textvariable=self.ollama_url_var, width=40)
        self.ollama_url_entry.pack(pady=5)
        tk.Label(self, text="Model Name:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.model_entry = tk.Entry(self, textvariable=self.model_var, width=40)
        self.model_entry.pack(pady=5)
        tk.Label(self, text="Coqui TTS Server URL:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.tts_url_entry = tk.Entry(self, textvariable=self.tts_url_var, width=40)
        self.tts_url_entry.pack(pady=5)
        tk.Label(self, text="Speaker ID (e.g., p267):", bg="#2b2b2b", fg="white").pack(pady=5)
        self.speaker_id_entry = tk.Entry(self, textvariable=self.speaker_id_var, width=40)
        self.speaker_id_entry.pack(pady=5)

        prompt_frame = tk.Frame(self, bg="#2b2b2b")
        prompt_frame.pack(pady=5, padx=10, fill=tk.X)
        tk.Label(prompt_frame, text="System Prompt:", bg="#2b2b2b", fg="white").pack(side=tk.TOP, anchor=tk.W)
        self.system_prompt_entry = tk.Text(prompt_frame, height=10, width=40)
        self.system_prompt_entry.insert(tk.END, SYSTEM_PROMPT)
        self.system_prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.send_button = tk.Button(prompt_frame, text="Send", command=self.send_prompt_action)
        self.send_button.pack(side=tk.RIGHT, padx=5)

        slider_frame = tk.Frame(self, bg="#2b2b2b")
        slider_frame.pack(pady=10)
        pitch_frame = tk.Frame(slider_frame, bg="#2b2b2b")
        pitch_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(pitch_frame, text="Pitch Shift (semitones):", bg="#2b2b2b", fg="white").pack()
        self.pitch_slider = tk.Scale(pitch_frame, from_=-12, to=12, orient=tk.HORIZONTAL)
        self.pitch_slider.set(self.config["pitch"])
        self.pitch_slider.pack()
        interval_frame = tk.Frame(slider_frame, bg="#2b2b2b")
        interval_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(interval_frame, text="Speech Interval (seconds):", bg="#2b2b2b", fg="white").pack()
        self.interval_slider = tk.Scale(interval_frame, from_=0.5, to=10, resolution=0.5, orient=tk.HORIZONTAL)
        self.interval_slider.set(self.config["interval"])
        self.interval_slider.pack()
        variation_frame = tk.Frame(slider_frame, bg="#2b2b2b")
        variation_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(variation_frame, text="Speech Interval Variation (seconds):", bg="#2b2b2b", fg="white").pack()
        self.variation_slider = tk.Scale(variation_frame, from_=0, to=5, resolution=0.5, orient=tk.HORIZONTAL)
        self.variation_slider.set(self.config["variation"])
        self.variation_slider.pack()

        button_frame = tk.Frame(self, bg="#2b2b2b")
        button_frame.pack(pady=10)
        self.start_button = tk.Button(button_frame, text="Start", command=self.start_oracle)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_oracle)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(button_frame, text="Save Config", command=self.save_config_action)
        self.save_button.pack(side=tk.LEFT, padx=5)

        tk.Label(self, text="Console Output:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.console_text = scrolledtext.ScrolledText(self, height=15, width=70, state='disabled', bg="black", fg="green")
        self.console_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

    def enable_send_and_start(self):
        """Re-enable Send, Start, Save Config, and text inputs if not running."""
        if not self.is_running and not self.start_lock:
            self.send_button.config(state=tk.NORMAL)
            self.start_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.ollama_url_entry.config(state=tk.NORMAL)
            self.model_entry.config(state=tk.NORMAL)
            self.tts_url_entry.config(state=tk.NORMAL)
            self.speaker_id_entry.config(state=tk.NORMAL)
            self.system_prompt_entry.config(state=tk.NORMAL)
        self.send_enabled = True

    def save_config_action(self):
        """Wrapper for save_config to prevent overlap with Start."""
        if not self.start_lock:
            save_config(self)

    def verify_model(self, model):
        temp_session = setup_session(OLLAMA_URL)
        payload = {"model": model, "prompt": "test", "stream": False}
        for attempt in range(3):
            try:
                response = temp_session.post(OLLAMA_URL, json=payload, timeout=15)
                response.raise_for_status()
                return True
            except requests.RequestException as e:
                logger.error(f"Model verification attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        messagebox.showerror("Model Error", f"Model '{model}' not found or unavailable on Ollama server after retries.")
        return False

    def start_oracle(self):
        if self.start_lock or self.is_running:
            return  # Ignore if already starting or running
        self.start_lock = True  # Lock to prevent spam
        self.start_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.ollama_url_entry.config(state=tk.DISABLED)
        self.model_entry.config(state=tk.DISABLED)
        self.tts_url_entry.config(state=tk.DISABLED)
        self.speaker_id_entry.config(state=tk.DISABLED)
        self.system_prompt_entry.config(state=tk.DISABLED)

        global OLLAMA_URL, SYSTEM_PROMPT, TTS_SERVER_URL
        OLLAMA_URL = self.ollama_url_var.get()
        model = self.model_var.get()
        SYSTEM_PROMPT = self.system_prompt_entry.get("1.0", tk.END).strip()
        TTS_SERVER_URL = self.tts_url_var.get()
        speaker_id = self.speaker_id_var.get()

        if not all([OLLAMA_URL, model, TTS_SERVER_URL, speaker_id]):
            messagebox.showerror("Input Error", "Please fill all fields.")
            self.start_lock = False
            self.after(0, self.enable_send_and_start)
            return

        self.stop_oracle()

        if not self.verify_model(model):
            self.start_lock = False
            self.after(0, self.enable_send_and_start)
            return

        self.session = setup_session(OLLAMA_URL)
        self.is_running = True
        self.stop_event.clear()
        self.stop_button.config(state=tk.NORMAL, bg="red")
        self.pitch_slider.config(state=tk.DISABLED)
        self.interval_slider.config(state=tk.DISABLED)
        self.variation_slider.config(state=tk.DISABLED)

        self.generator_thread = threading.Thread(
            target=generate_wisdom,
            args=(self.session, self.wisdom_queue, model, self.stop_event),
            daemon=True
        )
        self.tts_thread = threading.Thread(
            target=text_to_speech,
            args=(self.wisdom_queue, self.audio_queue, speaker_id, self.pitch_slider.get, self.stop_event),
            daemon=True
        )
        self.playback_thread = threading.Thread(
            target=play_audio,
            args=(self.audio_queue, self.stop_event, self.interval_slider.get, self.variation_slider.get, self),
            daemon=True
        )

        self.generator_thread.start()
        self.tts_thread.start()
        self.playback_thread.start()

        # Unlock after threads are running
        self.after(500, lambda: setattr(self, 'start_lock', False))  # 500ms debounce

    def stop_oracle(self):
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
                try:
                    self.wisdom_queue.get_nowait()
                except queue.Empty:
                    break
            while not self.audio_queue.empty():
                try:
                    _, wav_path, _ = self.audio_queue.get_nowait()
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                except queue.Empty:
                    break

            if self.session:
                self.session.close()
                self.session = None

            self.start_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.ollama_url_entry.config(state=tk.NORMAL)
            self.model_entry.config(state=tk.NORMAL)
            self.tts_url_entry.config(state=tk.NORMAL)
            self.speaker_id_entry.config(state=tk.NORMAL)
            self.system_prompt_entry.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL, bg="lightgray")
            self.pitch_slider.config(state=tk.NORMAL)
            self.interval_slider.config(state=tk.NORMAL)
            self.variation_slider.config(state=tk.NORMAL)
            logger.info("Oracle stopped.")

    def send_prompt_action(self):
        if not self.send_enabled or self.start_lock:
            return  # Ignore if throttled or starting
        self.send_enabled = False
        self.send_button.config(state=tk.DISABLED)
        self.start_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.ollama_url_entry.config(state=tk.DISABLED)
        self.model_entry.config(state=tk.DISABLED)
        self.tts_url_entry.config(state=tk.DISABLED)
        self.speaker_id_entry.config(state=tk.DISABLED)
        self.system_prompt_entry.config(state=tk.DISABLED)

        global OLLAMA_URL, TTS_SERVER_URL
        OLLAMA_URL = self.ollama_url_var.get()
        model = self.model_var.get()
        prompt = self.system_prompt_entry.get("1.0", tk.END).strip()
        TTS_SERVER_URL = self.tts_url_var.get()
        speaker_id = self.speaker_id_var.get()

        if not all([OLLAMA_URL, model, prompt, TTS_SERVER_URL, speaker_id]):
            messagebox.showwarning("Input Error", "Please fill all fields.")
            self.enable_send_and_start()
            return

        send_session = setup_session(OLLAMA_URL)
        if not self.verify_model(model):
            send_session.close()
            self.enable_send_and_start()
            return

        threading.Thread(
            target=send_prompt,
            args=(send_session, self.send_wisdom_queue, model, prompt, self),
            daemon=True
        ).start()

def main():
    app = InfiniteOracleGUI()
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nThe Infinite Oracle rests... for now.")
