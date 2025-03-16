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
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"  # Match first script
DEFAULT_LM_STUDIO_URL = "http://localhost:1234/v1/completions"
DEFAULT_OLLAMA_MODEL = "llama3.2:latest"
DEFAULT_LM_STUDIO_MODEL = "llama-3.2-1b-instruct"
DEFAULT_TTS_URL = "http://localhost:5002/api/tts"  # Match first script
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

# Global playback lock
playback_lock = threading.Lock()

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

def setup_session(url):
    session = requests.Session()
    retry_strategy = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])  # Match first script
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    return session

def ping_server(server_url, server_type, model):
    session = setup_session(server_url)
    try:
        base_url = server_url.rsplit('/api/generate', 1)[0] if server_type == "Ollama" else server_url.rsplit('/v1/completions', 1)[0]
        ping_url = base_url + ("/api/tags" if server_type == "Ollama" else "/v1/models")
        response = session.get(ping_url, timeout=5)
        response.raise_for_status()
        
        payload = (
            {"model": model, "prompt": "test", "stream": False}
            if server_type == "Ollama"
            else {"model": model, "prompt": "test", "max_tokens": 10, "temperature": 0.7}
        )
        response = session.post(server_url, json=payload, timeout=15)  # Match first script's verification timeout
        response.raise_for_status()
        
        logger.info(f"{server_type} server at {server_url} is reachable with model '{model}'.")
        return True, ""
    except requests.RequestException as e:
        return False, f"{server_type} error at {server_url}: {str(e)}"
    finally:
        session.close()

def generate_wisdom(session, wisdom_queue, model, server_type, stop_event, get_request_interval_func):
    while not stop_event.is_set():
        payload = (
            {"model": model, "prompt": SYSTEM_PROMPT, "stream": False}
            if server_type == "Ollama"
            else {"model": model, "prompt": SYSTEM_PROMPT, "max_tokens": 300, "temperature": 0.7}
        )
        try:
            response = session.post(SERVER_URL, json=payload, timeout=60)  # Match first script
            response.raise_for_status()
            data = response.json()
            wisdom = (
                data.get("response", "").strip() if server_type == "Ollama"
                else data.get("choices", [{}])[0].get("text", "").strip()
            )
            if wisdom and not stop_event.is_set():
                logger.info(f"Generated wisdom (length: {len(wisdom)} chars): {wisdom}")
                print(f"The Infinite Oracle speaks: {wisdom}", end="\n\n")
                wisdom_queue.put(wisdom)
        except requests.RequestException as e:
            logger.error(f"{server_type} connection error: {e}")
            print(f"{server_type} error: {str(e)}. Retrying in {get_request_interval_func()}s...")
        interval = max(0.1, get_request_interval_func())
        time.sleep(interval)

def send_prompt(session, wisdom_queue, model, server_type, prompt, gui):
    payload = (
        {"model": model, "prompt": prompt, "stream": False}
        if server_type == "Ollama"
        else {"model": model, "prompt": prompt, "max_tokens": 300, "temperature": 0.7}
    )
    try:
        response = session.post(SERVER_URL, json=payload, timeout=60)  # Match first script
        response.raise_for_status()
        data = response.json()
        wisdom = (
            data.get("response", "").strip() if server_type == "Ollama"
            else data.get("choices", [{}])[0].get("text", "").strip()
        )
        if wisdom:
            logger.info(f"Generated wisdom (length: {len(wisdom)} chars): {wisdom}")
            print(f"The Infinite Oracle speaks: {wisdom}", end="\n\n")
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
                    audio = normalize(audio)
                    logger.info(f"Playing audio for: {wisdom[:50]}...")
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
            "interval": 2.0,
            "variation": 0,
            "request_interval": 1.0
        },
        "LM Studio": {
            "server_type": "LM Studio",
            "server_url": DEFAULT_LM_STUDIO_URL,
            "model": DEFAULT_LM_STUDIO_MODEL,
            "tts_url": DEFAULT_TTS_URL,
            "speaker_id": DEFAULT_SPEAKER_ID,
            "pitch": 0,
            "interval": 2.0,
            "variation": 0,
            "request_interval": 1.0
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
        "interval": gui.interval_slider.get(),
        "variation": gui.variation_slider.get(),
        "request_interval": gui.request_interval_slider.get()
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
        self.server_type_var = tk.StringVar(value="Ollama")  # Default to Ollama to match first script
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

        self.create_widgets()
        sys.stdout = ConsoleRedirector(self.console_text)
        self.server_type_var.trace("w", self.update_from_config)

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

    def create_widgets(self):
        self.configure(bg="#2b2b2b")
        tk.Label(self, text="Server Type:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.server_type_menu = tk.OptionMenu(self, self.server_type_var, "Ollama", "LM Studio")
        self.server_type_menu.pack(pady=5)
        tk.Label(self, text="Server URL:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.server_url_entry = tk.Entry(self, textvariable=self.server_url_var, width=40)
        self.server_url_entry.pack(pady=5)
        self.server_url_entry.bind("<KeyRelease>", lambda e: setattr(self, 'url_modified', True))
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
        self.pitch_slider.set(self.config["Ollama"]["pitch"])
        self.pitch_slider.pack()
        interval_frame = tk.Frame(slider_frame, bg="#2b2b2b")
        interval_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(interval_frame, text="Speech Interval (Start mode, seconds):", bg="#2b2b2b", fg="white").pack()
        self.interval_slider = tk.Scale(interval_frame, from_=0.5, to=10, resolution=0.5, orient=tk.HORIZONTAL)
        self.interval_slider.set(self.config["Ollama"]["interval"])
        self.interval_slider.pack()
        variation_frame = tk.Frame(slider_frame, bg="#2b2b2b")
        variation_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(variation_frame, text="Speech Interval Variation (Start mode, seconds):", bg="#2b2b2b", fg="white").pack()
        self.variation_slider = tk.Scale(variation_frame, from_=0, to=5, resolution=0.5, orient=tk.HORIZONTAL)
        self.variation_slider.set(self.config["Ollama"]["variation"])
        self.variation_slider.pack()
        request_interval_frame = tk.Frame(slider_frame, bg="#2b2b2b")
        request_interval_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(request_interval_frame, text="Request Interval (Start mode, seconds):", bg="#2b2b2b", fg="white").pack()
        self.request_interval_slider = tk.Scale(request_interval_frame, from_=0.5, to=60, resolution=0.5, orient=tk.HORIZONTAL)
        self.request_interval_slider.set(self.config["Ollama"]["request_interval"])
        self.request_interval_slider.pack()

        button_frame = tk.Frame(self, bg="#2b2b2b")
        button_frame.pack(pady=10)
        self.start_button = tk.Button(button_frame, text="Start", command=self.start_oracle)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_oracle)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        self.save_button = tk.Button(button_frame, text="Save Config", command=self.save_config_action)
        self.save_button.pack(side=tk.LEFT, padx=5)

        tk.Label(self, text="Console Output:", bg="#2b2b2b", fg="white").pack(pady=5)
        self.console_text = scrolledtext.ScrolledText(self, height=15, width=100, state='disabled', bg="black", fg="green")
        self.console_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

    def update_from_config(self, *args):
        self.config = load_config()
        server_type = self.server_type_var.get()
        config = self.config[server_type]
        self.server_url_var.set(config["server_url"])
        self.model_var.set(config["model"])
        self.tts_url_var.set(config["tts_url"])
        self.speaker_id_var.set(config["speaker_id"])
        self.pitch_slider.set(config["pitch"])
        self.interval_slider.set(config["interval"])
        self.variation_slider.set(config["variation"])
        self.request_interval_slider.set(config["request_interval"])
        self.url_modified = False
        logger.info(f"Loaded config for {server_type} from {CONFIG_FILE}")

    def disable_controls(self):
        self.start_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.DISABLED)
        self.server_type_menu.config(state=tk.DISABLED)
        self.server_url_entry.config(state=tk.DISABLED)
        self.model_entry.config(state=tk.DISABLED)
        self.tts_url_entry.config(state=tk.DISABLED)
        self.speaker_id_entry.config(state=tk.DISABLED)
        self.system_prompt_entry.config(state=tk.DISABLED)
        self.request_interval_slider.config(state=tk.DISABLED)
        self.update()

    def enable_send_and_start(self):
        if not self.is_running and not self.start_lock:
            self.send_button.config(state=tk.NORMAL)
            self.start_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.server_type_menu.config(state=tk.NORMAL)
            self.server_url_entry.config(state=tk.NORMAL)
            self.model_entry.config(state=tk.NORMAL)
            self.tts_url_entry.config(state=tk.NORMAL)
            self.speaker_id_entry.config(state=tk.NORMAL)
            self.system_prompt_entry.config(state=tk.NORMAL)
            self.request_interval_slider.config(state=tk.NORMAL)
            self.send_enabled = True
        self.update()

    def save_config_action(self):
        if not self.start_lock and self.send_enabled:
            self.disable_controls()
            self.after(100, lambda: save_config(self))
            self.after(150, self.enable_send_and_start)

    def verify_server(self, server_url, server_type, model):
        max_attempts = 3  # Match first script
        for attempt in range(max_attempts):
            is_alive, error_msg = ping_server(server_url, server_type, model)
            if is_alive:
                return True
            logger.error(f"Attempt {attempt + 1}/{max_attempts}: {error_msg}")
            if attempt < max_attempts - 1:
                time.sleep(2)
        messagebox.showerror("Server Error", f"Failed to connect to {server_type} at {server_url} after {max_attempts} attempts: {error_msg}")
        return False

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

        self.session = setup_session(SERVER_URL)
        self.is_running = True
        self.stop_event.clear()
        self.stop_button.config(state=tk.NORMAL, bg="red")
        self.pitch_slider.config(state=tk.DISABLED)
        self.interval_slider.config(state=tk.DISABLED)
        self.variation_slider.config(state=tk.DISABLED)
        self.request_interval_slider.config(state=tk.DISABLED)

        self.generator_thread = threading.Thread(
            target=generate_wisdom,
            args=(self.session, self.wisdom_queue, model, server_type, self.stop_event, self.request_interval_slider.get),
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

            self.start_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.save_button.config(state=tk.NORMAL)
            self.server_type_menu.config(state=tk.NORMAL)
            self.server_url_entry.config(state=tk.NORMAL)
            self.model_entry.config(state=tk.NORMAL)
            self.tts_url_entry.config(state=tk.NORMAL)
            self.speaker_id_entry.config(state=tk.NORMAL)
            self.system_prompt_entry.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL, bg="lightgray")
            self.pitch_slider.config(state=tk.NORMAL)
            self.interval_slider.config(state=tk.NORMAL)
            self.variation_slider.config(state=tk.NORMAL)
            self.request_interval_slider.config(state=tk.NORMAL)
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

        send_session = setup_session(SERVER_URL)
        if not self.verify_server(SERVER_URL, server_type, model):
            send_session.close()
            self.send_enabled = True
            self.after(0, self.enable_send_and_start)
            return

        threading.Thread(
            target=send_prompt,
            args=(send_session, self.send_wisdom_queue, model, server_type, prompt, self),
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
