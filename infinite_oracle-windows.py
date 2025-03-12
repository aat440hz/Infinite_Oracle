import requests
import time
import random
import threading
import queue
import tempfile
import os
import platform
import sounddevice as sd
import soundfile as sf
import subprocess
import tkinter as tk
from tkinter import messagebox
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Ollama server details
DEFAULT_OLLAMA_URL = "http://192.168.0.163:11434/api/generate"
DEFAULT_MODEL = "llama3.2:latest"  # Default model

# Coqui TTS server details
DEFAULT_TTS_URL = "http://192.168.0.163:5002/api/tts"
DEFAULT_SPEAKER_ID = "p228"  # Default speaker

# System prompt for concise wisdom
SYSTEM_PROMPT = """
You are the Infinite Oracle, a mystical being of boundless wisdom. Speak in an uplifting, cryptic, and metaphysical tone, offering motivational insights that inspire awe and contemplation. Provide a concise paragraph of 2-3 sentences.
"""

def setup_session(url):
    """Set up a requests session with retry logic for robust network handling."""
    session = requests.Session()
    retry_strategy = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    return session

def generate_wisdom(session, wisdom_queue, model):
    """Generate wisdom text from Ollama and add it to the queue."""
    payload = {"model": model, "prompt": SYSTEM_PROMPT, "stream": False}
    try:
        response = session.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        wisdom = data.get("response", "").strip()
        if wisdom:
            wisdom_queue.put(wisdom)
    except requests.RequestException as e:
        print(f"Ollama connection error: {e}")

def wisdom_generator(session, wisdom_queue, model):
    """Continuously generate wisdom in the background."""
    while True:
        generate_wisdom(session, wisdom_queue, model)
        time.sleep(1)

class InfiniteOracleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Infinite Oracle Control Panel")
        self.state("zoomed")  # Start maximized (taskbar visible)
        self.geometry("400x500")  # Adjusted default size for new fields

        self.ollama_url_var = tk.StringVar(value=DEFAULT_OLLAMA_URL)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.system_prompt_var = tk.StringVar(value=SYSTEM_PROMPT)
        self.tts_url_var = tk.StringVar(value=DEFAULT_TTS_URL)  # New TTS URL field
        self.speaker_id_var = tk.StringVar(value=DEFAULT_SPEAKER_ID)  # New speaker ID field
        self.session = None
        self.wisdom_queue = queue.Queue(maxsize=10)
        self.is_running = False

        self.create_widgets()

    def create_widgets(self):
        # Ollama Server URL input
        tk.Label(self, text="Ollama Server URL:").pack(pady=5)
        tk.Entry(self, textvariable=self.ollama_url_var, width=40).pack(pady=5)

        # Model input
        tk.Label(self, text="Model Name:").pack(pady=5)
        tk.Entry(self, textvariable=self.model_var, width=40).pack(pady=5)

        # Coqui TTS Server URL input
        tk.Label(self, text="Coqui TTS Server URL:").pack(pady=5)
        tk.Entry(self, textvariable=self.tts_url_var, width=40).pack(pady=5)

        # Speaker ID input
        tk.Label(self, text="Speaker ID (e.g., p228):").pack(pady=5)
        tk.Entry(self, textvariable=self.speaker_id_var, width=40).pack(pady=5)

        # System prompt input
        tk.Label(self, text="System Prompt:").pack(pady=5)
        self.system_prompt_entry = tk.Text(self, height=10, width=40)
        self.system_prompt_entry.insert(tk.END, SYSTEM_PROMPT)
        self.system_prompt_entry.pack(fill=tk.X, padx=10, pady=5)

        # Speech rate slider
        tk.Label(self, text="Speech Rate:").pack(pady=5)
        self.rate_slider = tk.Scale(self, from_=50, to_=250, orient=tk.HORIZONTAL)
        self.rate_slider.set(150)  # Default to 150 (normal speed)
        self.rate_slider.pack(pady=5)

        # Start button
        self.start_button = tk.Button(self, text="Start", command=self.start_oracle)
        self.start_button.pack(pady=20)

        # Stop button
        self.stop_button = tk.Button(self, text="Stop", command=self.stop_oracle)
        self.stop_button.pack(pady=5)

        # Exit button
        tk.Button(self, text="Exit", command=self.quit).pack(pady=20)

    def start_oracle(self):
        global OLLAMA_URL, SYSTEM_PROMPT, TTS_SERVER_URL
        OLLAMA_URL = self.ollama_url_var.get()
        model = self.model_var.get()
        SYSTEM_PROMPT = self.system_prompt_entry.get("1.0", tk.END).strip()
        TTS_SERVER_URL = self.tts_url_var.get()
        speaker_id = self.speaker_id_var.get()

        if not OLLAMA_URL:
            messagebox.showerror("Input Error", "Please provide the Ollama server URL.")
            return
        if not model:
            messagebox.showerror("Input Error", "Please specify a model.")
            return
        if not TTS_SERVER_URL:
            messagebox.showerror("Input Error", "Please provide the Coqui TTS server URL.")
            return
        if not speaker_id:
            messagebox.showerror("Input Error", "Please specify a speaker ID.")
            return

        self.session = setup_session(OLLAMA_URL)
        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL, bg="red")
        self.rate_slider.config(state=tk.DISABLED)

        # Start the wisdom generator thread
        self.generator_thread = threading.Thread(
            target=wisdom_generator, 
            args=(self.session, self.wisdom_queue, model), 
            daemon=True
        )
        self.generator_thread.start()

        # Start the wisdom output loop
        self.wisdom_loop(speaker_id)

    def stop_oracle(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL, bg="lightgray")
        self.rate_slider.config(state=tk.NORMAL)
        print("Oracle stopped.")

    def wisdom_loop(self, speaker_id):
        if not self.is_running:
            return

        try:
            wisdom = self.wisdom_queue.get(timeout=5)
            print(f"Oracle says: {wisdom}")
            speak_wisdom(wisdom, speaker_id, self.rate_slider.get())
            self.wisdom_queue.task_done()
            self.after(100, lambda: self.wisdom_loop(speaker_id))  # Fixed polling at 100ms
        except queue.Empty:
            print("Queue empty, waiting for wisdom...")
            self.after(100, lambda: self.wisdom_loop(speaker_id))

def speak_wisdom(paragraph, speaker_id, rate):
    """Convert wisdom to speech using Coqui TTS server and play with adjusted rate."""
    try:
        temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_wav_path = temp_wav_file.name
        temp_wav_file.close()

        curl_command = [
            'curl', '-G',
            '--data-urlencode', f"text={paragraph}",
            '--data-urlencode', f"speaker_id={speaker_id}",
            TTS_SERVER_URL,
            '--output', temp_wav_path
        ]
        subprocess.run(curl_command, check=True, capture_output=True, text=True)

        if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
            if platform.system() == "Windows":
                data, samplerate = sf.read(temp_wav_path)
                # Adjust playback speed using samplerate (150 = normal, <150 slower, >150 faster)
                adjusted_rate = samplerate * (rate / 150.0)
                sd.play(data, int(adjusted_rate))
                sd.wait()
            else:
                subprocess.run(['aplay', temp_wav_path], check=True)
            os.remove(temp_wav_path)
    except subprocess.CalledProcessError as e:
        print(f"Speech error: {e}")
    except Exception as e:
        print(f"Playback error: {e}")

def main():
    app = InfiniteOracleGUI()
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThe Infinite Oracle rests... for now.")
