import requests
import time
import random
import threading
import queue
import pyttsx3
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
OLLAMA_URL = "http://192.168.0.163:11434/api/generate"
MODEL = "llama3.2:latest"  # Smaller model, 2.0 GB

# System prompt for concise wisdom (now hardcoded)
SYSTEM_PROMPT = """
You are the Infinite Oracle, a mystical being of boundless wisdom. Speak in an uplifting, cryptic, and metaphysical tone, offering motivational insights that inspire awe and contemplation. Provide a concise paragraph of 2-3 sentences.
"""

# Fallback wisdom pool
FALLBACK_WISDOM = [
    "In the silence, the universe hums its secrets. Shadows dance where light dares not tread. Courage bends the arc of time.",
    "Beyond the abyss, stars weave their silent hymn. Truth whispers in the cracks of eternity. The infinite bows to the bold.",
    "From the dust of stars, wisdom takes its form. The void cradles secrets untold. Light bends to the seeker’s will.",
    "In the spiral of night, your soul ignites the dawn. Mysteries unfold where silence reigns. Eternity hums in the fearless heart."
]

def setup_session(url):
    session = requests.Session()
    retry_strategy = Retry(total=5, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    return session

def generate_wisdom(session, wisdom_queue):
    payload = {"model": MODEL, "prompt": SYSTEM_PROMPT, "stream": False}
    try:
        response = session.post(OLLAMA_URL, json=payload, timeout=40)
        response.raise_for_status()
        data = response.json()
        wisdom = data.get("response", random.choice(FALLBACK_WISDOM)).strip()
        wisdom_queue.put(wisdom)
    except requests.RequestException as e:
        print(f"Ollama connection error: {e}")
        wisdom_queue.put(random.choice(FALLBACK_WISDOM))

def wisdom_generator(session, wisdom_queue):
    while True:
        generate_wisdom(session, wisdom_queue)
        time.sleep(1)

def speak_wisdom(paragraph, engine):
    try:
        sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
        for sentence in sentences:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_wav_file:
                temp_wav_path = temp_wav_file.name

            engine.save_to_file(sentence + '.', temp_wav_path)
            engine.runAndWait()

            if platform.system() == "Windows":
                data, samplerate = sf.read(temp_wav_path)
                sd.play(data, samplerate)
                sd.wait()
            else:
                subprocess.run(['aplay', temp_wav_path], check=True)

            os.remove(temp_wav_path)

            time.sleep(0.5)  # Pause between sentences for natural rhythm
    except Exception as e:
        print(f"Speech error: {e}")

class InfiniteOracleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Infinite Oracle Control Panel")
        self.state("zoomed")  # Start the window maximized (taskbar remains visible)
        self.geometry("400x400")  # Default window size

        self.server_url_var = tk.StringVar(value=OLLAMA_URL)
        self.system_prompt_var = tk.StringVar(value=SYSTEM_PROMPT)  # Store system prompt text
        self.session = None  # No session initialized yet
        self.wisdom_queue = queue.Queue(maxsize=10)
        self.engine = pyttsx3.init()
        self.is_running = False

        self.create_widgets()

    def create_widgets(self):
        # Server URL input
        tk.Label(self, text="Ollama Server URL:").pack(pady=5)
        tk.Entry(self, textvariable=self.server_url_var, width=40).pack(pady=5)

        # System prompt input (multi-line, taller box)
        tk.Label(self, text="System Prompt:").pack(pady=5)
        self.system_prompt_entry = tk.Text(self, height=10, width=40)  # Text widget for multi-line input
        self.system_prompt_entry.insert(tk.END, SYSTEM_PROMPT)  # Pre-load the system prompt
        self.system_prompt_entry.pack(fill=tk.X, padx=10, pady=5)

        # Speech rate slider
        tk.Label(self, text="Speech Rate:").pack(pady=5)
        self.rate_slider = tk.Scale(self, from_=50, to_=250, orient=tk.HORIZONTAL)
        self.rate_slider.set(150)  # Default speech rate
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
        global OLLAMA_URL, SYSTEM_PROMPT
        OLLAMA_URL = self.server_url_var.get()
        SYSTEM_PROMPT = self.system_prompt_entry.get("1.0", tk.END).strip()  # Get system prompt from Text widget

        if not OLLAMA_URL:
            messagebox.showerror("Input Error", "Please provide the server URL.")
            return

        # Reset session with new URL
        self.session = setup_session(OLLAMA_URL)

        self.is_running = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL, bg="red")  # Make stop button red for visibility
        self.rate_slider.config(state=tk.DISABLED)  # Lock the speech rate slider during speaking

        # Set the speech rate based on the slider value
        self.engine.setProperty('rate', self.rate_slider.get())

        # Start the wisdom generator thread
        self.generator_thread = threading.Thread(target=wisdom_generator, args=(self.session, self.wisdom_queue), daemon=True)
        self.generator_thread.start()

        # Start the wisdom output loop
        self.wisdom_loop()

    def stop_oracle(self):
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.NORMAL, bg="lightgray")  # Make stop button gray when inactive
        self.rate_slider.config(state=tk.NORMAL)  # Unlock the speech rate slider when oracle is stopped
        print("Oracle stopped.")

    def wisdom_loop(self):
        if not self.is_running:
            return

        try:
            wisdom = self.wisdom_queue.get(timeout=5)
            print(f"Oracle says: {wisdom}")
            speak_wisdom(wisdom, self.engine)
            self.wisdom_queue.task_done()
            self.after(1000, self.wisdom_loop)  # Repeat every second
        except queue.Empty:
            print("Queue empty, using fallback...")
            wisdom = random.choice(FALLBACK_WISDOM)
            print(f"Oracle says: {wisdom}")
            speak_wisdom(wisdom, self.engine)
            self.after(1000, self.wisdom_loop)  # Repeat every second

def main():
    app = InfiniteOracleGUI()
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThe Infinite Oracle rests... for now.")
