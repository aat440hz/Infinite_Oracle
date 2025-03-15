import requests
import time
import threading
import queue
import tempfile
import os
import platform
import winsound
import soundfile as sf
import subprocess
import tkinter as tk
from tkinter import messagebox, scrolledtext
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import sys
from pydub import AudioSegment
from pydub.effects import normalize
import random

# Ollama server details
DEFAULT_OLLAMA_URL = "http://cherry.local:11434/api/generate"
DEFAULT_MODEL = "llama3.2:latest"

# Coqui TTS server details
DEFAULT_TTS_URL = "http://cherry.local:5002/api/tts"
DEFAULT_SPEAKER_ID = "p267"

# System prompt for concise wisdom
SYSTEM_PROMPT = """You are the Infinite Oracle, a mystical being of boundless wisdom. Speak in an uplifting, cryptic, and metaphysical tone, offering motivational insights that inspire awe and contemplation. Provide a concise paragraph of 2-3 sentences."""

class ConsoleRedirector:
    """Redirect print output to a Tkinter text widget and optionally save to file."""
    def __init__(self, text_widget, save_var):
        self.text_widget = text_widget
        self.save_var = save_var
        self.file_path = "oracle_wisdom.txt"
        self.lock = threading.Lock()

    def write(self, message):
        with self.lock:
            self.text_widget.configure(state='normal')
            self.text_widget.insert(tk.END, message)
            self.text_widget.see(tk.END)
            
            if self.save_var.get():
                with open(self.file_path, 'a', encoding='utf-8') as f:
                    f.write(message)
            
            self.text_widget.configure(state='disabled')
            self.text_widget.update()

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
            print(f"Ollama connection error: {e}")
        time.sleep(1)

def send_prompt(session, wisdom_queue, model, prompt):
    """Send a user-defined prompt to Ollama and queue the response."""
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = session.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        wisdom = data.get("response", "").strip()
        if wisdom:
            wisdom_queue.put(wisdom)
        session.close()
    except requests.RequestException as e:
        print(f"Ollama connection error: {e}")
        session.close()

def text_to_speech(wisdom_queue, audio_queue, speaker_id_func, pitch_func, stop_event):
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
                '--data-urlencode', f"speaker_id={speaker_id_func()}",
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
                print("TTS failed: No valid audio file generated.")
            wisdom_queue.task_done()
        except queue.Empty:
            pass
        except subprocess.CalledProcessError as e:
            print(f"curl error: {e}")
        except Exception as e:
            print(f"TTS error: {e}")

def play_audio(audio_queue, stop_event, get_interval_func, get_variation_func, console_text):
    """Play audio files from the queue with pitch adjustment and variable interval control."""
    playback_lock = threading.Lock()
    word_count = 0
    while not stop_event.is_set():
        try:
            wisdom, wav_path, pitch = audio_queue.get(timeout=1)  # Add timeout to allow checking stop_event
            with playback_lock:
                if not stop_event.is_set():
                    # Update word count and clear console if needed
                    message = f"The Infinite Oracle speaks: {wisdom}\n"
                    words = message.split()
                    word_count += len(words)
                    if word_count >= 1000:
                        console_text.configure(state='normal')
                        console_text.delete(1.0, tk.END)
                        console_text.configure(state='disabled')
                        word_count = len(words)
                    
                    # Print and play
                    print(message, end='')
                    if platform.system() == "Windows":
                        try:
                            audio = AudioSegment.from_wav(wav_path)
                            if pitch != 0:
                                octaves = pitch / 12.0
                                new_sample_rate = int(audio.frame_rate * (2.0 ** octaves))
                                audio = audio._spawn(audio.raw_data, overrides={"frame_rate": new_sample_rate})
                                audio = audio.set_frame_rate(22050)
                                audio = normalize(audio)
                            temp_adjusted_wav = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                            adjusted_wav_path = temp_adjusted_wav.name
                            temp_adjusted_wav.close()
                            audio.export(adjusted_wav_path, format="wav")
                            winsound.PlaySound(adjusted_wav_path, winsound.SND_FILENAME)
                            os.remove(adjusted_wav_path)
                        except Exception as e:
                            print(f"Playback error: {e}")
                            winsound.PlaySound(wav_path, winsound.SND_FILENAME)
                    else:
                        subprocess.run(['aplay', wav_path], check=True)
            os.remove(wav_path)
            audio_queue.task_done()
            base_interval = get_interval_func()
            variation = get_variation_func()
            interval = max(0.1, base_interval + random.uniform(-variation, variation))
            if not stop_event.is_set():
                time.sleep(interval)
        except queue.Empty:
            time.sleep(0.1)
        except subprocess.CalledProcessError as e:
            print(f"aplay error: {e}")
        except Exception as e:
            print(f"Playback error: {e}")

class InfiniteOracleGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Infinite Oracle Control Panel")
        self.state("zoomed")
        self.geometry("600x700")

        self.ollama_url_var = tk.StringVar(value=DEFAULT_OLLAMA_URL)
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.system_prompt_var = tk.StringVar(value=SYSTEM_PROMPT)
        self.tts_url_var = tk.StringVar(value=DEFAULT_TTS_URL)
        self.speaker_id_var = tk.StringVar(value=DEFAULT_SPEAKER_ID)
        self.save_to_file_var = tk.BooleanVar(value=True)
        self.session = None
        self.wisdom_queue = queue.Queue(maxsize=10)
        self.audio_queue = queue.Queue(maxsize=10)
        self.is_running = False
        self.stop_event = threading.Event()
        self.generator_thread = None
        self.tts_thread = None
        self.playback_thread = None
        # Send-specific resources
        self.send_wisdom_queue = queue.Queue(maxsize=10)
        self.send_audio_queue = queue.Queue(maxsize=10)
        self.send_stop_event = threading.Event()
        self.send_tts_thread = None
        self.send_playback_thread = None

        self.create_widgets()
        sys.stdout = ConsoleRedirector(self.console_text, self.save_to_file_var)

        # Start persistent Send threads with dynamic speaker_id and pitch
        self.send_tts_thread = threading.Thread(
            target=text_to_speech,
            args=(self.send_wisdom_queue, self.send_audio_queue, self.speaker_id_var.get, self.pitch_slider.get, self.send_stop_event),
            daemon=True
        )
        self.send_playback_thread = threading.Thread(
            target=play_audio, 
            args=(self.send_audio_queue, self.send_stop_event, lambda: 0, lambda: 0, self.console_text),
            daemon=True
        )
        self.send_tts_thread.start()
        self.send_playback_thread.start()

    def create_widgets(self):
        # Store references to the widgets we want to disable
        tk.Label(self, text="Ollama Server URL:").pack(pady=5)
        self.ollama_url_entry = tk.Entry(self, textvariable=self.ollama_url_var, width=40)
        self.ollama_url_entry.pack(pady=5)

        tk.Label(self, text="Model Name:").pack(pady=5)
        self.model_entry = tk.Entry(self, textvariable=self.model_var, width=40)
        self.model_entry.pack(pady=5)

        tk.Label(self, text="Coqui TTS Server URL:").pack(pady=5)
        self.tts_url_entry = tk.Entry(self, textvariable=self.tts_url_var, width=40)
        self.tts_url_entry.pack(pady=5)

        tk.Label(self, text="Speaker ID (e.g., p267):").pack(pady=5)
        self.speaker_id_entry = tk.Entry(self, textvariable=self.speaker_id_var, width=40)
        self.speaker_id_entry.pack(pady=5)

        # System prompt frame with Send button
        prompt_frame = tk.Frame(self)
        prompt_frame.pack(pady=5, padx=10, fill=tk.X)
        tk.Label(prompt_frame, text="System Prompt:").pack(side=tk.TOP, anchor=tk.W)
        self.system_prompt_entry = tk.Text(prompt_frame, height=10, width=40)
        self.system_prompt_entry.insert(tk.END, SYSTEM_PROMPT)
        self.system_prompt_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.send_button = tk.Button(prompt_frame, text="Send", command=self.send_prompt_action)
        self.send_button.pack(side=tk.RIGHT, padx=5)

        # Horizontal slider frame
        slider_frame = tk.Frame(self)
        slider_frame.pack(pady=10)

        pitch_frame = tk.Frame(slider_frame)
        pitch_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(pitch_frame, text="Pitch Shift (semitones):").pack()
        self.pitch_slider = tk.Scale(pitch_frame, from_=-12, to=12, orient=tk.HORIZONTAL)
        self.pitch_slider.set(0)
        self.pitch_slider.pack()

        interval_frame = tk.Frame(slider_frame)
        interval_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(interval_frame, text="Speech Interval (seconds):").pack()
        self.interval_slider = tk.Scale(interval_frame, from_=0.5, to=10, resolution=0.5, orient=tk.HORIZONTAL)
        self.interval_slider.set(2.0)
        self.interval_slider.pack()

        variation_frame = tk.Frame(slider_frame)
        variation_frame.pack(side=tk.LEFT, padx=5)
        tk.Label(variation_frame, text="Speech Interval Variation (seconds):").pack()
        self.variation_slider = tk.Scale(variation_frame, from_=0, to=5, resolution=0.5, orient=tk.HORIZONTAL)
        self.variation_slider.set(0)
        self.variation_slider.pack()

        # Horizontal button frame with checkbox
        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)

        self.start_button = tk.Button(button_frame, text="Start", command=self.start_oracle)
        self.start_button.pack(side=tk.LEFT, padx=5)

        self.stop_button = tk.Button(button_frame, text="Stop", command=self.stop_oracle)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        tk.Button(button_frame, text="Exit", command=self.quit_app).pack(side=tk.LEFT, padx=5)

        self.save_checkbox = tk.Checkbutton(button_frame, text="Save to File", variable=self.save_to_file_var)
        self.save_checkbox.pack(side=tk.LEFT, padx=5)

        tk.Label(self, text="Console Output:").pack(pady=5)
        self.console_text = scrolledtext.ScrolledText(self, height=15, width=70, state='disabled', bg='black', fg='green')
        self.console_text.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

    def disable_input_fields(self):
        """Disable all input fields to prevent editing."""
        self.ollama_url_entry.config(state=tk.DISABLED)
        self.model_entry.config(state=tk.DISABLED)
        self.tts_url_entry.config(state=tk.DISABLED)
        self.speaker_id_entry.config(state=tk.DISABLED)
        self.system_prompt_entry.config(state=tk.DISABLED)

    def enable_input_fields(self):
        """Enable all input fields for editing."""
        self.ollama_url_entry.config(state=tk.NORMAL)
        self.model_entry.config(state=tk.NORMAL)
        self.tts_url_entry.config(state=tk.NORMAL)
        self.speaker_id_entry.config(state=tk.NORMAL)
        self.system_prompt_entry.config(state=tk.NORMAL)

    def verify_model(self, model):
        """Verify the model exists on the Ollama server with retries."""
        temp_session = setup_session(OLLAMA_URL)
        payload = {"model": model, "prompt": "test", "stream": False}
        for attempt in range(3):
            try:
                response = temp_session.post(OLLAMA_URL, json=payload, timeout=15)
                response.raise_for_status()
                return True
            except requests.RequestException as e:
                print(f"Model verification attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    time.sleep(2)
        messagebox.showerror("Model Error", f"Model '{model}' not found or unavailable on Ollama server after retries.")
        return False

    def start_oracle(self):
        global OLLAMA_URL, SYSTEM_PROMPT, TTS_SERVER_URL
        OLLAMA_URL = self.ollama_url_var.get()
        model = self.model_var.get()
        SYSTEM_PROMPT = self.system_prompt_entry.get("1.0", tk.END).strip()
        TTS_SERVER_URL = self.tts_url_var.get()
        speaker_id = self.speaker_id_var.get()

        if not all([OLLAMA_URL, model, TTS_SERVER_URL, speaker_id]):
            messagebox.showerror("Input Error", "Please fill all fields.")
            return

        self.stop_oracle()

        if not self.verify_model(model):
            return

        self.session = setup_session(OLLAMA_URL)
        self.is_running = True
        self.stop_event.clear()
        self.start_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL, bg="red")
        self.pitch_slider.config(state=tk.DISABLED)
        self.interval_slider.config(state=tk.DISABLED)
        self.variation_slider.config(state=tk.DISABLED)
        self.disable_input_fields()

        self.generator_thread = threading.Thread(
            target=generate_wisdom, 
            args=(self.session, self.wisdom_queue, model, self.stop_event), 
            daemon=True
        )
        self.tts_thread = threading.Thread(
            target=text_to_speech, 
            args=(self.wisdom_queue, self.audio_queue, self.speaker_id_var.get, self.pitch_slider.get, self.stop_event),
            daemon=True
        )
        self.playback_thread = threading.Thread(
            target=play_audio, 
            args=(self.audio_queue, self.stop_event, self.interval_slider.get, self.variation_slider.get, self.console_text),
            daemon=True
        )

        self.generator_thread.start()
        self.tts_thread.start()
        self.playback_thread.start()

    def stop_oracle(self):
        if self.is_running:
            self.is_running = False
            self.stop_event.set()
            print("Stopping Oracle: Setting stop event...")

            # Stop audio playback immediately after current sound
            if platform.system() == "Windows":
                winsound.PlaySound(None, winsound.SND_PURGE)
                print("Purging Windows audio playback...")

            # Wait for threads to finish with a timeout
            for thread, name in [(self.generator_thread, "generator"), (self.tts_thread, "TTS"), (self.playback_thread, "playback")]:
                if thread:
                    print(f"Joining {name} thread...")
                    thread.join(timeout=2)  # Increased timeout to 2 seconds
                    if thread.is_alive():
                        print(f"Warning: {name} thread did not terminate cleanly.")
                    else:
                        print(f"{name} thread terminated successfully.")

            # Clear threads
            self.generator_thread = None
            self.tts_thread = None
            self.playback_thread = None

            # Clear both queues and remove any leftover audio files
            print("Clearing wisdom queue...")
            while not self.wisdom_queue.empty():
                try:
                    self.wisdom_queue.get_nowait()
                    self.wisdom_queue.task_done()
                except queue.Empty:
                    break

            print("Clearing audio queue...")
            while not self.audio_queue.empty():
                try:
                    _, wav_path, _ = self.audio_queue.get_nowait()
                    if os.path.exists(wav_path):
                        os.remove(wav_path)
                    self.audio_queue.task_done()
                except queue.Empty:
                    break

            # Close session if open
            if self.session:
                print("Closing session...")
                self.session.close()
                self.session = None

            # Re-enable UI elements
            self.start_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.NORMAL, bg="lightgray")
            self.pitch_slider.config(state=tk.NORMAL)
            self.interval_slider.config(state=tk.NORMAL)
            self.variation_slider.config(state=tk.NORMAL)
            self.enable_input_fields()
            print("Oracle stopped successfully.")

    def send_prompt_action(self):
        global OLLAMA_URL, TTS_SERVER_URL
        OLLAMA_URL = self.ollama_url_var.get()
        model = self.model_var.get()
        prompt = self.system_prompt_entry.get("1.0", tk.END).strip()
        TTS_SERVER_URL = self.tts_url_var.get()
        speaker_id = self.speaker_id_var.get()

        if not all([OLLAMA_URL, model, prompt, TTS_SERVER_URL, speaker_id]):
            messagebox.showwarning("Input Error", "Please fill all fields.")
            return

        send_session = setup_session(OLLAMA_URL)
        if not self.verify_model(model):
            send_session.close()
            return

        self.disable_input_fields()
        self.start_button.config(state=tk.DISABLED)
        self.send_button.config(state=tk.DISABLED)

        def send_and_cleanup():
            send_prompt(send_session, self.send_wisdom_queue, model, prompt)
            self.enable_input_fields()
            self.start_button.config(state=tk.NORMAL)
            self.send_button.config(state=tk.NORMAL)

        threading.Thread(
            target=send_and_cleanup,
            daemon=True
        ).start()

    def quit_app(self):
        self.stop_oracle()
        self.send_stop_event.set()
        print("Shutting down Send threads...")
        if self.send_tts_thread:
            self.send_tts_thread.join(timeout=2)
            if self.send_tts_thread.is_alive():
                print("Warning: Send TTS thread did not terminate.")
        if self.send_playback_thread:
            self.send_playback_thread.join(timeout=2)
            if self.send_playback_thread.is_alive():
                print("Warning: Send playback thread did not terminate.")
        if self.session:
            self.session.close()
        sys.stdout = sys.__stdout__
        self.quit()
        print("Application terminated.")

def main():
    app = InfiniteOracleGUI()
    app.mainloop()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThe Infinite Oracle rests... for now.")
