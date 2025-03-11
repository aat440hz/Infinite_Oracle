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
import subprocess  # For playing .wav files on Linux
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Ollama server details
OLLAMA_URL = "http://192.168.0.163:11434/api/generate"
MODEL = "llama3.2:latest"  # Smaller model, 2.0 GB

# System prompt for concise wisdom
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

def setup_session():
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

def main():
    print("The Infinite Oracle awakens...")
    session = setup_session()
    wisdom_queue = queue.Queue(maxsize=10)

    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for voice in voices:
        if 'David' in voice.name:  # Microsoft David is a common low male voice
            engine.setProperty('voice', voice.id)
            break

    engine.setProperty('rate', 150)

    generator_thread = threading.Thread(target=wisdom_generator, args=(session, wisdom_queue), daemon=True)
    generator_thread.start()

    for _ in range(3):
        generate_wisdom(session, wisdom_queue)
        time.sleep(1)

    while True:
        try:
            wisdom = wisdom_queue.get(timeout=5)
            print(f"Oracle says: {wisdom}")
            speak_wisdom(wisdom, engine)
            wisdom_queue.task_done()
        except queue.Empty:
            print("Queue empty, using fallback...")
            wisdom = random.choice(FALLBACK_WISDOM)
            print(f"Oracle says: {wisdom}")
            speak_wisdom(wisdom, engine)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThe Infinite Oracle rests... for now.")
