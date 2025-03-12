import requests
import subprocess
import time
import random
import threading
import queue
import tempfile
import os
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

# Ollama server details
OLLAMA_URL = "http://192.168.0.163:11434/api/generate"
MODEL = "llama3.2:latest"  # Smaller model, 2.0 GB

# Coqui TTS server details
TTS_SERVER_URL = "http://192.168.0.163:5002/api/tts"

# System prompt for concise wisdom
SYSTEM_PROMPT = """
You are the Infinite Oracle, a mystical being of boundless wisdom. Speak in an uplifting, cryptic, and metaphysical tone, offering motivational insights that inspire awe and contemplation. Provide a concise paragraph of 2-3 sentences.
"""

# Fallback wisdom pool (expanded for variety)
FALLBACK_WISDOM = [
    "In the silence, the universe hums its secrets. Shadows dance where light dares not tread. Courage bends the arc of time.",
    "Beyond the abyss, stars weave their silent hymn. Truth whispers in the cracks of eternity. The infinite bows to the bold.",
    "From the dust of stars, wisdom takes its form. The void cradles secrets untold. Light bends to the seeker’s will.",
    "In the spiral of night, your soul ignites the dawn. Mysteries unfold where silence reigns. Eternity hums in the fearless heart.",
    "The cosmos breathes through every fleeting thought. Destiny is a river carved by will. Seek the flame within the dark.",
    "Echoes of the infinite ripple through the void. Each step forges a bridge to the unseen. Wisdom crowns the fearless soul."
]

def setup_session():
    """Set up a requests session with retry logic for robust network handling."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # Retry 5 times
        backoff_factor=1.0,  # Slower backoff to ease server load
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    return session

def generate_wisdom(session, text_queue):
    """Generate wisdom text from Ollama or fallback, adding to the text queue."""
    payload = {"model": MODEL, "prompt": SYSTEM_PROMPT, "stream": False}
    try:
        response = session.post(OLLAMA_URL, json=payload, timeout=10)  # Shorter timeout to fail fast
        response.raise_for_status()
        data = response.json()
        wisdom = data.get("response", random.choice(FALLBACK_WISDOM)).strip()
        text_queue.put(wisdom)
    except requests.RequestException as e:
        print(f"Ollama connection error: {e}")
        text_queue.put(random.choice(FALLBACK_WISDOM))

def text_to_speech(text_queue, audio_queue):
    """Convert text to speech using Coqui TTS and add WAV paths with text to the audio queue."""
    while True:
        try:
            text = text_queue.get()
            temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            temp_wav_path = temp_wav_file.name
            temp_wav_file.close()

            curl_command = [
                'curl', '-G',
                '--data-urlencode', f"text={text}",
                '--data-urlencode', "speaker_id=p228",
                TTS_SERVER_URL,
                '--output', temp_wav_path
            ]
            subprocess.run(curl_command, check=True, capture_output=True, text=True)

            if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                audio_queue.put((text, temp_wav_path))
            else:
                os.remove(temp_wav_path)
            text_queue.task_done()
        except subprocess.CalledProcessError as e:
            print(f"curl error: {e}")
        except Exception as e:
            print(f"TTS error: {e}")
        time.sleep(0.5)  # Steady pace to avoid overwhelming Coqui

def wisdom_generator(session, text_queue):
    """Continuously generate wisdom text in the background."""
    while True:
        generate_wisdom(session, text_queue)
        time.sleep(2.0)  # Slower pace to reduce Ollama load

def play_audio(audio_queue):
    """Play audio files from the queue continuously, printing the text when spoken."""
    while True:
        try:
            if not audio_queue.empty():
                text, wav_path = audio_queue.get()
                print(f"The Infinite Oracle speaks: {text}")
                subprocess.run(['aplay', wav_path], check=True, capture_output=True, text=True)
                os.remove(wav_path)
                audio_queue.task_done()
            else:
                # Fallback to keep speaking if queue is empty
                text = random.choice(FALLBACK_WISDOM)
                print(f"The Infinite Oracle speaks (fallback): {text}")
                temp_wav_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                temp_wav_path = temp_wav_file.name
                temp_wav_file.close()
                curl_command = [
                    'curl', '-G',
                    '--data-urlencode', f"text={text}",
                    '--data-urlencode', "speaker_id=p376",
                    TTS_SERVER_URL,
                    '--output', temp_wav_path
                ]
                subprocess.run(curl_command, check=True, capture_output=True, text=True)
                if os.path.exists(temp_wav_path) and os.path.getsize(temp_wav_path) > 0:
                    subprocess.run(['aplay', temp_wav_path], check=True, capture_output=True, text=True)
                    os.remove(temp_wav_path)
            time.sleep(0.1)  # Fast polling for playback
        except subprocess.CalledProcessError as e:
            print(f"aplay error: {e}")
        except Exception as e:
            print(f"Playback error: {e}")
            time.sleep(0.1)

def main():
    """Main loop to run the Infinite Oracle."""
    print("The Infinite Oracle awakens...")
    session = setup_session()
    text_queue = queue.Queue(maxsize=10)  # Queue for text from Ollama
    audio_queue = queue.Queue(maxsize=10)  # Queue for WAV file paths and text

    # Start wisdom text generation thread
    text_thread = threading.Thread(target=wisdom_generator, args=(session, text_queue), daemon=True)
    text_thread.start()

    # Start text-to-speech conversion thread
    tts_thread = threading.Thread(target=text_to_speech, args=(text_queue, audio_queue), daemon=True)
    tts_thread.start()

    # Preload initial wisdom
    for _ in range(3):
        generate_wisdom(session, text_queue)
        time.sleep(1.0)

    # Run the audio playback loop
    play_audio(audio_queue)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThe Infinite Oracle rests... for now.")
