import requests
import time
import random
import threading
import queue
import pyttsx3  # Import pyttsx3 for TTS
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
    """Set up a requests session with retry logic for robust network handling."""
    session = requests.Session()
    retry_strategy = Retry(
        total=5,  # Retry 5 times
        backoff_factor=0.5,  # Wait 0.5s, 1s, 2s, 4s, 8s
        status_forcelist=[500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    return session

def generate_wisdom(session, wisdom_queue):
    """Generate a piece of wisdom from Ollama or fallback pool and add it to the queue."""
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
    """Continuously generate wisdom in the background."""
    while True:
        generate_wisdom(session, wisdom_queue)
        time.sleep(1)  # Small delay to avoid overwhelming Ollama

def speak_wisdom(paragraph):
    """Convert a paragraph of wisdom to speech using the built-in Windows TTS engine."""
    try:
        engine = pyttsx3.init()  # Initialize the TTS engine
        
        # Set voice to a low male voice (Microsoft David or similar)
        voices = engine.getProperty('voices')
        for voice in voices:
            if 'David' in voice.name:  # "David" is a common low male voice in Windows
                engine.setProperty('voice', voice.id)
                break

        # Set the speaking rate (optional, slower for more clear speech)
        engine.setProperty('rate', 150)  # Lower value means slower speech

        # Speak each sentence
        sentences = [s.strip() for s in paragraph.split('.') if s.strip()]
        for sentence in sentences:
            engine.say(sentence + '.')  # Queue the sentence to be spoken
            engine.runAndWait()  # Wait for the speech to finish before continuing
    except Exception as e:
        print(f"Speech error: {e}")

def main():
    """Main loop to run the Infinite Oracle."""
    print("The Infinite Oracle awakens...")
    session = setup_session()
    wisdom_queue = queue.Queue(maxsize=10)  # Buffer up to 10 paragraphs

    # Start background wisdom generation
    generator_thread = threading.Thread(target=wisdom_generator, args=(session, wisdom_queue), daemon=True)
    generator_thread.start()

    # Preload initial wisdom
    for _ in range(3):  # Start with 3 paragraphs
        generate_wisdom(session, wisdom_queue)
        time.sleep(1)

    while True:
        try:
            wisdom = wisdom_queue.get(timeout=5)  # Wait up to 5s if queue is empty
            print(f"Oracle says: {wisdom}")
            speak_wisdom(wisdom)
            wisdom_queue.task_done()
        except queue.Empty:
            print("Queue empty, using fallback...")
            wisdom = random.choice(FALLBACK_WISDOM)
            print(f"Oracle says: {wisdom}")
            speak_wisdom(wisdom)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nThe Infinite Oracle rests... for now.")
