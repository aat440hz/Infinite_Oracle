# Infinite Oracle

The **Infinite Oracle** is a mystical wisdom generator that uses the Ollama model for generating insightful, motivational, and metaphysical wisdom. The wisdom is then converted into speech using a Text-to-Speech (TTS) engine and spoken aloud with a male voice.

## Features

- **Wisdom Generation**: The Infinite Oracle generates thought-provoking, cryptic, and motivational wisdom using the Ollama model (Llama 3.2).
- **Text-to-Speech**: The wisdom is converted into speech using:
  - **Linux**: Flite's `kal` voice
  - **Windows**: Built-in Windows TTS for a deep male voice with improved playback using `sounddevice` and `soundfile`
- **Background Operation**: The script continuously generates new wisdom in the background and speaks it aloud.

## Prerequisites

Before running the application, ensure that you have the following:

- **Python 3.10+** installed
- **Flite TTS Engine** installed (for Linux)
- **Ollama model** setup and running

## Installation

### **Linux Installation**

1. **Install Flite TTS Engine:**

```bash
sudo apt-get install flite
```

2. **Install Python Dependencies:**

```bash
pip install -r requirements.txt
```

3. **Start the Infinite Oracle:**

```bash
python infinite_oracle.py
```

### **Windows Installation**

1. **Install Python 3.10+**
   - Download from [python.org](https://www.python.org/downloads/) and ensure you check the box for **"Add Python to PATH"** during installation.

2. **Install Required Libraries:**

```powershell
pip install requests pyttsx3 sounddevice soundfile
```

3. **Start the Infinite Oracle:**

```powershell
python infinite_oracle.py
```

### **Configure the Script**

If you want to modify the wisdom generation prompt, TTS voice, or model URL, you can update the respective variables in the script:
- **OLLAMA_URL**: The URL of the Ollama model for generating wisdom.
- **SYSTEM_PROMPT**: The system prompt for generating cryptic and motivational wisdom.
- **TTS Voice (Windows Only)**: The script uses the built-in `pyttsx3` library, which defaults to the system's deep male voice for a rich tone.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
