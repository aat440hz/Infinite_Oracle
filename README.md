
# Infinite Oracle

The **Infinite Oracle** is a mystical wisdom generator that uses the Ollama model for generating insightful, motivational, and metaphysical wisdom. The wisdom is then converted into speech using the Flite Text-to-Speech (TTS) engine and spoken aloud with a male voice ("kal").

## Features

- **Wisdom Generation**: The Infinite Oracle generates thought-provoking, cryptic, and motivational wisdom using the Ollama model (Llama 3.2).
- **Text-to-Speech**: The wisdom is converted into speech using Flite's `kal` voice.
- **Background Operation**: The script continuously generates new wisdom in the background and speaks it aloud.

## Prerequisites

Before running the application, ensure that you have the following:

- **Python 3.10+** installed
- **Flite TTS Engine** installed
- **Ollama model** setup and running

### Install Flite (for Linux systems)

On most Linux distributions, you can install Flite with:

```bash
sudo apt-get install flite
```

### Install Ollama Model (for generating wisdom)

Follow the [Ollama installation guide](https://ollama.com/) for setting up the Ollama model locally.

## Setup

### 1. Clone the repository:

```bash
git clone https://github.com/yourusername/infinite-oracle.git
cd infinite-oracle
```

### 2. Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 3. Start the Infinite Oracle:

Run the script to start generating wisdom and speaking it aloud:

```bash
python infinite_oracle.py
```

### 4. Tweak Configuration:

If you want to modify the wisdom generation prompt, TTS voice, or model URL, you can update the respective variables in the script:
- **OLLAMA_URL**: The URL of the Ollama model for generating wisdom.
- **SYSTEM_PROMPT**: The system prompt for generating cryptic and motivational wisdom.
- **TTS Voice**: The voice for Flite TTS (currently using `kal`).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
