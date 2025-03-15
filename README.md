# Infinite Oracle
The Infinite Oracle is a mystical, AI-driven voice of boundless wisdom, weaving cryptic and uplifting insights in real-time on Windows. Powered by Ollama for text generation and Coqui TTS for speech synthesis, it speaks truths to inspire awe and contemplation through an interactive GUI.

## Features
- **Boundless Wisdom:** Generates concise, metaphysical paragraphs via Ollama.
- **Ethereal Voice:** Converts text to speech with Coqui TTS, configurable via speaker ID.
- **Interactive GUI:** Start, Stop, Send, and sliders for pitch, interval, and variation.
- **Local Deployment:** Run Ollama and Coqui TTS locally—no cloud needed.

## Prerequisites
- **Ollama Server:** For wisdom generation.
- **Coqui TTS Server:** For speech synthesis.

## Setup Instructions

### Running Locally
Host Ollama and Coqui TTS on your Windows machine:

#### Ollama
1. Download `ollama.exe` from [ollama.com](https://ollama.com/).
2. Open a terminal (e.g., Command Prompt or PowerShell):

   ``ollama.exe pull llama3.2:latest``

   ``ollama.exe serve``
4. Runs at `http://localhost:11434` by default.

#### Coqui TTS
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop) for Windows.
2. Open a terminal and run:
   ``docker run -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts``
3. Inside the container:
   ``python3 TTS/server/server.py --model_name tts_models/en/ljspeech/tacotron2-DDC``
4. Access at `http://localhost:5002`.

### Installation
1. Download the latest `InfiniteOracle.exe` from the [Releases](https://github.com/yourusername/infinite-oracle/releases) page.
2. Place it in a folder of your choice (e.g., `C:\InfiniteOracle`).

### Usage
1. Ensure Ollama (`http://localhost:11434`) and Coqui TTS (`http://localhost:5002`) are running.
2. Double-click `InfiniteOracle.exe` to launch.
