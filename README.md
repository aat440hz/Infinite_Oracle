# Infinite Oracle

Welcome to the Infinite Oracle—a mystical standalone application that channels boundless wisdom. Powered by AI language models and text-to-speech, this executable delivers motivational insights with a metaphysical flair. This README will guide you through setting up the required services on Windows using Docker Desktop for Coqui TTS, and standalone installers for Ollama and LM Studio.

## Features
- **Mystical Wisdom**: Generate concise, awe-inspiring paragraphs.
- **Customizable Voice**: Adjust pitch and reverb for a cosmic vibe.
- **Interactive GUI**: Control the Oracle with a sleek interface.
- **Recording**: Save the Oracle’s voice as WAV files in `OracleRecordings`.
- **Dual AI Support**: Works with Ollama or LM Studio for text generation.

## Prerequisites
- Windows 10/11 (64-bit)
- Internet connection (for downloads and API calls)
- ~5 GB free disk space (for Docker, models, and the `.exe`)

## Setup Instructions

### 1. Install Docker Desktop
Docker runs the Coqui TTS server, which powers the Oracle’s voice at `http://localhost:5002/api/tts`.

1. **Download Docker Desktop**:
   - Go to [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop/).
   - Click "Download for Windows" and run `Docker Desktop Installer.exe`.

2. **Install Docker**:
   - Follow the prompts (enable WSL 2 if prompted; it’s recommended).
   - Restart your PC if required.

3. **Launch Docker Desktop**:
   - Open Docker Desktop from the Start menu.
   - Wait for the green "running" status in the bottom-left corner.

### 2. Launch Coqui TTS Server in Docker
The Coqui TTS server turns text into speech for the Oracle.

1. **Open a Terminal**:
   - In Docker Desktop, click the whale icon in the system tray, then "Dashboard".
   - Go to the "CLI" tab, or use Command Prompt/PowerShell.

2. **Pull and Run the Coqui TTS Container**:
   - Run these commands once:
     ```bash
     docker pull coqui/tts
     docker run -d -p 5002:5002 --name coqui-tts coqui/tts --port 5002
     ```
   - `-d`: Runs in the background.
   - `-p 5002:5002`: Maps port 5002 (the Oracle’s default).
   - `--name coqui-tts`: Names the container.

3. **Verify It’s Running**:
   - In Docker Desktop, under "Containers", see `coqui-tts` listed as "Running".
   - Test: Open a browser to `http://localhost:5002`. You’ll see a TTS API response (or an error if no text is sent).

4. **Manage It**:
   - **Stop**: In Docker Desktop, click `coqui-tts` under "Containers" and hit the "Stop" button.
   - **Restart**: Click "Start" next to `coqui-tts` if stopped. No need to rerun commands after the first time!

### 3. Install Ollama
Ollama generates wisdom at `http://localhost:11434/api/chat`.

1. **Download Ollama**:
   - Visit [Ollama’s website](https://ollama.com/).
   - Click "Download for Windows" to get `ollama-windows-amd64.exe`.

2. **Install Ollama**:
   - Run the `.exe` and follow the prompts (defaults are fine).

3. **Run Ollama**:
   - Open Command Prompt or PowerShell.
   - Start Ollama with the default model:
     ```bash
     ollama run llama3.2:latest
     ```
   - Downloads `llama3.2:latest` (the Oracle’s default) and starts the server.
   - Keep the terminal open.

4. **Verify**:
   - In a browser, visit `http://localhost:11434`. You’ll see Ollama’s welcome page.

### 4. Install LM Studio (Alternative to Ollama)
LM Studio is an optional AI backend at `http://localhost:1234/v1/chat/completions`.

1. **Download LM Studio**:
   - Go to [LM Studio’s website](https://lmstudio.ai/).
   - Click "Download for Windows" to get the installer.

2. **Install LM Studio**:
   - Run the installer (e.g., `LMStudio-windows-x64.exe`) and follow the prompts.

3. **Launch LM Studio**:
   - Open LM Studio from the Start menu.
   - In the app:
     - Go to "Models" tab, search for `qwen2.5-1.5b-instruct`, and download it (the Oracle’s default).
     - Go to "Server" tab, select the model, and click "Start Server".

4. **Verify**:
   - Confirm the server runs at `http://localhost:1234` (check the app’s status).

### 5. Run Infinite Oracle
With the servers ready, launch the Oracle executable.

1. **Download the Executable**:
   - Grab `infinite_oracle.exe` from [Releases](https://github.com/aat440hz/infinite-oracle/releases).
   - Place it in a folder (e.g., `C:\InfiniteOracle`).

2. **Launch the Oracle**:
   - Double-click `infinite_oracle.exe`.
   - The GUI opens—all dependencies are bundled.

## Usage
- **Start Mode**: Click "Start" for continuous wisdom every "Speech Interval" seconds.
- **Send Mode**: Edit the prompt and click "Send" for one-off wisdom.
- **Record**: Toggle "Record" to save WAV files in an `OracleRecordings` folder next to the `.exe`.
- **Server Choice**: Select "Ollama" or "LM Studio" in the GUI dropdown.

## Troubleshooting
- **No Sound**: Check Docker Desktop; ensure `coqui-tts` is "Running" (click "Start" if stopped).
- **Ollama/LM Studio Fails**: Verify they’re active and ports (11434/1234) aren’t blocked (`netstat -aon | findstr 11434`).
- **EXE Won’t Run**: Ensure Docker and at least one AI server (Ollama or LM Studio) are up.
