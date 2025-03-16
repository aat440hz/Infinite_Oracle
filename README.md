# Infinite Oracle

A mystical AI-powered conversational tool that channels boundless wisdom through an uplifting, cryptic voice. Integrating language models (Ollama or LM Studio) with text-to-speech (Coqui TTS), it inspires awe and contemplation. Speak to the Oracle, and it remembers your interactions—perfect for cosmic chats!

## Features
- **Conversational Memory**: Remembers your name and past exchanges within a session.
- **Dual Backend Support**: Works with Ollama (`/api/chat`) or LM Studio (`/v1/chat/completions`).
- **TTS Integration**: Voices wisdom using Coqui TTS with customizable pitch and speaker.
- **GUI Control**: Start, stop, and send prompts via an intuitive interface.

## Prerequisites
- **Windows**
- **Docker Desktop**: For running Coqui TTS in a container.
- **NVIDIA GPU (Optional)**: Enhances performance (e.g., GTX 1050 Ti with 4GB VRAM).

## Installation

### 1. Install Ollama
Ollama runs locally and hosts language models like Phi-3 Mini.

1. **Download**:
   - Grab the Windows installer from [Ollama’s official site](https://ollama.com/).
   - Run `OllamaSetup.exe` and follow the prompts.

2. **Run Ollama**:
   - Open Command Prompt: `ollama serve`.
   - It binds to `http://localhost:11434`.

3. **Pull a Model**:
   - `ollama pull phi3` (Phi-3 Mini, ~2.5GB in Q4_K_M).
   - Verify: `curl http://localhost:11434/api/tags`.

4. **Optional: LAN Access**:
   - Set `OLLAMA_HOST=0.0.0.0` in Command Prompt before `ollama serve` for network access.

### 2. Install LM Studio
LM Studio is an alternative backend with a GUI and server mode.

1. **Download**:
   - Get it from [LM Studio’s site](https://lmstudio.ai/).
   - Install via the `.exe` (e.g., to `C:\Program Files\LM Studio`).

2. **Setup**:
   - Launch LM Studio.
   - Search and download `Phi-3-mini-4k-instruct` (Q4_K_M recommended for 4GB VRAM GPUs).
   - Go to “Local Server” tab, select the model, and start the server (`http://localhost:1234`).

3. **Verify**:
   - `curl http://localhost:1234/v1/models` should list your model.

### 3. Install Coqui TTS (via Docker Desktop)
Coqui TTS turns text into speech, running in a Docker container.

1. **Install Docker Desktop**:
   - Download from [Docker’s site](https://www.docker.com/products/docker-desktop/).
   - Install and enable WSL 2 or Hyper-V backend (for GPU support).
   - Ensure NVIDIA Container Toolkit is enabled in Docker settings if using GPU.

2. **Create Batch File**:
   - Save as `start_tts.bat`:
     ```
     @echo off
     echo Starting Coqui TTS Docker container...
     docker run --rm -p 5002:5002 --gpus all --entrypoint /bin/bash ghcr.io/coqui-ai/tts -c "python3 TTS/server/server.py --model_name tts_models/en/vctk/vits --use_cuda true"
     echo TTS server running on port 5002. Stop via Docker Desktop.
     pause
     ```

3. **Run TTS**:
   - Open Docker Desktop (ensure it’s running).
   - Double-click `start_tts.bat` or run in Command Prompt.
   - Docker pulls `ghcr.io/coqui-ai/tts` and starts the TTS server.
   - Verify: `curl "http://localhost:5002/api/tts?text=Hello&speaker_id=p267"`.

4. **Stop TTS**:
   - In Docker Desktop, find the `ghcr.io/coqui-ai/tts` container (random name).
   - Click “Stop” to shut it down cleanly (auto-deletes with `--rm`).

5. **Headless Option**:
   - Create `start_tts_hidden.vbs`:
     ```
     Set WshShell = CreateObject("WScript.Shell")
     WshShell.Run "cmd /c start_tts.bat", 0, False
     Set WshShell = Nothing
     ```
   - Double-click to run silently. Stop via Docker Desktop.

### 4. Install Infinite Oracle
1. **Download the Executable**:
   - Grab `oracle.exe` from the [Releases page](https://github.com/<your-username>/infinite-oracle/releases).
   - Place it in a folder (e.g., `C:\InfiniteOracle`).

2. **Run the Oracle**:
   - Double-click `oracle.exe`.
   - The GUI launches—select Ollama or LM Studio, tweak settings, and start chatting!

## Usage
- **Start**: Click “Start” for continuous wisdom (TTS-enabled).
- **Send**: Enter a prompt (e.g., “What’s my name?”) and click “Send.”
- **Stop**: Click “Stop” to reset the conversation.
- **Config**: Adjust server URLs, models, and TTS settings in the GUI; saved to `oracle_config.json`.

## Notes
- **GPU**: Phi-3 Mini (Q4_K_M) can run on a 4GB GPU (e.g., 1050 Ti). Coqui TTS uses CUDA if available.
- **Memory**: Conversation history resets on “Stop.” Edit the source to persist if desired.
- **Docker**: TTS runs in Docker; ensure port `5002` is free.
- **FFmpeg**: Bundled with `oracle.exe`—no separate install needed!

## Troubleshooting
- **Connection Errors**: Check servers are running (`curl` test above). Allow ports `11434`, `1234`, `5002` in Firewall if LAN issues arise.
- **No Audio**: Verify TTS server is up.
- **Slow Response**: Lower model quantization or offload to CPU if GPU’s maxed.
