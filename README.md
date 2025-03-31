# Infinite Oracle

Welcome to **Infinite Oracle**, a mystical application that channels boundless wisdom through an ethereal voice. This executable brings you cryptic, uplifting insights powered by advanced AI models and text-to-speech technology.

This README will guide you through downloading and setting up the necessary dependencies to run `infinite_oracle.exe` on Windows.

## Prerequisites

To use Infinite Oracle, you'll need to install and configure the following tools:
1. **`Ollama`** and/or **`LM Studio`** (choose one for the AI backend)
2. **Docker Desktop** (to run the Coqui TTS Server and Whisper ASR Webservice)

Below are step-by-step instructions for setting up each component.

---

## Step 1: Download infinite_oracle.zip

1. Download the latest `infinite_oracle.zip` (~3 GB) from the following Google Drive link:
   - **[Download Infinite Oracle](https://drive.google.com/file/d/1Tf_A8J9BmAkf8wgbs-AGSDqc4uMeAKLr/view?usp=sharing)**
2. Once downloaded, extract the ZIP file to a folder on your computer.

---

## Step 2: Install and Configure the AI Backend

Infinite Oracle supports two AI backends: **Ollama** or **LM Studio**. Choose one based on your preference.

### Option 1: Ollama
Ollama is a lightweight, open-source tool for running large language models locally.

1. **Download Ollama**:
   - Visit [ollama.com](https://ollama.com).
   - Click the "Download for Windows" button and install it by following the installer prompts.

2. **Install a Model**:
   - Open a Command Prompt (`cmd`) or PowerShell.
   - Run the following command to pull a model (e.g., LLaMA 3.2):
     ```
     ollama pull llama3.2:latest
     ```
   - Wait for the download to complete.

3. **Run Ollama**:
   - Start Ollama by running:
     ```
     ollama serve
     ```
   - Keep the terminal window open. Ollama will run on `http://localhost:11434` by default.

### Option 2: LM Studio
LM Studio is a user-friendly interface for running AI models with a graphical setup.

1. **Download LM Studio**:
   - Go to [lmstudio.ai](https://lmstudio.ai).
   - Download the Windows version and install it.

2. **Load a Model**:
   - Open LM Studio.
   - In the "Models" tab, search for a model (e.g., "qwen2.5-1.5b-instruct").
   - Download and load the model.

3. **Start the Server**:
   - Go to the "Server" tab in LM Studio.
   - Click "Start Server". It will run on `http://localhost:1234` by default.
   - Keep LM Studio open while using Infinite Oracle.

---

## Step 3: Install Docker Desktop

Docker Desktop is required to run the Coqui TTS Server and Whisper ASR Webservice as containers.

1. **Download Docker Desktop**:
   - Visit [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop/).
   - Click "Download for Windows" and install it.

2. **Install Docker Desktop**:
   - Run the installer and follow the prompts.
   - Ensure "Enable Hyper-V" and "Install WSL 2" are checked if prompted (required for Windows 10/11).
   - Restart your computer if required.

3. **Verify Installation**:
   - Open Docker Desktop to ensure it’s running (you’ll see a whale icon in the system tray).
   - Open a Command Prompt and run:
     ```
     docker --version
     ```
   - You should see the Docker version (e.g., `Docker version 20.10.21`).

---

## Step 4: Set Up Coqui TTS Server

The Coqui TTS Server converts text to speech for Infinite Oracle using the VCTK VITS model, which supports multiple voices via speaker IDs (e.g., `p267`).

1. **Run the Coqui TTS Server with GPU Support**:
   - Open a Command Prompt or PowerShell.
   - Run the following command to start the container interactively with GPU support:
     ```
     docker run --rm -it -p 5002:5002 --gpus all --entrypoint /bin/bash ghcr.io/coqui-ai/tts
     ```
   - Once inside the container, run this command to start the TTS server with the VCTK VITS model:
     ```
     python3 TTS/server/server.py --model_name tts_models/en/vctk/vits --use_cuda true
     ```
   - Keep the terminal open. The server will run on `http://localhost:5002`.

   **Note**: This requires an NVIDIA GPU with CUDA installed. If you don’t have a compatible GPU, see the CPU-only alternative below.

2. **CPU-Only Alternative** (if no GPU):
   - If you lack an NVIDIA GPU, use this command instead:
     ```
     docker run --rm -it -p 5002:5002 --entrypoint /bin/bash ghcr.io/coqui-ai/tts-cpu
     ```
   - Inside the container, run:
     ```
     python3 TTS/server/server.py --model_name tts_models/en/vctk/vits
     ```
   - This will run on CPU and still work with Infinite Oracle.

3. **Verify It’s Running**:
   - Open a browser and go to `http://localhost:5002`.
   - You should see a simple API page. If not, ensure Docker Desktop is running and retry.

---

## Step 5: Set Up Whisper ASR Webservice

The Whisper ASR Webservice enables voice input by transcribing spoken audio using OpenAI's Whisper model.

1. **Pull the Whisper ASR Webservice Docker Image**:
   - In a Command Prompt or PowerShell, run:
     ```
     docker pull onerahmet/openai-whisper-asr-webservice:latest
     ```

2. **Run the Whisper ASR Webservice**:
   - Start the server with:
     ```
     docker run -d -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest
     ```
   - This maps the server to `http://localhost:9000` and runs it in the background using the "base" model.

3. **Optional GPU Support**:
   - If you have an NVIDIA GPU, use this command instead:
     ```
     docker run -d --gpus all -p 9000:9000 -e ASR_MODEL=base -e ASR_ENGINE=openai_whisper onerahmet/openai-whisper-asr-webservice:latest-gpu
     ```
   - Ensure your NVIDIA drivers and CUDA toolkit are installed.

4. **Verify It’s Running**:
   - Open a browser and go to `http://localhost:9000/docs` to see the Swagger API documentation.
   - If accessible, the server is ready. Check Docker Desktop if you encounter issues.

---

## Step 6: Run infinite_oracle.exe

1. **Ensure All Services Are Running**:
   - Ollama (`http://localhost:11434`) or LM Studio (`http://localhost:1234`).
   - Coqui TTS Server (`http://localhost:5002`).
   - Whisper ASR Webservice (`http://localhost:9000`).

2. **Launch the Executable**:
   - Double-click `infinite_oracle.exe` in the folder where you extracted it.
   - A control panel will open.

3. **Configure the Control Panel**:
   - **Server Type**: Select "Ollama" or "LM Studio" based on your choice.
   - **Server URL**: Use `http://localhost:11434/api/chat` for Ollama or `http://localhost:1234/v1/chat/completions` for LM Studio.
   - **Model Name**: Enter the model you installed (e.g., `llama3.2:latest` for Ollama or `qwen2.5-1.5b-instruct` for LM Studio).
   - **Coqui TTS Server URL**: Leave as `http://localhost:5002`.
   - **Speaker ID**: Default is `p267` (try others like `p225` or `p240` for different voices).
   - **Whisper Server URL**: Leave as `http://localhost:9000`.

4. **Start the Oracle**:
   - Click "Start" to begin generating wisdom in "Start Mode" (continuous generation).
   - Use "Send" to input a custom prompt manually.
   - Use "Listen" to speak a prompt (requires a microphone).

5. **Interact**:
   - The Oracle will speak its wisdom aloud using the VCTK voices. Adjust settings like pitch, reverb, or intervals as desired.
   - Enable "Record" to save audio files to an `OracleRecordings` folder.

---

## Troubleshooting

- **Ollama/LM Studio Not Responding**:
  - Ensure the server is running and the URL/model name matches your setup.
- **No Sound**:
  - Check that the Coqui TTS Server is running with the VCTK VITS model (`docker ps` in Command Prompt) and accessible at `http://localhost:5002`.
- **Listen Button Fails**:
  - Verify the Whisper ASR Webservice is running at `http://localhost:9000` (`docker ps` to check).
- **Docker Issues**:
  - Restart Docker Desktop and rerun the `docker run` commands.
- **CUDA Errors**:
  - Ensure NVIDIA drivers and CUDA toolkit are installed if using GPU versions. Otherwise, use CPU-only commands.
---

## Features

- **Start Mode**: Continuous wisdom generation with customizable intervals.
- **Send Mode**: Send a specific prompt to the Oracle.
- **Listen Mode**: Speak to the Oracle and hear its response.
- **Voice Variety**: Change voices using Speaker IDs (e.g., `p267`, `p225`, `p240`).
- **Effects**: Adjust pitch and reverb for a mystical voice.
- **Recording**: Save the Oracle’s wisdom as WAV files.

---

## Notes

- Keep all terminal windows or Docker Desktop open while using the app.
- The default settings should work if you followed the steps above.
- For advanced customization, explore the control panel options.
