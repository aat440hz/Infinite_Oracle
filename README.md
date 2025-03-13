Infinite Oracle (Windows Version)

The Infinite Oracle is a mystical text-to-speech application that generates cryptic, uplifting wisdom using Ollama and Coqui TTS, with a Tkinter GUI featuring a pitch slider and embedded console. This version is tailored for Windows, built as a standalone .exe with no console popups.

Features
- Generates wisdom from Ollama (llama3.2:latest) at http://192.168.0.163:11434.
- Converts text to speech via Coqui TTS (p228 voice) at http://192.168.0.163:5002.
- GUI with input fields, pitch control (-12 to +12 semitones), and embedded console output.
- Pitch adjustment without speed change using pydub.
- No CMD or curl console popups.

Dependencies
- Python 3.8+: For building from source (not needed for .exe).
- Libraries (if building):
  - requests
  - winsound (built-in on Windows)
  - soundfile
  - subprocess (built-in)
  - tkinter (built-in)
  - pydub
- External Binaries (bundled in .exe):
  - ffmpeg.exe: For pydub audio processing.
  - curl.exe: Assumed in PATH (Windows 10/11 default) or bundled if specified.

Installation

Option 1: Run Prebuilt .exe
1. Download:
   - Grab infinite_oracle.exe from the Releases page (https://github.com/yourusername/yourrepo/releases) (see note below on file size).
   - Size: ~54 MB (includes ffmpeg).
2. Run:
   infinite_oracle.exe
   - No additional setup needed—fully portable.

Option 2: Build from Source
1. Install Python:
   - Download from https://www.python.org/downloads/.
2. Install Dependencies:
   pip install requests soundfile pydub
3. Install ffmpeg:
   - Download from https://github.com/GyanD/codexffmpeg/releases (e.g., ffmpeg-release-essentials.zip).
   - Extract and add ffmpeg.exe to PATH (e.g., C:\ffmpeg\bin) or place in project folder.
4. Get curl (if not in PATH):
   - Comes with Windows 10/11, or download from https://curl.se/windows/.
5. Build .exe:
   pip install pyinstaller
   pyinstaller --onefile --noconsole --add-binary "C:\ffmpeg\bin\ffmpeg.exe;." infinite_oracle.py
   - Replace C:\ffmpeg\bin\ffmpeg.exe with your path.
   - Find infinite_oracle.exe in dist.

Running
- Launch infinite_oracle.exe.
- Configure fields (Ollama URL, model, TTS URL, speaker ID) if different from defaults.
- Adjust pitch slider (-12 deep, +12 high).
- Click "Start" to hear the Oracle’s wisdom in the GUI console.

Notes
- File Size: The .exe is ~54 MB due to ffmpeg bundling, exceeding GitHub’s 25 MB limit. Use Git LFS or download from Releases.
- Requires Ollama and Coqui TTS servers running at specified URLs.

Using Git LFS
1. Install Git LFS: git lfs install
2. Track .exe: git lfs track "*.exe"
3. Add, commit, push:
   git add infinite_oracle.exe
   git commit -m "Add Windows exe with LFS"
   git push

Releases
- Alternatively, upload infinite_oracle.exe as a release asset:
  - Go to GitHub repo > "Releases" > "New release".
  - Attach infinite_oracle.exe—GitHub allows up to 2 GB for release assets.

 Infinite Oracle (Linux Version)

The Infinite Oracle is a mystical text-to-speech application that generates cryptic, uplifting wisdom using Ollama and Coqui TTS, running as a command-line script on Linux with continuous playback.

Features
- Generates wisdom from Ollama (llama3.2:latest) at http://192.168.0.163:11434.
- Converts text to speech via Coqui TTS (p228 voice) at http://192.168.0.163:5002.
- Continuous playback with aplay.
- No GUI—pure terminal experience.

Dependencies
- Python 3.8+
- Libraries:
  - requests
  - subprocess (built-in)
- System Tools:
  - curl: For TTS requests.
  - aplay: For audio playback (part of alsa-utils).

Installation
1. Install Python:
   sudo apt update
   sudo apt install python3 python3-pip
2. Install Dependencies:
   pip3 install requests
3. Install System Tools:
   sudo apt install curl alsa-utils
4. Clone or Download:
   git clone https://github.com/yourusername/yourrepo.git
   cd yourrepo

Running
1. Ensure Ollama and Coqui TTS servers are running at specified URLs.
2. Execute:
   python3 infinite_oracle.py
   - Outputs wisdom to terminal and plays via aplay.
3. Stop with Ctrl+C—see "The Infinite Oracle rests...".

Notes
- No .exe—runs as a Python script.
- Fixed to p228 speaker; edit text_to_speech for others (e.g., p376).
- Lightweight—no GUI or audio processing beyond playback.

Customization
- Edit OLLAMA_URL, MODEL, TTS_SERVER_URL, or SYSTEM_PROMPT in the script to tweak behavior.
