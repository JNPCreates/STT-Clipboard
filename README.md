# STT Clipboard

Record your voice, stop recording, and paste the transcription wherever you want. STT Clipboard transcribes your audio locally with Whisper and copies the finished text straight to your clipboard.

## Get started

Download **both Windows ZIPs** from [Releases](https://github.com/JNPCreates/STT-Clipboard/releases/latest) and extract them into the **same folder**, then open **STT_Clipboard.exe**. Keep the extracted files together—no Python setup needed. A compatible NVIDIA GPU is used when available.

Choose your microphone in **Settings → Audio**, click **Start Recording**, speak, then click **Stop Recording**. Paste your text with **Ctrl+V**.

## Features

- Standard recording or continuous real-time transcription.
- Microphone selection and a small input meter, with a **QUIET** warning for low input.
- Transcription history: keep the last **10** by default, or choose **1–1000** in Settings.
- Multiple Whisper models, listed by size. Choose whichever suits you best; Base is the default.
- A resizable window, configurable recording hotkey, and always-on-top option.
- Retry failed transcriptions without recording again. Audio stays in memory and is discarded when the app closes.

Models aren't included in the ZIP. Your selected model downloads automatically when first needed—after Stop in standard mode, or before recording in real-time mode—and is reused afterward. Models are cached in `%USERPROFILE%\.cache\whisper`.

Currently supports English transcription. Settings and text history are saved beside the app; audio files aren't saved.

## Why is the Windows download large?

The ready-to-run app includes PyTorch and NVIDIA GPU libraries, which account for almost all of its roughly **2.5 GB** download. Models aren't bundled. It's provided as two ZIPs to fit GitHub's per-file release limit.

## Run the source

The source contains only the app, its icon and sound, dependencies list, and these docs. Use Windows and Python 3.10. If compatible dependencies are already installed, just run the last command:

```powershell
python -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python STT_to_clipboard.py
```

[MIT License](LICENSE.txt)
