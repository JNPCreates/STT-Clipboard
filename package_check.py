"""Explicit frozen-app diagnostics; does not record or change the clipboard/history."""
import argparse
import json
import sys
import tempfile
import threading
import time
import traceback
import wave
from pathlib import Path


def run(resource_dir):
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', type=Path, required=True, metavar='REPORT_JSON')
    parser.add_argument('--fixture', type=Path)
    parser.add_argument('--model', default='base')
    args = parser.parse_args()
    report = {'frozen': bool(getattr(sys, 'frozen', False)), 'ok': False}
    started = time.monotonic()
    root = None
    try:
        import tkinter as tk
        from tkinter import ttk
        import pyaudio
        import torch
        from audio_input import InputLevel, input_devices
        from speech_engine import SpeechBackend, RollingTranscript, RATE, WINDOW_SECONDS
        from transcription_history import TranscriptionHistory
        resources = Path(resource_dir)
        for name in ('stt_Clipboard.ico', 'chimes.wav', 'notification sound.wav'):
            if not (resources / name).is_file():
                raise RuntimeError(f'Bundled resource missing: {name}')
        root = tk.Tk()
        root.withdraw()
        root.iconbitmap(str(resources / 'stt_Clipboard.ico'))
        ttk.Notebook(root)
        root.update_idletasks()
        report['tkinter'] = 'ok'
        audio = pyaudio.PyAudio()
        try:
            report['audio_devices'] = audio.get_device_count()
            report['microphone_inputs'] = len(input_devices(audio))
        finally:
            audio.terminate()
        backend = SpeechBackend(resources)
        report['torch_cuda_build'] = torch.version.cuda
        meter = InputLevel()
        assert meter.update(b'\x00\x00' * 32000)['quiet']
        report['input_meter'] = 'ok'
        route = backend.prepare(True, threading.Event(), args.model)
        report['device'] = backend.device
        if args.fixture:
            with wave.open(str(args.fixture), 'rb') as wav:
                if (wav.getframerate(), wav.getnchannels(), wav.getsampwidth()) != (RATE, 1, 2):
                    raise ValueError('Fixture must be mono PCM16 at 16 kHz.')
                pcm = wav.readframes(wav.getnframes())
            report['fixture_seconds'] = len(pcm) / (RATE * 2)
            report['standard_text'] = backend.transcribe(pcm, True, route, threading.Event())
            rolling = RollingTranscript()
            offset = 0
            window = WINDOW_SECONDS * RATE * 2
            while offset < len(pcm):
                snapshot = pcm[offset:offset + window]
                final = offset + window >= len(pcm)
                words = backend.transcribe_words(snapshot, route, threading.Event())
                text, trim = rolling.update(words, offset / (RATE * 2), commit=not final, final=final)
                if final:
                    report['realtime_text'] = text
                    break
                offset = int(trim * RATE) * 2
            if not report['standard_text'] or not report.get('realtime_text'):
                raise RuntimeError('Speech fixture produced no text.')
        with tempfile.TemporaryDirectory() as directory:
            history = TranscriptionHistory(Path(directory) / 'history.json')
            history.add('Packaging test')
            assert TranscriptionHistory(history.path).entries[0]['text'] == 'Packaging test'
        report['history'] = 'ok'
        report['ok'] = True
    except Exception:
        report['error'] = traceback.format_exc()
    finally:
        if root is not None:
            root.destroy()
        report['elapsed_seconds'] = round(time.monotonic() - started, 2)
        args.self_test.write_text(json.dumps(report, indent=2), encoding='utf-8')
    return 0 if report['ok'] else 1
