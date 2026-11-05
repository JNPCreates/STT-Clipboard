"""Microphone capture and serialized Whisper previews, independent of Tkinter."""
import re
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from whisper_models import Cancelled, MODEL_NAMES, ensure_checkpoint
from audio_input import InputLevel, resolve_microphone

RATE = 16000
WINDOW_SECONDS = 28
COMMIT_SECONDS = 24
OVERLAP_SECONDS = 2


class SpeechBackend:
    """Keep local Whisper loaded; timestamped words support continuous dictation."""

    def __init__(self, app_dir):
        self.app_dir = Path(app_dir)
        self.model = None
        self.model_name = 'base'
        self.lock = threading.Lock()

    def prepare(self, live, cancelled, model_name='base', progress=lambda *_: None):
        if model_name not in MODEL_NAMES:
            raise ValueError(f'Unknown Whisper model: {model_name}')
        if cancelled.is_set():
            raise Cancelled()
        with self.lock:
            if cancelled.is_set():
                raise Cancelled()
            if self.model is None or self.model_name != model_name:
                import torch
                import whisper
                checkpoint = ensure_checkpoint(self.app_dir, model_name,
                    whisper._MODELS[model_name], cancelled, progress)
                if cancelled.is_set():
                    raise Cancelled()
                progress('loading', f'Loading {model_name}…')
                # Release the previous model before allocating another on the GPU.
                self.model = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                torch.set_num_threads(4)
                self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                self.model = whisper.load_model(str(checkpoint), device=self.device)
                self.model.set_alignment_heads(whisper._ALIGNMENT_HEADS[model_name])
                self.model_name = model_name
        return 'local'

    def transcribe(self, pcm, final, route, cancelled):
        result = self._decode(pcm, cancelled, timestamps=False)
        return ' '.join(segment['text'].strip() for segment in result).strip()

    def transcribe_words(self, pcm, route, cancelled):
        result = self._decode(pcm, cancelled, timestamps=True)
        return [Word(w['word'], float(w['start']), float(w['end']))
                for segment in result for w in segment.get('words', []) if w['word'].strip()]

    def _decode(self, pcm, cancelled, timestamps):
        import numpy as np
        with self.lock:
            if cancelled.is_set():
                raise Cancelled()
            audio = np.frombuffer(pcm, dtype='<i2').astype(np.float32) / 32768.0
            blocks = audio[:len(audio) // 320 * 320].reshape(-1, 320)
            if not len(blocks) or np.count_nonzero(np.sqrt(np.mean(blocks * blocks, axis=1)) > .003) < 4:
                return []
            audio = np.concatenate((audio, np.zeros(4800, dtype=np.float32)))
            result = self.model.transcribe(audio, language='en', task='transcribe',
                fp16=self.device == 'cuda', temperature=0, beam_size=1,
                condition_on_previous_text=False, verbose=None, word_timestamps=timestamps)
            return [s for s in result['segments']
                    if s.get('no_speech_prob', 0) < .65 and s.get('avg_logprob', 0) > -1]


@dataclass
class Word:
    text: str
    start: float
    end: float


class RollingTranscript:
    """Commit whole words with lookahead; retain audio context for the next window."""

    def __init__(self):
        self.committed = []
        self.frontier = 0.0

    @staticmethod
    def token(word):
        return re.sub(r'[^\w]', '', word.text).casefold()

    def pending(self, words, offset):
        absolute = [Word(w.text, w.start + offset, w.end + offset) for w in words]
        if not self.committed:
            return [w for w in absolute if (w.start + w.end) / 2 >= self.frontier]
        # Align the overlapping words to the committed suffix. Timestamps prevent
        # collapsing an intentional repeated word or phrase later in the recording.
        for count in range(min(8, len(self.committed), len(absolute)), 0, -1):
            suffix = self.committed[-count:]
            matches = []
            for index in range(len(absolute) - count + 1):
                candidate = absolute[index:index + count]
                distance = abs(candidate[-1].end - suffix[-1].end)
                if distance <= .75 and all(self.token(a) == self.token(b)
                                           for a, b in zip(candidate, suffix)):
                    matches.append((distance, index + count))
            if matches:
                return absolute[min(matches)[1]:]
        # A changed overlap hypothesis has no lexical anchor. Use ownership in time.
        return [w for w in absolute if (w.start + w.end) / 2 >= self.frontier]

    def update(self, words, offset, commit=False, final=False):
        pending = self.pending(words, offset)
        if final:
            self.committed.extend(pending)
            return self.text(), None
        if not commit:
            return self.text(pending), None
        cutoff = offset + COMMIT_SECONDS
        count = 0
        for word in pending:
            if word.end > cutoff:
                break
            count += 1
        accepted = pending[:count]
        boundary = cutoff
        if accepted and count < len(pending):
            # Place the seam between complete words, with the next words still provisional.
            boundary = min(cutoff, (accepted[-1].end + pending[count].start) / 2)
        self.committed.extend(accepted)
        self.frontier = max(self.frontier, boundary)
        return self.text(pending[count:]), max(offset + .1, boundary - OVERLAP_SECONDS)

    def text(self, pending=()):
        return ' '.join(w.text.strip() for w in [*self.committed, *pending]).strip()


@dataclass
class Session:
    id: int
    live: bool
    model_name: str = 'base'
    stop: threading.Event = field(default_factory=threading.Event)
    cancelled: threading.Event = field(default_factory=threading.Event)
    captured: threading.Event = field(default_factory=threading.Event)
    finished: threading.Event = field(default_factory=threading.Event)
    changed: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    audio: bytearray = field(default_factory=bytearray)
    error: str = ''


class SpeechEngine:
    def __init__(self, backend, capture=None, preview_interval=.65):
        self.backend = backend
        self.events = queue.Queue()
        self.capture = capture or self._microphone
        self.preview_interval = preview_interval
        self.current = None
        self.pending_audio = None
        self.sequence = 0
        self.capture_lock = threading.Lock()

    def emit(self, session, kind, value=''):
        if not session.cancelled.is_set():
            self.events.put((session.id, kind, value))

    def start(self, live=True, microphone=None, model_name='base'):
        self.cancel()
        self.sequence += 1
        session = Session(self.sequence, live, model_name)
        self.current = session
        threading.Thread(target=self._run, args=(session, microphone), daemon=True).start()
        return session.id

    def stop(self):
        if self.current:
            self.current.stop.set()
            self.current.changed.set()

    def retry(self, model_name='base'):
        """Retry the most recent failed standard recording without opening the mic."""
        if self.pending_audio is None:
            raise ValueError('There is no failed recording to retry.')
        self.cancel()
        self.sequence += 1
        session = Session(self.sequence, False, model_name)
        session.audio.extend(self.pending_audio)
        session.captured.set()
        session.stop.set()
        self.current = session
        threading.Thread(target=self._run, args=(session, None), daemon=True).start()
        return session.id

    def cancel(self):
        if self.current:
            self.current.cancelled.set()
            self.current.stop.set()
            self.current.changed.set()
            self.current = None

    def _capture(self, session, microphone):
        try:
            with self.capture_lock:
                if not session.stop.is_set():
                    self.capture(session, microphone)
        except Exception as exc:
            session.error = f'Microphone error: {exc}'
        finally:
            session.captured.set()
            session.changed.set()
            if session.finished.is_set() or session.cancelled.is_set():
                with session.lock:
                    session.audio.clear()

    def _microphone(self, session, microphone):
        import pyaudio
        audio = pyaudio.PyAudio()
        stream = None
        try:
            microphone = resolve_microphone(audio, microphone)
            stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE,
                                input=True, frames_per_buffer=1024, input_device_index=microphone)
            self.emit(session, 'recording')
            meter = InputLevel(RATE)
            while not session.stop.is_set():
                block = stream.read(1024, exception_on_overflow=False)
                with session.lock:
                    session.audio.extend(block)
                session.changed.set()
                self.emit(session, 'level', meter.update(block))
        finally:
            try:
                if stream is not None:
                    try:
                        stream.stop_stream()
                    finally:
                        stream.close()
            finally:
                audio.terminate()

    def _run(self, session, microphone):
        try:
            if session.live:
                self.emit(session, 'loading')
                route = self.backend.prepare(True, session.stop, session.model_name,
                    lambda kind, value: self.emit(session, kind, value))
                if session.cancelled.is_set():
                    return
                if session.stop.is_set():
                    self.emit(session, 'final', '')
                    return
            if not session.captured.is_set():
                capture = threading.Thread(target=self._capture, args=(session, microphone), daemon=True)
                capture.start()
            if session.live:
                self._stream(session, route)
                return
            finalizing = False
            while not session.captured.is_set():
                if session.cancelled.wait(.05):
                    return
                if session.stop.is_set() and not finalizing:
                    finalizing = True
                    self.emit(session, 'finalizing')
            if session.cancelled.is_set():
                return
            if session.error:
                raise RuntimeError(session.error)
            if not finalizing:
                self.emit(session, 'finalizing')
            with session.lock:
                pcm = bytes(session.audio)
            text = ''
            if pcm:
                # Stop means capture is complete, not that preparation is cancelled.
                # Keep the recording in memory while downloading/loading the model.
                route = self.backend.prepare(False, session.cancelled, session.model_name,
                    lambda kind, value: self.emit(session, kind, value))
                if session.cancelled.is_set():
                    return
                self.emit(session, 'finalizing')
                text = self.backend.transcribe(pcm, True, route, session.cancelled)
                if not session.cancelled.is_set():
                    self.pending_audio = None
            self.emit(session, 'final', text)
        except Cancelled:
            if not session.cancelled.is_set():
                self.emit(session, 'final', '')
        except Exception as exc:
            session.stop.set()
            if not session.live and session.captured.is_set() and not session.cancelled.is_set():
                with session.lock:
                    if session.audio:
                        self.pending_audio = bytes(session.audio)
            self.emit(session, 'error', str(exc))
        finally:
            session.finished.set()
            if session.captured.is_set():
                with session.lock:
                    session.audio.clear()

    def _stream(self, session, route):
        transcript = RollingTranscript()
        offset_samples = 0
        window_bytes = WINDOW_SECONDS * RATE * 2
        previous_size = 0
        due = 0
        finalizing = False
        try:
            while not session.cancelled.is_set():
                if session.stop.is_set() and not finalizing:
                    finalizing = True
                    self.emit(session, 'finalizing')
                captured = session.captured.is_set()
                if captured and session.error:
                    raise RuntimeError(session.error)
                with session.lock:
                    size = len(session.audio)
                    full_window = size >= window_bytes
                    ready = captured or full_window or (not session.stop.is_set() and size >= RATE
                        and size != previous_size and time.monotonic() >= due)
                    pcm = bytes(session.audio[:window_bytes]) if ready else None
                if pcm is None:
                    session.changed.wait(.05)
                    session.changed.clear()
                    continue
                final = captured and size <= window_bytes
                if final and not finalizing:
                    self.emit(session, 'finalizing')
                started = time.monotonic()
                words = self.backend.transcribe_words(pcm, route, session.cancelled) if pcm else []
                if session.cancelled.is_set():
                    return
                text, trim_to = transcript.update(words, offset_samples / RATE,
                                                   commit=full_window and not final, final=final)
                self.emit(session, 'final' if final else 'partial', text)
                if final:
                    return
                if trim_to is not None:
                    trim_samples = int(trim_to * RATE) - offset_samples
                    with session.lock:
                        del session.audio[:trim_samples * 2]
                    offset_samples += trim_samples
                    previous_size = 0
                else:
                    previous_size = size
                due = started + self.preview_interval
        except Exception:
            # Preserve completed words if a later decode fails; the UI retains this draft.
            if transcript.committed:
                self.emit(session, 'recovered', transcript.text())
            raise
