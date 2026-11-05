import threading
import time
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch, Mock

from speech_engine import SpeechBackend, SpeechEngine, Session, RATE, WINDOW_SECONDS, Word, RollingTranscript
from transcription_history import TranscriptionHistory


class FakeBackend:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.block_preview = False

    def prepare(self, live, cancelled, model_name='base', progress=lambda *_: None):
        return 'fake'

    def transcribe(self, pcm, final, route, cancelled):
        self.calls.append((len(pcm), final))
        if not final:
            self.entered.set()
            if self.block_preview:
                self.release.wait(3)
        return 'final corrected sentence'

    def transcribe_words(self, pcm, route, cancelled):
        text = self.transcribe(pcm, False, route, cancelled)
        return [Word(text, 0, len(pcm) / (RATE * 2))]


def capture_until_stop(session, microphone):
    with session.lock:
        session.audio.extend(b'\x01\x01' * RATE)
    session.changed.set()
    session.stop.wait(3)
    # Simulate the final microphone read arriving after Stop.
    with session.lock:
        session.audio.extend(b'\x01\x01' * 1024)


def wait_terminal(engine, timeout=4):
    events = []
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        item = engine.events.get(timeout=max(.01, deadline - time.monotonic()))
        events.append(item)
        if item[1] in ('final', 'error'):
            return events
    raise AssertionError('No terminal event')


class EngineTests(unittest.TestCase):
    def test_stop_waits_for_capture_and_inflight_preview(self):
        backend = FakeBackend()
        backend.block_preview = True
        engine = SpeechEngine(backend, capture_until_stop)
        engine.start()
        self.assertTrue(backend.entered.wait(2))
        engine.stop()
        self.assertTrue(engine.current.captured.wait(2))
        backend.release.set()
        events = wait_terminal(engine)
        self.assertEqual(backend.calls, [(RATE * 2, False), ((RATE + 1024) * 2, False)])
        self.assertEqual([x[1] for x in events].count('final'), 1)
        self.assertEqual(events[-1][2], 'final corrected sentence')

    def test_standard_only_decodes_after_stop(self):
        backend = FakeBackend()
        engine = SpeechEngine(backend, capture_until_stop)
        engine.start(live=False)
        time.sleep(.1)
        self.assertEqual(backend.calls, [])
        engine.stop()
        wait_terminal(engine)
        self.assertEqual(backend.calls, [((RATE + 1024) * 2, True)])

    def test_cancelled_inference_cannot_publish_late_results(self):
        backend = FakeBackend()
        backend.block_preview = True
        engine = SpeechEngine(backend, capture_until_stop)
        old_id = engine.start()
        self.assertTrue(backend.entered.wait(2))
        engine.cancel()
        backend.release.set()
        engine.start(live=False)
        engine.stop()
        events = wait_terminal(engine)
        self.assertFalse(any(i == old_id and kind in ('partial', 'final') for i, kind, _ in events))

    def test_capture_failure_is_recoverable(self):
        def failure(*_):
            raise OSError('device unavailable')
        engine = SpeechEngine(FakeBackend(), failure)
        engine.start()
        self.assertIn('device unavailable', wait_terminal(engine)[-1][2])
        engine.capture = lambda *_: None
        engine.start()
        self.assertEqual(wait_terminal(engine)[-1][1:], ('final', ''))

    def test_realtime_capture_continues_past_thirty_seconds_and_cleans_up(self):
        stream = Mock()
        def read(*args, **kwargs):
            if stream.read.call_count >= 625:
                session.stop.set()
            return b'\x00\x00' * 1024
        stream.read.side_effect = read
        audio = Mock()
        audio.open.return_value = stream
        fake = Mock()
        fake.PyAudio.return_value = audio
        engine = SpeechEngine(FakeBackend())
        session = Session(1, True)
        with patch.dict('sys.modules', {'pyaudio': fake}):
            engine._capture(session, None)
        self.assertEqual(len(session.audio), RATE * 40 * 2)
        self.assertTrue(session.stop.is_set())
        self.assertTrue(session.captured.is_set())
        stream.close.assert_called_once()
        audio.terminate.assert_called_once()

    def test_model_stays_loaded_between_recordings(self):
        backend = SpeechBackend('.')
        backend.model = object()
        self.assertEqual(backend.prepare(True, threading.Event()), 'local')
        self.assertEqual(backend.prepare(False, threading.Event()), 'local')

    def test_three_minutes_roll_without_missing_or_duplicating_words(self):
        import numpy as np
        class TimestampBackend(FakeBackend):
            def transcribe_words(self, pcm, route, cancelled):
                self.calls.append(len(pcm))
                samples = np.frombuffer(pcm, dtype='<i2')
                edges = [0, *(np.flatnonzero(np.diff(samples)) + 1).tolist(), len(samples)]
                return [Word(f'word{samples[a]}', a / RATE + .1, b / RATE - .1)
                        for a, b in zip(edges, edges[1:]) if b - a >= RATE / 2]
        backend = TimestampBackend()
        def capture(session, microphone):
            with session.lock:
                session.audio.extend(np.repeat(np.arange(180, dtype='<i2'), RATE).tobytes())
        engine = SpeechEngine(backend, capture)
        engine.start()
        events = wait_terminal(engine, timeout=10)
        self.assertEqual(events[-1][1], 'final')
        self.assertEqual(events[-1][2].split(), [f'word{i}' for i in range(180)])
        self.assertTrue(all(size <= WINDOW_SECONDS * RATE * 2 for size in backend.calls))
        self.assertGreater(len(backend.calls), 6)
        self.assertEqual(sum(kind == 'final' for _, kind, _ in events), 1)
        self.assertTrue(any(kind == 'partial' for _, kind, _ in events))

    def test_overlap_timing_jitter_preserves_intentional_repetition(self):
        transcript = RollingTranscript()
        words = [Word(str(i) if i < 22 else 'very', i + .1, i + .9) for i in range(28)]
        _, trim = transcript.update(words, 0, commit=True)
        self.assertEqual(trim, 22)
        tail = [Word('very', i + .2, i + 1) for i in range(6)]
        text, _ = transcript.update(tail, trim, final=True)
        self.assertEqual(text.split(), [str(i) for i in range(22)] + ['very'] * 6)

    def test_inference_error_stops_capture(self):
        backend = FakeBackend()
        backend.transcribe = Mock(side_effect=OSError('connection lost'))
        engine = SpeechEngine(backend, capture_until_stop)
        engine.start()
        self.assertIn('connection lost', wait_terminal(engine)[-1][2])
        self.assertTrue(engine.current.stop.is_set())
        self.assertTrue(engine.current.captured.wait(2))


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import runpy
        import tkinter as tk
        from pathlib import Path
        cls.original_config = Path('stt_settings.json').read_bytes() if Path('stt_settings.json').exists() else None
        cls.patches = [patch.object(tk.Tk, 'mainloop'), patch('keyboard.add_hotkey'),
                       patch('keyboard.unhook_all'), patch('pyperclip.copy'),
                       patch('winsound.PlaySound'), patch('tkinter.messagebox.showerror')]
        cls.mocks = [p.start() for p in cls.patches]
        result = runpy.run_path('STT_to_clipboard.py')
        cls.g = result['toggle_recording'].__globals__
        cls.g['root'].withdraw()
        cls.real_save_config = staticmethod(cls.g['save_config'])
        cls.g['save_config'] = Mock()
        cls.temp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        from pathlib import Path
        cls.g['close_app']()
        for p in reversed(cls.patches):
            p.stop()
        assert (Path('stt_settings.json').read_bytes() if Path('stt_settings.json').exists() else None) == cls.original_config
        cls.temp.cleanup()

    def setUp(self):
        self.g['state'] = 'idle'
        self.g['restart_allowed_at'] = 0.0
        self.g['real_time_mode'] = False
        self.g['input_quiet'] = False
        self.g['app_config']['microphone'] = None
        self.g['engine'].pending_audio = None
        self.g['app_config']['whisper_model'] = 'base'
        self.g['app_config']['history_limit'] = 10
        self.g['history'] = TranscriptionHistory(Path(self.temp.name) / (self._testMethodName + '.json'))
        self.g['active_id'] = 7
        self.g['session_preview'] = ''
        self.g['previous_transcript'] = 'previous draft'
        self.g['replace_transcript']('previous draft')
        self.mocks[3].reset_mock()
        self.mocks[4].reset_mock()

    def deliver(self, kind, value, session=7):
        self.g['engine'].events.put((session, kind, value))
        self.g['process_events']()

    def text(self):
        return self.g['transcription_text_area'].get('1.0', 'end-1c')

    def test_model_and_history_settings_persist_and_apply_to_next_recording(self):
        import json
        path = Path(self.temp.name) / 'settings.json'
        with patch.dict(self.g, CONFIG_FILE=str(path), save_config=self.real_save_config):
            self.g['model_combo'].current(self.g['MODEL_NAMES'].index('tiny'))
            self.g['select_model']()
            self.g['history_limit_var'].set('25')
            self.g['apply_history_limit']()
            saved = json.loads(path.read_text())
            self.assertEqual(saved['whisper_model'], 'tiny')
            self.assertEqual(saved['history_limit'], 25)
            self.assertEqual(self.g['history'].limit, 25)
            self.assertIn('Last 25', self.g['history_summary'].get())
            with patch.object(self.g['engine'], 'start', return_value=7) as start:
                self.g['toggle_recording']()
                start.assert_called_once_with(False, self.g['app_config'].get('microphone'), 'tiny')
            self.assertEqual(str(self.g['model_combo']['state']), 'disabled')
            with patch.object(self.g['engine'], 'stop'):
                self.g['toggle_recording']()
            self.assertEqual(self.g['state'], 'finalizing')
            self.deliver('download', 42)
            self.assertEqual(self.g['status_indicator']['text'], 'DL 42%')
            self.assertEqual(str(self.g['record_button']['state']), 'disabled')
            self.deliver('final', '')
            self.assertEqual(self.text(), 'previous draft')
            self.mocks[3].assert_not_called()

    def test_retry_failed_audio_uses_selected_model_without_starting_capture(self):
        self.g['engine'].pending_audio = b'kept recording'
        self.g['real_time_mode'] = True
        self.g['app_config']['whisper_model'] = 'tiny'
        self.g['refresh_controls']()
        self.assertEqual(str(self.g['retry_button']['state']), 'normal')
        with patch.object(self.g['engine'], 'retry', return_value=7) as retry, \
                patch.object(self.g['engine'], 'start') as start:
            self.g['retry_recording']()
            retry.assert_called_once_with('tiny')
            start.assert_not_called()
            self.assertFalse(self.g['real_time_mode'])
            self.assertEqual(self.g['state'], 'finalizing')
            self.assertEqual(str(self.g['retry_button']['state']), 'disabled')
        self.g['engine'].pending_audio = None
        self.deliver('final', 'recovered transcript')
        self.assertEqual(self.g['history'].entries[0]['text'], 'recovered transcript')
        self.mocks[3].assert_called_once_with('recovered transcript')

    def test_meter_warns_recovers_and_ignores_levels_after_stop(self):
        self.g['state'] = 'recording'
        self.deliver('level', {'level': 0, 'quiet': True})
        self.assertEqual(self.g['status_indicator']['text'], 'QUIET')
        self.assertEqual(self.g['status_indicator']['bg'], '#ffaa00')
        self.deliver('level', {'level': .8, 'quiet': False})
        self.assertIn('REC', self.g['status_indicator']['text'])
        self.assertGreater(self.g['meter_canvas'].coords(self.g['meter_fill'])[2], 0)
        self.deliver('finalizing', '')
        self.deliver('level', {'level': .9, 'quiet': True})
        self.assertIn('WAIT', self.g['status_indicator']['text'])
        self.assertEqual(self.g['meter_canvas'].coords(self.g['meter_fill'])[2], 0)

    def test_microphone_selection_persists_identity_and_locks_while_recording(self):
        import json
        selection = {'index': 8, 'name': 'USB microphone', 'host_api': 'MME'}
        self.g['microphone_choices'] = [None, selection]
        self.g['microphone_combo'].config(values=['Default', 'USB microphone'])
        self.g['microphone_combo'].current(1)
        path = Path(self.temp.name) / 'microphone-settings.json'
        with patch.dict(self.g, CONFIG_FILE=str(path), save_config=self.real_save_config):
            self.g['select_microphone']()
        self.assertEqual(json.loads(path.read_text())['microphone'], selection)
        with patch.object(self.g['engine'], 'start', return_value=7) as start:
            self.g['toggle_recording']()
            start.assert_called_once_with(False, selection, 'base')
        self.assertEqual(str(self.g['microphone_combo']['state']), 'disabled')

    def test_microphone_refresh_finishes_on_ui_thread(self):
        selected = {'index': 2, 'name': 'Test microphone', 'host_api': 'MME'}
        with patch.dict(self.g, list_microphones=Mock(return_value=[selected])):
            self.g['refresh_microphones']()
            deadline = time.monotonic() + 2
            while self.g['microphones_loading'] and time.monotonic() < deadline:
                self.g['process_events']()
                time.sleep(.01)
            self.assertFalse(self.g['microphones_loading'])
            self.assertEqual(self.g['microphone_choices'], [None, selected])
            self.assertEqual(self.g['microphone_combo'].current(), 0)

    def test_failed_settings_save_preserves_previous_file(self):
        path = Path(self.temp.name) / 'settings-failure.json'
        path.write_text('{"whisper_model":"base"}')
        with patch.dict(self.g, CONFIG_FILE=str(path)), patch('os.replace', side_effect=OSError('disk full')):
            self.assertFalse(self.real_save_config({'whisper_model': 'tiny'}))
        self.assertEqual(path.read_text(), '{"whisper_model":"base"}')

    def test_preview_replaced_final_copied_once(self):
        self.g['state'] = 'recording'
        self.deliver('partial', 'first guess')
        self.deliver('partial', 'revised guess')
        self.assertEqual(self.text(), 'revised guess')
        self.mocks[3].assert_not_called()
        self.deliver('final', 'correct sentence')
        self.assertEqual(self.text(), 'correct sentence')
        self.mocks[3].assert_called_once_with('correct sentence')
        self.mocks[4].assert_called_once()
        self.assertEqual(self.g['state'], 'idle')

    def test_silence_restores_draft_without_copy(self):
        self.g['state'] = 'recording'
        self.deliver('partial', 'false guess')
        self.deliver('final', '')
        self.assertEqual(self.text(), 'previous draft')
        self.mocks[3].assert_not_called()

    def test_cancel_ignores_late_final(self):
        self.g['state'] = 'recording'
        self.deliver('partial', 'unfinished')
        self.g['cancel_recording']()
        self.deliver('final', 'stale result')
        self.assertEqual(self.text(), 'previous draft')
        self.mocks[3].assert_not_called()

    def test_error_restores_draft_and_controls(self):
        self.g['state'] = 'recording'
        self.deliver('partial', 'unfinished')
        self.deliver('error', 'service unavailable')
        self.assertEqual(self.text(), 'previous draft')
        self.assertEqual(str(self.g['record_button']['state']), 'disabled')
        self.assertEqual(self.g['state'], 'idle')
        self.mocks[3].assert_not_called()

    def test_stop_double_click_and_fast_final_cannot_restart(self):
        self.g['state'] = 'recording'
        with patch.object(self.g['engine'], 'stop') as stop, patch.object(self.g['engine'], 'start', return_value=8) as start:
            self.g['toggle_recording']()
            self.assertEqual(self.g['state'], 'finalizing')
            self.assertEqual(str(self.g['record_button']['state']), 'disabled')
            self.g['toggle_recording']()
            stop.assert_called_once()
            start.assert_not_called()
            self.deliver('final', 'long recording preserved')
            self.g['toggle_recording']()
            start.assert_not_called()
            self.assertEqual(self.text(), 'long recording preserved')
            self.assertEqual(self.g['history'].entries[0]['text'], 'long recording preserved')
            deadline = self.g['restart_allowed_at']
            with patch('time.monotonic', return_value=deadline + 1):
                self.g['toggle_recording'](requested_at=deadline - .2)
                start.assert_not_called()
                self.g['release_restart_guard']()
                self.assertEqual(str(self.g['record_button']['state']), 'normal')
                self.g['toggle_recording']()
                start.assert_called_once_with(False, self.g['app_config'].get('microphone'), self.g['app_config']['whisper_model'])

    def test_standard_default_and_fixed_labels(self):
        with patch.object(self.g['root'], 'winfo_width', return_value=600):
            self.g['refresh_controls']()
            self.assertEqual(self.g['record_button']['text'], 'Start Recording')
            self.assertFalse(self.g['real_time_var'].get())
            self.g['toggle_mode']()
            self.assertTrue(self.g['real_time_mode'])
            self.assertIn('REAL-TIME ON', self.g['transcription_frame']['text'])
            self.assertTrue(self.g['real_time_var'].get())
            self.assertEqual(self.g['record_button']['text'], 'Start Recording')
            self.assertEqual(self.g['mode_button']['text'], 'Real-time')
            self.g['toggle_mode']()
            self.assertFalse(self.g['real_time_mode'])

    def test_realtime_error_retains_current_transcription(self):
        self.g['real_time_mode'] = True
        self.g['state'] = 'recording'
        self.deliver('partial', 'Minutes of dictation already recognized.')
        self.deliver('error', 'Decoder error')
        self.assertEqual(self.text(), 'Minutes of dictation already recognized.')
        self.mocks[3].assert_not_called()

    def test_settings_open_directly_and_main_window_resizes(self):
        self.assertNotIn('settings_menu', self.g)
        self.assertNotIn('compact_mode_var', self.g)
        self.g['settings_button'].invoke()
        self.g['root'].update_idletasks()
        self.assertEqual(self.g['settings_notebook'].select(), str(self.g['settings_frame']))
        self.g['close_settings']()
        self.g['root'].geometry('700x440')
        self.g['root'].update_idletasks()
        self.assertEqual(self.g['root'].resizable(), (1, 1))

    def test_history_only_records_nonempty_final_and_can_restore(self):
        self.g['state'] = 'recording'
        self.deliver('partial', 'partial text')
        self.assertEqual(self.g['history'].entries, [])
        self.deliver('final', 'complete text')
        self.assertEqual(len(self.g['history'].entries), 1)
        entry = self.g['history'].entries[0]
        self.g['replace_transcript']('newer draft')
        self.mocks[3].reset_mock()
        self.assertTrue(self.g['restore_history_entry'](entry))
        self.assertEqual(self.text(), 'complete text')
        self.mocks[3].assert_not_called()
        self.g['state'] = 'recording'
        self.assertFalse(self.g['restore_history_entry']({'text': 'interrupt'}))
        self.g['active_id'] = 9
        self.deliver('final', '', session=9)
        self.assertEqual(len(self.g['history'].entries), 1)

    def test_history_window_empty_and_populated(self):
        self.g['open_history']()
        try:
            dialog = self.g['history_window']
            self.assertTrue(dialog.winfo_exists())
            self.g['remember_transcription']('A completed recording to restore later.')
            self.g['root'].update_idletasks()
            self.assertEqual(self.g['history_button']['text'], 'History (1)')
            def descendants(widget):
                for child in widget.winfo_children():
                    yield child
                    yield from descendants(child)
            copy = next(w for w in descendants(dialog)
                        if w.winfo_class() == 'TButton' and w['text'] == 'Copy to clipboard')
            copy.invoke()
            self.mocks[3].assert_called_once_with('A completed recording to restore later.')
            self.g['history_restore_button'].invoke()
            self.assertEqual(self.text(), 'A completed recording to restore later.')
        finally:
            self.g['close_settings']()


class HistoryTests(unittest.TestCase):
    def test_last_ten_survive_restart_with_exact_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.json'
            history = TranscriptionHistory(path)
            for i in range(12):
                history.add(f'Transcription {i}\nUnicode: café. Full text retained.')
            history.add('   ')
            loaded = TranscriptionHistory(path)
            self.assertEqual(len(loaded.entries), 10)
            self.assertEqual(loaded.entries[0]['text'], 'Transcription 11\nUnicode: café. Full text retained.')
            self.assertTrue(loaded.entries[-1]['text'].startswith('Transcription 2\n'))
            self.assertEqual([p.name for p in Path(directory).iterdir()], ['history.json'])

    def test_failed_save_preserves_previous_file_and_new_text_in_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.json'
            history = TranscriptionHistory(path)
            history.add('previous')
            with patch('transcription_history.os.replace', side_effect=OSError('disk unavailable')):
                with self.assertRaises(OSError):
                    history.add('new text')
            self.assertEqual(TranscriptionHistory(path).entries[0]['text'], 'previous')
            self.assertEqual(history.entries[0]['text'], 'new text')
            self.assertEqual(len(list(Path(directory).iterdir())), 1)

    def test_corrupt_history_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'history.json'
            path.write_text('broken json', encoding='utf-8')
            history = TranscriptionHistory(path)
            self.assertTrue(history.load_error)
            with self.assertRaises(OSError):
                history.add('still recoverable in memory')
            self.assertEqual(path.read_text(), 'broken json')


if __name__ == '__main__':
    unittest.main()
