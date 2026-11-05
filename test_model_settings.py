import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

from speech_engine import SpeechBackend, SpeechEngine
from test_speech_engine import wait_terminal
from transcription_history import TranscriptionHistory
from whisper_models import Cancelled, ensure_checkpoint, normalize_model


class ModelSettingsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.folder = Path(self.directory.name)
        self.cache = self.folder / 'cache'
        self.payload = b'valid test checkpoint' * 100
        self.url = 'https://example.invalid/' + hashlib.sha256(self.payload).hexdigest() + '/tiny.pt'
        self.cache_patch = patch('whisper_models.cache_directory', return_value=self.cache)
        self.cache_patch.start()
        self.addCleanup(self.cache_patch.stop)

    def response(self, data=None):
        response = io.BytesIO(self.payload if data is None else data)
        response.headers = {'Content-Length': str(len(response.getvalue()))}
        return response

    def test_download_verified_and_reused_without_network(self):
        progress = Mock()
        with patch('urllib.request.urlopen', return_value=self.response()) as request:
            path = ensure_checkpoint(self.folder, 'tiny', self.url, threading.Event(), progress)
            self.assertEqual(path.read_bytes(), self.payload)
            self.assertEqual(ensure_checkpoint(self.folder, 'tiny', self.url,
                threading.Event(), progress), path)
            request.assert_called_once()
        self.assertIn(unittest.mock.call('download', 100), progress.call_args_list)
        self.assertEqual(list(self.cache.glob('*.tmp')), [])

    def test_failed_verification_preserves_existing_checkpoint(self):
        self.cache.mkdir()
        target = self.cache / 'tiny.pt'
        target.write_bytes(b'old checkpoint')
        with patch('urllib.request.urlopen', return_value=self.response(b'broken')):
            with self.assertRaisesRegex(OSError, 'verification'):
                ensure_checkpoint(self.folder, 'tiny', self.url, threading.Event(), Mock())
        self.assertEqual(target.read_bytes(), b'old checkpoint')
        self.assertEqual(list(self.cache.glob('*.tmp')), [])

    def test_cancel_download_removes_partial_and_allows_retry(self):
        stopped = threading.Event()
        def progress(kind, value):
            if kind == 'download' and value > 0:
                stopped.set()
        with patch('urllib.request.urlopen', return_value=self.response()):
            with self.assertRaises(Cancelled):
                ensure_checkpoint(self.folder, 'tiny', self.url, stopped, progress)
        self.assertEqual(list(self.cache.iterdir()), [])
        with patch('urllib.request.urlopen', return_value=self.response()):
            self.assertTrue(ensure_checkpoint(self.folder, 'tiny', self.url,
                                             threading.Event(), Mock()).is_file())

    def test_switch_model_reloads_once_and_reuses_selection(self):
        torch, whisper = Mock(), Mock()
        torch.cuda.is_available.return_value = False
        whisper._MODELS = {'tiny': self.url, 'base': self.url}
        whisper._ALIGNMENT_HEADS = {'tiny': b'tiny', 'base': b'base'}
        backend = SpeechBackend(self.folder)
        with patch.dict('sys.modules', torch=torch, whisper=whisper), \
                patch('speech_engine.ensure_checkpoint', return_value=self.folder / 'test.pt'):
            backend.prepare(False, threading.Event(), 'tiny')
            backend.prepare(True, threading.Event(), 'tiny')
            self.assertEqual(whisper.load_model.call_count, 1)
            backend.prepare(False, threading.Event(), 'base')
            self.assertEqual(whisper.load_model.call_count, 2)
            self.assertEqual(backend.model_name, 'base')

    def test_stop_during_preparation_does_not_open_microphone(self):
        entered = threading.Event()
        backend, capture = Mock(), Mock()
        def prepare(live, stopped, model_name, progress):
            self.assertEqual(model_name, 'tiny')
            entered.set()
            stopped.wait(2)
            raise Cancelled()
        backend.prepare.side_effect = prepare
        engine = SpeechEngine(backend, capture)
        engine.start(True, model_name='tiny')
        self.assertTrue(entered.wait(2))
        engine.stop()
        self.assertEqual(wait_terminal(engine)[-1][1:], ('final', ''))
        capture.assert_not_called()

    def test_download_failure_is_reported_before_microphone_opens(self):
        backend, capture = Mock(), Mock()
        backend.prepare.side_effect = OSError('Network unavailable')
        engine = SpeechEngine(backend, capture)
        engine.start(True, model_name='tiny')
        self.assertEqual(wait_terminal(engine)[-1][1:], ('error', 'Network unavailable'))
        capture.assert_not_called()

    def test_model_aliases(self):
        self.assertEqual(normalize_model('large'), 'large-v3')
        self.assertEqual(normalize_model('large-v3-turbo'), 'turbo')
        self.assertEqual(normalize_model('unknown'), 'base')

    def test_standard_captures_before_prepare_and_retries_without_recapture(self):
        captured = threading.Event()
        payload = b'\x01\x01' * 16000
        backend = Mock()
        def prepare(live, cancelled, name, progress):
            self.assertTrue(captured.is_set())
            self.assertFalse(cancelled.is_set())
            progress('download', 50)
            if backend.prepare.call_count == 1:
                raise OSError('Network unavailable')
            return 'local'
        backend.prepare.side_effect = prepare
        backend.transcribe.return_value = 'recovered speech'
        def record(session, microphone):
            session.audio.extend(payload)
            captured.set()
            session.stop.wait(2)
        capture = Mock(side_effect=record)
        engine = SpeechEngine(backend, capture)
        engine.start(False, model_name='tiny')
        self.assertTrue(captured.wait(2))
        backend.prepare.assert_not_called()
        engine.stop()
        self.assertEqual(wait_terminal(engine)[-1][1], 'error')
        self.assertEqual(engine.pending_audio, payload)
        first = engine.current
        self.assertTrue(first.finished.wait(2))
        self.assertEqual(first.audio, bytearray())
        engine.retry('base')
        self.assertEqual(wait_terminal(engine)[-1][1:], ('final', 'recovered speech'))
        capture.assert_called_once()
        self.assertEqual(backend.prepare.call_args.args[2], 'base')
        self.assertEqual(backend.transcribe.call_args.args[0], payload)
        self.assertIsNone(engine.pending_audio)

    def test_empty_standard_recording_does_not_download(self):
        backend = Mock()
        engine = SpeechEngine(backend, lambda *_: None)
        engine.start(False)
        self.assertEqual(wait_terminal(engine)[-1][1:], ('final', ''))
        backend.prepare.assert_not_called()

    def test_public_release_omits_personal_data_and_personal_build_opts_in(self):
        from build_exe import copy_user_data
        project, app = self.folder / 'source', self.folder / 'release'
        project.mkdir()
        app.mkdir()
        (project / 'stt_settings.json').write_text('{"history_limit":25}')
        (project / 'transcription_history.json').write_text('["private transcript"]')
        copy_user_data(project, app)
        self.assertEqual(list(app.iterdir()), [])
        copy_user_data(project, app, include_personal=True)
        self.assertEqual(json.loads((app / 'stt_settings.json').read_text())['history_limit'], 25)
        self.assertEqual((app / 'transcription_history.json').read_text(), '["private transcript"]')

    def test_history_twenty_five_and_reduction_survive_restart(self):
        path = self.folder / 'history.json'
        history = TranscriptionHistory(path)
        history.set_limit(25)
        for index in range(30):
            history.add(str(index))
        reloaded = TranscriptionHistory(path, 25)
        self.assertEqual(len(reloaded.entries), 25)
        self.assertEqual(reloaded.entries[-1]['text'], '5')
        reloaded.set_limit(3)
        self.assertEqual([e['text'] for e in json.loads(path.read_text())], ['29', '28', '27'])

    def test_history_invalid_limits_and_failed_save_leave_history_intact(self):
        history = TranscriptionHistory(self.folder / 'history.json')
        history.add('keep this')
        for value in ('', 0, -1, 1001, '1.5', True):
            with self.assertRaises(ValueError):
                history.set_limit(value)
        with patch('transcription_history.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                history.set_limit(25)
        self.assertEqual(history.limit, 10)
        self.assertEqual(history.entries[0]['text'], 'keep this')


if __name__ == '__main__':
    unittest.main()
