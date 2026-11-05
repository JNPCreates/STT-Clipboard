from array import array
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch
import zipfile

from audio_input import InputLevel, input_devices, list_microphones, resolve_microphone


class AudioInputTests(unittest.TestCase):
    def audio(self):
        audio = Mock()
        audio.get_device_count.return_value = 3
        audio.get_device_info_by_index.side_effect = [
            {'name': 'Speakers', 'maxInputChannels': 0, 'hostApi': 0},
            {'name': 'USB microphone', 'maxInputChannels': 1, 'hostApi': 0},
            {'name': 'USB microphone', 'maxInputChannels': 1, 'hostApi': 1}]
        audio.get_host_api_info_by_index.side_effect = lambda i: {'name': ['MME', 'WASAPI'][i]}
        return audio

    def test_only_inputs_are_listed_with_driver_identity(self):
        devices = input_devices(self.audio())
        self.assertEqual([d['index'] for d in devices], [1, 2])
        self.assertEqual([d['host_api'] for d in devices], ['MME', 'WASAPI'])

    def test_device_reconnect_resolves_name_and_driver_not_old_index(self):
        selected = {'index': 99, 'name': 'USB microphone', 'host_api': 'WASAPI'}
        self.assertEqual(resolve_microphone(self.audio(), selected), 2)
        self.assertIsNone(resolve_microphone(self.audio(), None))
        self.assertEqual(resolve_microphone(self.audio(), 1), 1)

    def test_missing_device_does_not_silently_fall_back(self):
        with self.assertRaisesRegex(OSError, 'unavailable'):
            resolve_microphone(self.audio(), {'name': 'unplugged', 'host_api': 'MME'})

    def test_enumeration_releases_audio_on_failure(self):
        audio = self.audio()
        audio.get_device_count.side_effect = OSError('audio unavailable')
        with patch.dict('sys.modules', pyaudio=Mock(PyAudio=Mock(return_value=audio))):
            with self.assertRaises(OSError):
                list_microphones()
        audio.terminate.assert_called_once()

    def test_silence_warning_threshold_and_voice_recovery(self):
        meter = InputLevel()
        self.assertFalse(meter.update(b'\x00\x00' * 16000)['quiet'])
        self.assertEqual(meter.update(b'\x00\x00' * 16000), {'level': 0, 'quiet': True})
        voice = meter.update(array('h', [4000, -4000] * 512).tobytes())
        self.assertGreater(voice['level'], .5)
        self.assertFalse(voice['quiet'])
        self.assertFalse(meter.update(b'\x00\x00' * 16000)['quiet'])

    def test_meter_increases_with_input_and_stays_bounded(self):
        meter = InputLevel()
        levels = [meter.update(array('h', [amplitude] * 1024).tobytes())['level']
                  for amplitude in (0, 10, 100, 1000, 10000, -32768)]
        self.assertEqual(levels, sorted(levels))
        self.assertEqual(levels[-1], 1)

    def test_source_archive_includes_only_allowlisted_files(self):
        from release_files import SOURCE_FILES, create_archives
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            for name in SOURCE_FILES:
                (project / name).write_text('test')
            (project / '.env').write_text('not for release')
            (project / 'transcription_history.json').write_text('private')
            app = project / 'dist' / 'release' / 'STT_Clipboard'
            app.mkdir(parents=True)
            (app / 'STT_Clipboard.exe').write_bytes(b'fixture')
            _, source = create_archives(project, app, 'CPU')
            with zipfile.ZipFile(source) as archive:
                self.assertEqual(len(archive.namelist()), len(SOURCE_FILES))
                self.assertFalse(any('.env' in name or 'transcription_history.json' in name for name in archive.namelist()))

    def test_release_groups_stay_within_limit_without_omitting_files(self):
        from release_files import archive_groups
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            for i, size in enumerate((76, 53, 51, 28, 28, 27, 13, 10, 1)):
                path = Path(directory) / str(i)
                path.write_bytes(b'x' * size)
                paths.append(path)
            groups = archive_groups(paths, limit=150)
            self.assertEqual({p for group in groups for p in group}, set(paths))
            self.assertEqual(sum(len(group) for group in groups), len(paths))
            self.assertTrue(all(sum(p.stat().st_size for p in group) <= 150 for group in groups))
            with self.assertRaises(ValueError):
                archive_groups(paths, limit=50)


if __name__ == '__main__':
    unittest.main()
