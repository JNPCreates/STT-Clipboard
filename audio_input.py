"""Input device selection and lightweight PCM metering (no Whisper import)."""
from array import array
import math


def input_devices(audio):
    devices = []
    for index in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(index)
        if int(info['maxInputChannels']) > 0:
            host = audio.get_host_api_info_by_index(int(info['hostApi']))['name']
            devices.append({'index': index, 'name': info['name'], 'host_api': host})
    return devices


def list_microphones():
    import pyaudio
    audio = pyaudio.PyAudio()
    try:
        return input_devices(audio)
    finally:
        audio.terminate()


def resolve_microphone(audio, selection):
    if selection is None:
        return None  # Follow the current Windows default input device.
    devices = input_devices(audio)
    if isinstance(selection, int) and not isinstance(selection, bool):
        matches = [d for d in devices if d['index'] == selection]
    elif isinstance(selection, dict):
        matches = [d for d in devices if d['name'] == selection.get('name')
                   and d['host_api'] == selection.get('host_api')]
        # A device number may change after a reboot or reconnect.
        matches.sort(key=lambda d: d['index'] != selection.get('index'))
    else:
        matches = []
    if not matches:
        raise OSError('The selected microphone is unavailable. Open Settings > Audio, '
                      'refresh the list, and choose a connected microphone.')
    return matches[0]['index']


class InputLevel:
    """A logarithmic meter; QUIET indicates low input, not proven microphone mute."""
    def __init__(self, rate=16000):
        self.rate = rate
        self.quiet_samples = 0

    def update(self, pcm):
        samples = array('h', pcm)
        rms = math.sqrt(sum(float(s) * s for s in samples) / max(1, len(samples))) / 32768
        self.quiet_samples = self.quiet_samples + len(samples) if rms < .003 else 0
        db = 20 * math.log10(max(rms, 1e-9))
        return {'level': max(0.0, min(1.0, (db + 60) / 60)),
                'quiet': self.quiet_samples >= self.rate * 2}
