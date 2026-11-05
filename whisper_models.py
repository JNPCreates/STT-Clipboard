"""Whisper model choices and verified, cancellable downloads without a console."""
import hashlib
import os
from pathlib import Path
import tempfile
import urllib.request


class Cancelled(Exception):
    pass


# Approximate checkpoint sizes in decimal MB/GB, smallest first.
# English-only checkpoints are slightly smaller than their multilingual peers.
MODEL_CHOICES = (
    ('tiny.en', 'Tiny.en (English only) — 76 MB'),
    ('tiny', 'Tiny — 76 MB'),
    ('base.en', 'Base.en (English only) — 145 MB'),
    ('base', 'Base (default) — 145 MB'),
    ('small.en', 'Small.en (English only) — 484 MB'),
    ('small', 'Small — 484 MB'),
    ('medium.en', 'Medium.en (English only) — 1.53 GB'),
    ('medium', 'Medium — 1.53 GB'),
    ('turbo', 'Turbo (large-v3-turbo) — 1.62 GB'),
    ('large-v1', 'Large-v1 — 3.09 GB'),
    ('large-v2', 'Large-v2 — 3.09 GB'),
    ('large-v3', 'Large-v3 (large) — 3.09 GB'),
)
MODEL_NAMES = tuple(name for name, _ in MODEL_CHOICES)


def normalize_model(name):
    name = {'large': 'large-v3', 'large-v3-turbo': 'turbo'}.get(name, name)
    return name if name in MODEL_NAMES else 'base'


def cache_directory():
    return Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'whisper'


def checkpoint_path(app_dir, name):
    filename = ('large-v3-turbo' if name == 'turbo' else name) + '.pt'
    bundled = Path(app_dir) / 'models' / filename
    return bundled if bundled.is_file() else cache_directory() / filename


def ensure_checkpoint(app_dir, name, url, stopped, progress):
    """Never expose incomplete downloads as checkpoints or replace valid files."""
    expected = url.split('/')[-2]
    target = checkpoint_path(app_dir, name)

    def check_stop():
        if stopped.is_set():
            raise Cancelled()

    check_stop()
    for candidate in dict.fromkeys((target, cache_directory() / Path(url).name)):
        if candidate.is_file():
            progress('loading', f'Checking {name}…')
            digest = hashlib.sha256()
            with candidate.open('rb') as source:
                while block := source.read(1024 * 1024):
                    check_stop()
                    digest.update(block)
            if digest.hexdigest() == expected:
                return candidate
    # Downloads go to the user's cache, even if a bundled checkpoint is damaged.
    target = cache_directory() / Path(url).name
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        progress('download', 0)
        with urllib.request.urlopen(url, timeout=15) as source:
            total = int(source.headers.get('Content-Length', 0))
            digest = hashlib.sha256()
            received, last_percent = 0, -1
            with tempfile.NamedTemporaryFile(dir=target.parent, prefix='.download-',
                                             suffix='.tmp', delete=False) as output:
                temporary = output.name
                while True:
                    check_stop()
                    block = source.read(256 * 1024)
                    if not block:
                        break
                    output.write(block)
                    digest.update(block)
                    received += len(block)
                    percent = min(100, int(received * 100 / total)) if total else 0
                    if percent != last_percent:
                        progress('download', percent)
                        last_percent = percent
                output.flush()
                os.fsync(output.fileno())
        check_stop()
        if digest.hexdigest() != expected:
            raise OSError('The downloaded model failed verification. Please try again.')
        os.replace(temporary, target)
        return target
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
