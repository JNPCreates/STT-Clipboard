"""Create minimal source/app ZIPs and collect bundled dependency notices."""
import hashlib
import importlib.metadata as metadata
from pathlib import Path
import re
import shutil
import sys
import zipfile

SOURCE_FILES = (
    'STT_to_clipboard.py', 'speech_engine.py', 'whisper_models.py', 'audio_input.py',
    'transcription_history.py', 'package_check.py', 'build_exe.py',
    'prepare_build_environment.py', 'release_files.py', 'STT_Clipboard.spec',
    'Start STT Clipboard.cmd', 'Build STT Clipboard.cmd',
    'requirements.txt', 'requirements-build.txt', 'README.md', 'BUILDING.txt',
    'LICENSE.txt', 'THIRD_PARTY_NOTICES.txt', '.gitignore',
    'stt_Clipboard.ico', 'chimes.wav', 'notification sound.wav',
    'test_speech_engine.py', 'test_model_settings.py', 'test_audio_input.py',
)


def dependency_notices(app):
    destination = app / 'licenses'
    destination.mkdir(exist_ok=True)
    for dist in metadata.distributions():
        name = re.sub(r'[^A-Za-z0-9_.-]', '_', dist.metadata['Name'])
        for file in dist.files or []:
            relative = Path(file)
            if '..' in relative.parts or relative.is_absolute():
                continue
            if not relative.name.upper().startswith(('LICENSE', 'COPYING', 'NOTICE')):
                continue
            source = Path(dist.locate_file(file))
            if source.is_file():
                target = destination / name / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    python_license = Path(sys.base_prefix) / 'LICENSE.txt'
    if python_license.is_file():
        shutil.copy2(python_license, destination / 'Python-LICENSE.txt')
    for terms in (Path(sys.base_prefix) / 'tcl').glob('*/license.terms'):
        shutil.copy2(terms, destination / (terms.parent.name + '-license.terms'))


def archive_groups(paths, limit=1900 * 1024 * 1024):
    """Keep each ZIP below GitHub's 2 GiB limit, even for incompressible data."""
    groups, sizes = [[]], [0]
    for path in sorted(paths, key=lambda p: p.stat().st_size, reverse=True):
        size = path.stat().st_size
        if size > limit:
            raise ValueError(f'{path.name} is too large for a release ZIP.')
        index = min(range(len(groups)), key=sizes.__getitem__)
        if sizes[index] + size > limit:
            groups.append([])
            sizes.append(0)
            index = len(groups) - 1
        groups[index].append(path)
        sizes[index] += size
    return groups


def create_source_archive(project, destination):
    with zipfile.ZipFile(destination, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for filename in SOURCE_FILES:
            archive.write(project / filename, 'STT-Clipboard-Source/' + filename)


def write_checksums(paths, destination):
    lines = []
    for path in paths:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        lines.append(f'{digest.hexdigest()}  {path.name}')
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def create_archives(project, app, flavor):
    source_zip = app.parent / 'STT-Clipboard-Source.zip'
    groups = archive_groups([p for p in app.rglob('*') if p.is_file()])
    archives = []
    for index, group in enumerate(groups, 1):
        suffix = f'-{index}-of-{len(groups)}' if len(groups) > 1 else ''
        app_zip = app.parent / f'STT-Clipboard-Windows-{flavor}{suffix}.zip'
        with zipfile.ZipFile(app_zip, 'w', zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for path in sorted(group):
                archive.write(path, path.relative_to(app.parent))
        if app_zip.stat().st_size >= 2 * 1024**3:
            raise ValueError(f'{app_zip.name} exceeds GitHub\'s release limit.')
        archives.append(app_zip)
    create_source_archive(project, source_zip)
    archives.append(source_zip)
    write_checksums(archives, app.parent / 'SHA256SUMS.txt')
    return tuple(archives)
