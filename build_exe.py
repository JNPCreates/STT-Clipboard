"""Create a new portable release; never rebuild over an installed app's data."""
import importlib.metadata
import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from prepare_build_environment import prepare
from release_files import dependency_notices, create_archives


def copy_user_data(project, app, include_personal=False):
    """Public releases start with defaults and no transcription history."""
    if not include_personal:
        return
    settings_file = project / 'stt_settings.json'
    if settings_file.exists():
        settings = json.loads(settings_file.read_text(encoding='utf-8'))
        sound = Path(settings.get('notification_sound', 'notification sound.wav'))
        if sound.name in ('chimes.wav', 'notification sound.wav'):
            settings['notification_sound'] = sound.name
        elif sound.is_file():
            destination = app / '_internal' / 'custom_notification.wav'
            shutil.copy2(sound, destination)
            settings['notification_sound'] = 'custom_notification.wav'
        (app / 'stt_settings.json').write_text(json.dumps(settings, indent=2), encoding='utf-8')
    history = project / 'transcription_history.json'
    if history.exists():
        shutil.copy2(history, app / history.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--personal', action='store_true',
                        help='Include your settings and transcription history; do not distribute this build.')
    parser.add_argument('--zip', action='store_true', help='Create public app and source ZIPs.')
    args = parser.parse_args()
    if args.personal and args.zip:
        parser.error('--zip produces public downloads; omit --personal.')
    import torch
    flavor = 'CPU' if torch.version.cuda is None else 'GPU'
    project = Path(__file__).resolve().parent
    build_python = prepare(project)
    print(f'Isolated build Python: {build_python}', flush=True)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S') + '-' + flavor.lower()
    release = project / 'dist' / stamp
    logs = project / 'build' / stamp
    logs.mkdir(parents=True, exist_ok=False)
    versions = {'python': sys.version, 'flavor': flavor}
    for name in ('pyinstaller', 'pyinstaller-hooks-contrib', 'torch', 'openai-whisper',
                 'numpy', 'numba', 'llvmlite', 'tiktoken', 'pyaudio', 'pyperclip', 'keyboard'):
        versions[name] = importlib.metadata.version(name)
    (logs / 'environment.json').write_text(json.dumps(versions, indent=2), encoding='utf-8')
    print(f'Building a separate release in {release}', flush=True)
    print(f'Build log: {logs / "pyinstaller.log"}', flush=True)
    with (logs / 'pyinstaller.log').open('w', encoding='utf-8') as log:
        result = subprocess.run([str(build_python), '-m', 'PyInstaller', '--clean',
            '--distpath', str(release), '--workpath', str(logs / 'work'),
            str(project / 'STT_Clipboard.spec')], cwd=project, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        print(f'Build failed. See {logs / "pyinstaller.log"}', flush=True)
        return result.returncode
    app = release / 'STT_Clipboard'
    copy_user_data(project, app, args.personal)
    shutil.copy2(logs / 'environment.json', app / 'build-environment.json')
    shutil.copy2(project / 'BUILDING.txt', app / 'BUILDING.txt')
    for filename in ('README.md', 'LICENSE.txt', 'THIRD_PARTY_NOTICES.txt'):
        shutil.copy2(project / filename, app / filename)
    dependency_notices(app)
    if args.zip:
        for archive in create_archives(project, app, flavor):
            print(f'ZIP ready: {archive}', flush=True)
    print(f'Build complete: {app / "STT_Clipboard.exe"}', flush=True)
    print('PERSONAL build: contains your data; do not distribute.' if args.personal
          else 'Public build: default settings and no personal transcription history.', flush=True)
    print('Keep the entire STT_Clipboard folder together. Your previous installation is unchanged.', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
