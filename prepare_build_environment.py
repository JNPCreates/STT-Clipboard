"""Expose only required installed packages in a dedicated Windows build venv."""
import hashlib
import importlib.metadata as metadata
import json
import shutil
import subprocess
import sys
import venv
from pathlib import Path
from packaging.requirements import Requirement


def prepare(project):
    pending = ['pyinstaller', 'pyinstaller-hooks-contrib', 'openai-whisper',
               'pyaudio', 'pyperclip', 'keyboard']
    distributions = {}
    while pending:
        name = pending.pop()
        dist = metadata.distribution(name)
        key = dist.metadata['Name'].lower().replace('_', '-')
        if key in distributions:
            continue
        distributions[key] = dist
        for spec in dist.requires or []:
            requirement = Requirement(spec)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                pending.append(requirement.name)
    versions = {name: dist.version for name, dist in sorted(distributions.items())}
    identity = json.dumps({'source_python': sys.executable, 'versions': versions}, sort_keys=True)
    fingerprint = hashlib.sha256(identity.encode()).hexdigest()[:12]
    target = project / 'build' / ('python-' + fingerprint)
    python = target / 'Scripts' / 'python.exe'
    complete = target / 'stt-build-environment.json'
    if python.is_file() and complete.is_file():
        return python
    venv.EnvBuilder(with_pip=False).create(target)
    destination = target / 'Lib' / 'site-packages'
    links = {}
    for dist in distributions.values():
        for file in dist.files or []:
            first = Path(file).parts[0]
            if first in ('.', '..'):
                continue
            source = Path(dist.locate_file(first)).resolve()
            if not source.exists():
                continue
            dest = destination / first
            if source.is_dir():
                links[str(dest)] = str(source)
            elif not dest.exists():
                shutil.copy2(source, dest)
    plan = target / 'package-links.json'
    plan.write_text(json.dumps(links), encoding='utf-8')
    # Junctions expose installed binaries without copying gigabytes or changing the
    # original environment. Their targets are never removed by this helper.
    command = '''$plan = Get-Content -LiteralPath $args[0] -Raw | ConvertFrom-Json;
foreach ($item in $plan.PSObject.Properties) {
    if (-not (Test-Path -LiteralPath $item.Name)) {
        New-Item -ItemType Junction -Path $item.Name -Value $item.Value | Out-Null
    }
}'''
    script = target / 'link-packages.ps1'
    script.write_text(command, encoding='utf-8')
    subprocess.run(['powershell.exe', '-NoProfile', '-ExecutionPolicy', 'Bypass',
                    '-File', str(script), str(plan)], check=True)
    complete.write_text(identity, encoding='utf-8')
    return python
