# Build with the same Python environment used to test speech_engine.py.
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, copy_metadata

project = Path(SPECPATH)
datas = [(str(project / name), '.') for name in
          ('stt_Clipboard.ico', 'notification sound.wav', 'chimes.wav')]
datas += collect_data_files('whisper')
datas += copy_metadata('openai-whisper')
datas += copy_metadata('tiktoken')

a = Analysis(
    [str(project / 'STT_to_clipboard.py')],
    pathex=[str(project)],
    datas=datas,
    hiddenimports=['tiktoken_ext.openai_public'],
    # Keep source available for Numba/Whisper's runtime inspection.
    module_collection_mode={'whisper': 'pyz+py'},
    excludes=['IPython', 'matplotlib', 'pandas', 'scipy', 'sklearn',
              'tensorflow', 'jax', 'jaxlib', 'transformers', 'torchaudio',
              'torchvision', 'tensorboard', 'cv2', 'PyQt5', 'PyQt6', 'PySide6'],
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True,
    name='STT_Clipboard', console=False, debug=False, strip=False, upx=False,
    icon=str(project / 'stt_Clipboard.ico'),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name='STT_Clipboard')
