"""A configurable number of recent transcriptions. Audio is never stored here."""
import json
import os
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4


class TranscriptionHistory:
    @staticmethod
    def validate_limit(limit):
        if isinstance(limit, bool) or not str(limit).isdigit() or not 1 <= int(limit) <= 1000:
            raise ValueError('Choose a history limit from 1 to 1000.')
        return int(limit)

    def __init__(self, path, limit=10):
        self.path = Path(path)
        self.limit = self.validate_limit(limit)
        self.entries = []
        self.load_error = ''
        if self.path.exists():
            try:
                data = json.loads(self.path.read_text(encoding='utf-8'))
                if not isinstance(data, list):
                    raise ValueError('Expected a list of transcriptions')
                for entry in data:
                    if not (isinstance(entry, dict) and all(isinstance(entry.get(k), str)
                            for k in ('id', 'created_at', 'text'))):
                        raise ValueError('Invalid transcription entry')
                self.entries = data[:self.limit]
            except (OSError, ValueError) as exc:
                self.load_error = str(exc)

    def add(self, text):
        if not text.strip():
            return
        self.entries.insert(0, {'id': uuid4().hex,
            'created_at': datetime.now().astimezone().isoformat(timespec='seconds'), 'text': text})
        del self.entries[self.limit:]
        self._save()

    def set_limit(self, limit):
        limit = self.validate_limit(limit)
        old_limit, old_entries = self.limit, self.entries
        self.limit, self.entries = limit, self.entries[:limit]
        try:
            self._save()
        except OSError:
            self.limit, self.entries = old_limit, old_entries
            raise

    def _save(self):
        if self.load_error:
            raise OSError('Existing history could not be read; it has been left untouched. '
                          'New entries are available for this session only. ' + self.load_error)
        # Replace atomically so interrupted writes cannot truncate the previous history.
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                                             prefix='.history-', suffix='.tmp', delete=False) as f:
                temporary = f.name
                json.dump(self.entries, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary and os.path.exists(temporary):
                os.unlink(temporary)
