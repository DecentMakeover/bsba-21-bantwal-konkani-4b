"""Require a complete downloader receipt before ASR consumes a final audio path."""
import re
from pathlib import Path

VIDEO_ID = re.compile(r"[A-Za-z0-9_-]{11}")
AUDIO_SUFFIXES = {".webm", ".m4a", ".mp4", ".ogg", ".opus", ".mp3", ".wav", ".flac"}


def completed_ids(status_path):
    """Latest well-formed, newline-terminated receipt wins; damaged history is ignored."""
    latest = {}
    path = Path(status_path)
    if not path.exists():
        return set()
    for line in path.read_text(errors="replace").splitlines(keepends=True):
        if not line.endswith("\n") or "\x00" in line or "\ufffd" in line:
            continue
        fields = line.rstrip("\r\n").split("\t")
        if len(fields) < 3 or not VIDEO_ID.fullmatch(fields[1]):
            continue
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", fields[0]):
            continue
        if fields[2] in {"ok", "failed", "backoff", "netwait"}:
            latest[fields[1]] = fields[2]
    return {vid for vid, state in latest.items() if state == "ok"}


def ready_audio(src_dir, vid, complete):
    if vid not in complete:
        return None
    candidates = sorted(p for p in Path(src_dir).glob(f"{vid}.*")
                        if p.is_file() and p.suffix.lower() in AUDIO_SUFFIXES and p.stat().st_size > 0)
    return candidates[0] if candidates else None
