"""Polite, login-free audio downloader for a batch file (default notes/batch1.tsv).
One video at a time, random pause between videos, bandwidth cap, exponential back-off when YouTube asks to sign in.
Audio lands in data/audio_src/<id>.<ext>; status in notes/download_status.tsv; notes/download.done when finished."""
import csv
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "data/audio_src"
PARTIAL = ROOT / "data/audio_partial"  # fragments of failed downloads (kept, never deleted)
STATUS = ROOT / "notes/download_status.tsv"
PAUSE_S = (20, 60)
BACKOFF_START_S, BACKOFF_MAX_S = 30 * 60, 4 * 3600
NETWORK_DOWN = ("Failed to resolve", "Temporary failure in name resolution", "Network is unreachable",
                "Connection refused", "Connection reset", "timed out", "RemoteDisconnected", "No route to host")
NETWAIT_S = 5 * 60
BOT_CHECK = ("Sign in to confirm", "Please sign in", "HTTP Error 429", "Too Many Requests")


def log(vid, status, note=""):
    with STATUS.open("a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}\t{vid}\t{status}\t{note[:200]}\n")


def quarantine(vid):
    """Move any fragment a failed yt-dlp run left in SRC aside, so a retry can't mistake it for a finished file."""
    PARTIAL.mkdir(parents=True, exist_ok=True)
    for f in SRC.glob(f"{vid}.*"):
        f.rename(PARTIAL / f"{f.name}.{int(time.time())}")


def main(batch):
    SRC.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(batch), delimiter="\t", quoting=csv.QUOTE_NONE))
    random.Random(2).shuffle(rows)  # interleave categories so partial data stays balanced
    # retry earlier failures on restart; only successful downloads count as done
    status_rows = [l.split("\t") for l in STATUS.open(errors="replace")] if STATUS.exists() else []
    done = {r[1] for r in status_rows if len(r) >= 3 and r[2] == "ok"}  # tolerate lines damaged by a crash
    backoff = BACKOFF_START_S
    for r in rows:
        vid = r["id"]
        if vid in done or next(SRC.glob(f"{vid}.*"), None):
            continue
        attempts = 0
        while True:
            attempts += 1
            p = subprocess.run([ROOT / ".venv/bin/yt-dlp", "-q", "--no-warnings", "-f", "bestaudio", "--no-part",
                                "--limit-rate", "3M", "--sleep-requests", "2",
                                "-o", str(SRC / "%(id)s.%(ext)s"), f"https://youtu.be/{vid}"],
                               capture_output=True, text=True, timeout=3 * 3600)
            err = (p.stderr or "").strip().replace("\n", " ")
            if p.returncode == 0:
                log(vid, "ok"); backoff = BACKOFF_START_S; break
            quarantine(vid)
            if any(s in err for s in NETWORK_DOWN):  # our connection is down, not the video: wait and retry it
                log(vid, "netwait", err); time.sleep(NETWAIT_S); attempts = 0
                continue
            if "confirm your age" in err or "age-restricted" in err:  # needs a login; we never use one, so skip
                log(vid, "failed", "age-restricted: " + err); break
            if any(s in err for s in BOT_CHECK):
                log(vid, "backoff", f"{backoff // 60} min: {err}")
                time.sleep(backoff); backoff = min(backoff * 2, BACKOFF_MAX_S)
                continue
            if ("403" in err or "more expected" in err) and attempts < 3:  # transient: rejected URL / cut transfer
                time.sleep(90); continue
            log(vid, "failed", err); break
        time.sleep(random.uniform(*PAUSE_S))
    (ROOT / "notes/download.done").touch()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "notes/batch1.tsv")
