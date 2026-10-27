"""Inventory candidate channels (no downloads): resolve each seed video's channel, list its videos + streams
with durations, apply an optional title filter, and write notes/inventory.tsv + a per-channel hours summary."""
import csv
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
YTDLP = str(ROOT / ".venv/bin/yt-dlp")


def ytdlp_json(url, flat=True):
    args = [YTDLP, "-J", "--no-warnings", "--ignore-errors"] + (["--flat-playlist"] if flat else []) + [url]
    out = subprocess.run(args, capture_output=True, text=True, timeout=900).stdout
    return (json.loads(out) if out.strip() else None) or {}


def main():
    seeds = list(csv.DictReader(open(ROOT / "notes/seed_channels.tsv"), delimiter="\t"))
    rows, summary = [], []
    for s in seeds:
        info = ytdlp_json(f"https://youtu.be/{s['seed_video']}", flat=False)
        chan_url = info.get("channel_url")
        if not chan_url:
            summary.append((s["name"], "?", 0, 0.0)); continue
        filt = re.compile(s["title_filter"], re.I) if s["title_filter"] else None
        seen, n, secs = set(), 0, 0
        for tab in ("videos", "streams"):
            for e in ytdlp_json(f"{chan_url}/{tab}").get("entries") or []:
                if not e or e.get("id") in seen or (filt and not filt.search(e.get("title") or "")):
                    continue
                seen.add(e["id"]); d = e.get("duration") or 0
                n += 1; secs += d
                rows.append([s["name"], s["genre"], s["variety"], tab, e["id"], int(d), (e.get("title") or "").replace("\t", " ")])
        summary.append((s["name"], chan_url, n, secs / 3600))
        print(f"{s['name']}: {n} videos, {secs/3600:.1f} h", flush=True)
    with open(ROOT / "notes/inventory.tsv", "w") as f:
        f.write("channel\tgenre\tvariety\ttab\tid\tduration_s\ttitle\n")
        f.writelines("\t".join(map(str, r)) + "\n" for r in rows)
    with open(ROOT / "notes/inventory_summary.tsv", "w") as f:
        f.write("channel\tchannel_url\tvideos\thours\n")
        f.writelines(f"{a}\t{b}\t{c}\t{d:.1f}\n" for a, b, c, d in summary)
    print(f"TOTAL: {len(rows)} videos, {sum(int(r[5]) for r in rows)/3600:.0f} h")


if __name__ == "__main__":
    main()
