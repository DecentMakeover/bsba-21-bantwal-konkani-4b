"""Build batch 1 (~500 h raw) from notes/inventory.tsv: conversational + everyday Konkani first, sermons capped,
liturgy / music excluded. Writes notes/batch1.tsv (id, channel, category, duration_s, title)."""
import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MIN_S, MAX_S = 180, 3 * 3600          # skip shorts/trailers and multi-hour streams
SERMON_HOURS = 150
LITURGY = re.compile(r"ಮಾಗ್ಣಿಂ|prayer|rosary|ರೊಜಾರ್|novena|ನೊವೆನ|ಭಕ್ತಿಪಣ್|adoration|ಆರಾಧನ್|\bmass\b|ಮಿಸ್|ಮೀಸ್|hymn|ಗಿತಾಂ|"
                     r"ಗೀತ್|song|feast|flag|ತೇರ್ಸ್|ತೆರ್ಸ್|way of the cross|ಖುರ್ಸಾ|live|🅻🅸🆅🅴|🔴|choir|ಮಾಳ್|"
                     r"music|devotional|ಭಕ್ತಿಕ್|jubilee|ordination|funeral|wedding|lyric|psalm|praise|worship|ನವೆನ್|singing|reels|"
                     r"competition|ಕೀರ್ತನಾಂ|karaoke|teaser|trailer|promo", re.I)
CHANNELS = json.loads((ROOT / "notes/channels.json").read_text())  # format: examples/channels.example.json
CONVERSATION = re.compile("|".join(filter(None, [CHANNELS.get("conversation_shows", ""),
                                                  r"episode|\bep\.? ?\d|podcast|interview|ಸಂವಾದ್"])), re.I)
TALK = re.compile(r"\bfr\b|\bfr\.|father|\brev\b|\bsr\b|sister|bishop|ಬಾಪ್|reflection|sermon|retreat|talk", re.I)
EVERYDAY = CHANNELS["everyday"]        # channel -> category, all taken
SERMON_CHANNELS = set(CHANNELS["sermon"])  # capped at SERMON_HOURS
MIXED = set(CHANNELS["mixed"])          # split by title: conversation / talk (sermon pool) / other


def main():
    rows = [r for r in csv.DictReader(open(ROOT / "notes/inventory.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE)
            if MIN_S <= int(r["duration_s"]) <= MAX_S and not LITURGY.search(r["title"])]
    picked, sermon_pool, seen = [], [], set()
    for r in rows:
        if r["id"] in seen:
            continue
        seen.add(r["id"])
        ch, title = r["channel"], r["title"]
        if ch in EVERYDAY:
            picked.append((r, EVERYDAY[ch]))
        elif ch in MIXED:
            if CONVERSATION.search(title):
                picked.append((r, "conversation"))
            elif TALK.search(title):
                sermon_pool.append(r)
            else:
                picked.append((r, "mixed-other"))
        elif ch in SERMON_CHANNELS:
            sermon_pool.append(r)
    random.Random(1).shuffle(sermon_pool)
    budget = SERMON_HOURS * 3600
    for r in sermon_pool:
        if budget <= 0:
            break
        picked.append((r, "sermon")); budget -= int(r["duration_s"])
    hours = defaultdict(float)
    with open(ROOT / "notes/batch1.tsv", "w") as f:
        f.write("id\tchannel\tcategory\tduration_s\ttitle\n")
        for r, cat in picked:
            f.write(f"{r['id']}\t{r['channel']}\t{cat}\t{r['duration_s']}\t{r['title']}\n")
            hours[cat] += int(r["duration_s"]) / 3600
    for cat, h in sorted(hours.items(), key=lambda kv: -kv[1]):
        print(f"{cat:14s} {h:6.0f} h")
    print(f"{'TOTAL':14s} {sum(hours.values()):6.0f} h, {len(picked)} videos")


if __name__ == "__main__":
    main()
