"""Export transcripts as clean Mangalorean Kannada-script training text.

Per passage (data/transcripts/*.jsonl, raw Devanagari ASR):
  drop < MIN_WORDS words; drop >= 15-word passages whose Konkani-marker ratio < MIN_MARKER_RATIO (English/Kannada
  stretches); drop repetition loops; keep only the first copy of texts repeated >= DUP_MIN times (intros/outros).
Then Devanagari -> Kannada (aksharamukha) -> Mangalorean spelling (mangalorean_ortho), and passages of each video are
joined into documents of <= MAX_DOC_CHARS chars at passage boundaries. Validation = VALID_PCT% of videos per category
(at least 1, lowest id-hash first), so every content type is represented.
Writes data/corpus/{train,valid}.jsonl ({"text","id","channel","category"}) and notes/corpus_stats.md."""
import collections
import hashlib
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import csv  # noqa: E402

import langcheck  # noqa: E402
import mangalorean_ortho  # noqa: E402
from aksharamukha import transliterate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
MIN_WORDS, MIN_MARKER_RATIO, DUP_MIN, MAX_DOC_CHARS, VALID_PCT = 4, 0.03, 3, 1500, 2


def is_loop(words):
    return len(words) >= 10 and collections.Counter(words).most_common(1)[0][1] / len(words) > 0.4


def main():
    meta = {r["id"]: r for r in csv.DictReader(open(ROOT / "notes/batch1.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE)}
    files = sorted((ROOT / "data/transcripts").glob("*.jsonl"))
    videos = {f.stem: [json.loads(l)["deva"] for l in f.open()] for f in files}
    counts = collections.Counter(t for ps in videos.values() for t in ps)
    by_cat = collections.defaultdict(list)
    for vid in videos:
        by_cat[meta.get(vid, {}).get("category", "pilot")].append(vid)
    valid = set()
    for cat, vids in by_cat.items():
        vids.sort(key=lambda v: hashlib.md5(v.encode()).hexdigest())
        valid.update(vids[:max(1, round(len(vids) * VALID_PCT / 100))])
    seen, drops = set(), collections.Counter()
    stats = collections.defaultdict(lambda: collections.Counter())
    out_dir = ROOT / "data/corpus"; out_dir.mkdir(exist_ok=True)
    outs = {s: (out_dir / f"{s}.jsonl").open("w") for s in ("train", "valid")}
    for vid, passages in videos.items():
        m = meta.get(vid, {})
        cat = m.get("category", "pilot")
        split = "valid" if vid in valid else "train"
        kept = []
        for t in passages:
            w = t.split()
            if len(w) < MIN_WORDS:
                drops["short"] += 1; continue
            if len(w) >= 15 and sum(x in langcheck.MARKERS for x in w) / len(w) < MIN_MARKER_RATIO:
                drops["not_konkani"] += 1; continue
            if is_loop(w):
                drops["loop"] += 1; continue
            if counts[t] >= DUP_MIN:
                if t in seen:
                    drops["duplicate"] += 1; continue
                seen.add(t)
            kept.append(mangalorean_ortho.normalise(transliterate.process("Devanagari", "Kannada", t)))
        doc = ""
        for k in kept + [None]:
            if k is None or (doc and len(doc) + 1 + len(k) > MAX_DOC_CHARS):
                if doc:
                    outs[split].write(json.dumps({"text": doc, "id": vid, "channel": m.get("channel", ""),
                                                  "category": cat}, ensure_ascii=False) + "\n")
                    stats[(split, cat)]["docs"] += 1; stats[(split, cat)]["words"] += len(doc.split())
                doc = k or ""
            else:
                doc = f"{doc} {k}" if doc else k
        stats[(split, cat)]["videos"] += 1
    for f in outs.values():
        f.close()
    lines = ["# Corpus export stats", "", f"videos: {len(videos)}, passages: {sum(len(p) for p in videos.values())}",
             "dropped passages: " + ", ".join(f"{k} {v}" for k, v in drops.most_common()), "",
             "| split | category | videos | docs | words |", "|---|---|---|---|---|"]
    for (split, cat), c in sorted(stats.items(), key=lambda kv: (kv[0][0], -kv[1]["words"])):
        lines.append(f"| {split} | {cat} | {c['videos']} | {c['docs']} | {c['words']:,} |")
    tot = {s: sum(c["words"] for (sp, _), c in stats.items() if sp == s) for s in ("train", "valid")}
    lines += ["", f"total words: train {tot['train']:,}, valid {tot['valid']:,}"]
    (ROOT / "notes/corpus_stats.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
