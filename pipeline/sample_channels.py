"""Estimate how much of each inventoried channel is really Konkani: language-check N random videos per channel
(fixed seed) and extrapolate to hours. Writes notes/channel_langcheck.tsv (per video) and prints a per-channel summary."""
import collections
import csv
import random
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import langcheck  # noqa: E402

ROOT = langcheck.ROOT
N, THRESHOLD = 8, 0.06


def main():
    inv = list(csv.DictReader(open(ROOT / "notes/inventory.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE))
    by_chan = collections.defaultdict(list)
    for r in inv:
        by_chan[r["channel"]].append(r)
    rng = random.Random(0)
    cache = langcheck.cached()
    model = langcheck.onnx_asr.load_model("OpenVoiceOS/ai4bharat-indicconformer-kok-onnx",
                                          ROOT / "models/indicconformer-kok-onnx")
    out = open(ROOT / "notes/channel_langcheck.tsv", "w")
    out.write("channel\tid\ttab\tduration_s\tscore\n")
    for chan, rows in by_chan.items():
        picks = rng.sample(rows, min(N, len(rows)))
        scores = []
        with tempfile.TemporaryDirectory() as tmp:
            for r in picks:
                vid = r["id"]
                if vid not in cache:
                    try:
                        audio, _ = langcheck.sf.read(langcheck.sample(vid, tmp), dtype="float32")
                        cache[vid] = model.recognize(audio, sample_rate=16000)
                    except Exception:  # noqa: BLE001 - unavailable/private/members-only video
                        continue
                    with langcheck.CACHE.open("a") as f:
                        f.write(langcheck.json.dumps({"id": vid, "text": cache[vid]}, ensure_ascii=False) + "\n")
                words = cache[vid].split()
                score = sum(w in langcheck.MARKERS for w in words) / max(len(words), 1)
                scores.append(score)
                out.write(f"{chan}\t{vid}\t{r['tab']}\t{r['duration_s']}\t{score:.3f}\n"); out.flush()
        share = sum(s >= THRESHOLD for s in scores) / max(len(scores), 1)
        hours = sum(int(r["duration_s"]) for r in rows) / 3600
        print(f"{chan}: {sum(s >= THRESHOLD for s in scores)}/{len(scores)} Konkani -> ~{share * hours:.0f} of {hours:.0f} h",
              flush=True)


if __name__ == "__main__":
    main()
