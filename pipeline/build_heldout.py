"""Held-out-channel eval set (CPU): Konkani videos from channels NOT used in batch 1 (already downloaded during the
channel sampling). Transcribe (same ASR + VAD passages), clean + Mangalorean spelling exactly like export_corpus.py,
tokenize with the Gemma tokenizer into 1024-token blocks. Writes data/eval/heldout_{transcripts/,docs.jsonl,
gemma3_seq1024.npy} + manifest_heldout_v1.json. Usage: build_heldout.py"""
import csv
import hashlib
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402

import export_corpus  # noqa: E402
import langcheck  # noqa: E402
import mangalorean_ortho  # noqa: E402
import transcribe  # noqa: E402
from aksharamukha import transliterate  # noqa: E402

ROOT = transcribe.ROOT
TOK = "unsloth/gemma-3-4b-it"
HELDOUT_CHANNELS = set(json.loads((ROOT / "notes/channels.json").read_text())["heldout"])  # never used in training
SEQ = 1024


def main():
    inv = {r["id"]: r for r in csv.DictReader(open(ROOT / "notes/inventory.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE)}
    cache = langcheck.cached()
    out = ROOT / "data/eval"; (out / "heldout_transcripts").mkdir(parents=True, exist_ok=True)
    picks = []
    for f in sorted((ROOT / "data/audio_src").iterdir()):
        vid = f.name.rsplit(".", 1)[0]
        r = inv.get(vid)
        if not r or r["channel"] not in HELDOUT_CHANNELS:
            continue
        w = cache.get(vid, "").split()
        if w and sum(x in langcheck.MARKERS for x in w) / len(w) >= 0.06:
            picks.append((vid, f, r["channel"]))
    asr = transcribe.onnx_asr.load_model("OpenVoiceOS/ai4bharat-indicconformer-kok-onnx",
                                         ROOT / "models/indicconformer-kok-onnx").with_vad(transcribe.onnx_asr.load_vad("silero"))
    docs = []
    for vid, src, ch in picks:
        tj = out / "heldout_transcripts" / f"{vid}.jsonl"
        if not tj.exists():
            with tempfile.TemporaryDirectory() as tmp:
                subprocess.run([langcheck.BIN / "ffmpeg", "-loglevel", "error", "-y", "-i", src, "-ac", "1", "-ar", "16000",
                                f"{tmp}/a.wav"], check=True)
                with tj.open("w") as fh:
                    for p in transcribe.passages(asr.recognize(f"{tmp}/a.wav")):
                        fh.write(json.dumps({"id": vid, "deva": p["deva"]}, ensure_ascii=False) + "\n")
        kept = []
        for line in tj.open():
            t = json.loads(line)["deva"]; w = t.split()
            if len(w) < export_corpus.MIN_WORDS or export_corpus.is_loop(w):
                continue
            if len(w) >= 15 and sum(x in langcheck.MARKERS for x in w) / len(w) < export_corpus.MIN_MARKER_RATIO:
                continue
            kept.append(mangalorean_ortho.normalise(transliterate.process("Devanagari", "Kannada", t)))
        docs.append({"id": vid, "channel": ch, "text": " ".join(kept)})
        print(f"{vid} {ch}: {len(kept)} passages", flush=True)
    (out / "heldout_docs.jsonl").write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOK)
    ids = []
    for d in docs:
        ids += tok(d["text"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    n = len(ids) // SEQ
    arr = np.asarray(ids[: n * SEQ], dtype=np.uint32).reshape(n, SEQ)
    np.save(out / "heldout_gemma3_seq1024.npy", arr)
    sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()  # noqa: E731
    man = {"channels": sorted(HELDOUT_CHANNELS), "videos": len(docs), "words": sum(len(d["text"].split()) for d in docs),
           "tokens": len(ids), "blocks": n, "blocks_path": str(out / "heldout_gemma3_seq1024.npy"),
           "blocks_sha256": sha(out / "heldout_gemma3_seq1024.npy"), "docs_sha256": sha(out / "heldout_docs.jsonl"),
           "note": "Channels absent from batch 1; mostly sermons/talks plus one recipe video."}
    (out / "manifest_heldout_v1.json").write_text(json.dumps(man, indent=1) + "\n")
    print(json.dumps({k: v for k, v in man.items() if "sha" not in k}), flush=True)


if __name__ == "__main__":
    main()
