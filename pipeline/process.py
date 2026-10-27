"""Worker: for each downloaded batch video, language-check a 60 s middle sample; transcribe Konkani ones
(VAD passages, raw Devanagari + plain Kannada transliteration) to data/transcripts/<id>.jsonl.
Status in notes/process_status.tsv. Polls until notes/download.done exists and nothing is pending."""
import csv
import json
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import langcheck  # noqa: E402
import transcribe  # noqa: E402
from download_readiness import completed_ids, ready_audio  # noqa: E402

ROOT = transcribe.ROOT
SRC, OUT = ROOT / "data/audio_src", transcribe.OUT
STATUS = ROOT / "notes/process_status.tsv"
THRESHOLD = 0.06
SAMPLE_AT = (0.5, 0.25, 0.75)  # middle first; more samples only when it is (nearly) wordless - music, pause, cut
MIN_SAMPLE_WORDS, MIN_WORDS = 20, 5


def to_wav(src, wav, start=None, length=None):
    cut = (["-ss", str(start), "-t", str(length)] if start is not None else [])
    subprocess.run([langcheck.BIN / "ffmpeg", "-loglevel", "error", "-y", *cut, "-i", src, "-ac", "1", "-ar", "16000",
                    wav], check=True, timeout=3600)


def main(batch):
    ids = [r["id"] for r in csv.DictReader(open(batch), delimiter="\t", quoting=csv.QUOTE_NONE)]
    asr = transcribe.onnx_asr.load_model("OpenVoiceOS/ai4bharat-indicconformer-kok-onnx",
                                         ROOT / "models/indicconformer-kok-onnx")
    asr_vad = asr.with_vad(transcribe.onnx_asr.load_vad("silero"))
    cache = langcheck.cached()
    while True:
        handled = {l.split("\t")[0] for l in STATUS.open()} if STATUS.exists() else set()
        # A final-looking --no-part file can still be growing. Require success.
        complete = completed_ids(ROOT / "notes/download_status.tsv")
        pending = [(i, s) for i in ids if i not in handled and (s := ready_audio(SRC, i, complete))]
        if not pending:
            if (ROOT / "notes/download.done").exists():
                break
            time.sleep(60); continue
        for vid, src in pending:
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    dur = float(subprocess.run([langcheck.BIN / "ffprobe", "-v", "error", "-show_entries",
                                                "format=duration", "-of", "csv=p=0", src],
                                               capture_output=True, text=True).stdout.strip() or 0)
                    if vid not in cache:
                        texts = []
                        for frac in SAMPLE_AT:
                            to_wav(src, f"{tmp}/s.wav", max(0, dur * frac - 30), 60)
                            audio, _ = langcheck.sf.read(f"{tmp}/s.wav", dtype="float32")
                            texts.append(asr.recognize(audio, sample_rate=16000) if len(audio) >= 16000 else "")
                            if len(" ".join(texts).split()) >= MIN_SAMPLE_WORDS:
                                break
                        cache[vid] = " ".join(t for t in texts if t)
                        with langcheck.CACHE.open("a") as f:
                            f.write(json.dumps({"id": vid, "text": cache[vid]}, ensure_ascii=False) + "\n")
                    words = cache[vid].split()
                    score = sum(w in langcheck.MARKERS for w in words) / max(len(words), 1)
                    if len(words) < MIN_WORDS:
                        status = f"skip_empty\t\t{dur:.0f}"
                    elif score < THRESHOLD:
                        status = f"skip_lang\t{score:.3f}\t{dur:.0f}"
                    else:
                        to_wav(src, f"{tmp}/full.wav")
                        n = speech = 0
                        part = OUT / f"{vid}.jsonl.part"
                        with part.open("w") as f:
                            for p in transcribe.passages(asr_vad.recognize(f"{tmp}/full.wav")):
                                p = {"id": vid, "start": round(p["start"], 2), "end": round(p["end"], 2),
                                     "deva": p["deva"],
                                     "knda": transcribe.transliterate.process("Devanagari", "Kannada", p["deva"])}
                                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                                n += 1; speech += p["end"] - p["start"]
                        part.rename(OUT / f"{vid}.jsonl")
                        status = f"ok\t{score:.3f}\t{dur:.0f}\t{n}\t{speech:.0f}"
                except Exception as e:  # noqa: BLE001 - one bad file must not stop the worker
                    status = f"error\t\t\t{str(e)[:150]}"
            with STATUS.open("a") as f:
                f.write(f"{vid}\t{status}\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ROOT / "notes/batch1.tsv")
