"""Is a YouTube video actually Konkani? Download a 60 s sample from the middle, transcribe it with the Konkani ASR,
and score the share of words that are distinctively Konkani (curated markers, absent from Hindi/Marathi/English).
Transcripts are cached in notes/langcheck_cache.jsonl so re-scoring needs no download.
Usage: langcheck.py VIDEO_ID [...]  -> prints "id  score  n_words  sample_text"""
import json
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import onnx_asr  # noqa: E402
import soundfile as sf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / ".tools/bin"
SAMPLE_S = 60
CACHE = ROOT / "notes/langcheck_cache.jsonl"
MARKERS = set("""
आसा आसात आसलें आसले आसली आसता आसतना आसचें नाशिल्लें नाका नाकात
हांव हांवें म्हाका म्हज्या म्हजें म्हजी म्हजो आमी आमका आमच्या आमचें आमची आमचो तुमी तुमका तुमच्या तुमचें
तुका तुजें तुज्या ताका ताची ताचें ताच्या तिका तिचें तांका तांच्या ह्या हें हो ही तें तो ती
जाल्यार म्हळ्यार म्हणल्यार म्हणून म्हणुन म्हणटा म्हणटात म्हणलें म्हणाली म्हणालो
आनी पूण कित्याक कितें किदें खंय खंयच्या केन्ना कशें कसो कसलें
थावन थावून कडेन खातीर सांगातां सांगाता भितर भायर वयर सकयल मुखार फाटीं
जाता जातात जालें जाला जाली जालो जावन जावंक जायना करता करतात केलें केला केली करचें करुंक
वता वतात गेलें गेलो गेली येता येतात आयलें आयलो दिता दितात घेता घेतात सांगता सांगलें
भुरगीं भुरग्यांक घर घरा घरांत देवाच्या देवाक बरें बरी बरो
""".split())


def sample(vid, tmp):
    """Download the full audio once (kept in data/audio_src for reuse), then cut SAMPLE_S s from the middle."""
    src_dir = ROOT / "data/audio_src"
    src = next(src_dir.glob(f"{vid}.*"), None)
    if src is None:
        subprocess.run([ROOT / ".venv/bin/yt-dlp", "-q", "--no-warnings", "-f", "bestaudio", "--no-part",
                        "-o", str(src_dir / "%(id)s.%(ext)s"), f"https://youtu.be/{vid}"], check=True, timeout=3600)
        src = next(src_dir.glob(f"{vid}.*"))
    dur = float(subprocess.run([BIN / "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                src], capture_output=True, text=True).stdout.strip() or 0)
    wav = Path(tmp) / f"{vid}.wav"
    subprocess.run([BIN / "ffmpeg", "-loglevel", "error", "-y", "-ss", str(max(0, dur / 2 - SAMPLE_S / 2)), "-t",
                    str(SAMPLE_S), "-i", src, "-ac", "1", "-ar", "16000", wav], check=True, timeout=300)
    return wav


def cached():
    """Cached sample transcripts; skips damaged lines (e.g. NUL-filled after a power loss)."""
    out = {}
    if CACHE.exists():
        for line in CACHE.open(errors="replace"):
            try:
                d = json.loads(line); out[d["id"]] = d["text"]
            except (ValueError, KeyError, TypeError):
                continue
    return out


def main(ids):
    cache = cached()
    model = onnx_asr.load_model("OpenVoiceOS/ai4bharat-indicconformer-kok-onnx", ROOT / "models/indicconformer-kok-onnx")
    with tempfile.TemporaryDirectory() as tmp:
        for vid in ids:
            if vid not in cache:
                try:
                    audio, _ = sf.read(sample(vid, tmp), dtype="float32")
                    cache[vid] = model.recognize(audio, sample_rate=16000)
                except Exception as e:  # noqa: BLE001 - one bad video must not stop a batch
                    print(f"{vid}\tERROR\t{e}", flush=True); continue
                with CACHE.open("a") as f:
                    f.write(json.dumps({"id": vid, "text": cache[vid]}, ensure_ascii=False) + "\n")
            words = cache[vid].split()
            score = sum(w in MARKERS for w in words) / max(len(words), 1)
            print(f"{vid}\t{score:.2f}\t{len(words)}\t{' '.join(words[:12])}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
