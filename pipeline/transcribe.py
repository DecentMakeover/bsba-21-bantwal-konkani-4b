"""Transcribe 16 kHz mono WAVs with AI4Bharat IndicConformer Konkani (ungated MIT ONNX export) + Silero VAD.
Adjacent speech segments are merged into passages (gap < MAX_GAP_S, length <= MAX_PASSAGE_S).
Writes data/transcripts/<id>.jsonl: {"id","start","end","deva","knda"} per passage. Skips files already done.
Usage: transcribe.py [wav ...]   (default: every data/raw_audio/*.wav)"""
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import onnx_asr  # noqa: E402
from aksharamukha import transliterate  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
AUDIO, OUT = ROOT / "data/raw_audio", ROOT / "data/transcripts"
MAX_GAP_S, MAX_PASSAGE_S = 1.0, 30.0


def passages(segments):
    cur = None
    for s in segments:
        text = s.text.strip()
        if not text:
            continue
        if cur and s.start - cur["end"] < MAX_GAP_S and s.end - cur["start"] <= MAX_PASSAGE_S:
            cur["end"], cur["deva"] = s.end, f"{cur['deva']} {text}"
        else:
            if cur:
                yield cur
            cur = {"start": s.start, "end": s.end, "deva": text}
    if cur:
        yield cur


def main(paths):
    OUT.mkdir(parents=True, exist_ok=True)
    model = onnx_asr.load_model("OpenVoiceOS/ai4bharat-indicconformer-kok-onnx",
                                ROOT / "models/indicconformer-kok-onnx").with_vad(onnx_asr.load_vad("silero"))
    for wav in [Path(p) for p in paths] or sorted(AUDIO.glob("*.wav")):
        out = OUT / f"{wav.stem}.jsonl"
        if out.exists():
            continue
        tmp = out.with_suffix(".jsonl.part")
        n = speech = 0
        with tmp.open("w") as f:
            for p in passages(model.recognize(str(wav))):
                p = {"id": wav.stem, "start": round(p["start"], 2), "end": round(p["end"], 2), "deva": p["deva"],
                     "knda": transliterate.process("Devanagari", "Kannada", p["deva"])}
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
                n += 1; speech += p["end"] - p["start"]
        tmp.rename(out)
        print(f"{wav.stem}: {n} passages, {speech / 60:.1f} min speech", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
