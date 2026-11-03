# BSBA-21-Bantwal

Code for BSBA-21-Bantwal, an open language model adapted to **Mangalorean Konkani as it is actually spoken**, in Kannada script.
The model is on Hugging Face: [decentmakeover/bsba-21-bantwal-konkani-4b](https://huggingface.co/decentmakeover/bsba-21-bantwal-konkani-4b). This repository has the full pipeline: from public speech recordings to a
trained, evaluated and packaged model.

**Why the name?** Bantwal is my hometown, near Mangalore, and BSBA-21 is my batch: the 21st batch of BSBA.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1. Find sources | `pipeline/inventory.py`, `pipeline/sample_channels.py` | List a channel's videos and durations; estimate how much of each channel is Konkani |
| 2. Language check | `pipeline/langcheck.py` | Transcribe a 60 s sample and score Konkani marker words (separates Konkani from Kannada, Hindi, English) |
| 3. Download | `pipeline/download.py`, `pipeline/build_batch.py` | Slow, login-free audio download with back-off on bot checks, network waits and partial-file quarantine |
| 4. Transcribe | `pipeline/process.py`, `pipeline/transcribe.py` | AI4Bharat IndicConformer (Konkani, ONNX) + Silero VAD, merged into passages |
| 5. Mangalorean spelling | `pipeline/mangalorean_ortho.py` | Devanagari → Kannada script with Mangalorean spelling rules and a lexicon tie-break |
| 6. Corpus | `pipeline/export_corpus.py`, `pipeline/pretokenize.py`, `pipeline/build_heldout.py` | Clean, de-duplicate, split by video; held-out set from unseen channels |
| 7. Train | `training/train_cpt.py`, `training/prep_sft_v2.py`, `training/train_lora_v2.py` | Stage A dialect pass; stage B instruction tuning with Mangalorean replay. Resumable in fixed-length runs |
| 8. Evaluate | `eval/eval_suite.py`, `eval/generate_check.py` | Perplexity (in-domain + unseen channels), Konkani-Bench chrF, stays-in-Konkani, Mangalorean word share |
| 9. Package | `release/package_release.py` | Merge, verify, GGUF conversion and quantization, smoke test |
| 10. Demo | `demo/app.py` | Gradio chat app that runs the Q4_K_M file on a CPU with llama.cpp (replies cut at the first English line) |

`configs/` holds the settings used for the v1 run. All paths in the scripts and configs are relative to the repository
root, and models are named by their Hugging Face IDs, so the code runs from a fresh clone on any machine.

## Results

See the model card on Hugging Face: [decentmakeover/bsba-21-bantwal-konkani-4b](https://huggingface.co/decentmakeover/bsba-21-bantwal-konkani-4b). In short, against the Gemma 3 4B + Konkani LLM baseline: replies stay
in Konkani 90% vs 5%, English → Konkani chrF 31.7 vs 8.4, perplexity on unseen Mangalorean channels 15.8 vs 70.1.

## Not included

No audio, transcripts or training text. They come from public YouTube videos whose creators did not license them for
redistribution. The scripts rebuild the corpus from the sources. The channel list is not included either: put your own
in `notes/seed_channels.tsv` (format in `examples/seed_channels.example.tsv`: one video ID per channel, a genre, the
Konkani variety and an optional regular expression to keep only matching titles), and the channel groups used to
build the batch and the held-out set in `notes/channels.json` (format in `examples/channels.example.json`).

## Requirements

- Python 3.12, an NVIDIA GPU with 24 GB or more for training (we used one DGX Spark), `ffmpeg` on the PATH.
- `pip install -r requirements.txt` (install PyTorch for your CUDA version first).
- A Hugging Face account (`hf auth login`); accept the terms of the Gemma 3 and Konkani LLM model pages if asked.
- For packaging only: [llama.cpp](https://github.com/ggml-org/llama.cpp) built in `.tools/llama.cpp` (we used tag `b11500`).

## How to run

Every command runs from the repository root. Downloading and transcription run on the CPU.

```bash
# 0. One-time downloads
hf download OpenVoiceOS/ai4bharat-indicconformer-kok-onnx --local-dir models/indicconformer-kok-onnx
hf download konkani/konkani-bench benchmark.jsonl --repo-type dataset --local-dir data/eval/konkani-bench

# 1-2. Sources: list the channels' videos, then sample each channel for Konkani
mkdir -p notes && cp examples/seed_channels.example.tsv notes/seed_channels.tsv   # then edit it
python pipeline/inventory.py
python pipeline/sample_channels.py
python pipeline/mangalorean_ortho.py --build-lexicon      # Mangalorean spelling list from the video titles

# 3-4. Pick the batch, then download and transcribe (run the two side by side; both resume after a stop)
python pipeline/build_batch.py
python pipeline/download.py notes/batch1.tsv
python pipeline/process.py notes/batch1.tsv

# 5-6. Corpus, tokens and the held-out set
python pipeline/export_corpus.py
python pipeline/pretokenize.py unsloth/gemma-3-4b-it
python pipeline/build_heldout.py

# 7. Training. Each run stops after slot_budget_s seconds and resumes from its checkpoint;
#    repeat until the final/ folder exists (stage A: 608 steps, stage B: 296 steps)
until [ -d models/cpt-4b-v1/final ]; do python training/train_cpt.py configs/cpt_4b_v1.json || break; done
python training/prep_sft_v2.py
until [ -d models/sft-4b-v2/final ]; do python training/train_lora_v2.py configs/sft_4b_v2.json || break; done

# 8. Scorecard (written to notes/eval_v2/)
python eval/eval_suite.py configs/eval_v2.json

# 9. Merged model + GGUF files (written to release/v1/)
python release/package_release.py configs/package_v1.json
```

Notes:

- `build_batch.py` and `build_heldout.py` encode our channel choices (genre caps, which channels are held out). Adjust
  them to your channel list.
- The training scripts take `--dry` for a tiny CPU check of the whole loop before a real run.
- `configs/eval_v2.json` also scores our first stage-B attempt (`models/sft-4b-v1`), which this repository does not
  rebuild. Remove that entry from `models` unless you have it.
- `configs/v1_run_hashes.json` has the SHA-256 of the data files in our run. Your rebuilt corpus will differ; to freeze
  a run of your own, put your hashes in a config's `frozen_data` and the training script checks them before it starts.

## Licence

Code: MIT (see `LICENSE`). The model weights are under the Gemma Terms of Use; see the model card.

## Acknowledgements

AI4Bharat (Konkani speech recogniser), the Konkani LLM project (Reuben Chagas Fernandes and Gaurang S. Patkar),
Google (Gemma), and the Mangalorean creators who keep Konkani alive online.
