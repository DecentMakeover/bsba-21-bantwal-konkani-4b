# Chat demo

A small Gradio chat app for [BSBA-21-Bantwal](https://huggingface.co/decentmakeover/bsba-21-bantwal-konkani-4b) that
runs on a CPU, no GPU needed. It starts llama.cpp's server with the Q4_K_M model file (about 2.5 GB) and streams short
replies (one or two sentences), cutting them at the first mostly-English line. An option adds a rough reading in Roman
letters. Each visitor gets 20 replies an hour; conversations are not stored.

## Run it on your own machine

```bash
pip install -r requirements.txt
# get llama-server from a llama.cpp release for your system: https://github.com/ggml-org/llama.cpp/releases
LLAMA_BIN=/path/to/llama-server THREADS=4 python app.py
```

The model file downloads from Hugging Face on the first start. On Linux x64 you can leave out `LLAMA_BIN`: the app
downloads llama.cpp release b11500 itself (it needs the system libraries in `packages.txt`). `MODEL_PATH` points it at a
GGUF file you already have. `DEMO_MOCK=1` runs the interface without the model, for testing.

On a 2-core CPU a short reply takes about a minute; more cores make it faster.
