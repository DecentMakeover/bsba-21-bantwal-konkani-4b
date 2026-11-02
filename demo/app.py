"""Demo chat for BSBA-21-Bantwal, a Mangalorean Konkani model, on a free CPU Hugging Face Space.

llama.cpp's own server (release b11500, the build that made the GGUF) runs the Q4_K_M file on 2 CPU cores, and this
app streams its replies. Replies are short (MAX_NEW_TOKENS) and cut at the first mostly-English line; an optional rough
Roman-letter reading is added for readers who do not read Kannada script. Per-visitor limit: REPLIES_PER_HOUR.
Conversations are not stored or logged. DEMO_MOCK=1 runs the interface without the model (for local testing);
LLAMA_BIN and MODEL_PATH point at a local llama-server and GGUF instead of downloading them."""
import json, os, subprocess, tarfile, threading, time
from collections import defaultdict, deque
from pathlib import Path

import gradio as gr
import httpx
from aksharamukha import transliterate

MODEL_REPO = os.environ.get("MODEL_REPO", "decentmakeover/bsba-21-bantwal-konkani-4b")
MODEL_FILE = os.environ.get("MODEL_FILE", "bsba-21-bantwal-konkani-4b-v1-Q4_K_M.gguf")
LLAMA_URL = "https://github.com/ggml-org/llama.cpp/releases/download/b11500/llama-b11500-bin-ubuntu-x64.tar.gz"
THREADS, PORT = int(os.environ.get("THREADS", "2")), 8081
MOCK = os.environ.get("DEMO_MOCK") == "1"
MAX_NEW_TOKENS, MAX_TURNS, MAX_INPUT_CHARS, REPLIES_PER_HOUR = 96, 3, 400, 20
KONKANI_HINT = "Reply in Mangalorean Konkani, written in Kannada script, in one or two short sentences. "
HERE = Path(__file__).resolve().parent


def model_path():
    if os.environ.get("MODEL_PATH"):
        return os.environ["MODEL_PATH"]
    if (HERE / MODEL_FILE).exists():  # bundled with the Space while the model repo is private
        return str(HERE / MODEL_FILE)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(MODEL_REPO, MODEL_FILE)


def llama_server():
    if os.environ.get("LLAMA_BIN"):
        return os.environ["LLAMA_BIN"]
    exe = HERE / "llama-b11500" / "llama-server"
    if not exe.exists():
        tgz = HERE / "llama.tar.gz"
        with httpx.stream("GET", LLAMA_URL, follow_redirects=True, timeout=120) as r, open(tgz, "wb") as f:
            r.raise_for_status()
            for chunk in r.iter_bytes():
                f.write(chunk)
        with tarfile.open(tgz) as t:
            t.extractall(HERE, filter="data")
        tgz.unlink()
        exe.chmod(0o755)
    return str(exe)


def start_server():
    cmd = [llama_server(), "-m", model_path(), "-t", str(THREADS), "-c", "2048", "-np", "1",
           "--host", "127.0.0.1", "--port", str(PORT), "--log-disable"]
    subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(600):
        try:
            if httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=2).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise RuntimeError("llama-server did not start")


if not MOCK:
    start_server()

_recent = defaultdict(deque)
_lock = threading.Lock()


def visitor_key(request):
    if request is None:
        return "local"
    fwd = request.headers.get("x-forwarded-for", "")
    return fwd.split(",")[0].strip() or (request.client.host if request.client else "unknown")


def allow(key):
    now = time.time()
    with _lock:
        q = _recent[key]
        while q and now - q[0] > 3600:
            q.popleft()
        if len(q) >= REPLIES_PER_HOUR:
            return False
        q.append(now)
        return True


def text_of(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # Gradio 6 may send [{"type": "text", "text": ...}]
        return " ".join(c.get("text", "") for c in content if isinstance(c, dict))
    return str(content or "")


def build_messages(message, history, konkani):
    msgs = [{"role": h["role"], "content": text_of(h["content"]).split("\n\n_Roman reading:")[0]}
            for h in history if h.get("role") in ("user", "assistant")][-2 * MAX_TURNS:]
    msgs.append({"role": "user", "content": (KONKANI_HINT if konkani else "") + message})
    return msgs


def gemma_prompt(msgs):
    turns = "".join(f"<start_of_turn>{'model' if m['role'] == 'assistant' else 'user'}\n{m['content']}<end_of_turn>\n"
                    for m in msgs)
    return turns + "<start_of_turn>model\n"


def konkani_part(text, final=False):
    """Cut the reply at the first mostly-English line after some Konkani (the model sometimes adds English glosses).
    Returns (text, cut). While streaming, a short unfinished last line is not judged yet."""
    kept, lines = [], text.split("\n")
    for i, line in enumerate(lines):
        kn = sum("ಀ" <= ch <= "೿" for ch in line)
        la = sum(ch.isascii() and ch.isalpha() for ch in line)
        done = final or i < len(lines) - 1 or la >= 12
        if done and la > kn and la >= 4 and any("ಀ" <= ch <= "೿" for ch in "".join(kept)):
            return "\n".join(kept).rstrip(" \n-*_:"), True
        kept.append(line)
    return text, False


def roman(text):
    return transliterate.process("Kannada", "RomanReadable", text).replace("'", "")


def stream_reply(msgs):
    """Yield text pieces from llama-server; leaving the loop early closes the stream, which stops generation."""
    if MOCK:
        for w in f"(mock reply; the model is not loaded) {len(msgs)} messages, last: {msgs[-1]['content'][:80]}".split():
            time.sleep(0.05)
            yield w + " "
        return
    body = {"prompt": gemma_prompt(msgs), "n_predict": MAX_NEW_TOKENS, "temperature": 0.7, "top_p": 0.9,
            "repeat_penalty": 1.1, "stream": True, "cache_prompt": True}
    with httpx.stream("POST", f"http://127.0.0.1:{PORT}/completion", json=body, timeout=300) as r:
        for line in r.iter_lines():
            if line.startswith("data: "):
                d = json.loads(line[6:])
                if d.get("content"):
                    yield d["content"]
                if d.get("stop"):
                    return


def respond(message, history, konkani, show_roman, request: gr.Request = None):
    message = (message or "").strip()[:MAX_INPUT_CHARS]
    if not message:
        yield "Please type a message."
        return
    if not allow(visitor_key(request)):
        yield f"You have reached the limit of {REPLIES_PER_HOUR} replies an hour. Please try again later."
        return
    yield "…"
    out, raw = "", ""
    for piece in stream_reply(build_messages(message, history, konkani)):
        raw += piece
        out, cut = konkani_part(raw) if konkani else (raw, False)
        yield out
        if cut:
            break  # stop generating the English tail
    if konkani:
        out = konkani_part(raw, final=True)[0]
        yield out
    if show_roman and any("ಀ" <= ch <= "೿" for ch in out):  # only when the reply has Kannada script
        yield f"{out}\n\n_Roman reading: {roman(out)}_"


EXAMPLES = [
    ["Ask a shopkeeper the price of fish today."],
    ["Tell your mother you will be late for dinner tonight."],
    ["Greet a neighbour after Sunday Mass and ask about their family."],
    ["Tell your brother in Dubai that everyone at home misses him."],
    ["Invite a friend home for tea this evening."],
    ["Translate into Konkani: I will come to your house tomorrow evening."],
]

DESCRIPTION = """A small open model (4B) adapted to **Mangalorean Konkani as it is spoken**, written in Kannada script.
Type in English or Konkani. It runs on a free 2-core CPU, so a reply takes about a minute and appears word by word.
It makes things up, so do not rely on it for facts. Conversations are not stored.
[Model card](https://huggingface.co/decentmakeover/bsba-21-bantwal-konkani-4b) ·
[Code](https://github.com/DecentMakeover/bsba-21-bantwal-konkani-4b)"""

demo = gr.ChatInterface(
    respond,
    title="BSBA-21-Bantwal: Mangalorean Konkani chat",
    description=DESCRIPTION,
    examples=EXAMPLES,
    cache_examples=False,
    additional_inputs=[
        gr.Checkbox(value=True, label="Always answer in Mangalorean Konkani"),
        gr.Checkbox(value=False, label="Also show a rough reading in Roman letters"),
    ],
    additional_inputs_accordion=gr.Accordion("Options", open=False),
)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=1).launch()
