"""Demo chat for the Mangalorean Konkani model (Hugging Face Space, ZeroGPU).

The model answers in Kannada script; an optional rough Roman-letter reading is added for readers who do not read it.
Per-visitor limit (REPLIES_PER_HOUR), short history (MAX_TURNS) and a reply cap (MAX_NEW_TOKENS) keep GPU use small.
Conversations are not stored or logged. DEMO_MOCK=1 runs the interface without the model (for local testing)."""
import os, time, threading
from collections import defaultdict, deque

import gradio as gr
from aksharamukha import transliterate

MODEL_ID = os.environ.get("MODEL_ID", "decentmakeover/bsba-21-bantwal-konkani-4b")
MOCK = os.environ.get("DEMO_MOCK") == "1"
MAX_NEW_TOKENS, MAX_TURNS, MAX_INPUT_CHARS, REPLIES_PER_HOUR = 256, 4, 600, 20
KONKANI_HINT = "Reply in Mangalorean Konkani, written in Kannada script. "

try:
    import spaces  # ZeroGPU on Hugging Face; a no-op decorator elsewhere
    gpu = spaces.GPU(duration=60)
except ImportError:
    gpu = lambda f: f

if not MOCK:
    import torch
    from transformers import AutoModelForImageTextToText, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    device = "cuda" if torch.cuda.is_available() or os.environ.get("SPACES_ZERO_GPU") else "cpu"
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to(device).eval()

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


if not MOCK:
    class Halt(StoppingCriteria):
        def __init__(self, event):
            self.event = event

        def __call__(self, input_ids, scores, **kwargs):
            hit = self.event is not None and self.event.is_set()
            return torch.full((input_ids.shape[0],), hit, dtype=torch.bool, device=input_ids.device)


@gpu
def stream_reply(msgs, stop=None):
    if MOCK:
        for w in f"(mock reply; the model is not loaded) {len(msgs)} messages, last: {msgs[-1]['content'][:80]}".split():
            time.sleep(0.05)
            yield w + " "
        return
    x = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True).to(model.device)
    streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    halt = StoppingCriteriaList([Halt(stop)])
    kw = dict(**x, streamer=streamer, max_new_tokens=MAX_NEW_TOKENS, do_sample=True, temperature=0.7, top_p=0.9,
              repetition_penalty=1.1, stopping_criteria=halt)
    threading.Thread(target=model.generate, kwargs=kw, daemon=True).start()
    yield from streamer


def respond(message, history, konkani, show_roman, request: gr.Request = None):
    message = (message or "").strip()[:MAX_INPUT_CHARS]
    if not message:
        yield "Please type a message."
        return
    if not allow(visitor_key(request)):
        yield f"You have reached the limit of {REPLIES_PER_HOUR} replies an hour. Please try again later."
        return
    out, raw, stop = "", "", threading.Event()
    for piece in stream_reply(build_messages(message, history, konkani), stop):
        raw += piece
        out, cut = konkani_part(raw) if konkani else (raw, False)
        yield out
        if cut:
            stop.set()  # stop generating the English tail
            break
    if konkani:
        out = konkani_part(raw, final=True)[0]
        yield out
    if show_roman and any("ಀ" <= ch <= "೿" for ch in out):  # only when the reply has Kannada script
        yield f"{out}\n\n_Roman reading: {roman(out)}_"


EXAMPLES = [
    ["Greet a neighbour after Sunday Mass and ask about their family."],
    ["Tell your brother in Dubai that everyone at home misses him."],
    ["Ask a shopkeeper the price of fish today."],
    ["Explain briefly how to make pork bafat."],
    ["Describe what happens at a Mangalorean roce ceremony."],
    ["Translate into Konkani: I will come to your house tomorrow evening."],
]

DESCRIPTION = """A small open model (4B) adapted to **Mangalorean Konkani as it is spoken**, written in Kannada script.
Type in English or Konkani. It makes things up, so do not rely on it for facts.
Conversations are not stored. [Model card](https://huggingface.co/decentmakeover/bsba-21-bantwal-konkani-4b) · [Code](https://github.com/DecentMakeover/bsba-21-bantwal-konkani-4b)"""

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
    demo.queue(default_concurrency_limit=2).launch()
