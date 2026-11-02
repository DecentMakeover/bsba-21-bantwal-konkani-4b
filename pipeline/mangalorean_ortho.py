"""Mangalorean Konkani spelling for Kannada-script text transliterated from Devanagari.

Rules (Mangalorean print convention, as in the Konkani titles of Mangalorean channels):
  1. long vowels -> short: ಈ ಊ ಏ ಓ (and their signs) -> ಇ ಉ ಎ ಒ
  2. word-final bare consonant gets a virama: ಹಾಂವ -> ಹಾಂವ್, ಘರ -> ಘರ್
  3. schwa deletion, right to left: a bare consonant between a vowel-bearing syllable and a consonant+vowel
     loses its inherent 'a': ಕುಟಮಾಂ -> ಕುಟ್ಮಾಂ, ಭುರಗಿಂ -> ಭುರ್ಗಿಂ, ಉಲಯತಾ -> ಉಲಯ್ತಾ
When notes/mangalorean_lexicon.tsv exists (word<TAB>count, built from Mangalorean channel titles by
build_lexicon()), a word whose rule output is not in the lexicon is replaced by the most frequent lexicon spelling
among its schwa-deletion variants, if any.
Usage as a module: normalise(text). As a script: reads text on stdin, writes normalised text;
`mangalorean_ortho.py --build-lexicon` rebuilds the lexicon from notes/inventory.tsv titles."""
import collections
import functools
import csv
import itertools
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LEXICON_PATH = ROOT / "notes/mangalorean_lexicon.tsv"

CONS = {chr(c) for c in range(0x0C95, 0x0CBA)} | {"ೞ"}
VIRAMA, NUKTA = "್", "಼"
SIGNS = {chr(c) for c in range(0x0CBE, 0x0CCD)} | {"ೢ", "ೣ"}
TAILS = {"ಂ", "ಃ"}  # anusvara, visarga
INDEP = {chr(c) for c in range(0x0C85, 0x0C95)}
SHORTEN = str.maketrans({"ೀ": "ಿ", "ೂ": "ು", "ೇ": "ೆ", "ೋ": "ೊ",
                         "ಈ": "ಇ", "ಊ": "ಉ", "ಏ": "ಎ", "ಓ": "ಒ"})
WORD = re.compile(r"[ಀ-೿]+")


def _syllables(word):
    """Split into [consonant-or-vowel, vowel-sign/virama/'' , tail] units; None if the word has unknown chars."""
    out, i = [], 0
    while i < len(word):
        ch = word[i]
        if ch in CONS:
            base = ch; i += 1
            if i < len(word) and word[i] == NUKTA:
                base += NUKTA; i += 1
            mark = ""
            if i < len(word) and (word[i] in SIGNS or word[i] == VIRAMA):
                mark = word[i]; i += 1
            tail = ""
            while i < len(word) and word[i] in TAILS:
                tail += word[i]; i += 1
            out.append([base, mark, tail])
        elif ch in INDEP:
            i += 1; tail = ""
            while i < len(word) and word[i] in TAILS:
                tail += word[i]; i += 1
            out.append([ch, "V", tail])
        else:
            return None
    return out


def _bare(s):  # consonant carrying the inherent vowel, nothing attached
    return s[1] == "" and s[2] == "" and s[0][0] in CONS


def _has_vowel(s):
    return s[1] != VIRAMA


def _join(syl):
    return "".join(b + ("" if m == "V" else m) + t for b, m, t in syl)


def _load_lexicon():
    if not LEXICON_PATH.exists():
        return {}
    return {w: int(c) for w, c in (l.rstrip("\n").split("\t") for l in LEXICON_PATH.open())}


LEXICON = _load_lexicon()


LONG_TO_SHORT = {"\u0CC0": "\u0CBF", "\u0CC2": "\u0CC1", "\u0CC7": "\u0CC6", "\u0CCB": "\u0CCA",
                 "\u0C88": "\u0C87", "\u0C8A": "\u0C89", "\u0C8F": "\u0C8E", "\u0C93": "\u0C92"}
MAX_VARIANT_SLOTS = 8


@functools.lru_cache(maxsize=500_000)
def normalise_word(word):
    syl = _syllables(word)
    if not syl:
        return word.translate(SHORTEN)
    # long-vowel positions: sign in slot 1, or an independent vowel in slot 0
    vowel_slots = [(i, 1) for i, s in enumerate(syl) if s[1] in LONG_TO_SHORT] + \
                  [(i, 0) for i, s in enumerate(syl) if s[1] == "V" and s[0] in LONG_TO_SHORT]
    long_form = {k: syl[k[0]][k[1]] for k in vowel_slots}
    for i, j in vowel_slots:
        syl[i][j] = LONG_TO_SHORT[syl[i][j]]
    if len(syl) >= 2 and _bare(syl[-1]):
        syl[-1][1] = VIRAMA
    schwa_slots = [i for i in range(1, len(syl) - 1) if _bare(syl[i])]  # medial consonants that may lose 'a'
    for i in range(len(syl) - 2, 0, -1):
        if _bare(syl[i]) and _has_vowel(syl[i - 1]) and syl[i + 1][0][0] in CONS and _has_vowel(syl[i + 1]):
            syl[i][1] = VIRAMA
    ruled = _join(syl)
    n_slots = len(schwa_slots) + len(vowel_slots)
    if not LEXICON or n_slots == 0 or n_slots > MAX_VARIANT_SLOTS:
        return ruled
    best, best_n = ruled, LEXICON.get(ruled, 0)
    for mask in itertools.product((False, True), repeat=n_slots):
        cand = [list(x) for x in syl]
        for i, drop in zip(schwa_slots, mask):
            cand[i][1] = VIRAMA if drop else ""
        for (i, j), keep_long in zip(vowel_slots, mask[len(schwa_slots):]):
            cand[i][j] = long_form[(i, j)] if keep_long else LONG_TO_SHORT[long_form[(i, j)]]
        w = _join(cand)
        if LEXICON.get(w, 0) > best_n:
            best, best_n = w, LEXICON[w]
    return best


def normalise(text):
    return WORD.sub(lambda m: normalise_word(m.group()), text)


def build_lexicon(min_count=2):
    """Count Kannada-script words in titles of Konkani channels (inventory), keep words seen >= min_count times."""
    counts = collections.Counter()
    for r in csv.DictReader(open(ROOT / "notes/inventory.tsv"), delimiter="\t", quoting=csv.QUOTE_NONE):
        counts.update(w.replace("\u200d", "") for w in WORD.findall(r["title"]))
    with LEXICON_PATH.open("w") as f:
        for w, c in counts.most_common():
            if c >= min_count and len(w) > 1:
                f.write(f"{w}\t{c}\n")
    return sum(1 for c in counts.values() if c >= min_count)


if __name__ == "__main__":
    if sys.argv[1:] == ["--build-lexicon"]:
        print(build_lexicon(), "words")
    else:
        sys.stdout.write(normalise(sys.stdin.read()))
