# imports & utilities
import os
import sys
from collections import defaultdict, Counter
import math
import random
import numpy as np
from typing import List, Tuple, Dict, Iterable

# Simple text normalization
def normalize_name(s: str) -> str:
    return s.strip().lower()

# Safe display of generated names (capitalizes first letter)
def beautify_name(s: str) -> str:
    if not s:
        return s
    return s.capitalize()
# config
VOCAB_EXTRA = '.'   # start/end token
MAX_NAME_LEN = 20   # safety cap for generated names
DEFAULT_N = 2       # default Markov order (bigram)
LAPLACE_ALPHA = 1.0 # smoothing
SEED = 2147483647   # default RNG seed for reproducible sampling
MAX_GENERATION_ATTEMPTS = 10000  # to avoid infinite loops

# load dataset
DATA_PATH = '/kaggle/input/names-dataset-for-bigram-model/names.txt'

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset file not found at '{DATA_PATH}'. Upload names.txt to the working directory.")

with open(DATA_PATH, 'r', encoding='utf-8') as f:
    raw_words = [line.strip() for line in f.readlines() if line.strip()]

# normalize
words = [normalize_name(w) for w in raw_words]

print(f"Loaded {len(words)} names. Example: {words[:10]}")
# peek distinct lengths
lengths = [len(w) for w in words]
print(f"Shortest: {min(lengths)} chars, Longest: {max(lengths)} chars")
# vocabulary
chars = sorted(list(set(''.join(words))))
allowed = [c for c in chars if c.isalpha()]
if len(allowed) < len(chars):
    print("Note: dataset contained non-alpha characters; they will be ignored in vocabulary:", 
          sorted(set(chars) - set(allowed)))

vocab = [VOCAB_EXTRA] + allowed  # '.' at index 0
stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for s,i in stoi.items()}
V = len(vocab)
print(f"Vocabulary ({V} tokens): {vocab}")
# build n-gram counts up to order N (inclusive)
def build_ngram_counts(corpus: Iterable[str], max_order: int = 2) -> Dict[int, Dict[Tuple[str,...], Counter]]:
    counts = {n: defaultdict(Counter) for n in range(1, max_order+1)}
    for w in corpus:
        # wrap with start and end token
        seq = [VOCAB_EXTRA] + list(w) + [VOCAB_EXTRA]
        for n in range(1, max_order+1):
            # context length = n-1
            for i in range(len(seq) - (n-1)):
                context = tuple(seq[i:i + (n-1)]) if (n-1) > 0 else tuple()
                next_token = seq[i + (n-1)]
                counts[n][context][next_token] += 1
    return counts

# build counts up to trigram for better robustness
MAX_ORDER = 3
counts = build_ngram_counts(words, max_order=MAX_ORDER)
for n in range(1, MAX_ORDER+1):
    print(f"Order {n}: contexts = {len(counts[n])}")
# convert counters into probability tables with Laplace smoothing and prepare backoff lookup

def build_probability_tables(counts: Dict[int, Dict[Tuple[str,...], Counter]],
                             vocab: List[str],
                             alpha: float = 1.0) -> Dict[int, Dict[Tuple[str,...], Tuple[List[str], np.ndarray]]]:
    V = len(vocab)
    prob_tables = {n: {} for n in counts}
    for n, ctx_map in counts.items():
        for ctx, counter in ctx_map.items():
            # build array over the entire vocab to allow sampling of any token (smoothed)
            counts_arr = np.array([counter.get(tok, 0) for tok in vocab], dtype=float)
            counts_arr += alpha  # Laplace
            probs = counts_arr / counts_arr.sum()
            prob_tables[n][ctx] = (vocab, probs)
    return prob_tables

prob_tables = build_probability_tables(counts, vocab, alpha=LAPLACE_ALPHA)
# quick sanity check: pick a random context
some_n = 2
some_ctx = next(iter(prob_tables[some_n].keys()))
tokens_list, probs_arr = prob_tables[some_n][some_ctx]
print(f"Example context (order {some_n}): {some_ctx} -> first tokens: {tokens_list[:8]} | probs sum {probs_arr.sum():.6f}")
# function to fetch distribution given current context with backoff

def get_distribution_with_backoff(context: Tuple[str,...],
                                  prob_tables: Dict[int, Dict[Tuple[str,...], Tuple[List[str], np.ndarray]]],
                                  max_order: int) -> Tuple[List[str], np.ndarray, int]:
    # context is a tuple of last tokens (length can be up to max_order-1); contxt length -> decreasing
    for order in range(min(max_order, len(context)+1), 0, -1):
        ctx_len_needed = order - 1
        ctx = tuple(context[-ctx_len_needed:]) if ctx_len_needed > 0 else tuple()
        if ctx in prob_tables[order]:
            return prob_tables[order][ctx][0], prob_tables[order][ctx][1], order
    # fallback: unigram with empty context must exist
    return prob_tables[1][tuple()][0], prob_tables[1][tuple()][1], 1
# sampling function (single name)
def sample_name(start_prefix: str,
                prob_tables: Dict[int, Dict[Tuple[str,...], Tuple[List[str], np.ndarray]]],
                vocab: List[str],
                stoi: Dict[str,int],
                itos: Dict[int,str],
                max_order: int = 3,
                rng: np.random.Generator = None,
                max_len: int = 20) -> str:
    
    if rng is None:
        rng = np.random.default_rng()
    # normalize prefix
    prefix = ''.join([c for c in start_prefix.lower() if c.isalpha()])
    # verify prefix characters are in vocab
    for ch in prefix:
        if ch not in vocab:
            raise ValueError(f"Character '{ch}' not in vocabulary.")
    # initial sequence: '.' + prefix chars
    seq = [VOCAB_EXTRA] + list(prefix)
    # If prefix already ends with end token (unlikely), keep it
    # Now autoregressively sample
    for step in range(max_len):
        # context = last (max_order-1) tokens
        context = tuple(seq[-(max_order-1):]) if (max_order-1) > 0 else tuple()
        tokens_list, probs_arr, used_order = get_distribution_with_backoff(context, prob_tables, max_order)
        # sample one token index from tokens_list according to probs_arr
        next_token = rng.choice(tokens_list, p=probs_arr)
        if next_token == VOCAB_EXTRA:
            # end of name
            break
        seq.append(next_token)
        # safety: if name grows beyond cap, stop
        if len(seq) > max_len:
            break
    # build name removing leading dot and any trailing dot
    # seq = ['.', 'a', 'b', 'c'] -> name 'abc'
    name = ''.join([t for t in seq if t != VOCAB_EXTRA])
    return name

# quick test sampling (deterministic rng)
test_rng = np.random.default_rng(SEED)
print("Example sample (no prefix):", sample_name("", prob_tables, vocab, stoi, itos, max_order=MAX_ORDER, rng=test_rng, max_len=MAX_NAME_LEN))
print("Example sample (prefix 'am'):", sample_name("am", prob_tables, vocab, stoi, itos, max_order=MAX_ORDER, rng=test_rng, max_len=MAX_NAME_LEN))
# prompt and generation

def prompt_and_generate():
    print("Welcome to Skred's Markov (n-gram) Name Generator")
    start = input("Enter starting letters for the name (leave empty for any): ").strip().lower()
    try:
        k = int(input("How many names do you want to generate? (e.g. 10): ").strip())
        if k <= 0:
            raise ValueError()
    except Exception:
        print("Invalid number; defaulting to 10.")
        k = 10

    try:
        order = int(input(f"Choose Markov order (1..{MAX_ORDER} (N-grams; higher order models create more realistic names)) [default {DEFAULT_N}]: ").strip() or DEFAULT_N)
        if order < 1 or order > MAX_ORDER:
            print(f"Order must be between 1 and {MAX_ORDER}; using default {DEFAULT_N}")
            order = DEFAULT_N
    except Exception:
        order = DEFAULT_N

    # ask for seed for reproducibility
    seed_input = input(f"Enter integer RNG seed (or leave empty for random): ").strip()
    rng = np.random.default_rng(int(seed_input)) if seed_input else np.random.default_rng()

    # produce unique names
    generated = []
    attempts = 0
    while len(generated) < k and attempts < MAX_GENERATION_ATTEMPTS:
        name = sample_name(start, prob_tables, vocab, stoi, itos, max_order=order, rng=rng, max_len=MAX_NAME_LEN)
        pretty = beautify_name(name)
        if pretty and pretty not in generated:
            generated.append(pretty)
        attempts += 1

    print(f"\nGenerated {len(generated)} unique names (attempts {attempts}):")
    for i, nm in enumerate(generated, 1):
        print(f"{i:3d}. {nm}")
    if len(generated) < k:
        print(f"Could only generate {len(generated)} unique names after {attempts} attempts; try larger tolerance or different prefix.")

# run prompt
prompt_and_generate()
# compute negative log-likelihood on the training set using our prob_tables (with backoff)
def compute_nll_dataset(corpus: Iterable[str], prob_tables: Dict[int, Dict], max_order: int = 3) -> float:
    total_logprob = 0.0
    total_tokens = 0
    for w in corpus:
        seq = [VOCAB_EXTRA] + list(w) + [VOCAB_EXTRA]
        for i in range(1, len(seq)):
            # context is tokens before position i, but we only use up to max_order-1 last tokens
            left = tuple(seq[max(0, i - (max_order-1)):i])  # last up to max_order-1 tokens
            tokens_list, probs_arr, used_order = get_distribution_with_backoff(left, prob_tables, max_order)
            # find index of actual next token
            next_tok = seq[i]
            try:
                idx = tokens_list.index(next_tok)
            except ValueError:
                # shouldn't happen because smoothing gave positive prob to all vocab tokens
                prob = 1e-12
            else:
                prob = probs_arr[idx]
            total_logprob += math.log(prob)
            total_tokens += 1
    nll = - total_logprob / total_tokens
    return nll

nll_value = compute_nll_dataset(words, prob_tables, max_order=MAX_ORDER)
print(f"Training-set NLL (avg negative log prob per token): {nll_value:.4f}")
print(f"Perplexity (exp(NLL)): {math.exp(nll_value):.4f}")
