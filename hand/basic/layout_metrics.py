#
#  Modified by Mohammed Hamdan, 2025-2026, under CeCILL-C Article 5.3.2. The measured
#  divergence from the pinned upstream commit is recorded in release/NOTICE.md
#  section 1.1. This file remains governed by CeCILL-C; the repository's top-level
#  MIT LICENSE does not apply to it.
"""
Page-level metrics for HAND V2, implemented to match their published definitions.

Two families:

1. Layout metrics as defined by DAN (Coquenet et al., TPAMI 2023, arXiv 2203.12273, §4.2-4.3)
   and computed by the official code (github.com/FactoDeepLearning/DAN, basic/metric_manager.py,
   CeCILL-C; vendored under third_party/DAN). `str_to_graph_read` / `str_to_graph_rimes` are
   line-for-line ports of DAN's functions, kept separate from `hand/basic/metric_manager.py`
   whose generic `str_to_graph` builds a *different* graph (nesting-depth levels, no fixed
   hierarchy) and therefore a different LOER. Nothing in the frozen baseline is modified.

   LOER = graph edit distance(G_gt, G_pred) / (|V_gt| + |E_gt|), micro-averaged over documents.
   mAP_CER is re-used unchanged from the repository (identical to DAN's).

2. Reading-order-invariant text metrics from Vidal, Toselli, Ríos-Vila, Calvo-Zaragoza,
   "End-to-end page-level assessment of handwritten text recognition", Pattern Recognition 2023
   (arXiv 2301.05935): bag-of-words WER (bWER), Hungarian WER (hWER), and the reading-order
   distance NSFD they derive.  Definitions used here (their §3):
     - bWER (Eq. 8): (1/2N)(|N − |Y|| + Σ_v |f_X(v) − f_Y(v)|), N = |X| = GT words; equal to
       (max(|X|, |Y|) − |multiset intersection|) / N, the words that cannot be matched
       regardless of order.
     - hWER (Eq. 9-11): Hungarian assignment between X and Y padded with dummies, pair cost =
       character edit distance (dummy pairs cost |v|/2, optional positional regulariser
       γ|j−k|/N); hWER = (Σ δ(X_j, Y_k) − (D − b)/2) / N with δ the word-mismatch indicator,
       D the number of one-dummy pairs, b = ||X| − |Y||. Paper: bWER ≤ hWER.
     - ΔWER = WER − bWER: the part of WER attributable to reading order.
     - NSFD (Eq. 5): ρ = Σ_{(j,k) ∈ A} |j − k| / ⌊N²/2⌋, N = max(|X|, |Y|), over the assigned
       real pairs A of the hWER alignment.
   Tokenisation: Vidal et al. use whitespace-delimited words (punctuation attached); the
   default here is the repository's/DAN's convention (punctuation as separate words) so that
   ΔWER is consistent with the WER reported everywhere else. Both are available.
   These are computed after layout tokens are stripped, exactly as CER/WER are.

Everything here is pure Python/NumPy/SciPy/NetworkX and is unit-tested against hand-scored
cases in tests/test_layout_metrics.py.
"""
from collections import Counter

import editdistance
import networkx as nx
import numpy as np
from scipy.optimize import linear_sum_assignment

# READ 2016 layout tokens (begin -> end), as in DAN's read2016_formatter.SEM_MATCHING_TOKENS
READ_MATCHING_TOKENS = {"ⓑ": "Ⓑ", "ⓐ": "Ⓐ", "ⓟ": "Ⓟ", "ⓝ": "Ⓝ", "ⓢ": "Ⓢ"}
# RIMES 2009 layout tokens, as in DAN's rimes_formatter.SEM_MATCHING_TOKENS
RIMES_MATCHING_TOKENS = {"ⓑ": "Ⓑ", "ⓞ": "Ⓞ", "ⓡ": "Ⓡ", "ⓢ": "Ⓢ", "ⓦ": "Ⓦ", "ⓨ": "Ⓨ", "ⓟ": "Ⓟ"}


def keep_only_tokens(s, tokens):
    return "".join(c for c in s if c in tokens)


def keep_all_but_tokens(s, tokens):
    return "".join(c for c in s if c not in tokens)


# ----------------------------------------------------------------------------------------------
# 1. DAN layout graphs and LOER (ported from third_party/DAN/basic/metric_manager.py)
# ----------------------------------------------------------------------------------------------

def str_to_graph_read(s):
    """DAN's graph for READ 2016 single/double page: D -> P_i -> {N, S} -> {A, B}, with
    sequential edges between successive nodes of the same level. Only *begin* tokens count."""
    begin_layout_tokens = "".join(READ_MATCHING_TOKENS.keys())
    seq = keep_only_tokens(s, begin_layout_tokens)
    g = nx.DiGraph()
    g.add_node("D", type="document", level=4, page=0)
    num = {"ⓟ": 0, "ⓐ": 0, "ⓑ": 0, "ⓝ": 0, "ⓢ": 0}
    prev_top = prev_mid = prev_low = None
    for c in seq:
        num[c] += 1
        if c == "ⓟ":
            name = "P_{}".format(num[c])
            g.add_node(name, type="page", level=3, page=num["ⓟ"])
            g.add_edge("D", name)
            if prev_top:
                g.add_edge(prev_top, name)
            prev_top, prev_mid, prev_low = name, None, None
        if c in "ⓝⓢ":
            name = "{}_{}".format("N" if c == "ⓝ" else "S", num[c])
            g.add_node(name, type="number" if c == "ⓝ" else "section", level=2, page=num["ⓟ"])
            g.add_edge(prev_top, name)
            if prev_mid:
                g.add_edge(prev_mid, name)
            prev_mid, prev_low = name, None
        if c in "ⓐⓑ":
            name = "{}_{}".format("A" if c == "ⓐ" else "B", num[c])
            g.add_node(name, type="annotation" if c == "ⓐ" else "body", level=1, page=num["ⓟ"])
            g.add_edge(prev_mid, name)
            if prev_low:
                g.add_edge(prev_low, name)
            prev_low = name
    return g


def str_to_graph_rimes(s):
    """DAN's graph for RIMES 2009 pages: D -> one node per region, sequential edges."""
    begin_layout_tokens = "".join(RIMES_MATCHING_TOKENS.keys())
    seq = keep_only_tokens(s, begin_layout_tokens)
    g = nx.DiGraph()
    g.add_node("D", type="document", level=2, page=0)
    names = {"ⓑ": "B", "ⓞ": "O", "ⓡ": "R", "ⓢ": "S", "ⓦ": "W", "ⓨ": "Y", "ⓟ": "P"}
    num = {t: 0 for t in begin_layout_tokens}
    prev = None
    for c in seq:
        num[c] += 1
        name = "{}_{}".format(names[c], num[c])
        g.add_node(name, type=names[c], level=1, page=0)
        g.add_edge("D", name)
        if prev:
            g.add_edge(prev, name)
        prev = name
    return g


def graph_edit_distance(g1, g2):
    """Exact GED with unit costs; substitution free iff node types match (DAN's costs)."""
    best = None
    for v in nx.optimize_graph_edit_distance(
            g1, g2,
            node_ins_cost=lambda n: 1, node_del_cost=lambda n: 1,
            node_subst_cost=lambda a, b: 0 if a["type"] == b["type"] else 1,
            edge_ins_cost=lambda e: 1, edge_del_cost=lambda e: 1,
            edge_subst_cost=lambda a, b: 0 if a == b else 1):
        best = v
    return best


def graph_edit_distance_by_page(g1, g2):
    """DAN's page-wise GED for double pages: split both graphs by page index and sum."""
    n1 = len([n for n, d in g1.nodes(data=True) if d["type"] == "page"])
    n2 = len([n for n, d in g2.nodes(data=True) if d["type"] == "page"])
    if n1 <= 1 and n2 <= 1:
        return graph_edit_distance(g1, g2)
    pages1 = [g1.subgraph([n for n, d in g1.nodes(data=True) if d["page"] == p]) for p in range(1, n1 + 1)]
    pages2 = [g2.subgraph([n for n, d in g2.nodes(data=True) if d["page"] == p]) for p in range(1, n2 + 1)]
    edit = 0
    for i in range(max(len(pages1), len(pages2))):
        a = pages1[i] if i < len(pages1) else nx.DiGraph()
        b = pages2[i] if i < len(pages2) else nx.DiGraph()
        edit += graph_edit_distance(a, b)
    return edit


def loer_items_read(str_gt, str_pred):
    """(edit distance, |V_gt| + |E_gt|) for one READ document, DAN definition."""
    g_gt, g_pred = str_to_graph_read(str_gt), str_to_graph_read(str_pred)
    return graph_edit_distance_by_page(g_gt, g_pred), g_gt.number_of_nodes() + g_gt.number_of_edges()


def loer_items_rimes(str_gt, str_pred):
    g_gt, g_pred = str_to_graph_rimes(str_gt), str_to_graph_rimes(str_pred)
    return graph_edit_distance(g_gt, g_pred), g_gt.number_of_nodes() + g_gt.number_of_edges()


def loer(pairs, dataset="read"):
    """Micro-averaged LOER over (gt, pred) pairs. Predictions should already be post-processed
    with the dataset's PostProcessingModule (as DAN does) if that is the protocol being followed."""
    fn = loer_items_read if dataset == "read" else loer_items_rimes
    edits, norm = 0, 0
    for gt, pred in pairs:
        e, n = fn(gt, pred)
        edits += e
        norm += n
    return edits / norm if norm else float("nan")


# ----------------------------------------------------------------------------------------------
# 2. Reading-order-invariant text metrics (Vidal et al. 2023)
# ----------------------------------------------------------------------------------------------

def words_of(s, layout_tokens, tokenizer="dan"):
    """Tokenise. "dan": the repository's / DAN's WER convention (layout tokens removed,
    punctuation marks are separate words, newlines and spaces separate words) so that
    ΔWER = WER − bWER is consistent with the reported WER. "whitespace": Vidal et al.'s
    convention (any whitespace-delimited character sequence is a word; punctuation attached)."""
    s = keep_all_but_tokens(s, layout_tokens)
    if tokenizer == "dan":
        for p in "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~¬":
            s = s.replace(p, " " + p + " ")
    return s.replace("\n", " ").split()


def wer_counts(gt_words, pred_words):
    return editdistance.eval(gt_words, pred_words), len(gt_words)


def bwer_counts(gt_words, pred_words):
    """Bag-of-words WER numerator: words that cannot be matched regardless of order."""
    cg, cp = Counter(gt_words), Counter(pred_words)
    matched = sum((cg & cp).values())
    return max(len(gt_words), len(pred_words)) - matched, len(gt_words)


def hwer_counts(gt_words, pred_words, gamma=0.0):
    """Hungarian WER numerator, Vidal et al. 2023 Eq. 9-11.

    Both sequences are padded with dummy words λ to the same length. Pair costs:
    g(x, y) = character edit distance, g(v, λ) = g(λ, v) = |v| / 2, g(λ, λ) = 0, plus the
    positional regulariser γ·|j − k| / N (Eq. 11; γ = 0 unless requested). The minimum-cost
    assignment is found with the Hungarian algorithm. The error is then the number of
    assigned pairs whose words differ, δ(x, y) ∈ {0, 1} with δ(v, λ) = 1, minus (D − b) / 2,
    where D is the number of pairs involving exactly one dummy and b = ||X| − |Y||, so that a
    paired insertion + deletion counts as one substitution (Eq. 10).
    Returns (error_count, N_gt, assignment) with assignment = {gt index: pred index} over the
    real (non-dummy) pairs."""
    n, m = len(gt_words), len(pred_words)
    if n == 0 and m == 0:
        return 0.0, 0, {}
    size = max(n, m)
    N = max(n, 1)
    cost = np.zeros((size, size), dtype=np.float64)
    for i in range(size):
        for j in range(size):
            if i < n and j < m:
                cost[i, j] = editdistance.eval(gt_words[i], pred_words[j]) + gamma * abs(i - j) / N
            elif i < n:
                cost[i, j] = len(gt_words[i]) / 2.0
            elif j < m:
                cost[i, j] = len(pred_words[j]) / 2.0
    rows, cols = linear_sum_assignment(cost)
    mismatches, dummies = 0, 0
    assignment = {}
    for i, j in zip(rows, cols):
        real_i, real_j = i < n, j < m
        if real_i and real_j:
            assignment[int(i)] = int(j)
            mismatches += int(gt_words[i] != pred_words[j])
        elif real_i or real_j:
            mismatches += 1
            dummies += 1
    b = abs(n - m)
    error = mismatches - (dummies - b) / 2.0
    return float(error), n, assignment


def nsfd(assignment, n_gt, n_pred=None):
    """Normalised Spearman footrule distance, Vidal et al. 2023 Eq. 5:
    ρ = Σ_{(j,k) ∈ A} |j − k| / ⌊N² / 2⌋ with N = max(|X|, |Y|), over the assigned real pairs."""
    if n_pred is None:
        n_pred = n_gt
    N = max(n_gt, n_pred)
    denom = (N * N) // 2
    if denom == 0 or not assignment:
        return 0.0
    return sum(abs(j - k) for j, k in assignment.items()) / denom


def order_invariant_metrics(pairs, layout_tokens, tokenizer="dan", gamma=0.0):
    """Micro-averaged WER, bWER, hWER, ΔWER = WER - bWER and mean NSFD over (gt, pred) pairs."""
    wer_num = bwer_num = hwer_num = 0.0
    n_words = 0
    nsfds = []
    for gt, pred in pairs:
        g, p = words_of(gt, layout_tokens, tokenizer), words_of(pred, layout_tokens, tokenizer)
        e, n = wer_counts(g, p)
        wer_num += e
        b, _ = bwer_counts(g, p)
        bwer_num += b
        h, _, assign = hwer_counts(g, p, gamma)
        hwer_num += h
        nsfds.append(nsfd(assign, len(g), len(p)))
        n_words += n
    if n_words == 0:
        return {k: float("nan") for k in ("wer", "bwer", "hwer", "delta_wer", "nsfd")}
    return {"wer": wer_num / n_words, "bwer": bwer_num / n_words, "hwer": hwer_num / n_words,
            "delta_wer": (wer_num - bwer_num) / n_words, "nsfd": float(np.mean(nsfds))}
