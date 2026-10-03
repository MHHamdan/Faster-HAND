"""
Hand-scored test cases for hand_v2/metrics/layout_metrics.py.

Every expected value below was computed by hand from the definitions (DAN §4.2 for LOER,
Vidal et al. 2023 for bWER/hWER) before the code was run; the derivation is written next to
each case so a reader can check it without executing anything.

Run: python hand_v2/tests/test_layout_metrics.py
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from hand_v2.metrics.layout_metrics import (  # noqa: E402
    str_to_graph_read, loer_items_read, loer, order_invariant_metrics, words_of,
    bwer_counts, hwer_counts, nsfd, READ_MATCHING_TOKENS,
)

LT = "".join(READ_MATCHING_TOKENS.keys()) + "".join(READ_MATCHING_TOKENS.values())


def test_graph_shape_single_page():
    # ⓟ ⓝ 12 Ⓝ ⓢ ⓐ ann Ⓐ ⓑ body Ⓑ Ⓢ Ⓟ
    # Nodes: D, P_1, N_1, S_1, A_1, B_1  -> 6
    # Edges: D->P1, P1->N1, P1->S1, N1->S1 (sequential, level 2), S1->A1, S1->B1, A1->B1 (sequential, level 1) -> 7
    s = "ⓟⓝ12Ⓝⓢⓐann.Ⓐⓑbody textⒷⓈⓅ"
    g = str_to_graph_read(s)
    assert g.number_of_nodes() == 6, g.nodes
    assert g.number_of_edges() == 7, g.edges
    assert set(g.edges) == {("D", "P_1"), ("P_1", "N_1"), ("P_1", "S_1"), ("N_1", "S_1"),
                            ("S_1", "A_1"), ("S_1", "B_1"), ("A_1", "B_1")}


def test_loer_identity_is_zero():
    s = "ⓟⓝ12Ⓝⓢⓐann.Ⓐⓑbody textⒷⓈⓅ"
    e, n = loer_items_read(s, s)
    assert e == 0 and n == 13, (e, n)


def test_loer_missing_annotation():
    # GT as above (|V|+|E| = 13). Prediction drops the annotation:
    # pred nodes D,P1,N1,S1,B1 (5); edges D->P1, P1->N1, P1->S1, N1->S1, S1->B1 (5).
    # Cheapest edit: delete node A_1 (1) + delete edges S1->A1 and A1->B1 (2) = 3.  LOER = 3/13.
    gt = "ⓟⓝ12Ⓝⓢⓐann.Ⓐⓑbody textⒷⓈⓅ"
    pred = "ⓟⓝ12Ⓝⓢⓑbody textⒷⓈⓅ"
    e, n = loer_items_read(gt, pred)
    assert (e, n) == (3, 13), (e, n)
    assert abs(loer([(gt, pred)]) - 3 / 13) < 1e-9


def test_loer_text_errors_do_not_count():
    # Only layout tokens matter: a garbled body text with the same tokens has LOER 0.
    gt = "ⓟⓝ12Ⓝⓢⓐann.Ⓐⓑbody textⒷⓈⓅ"
    pred = "ⓟⓝ21Ⓝⓢⓐxxx.Ⓐⓑb0dy txetⒷⓈⓅ"
    assert loer_items_read(gt, pred)[0] == 0


def test_loer_extra_section():
    # GT: one section with one body.  Nodes D,P1,S1,B1 (4); edges D->P1, P1->S1, S1->B1 (3): norm 7.
    # Pred: two sections each with a body. Nodes D,P1,S1,B1,S2,B2 (6);
    # edges D->P1, P1->S1, S1->B1, P1->S2, S1->S2, S2->B2 (6). There is NO B1->B2 edge:
    # DAN resets the level-1 "previous" node whenever a level-2 token (ⓢ/ⓝ) is read.
    # Cheapest edit from GT to pred: insert S2, B2 (2) + insert edges P1->S2, S1->S2, S2->B2 (3) = 5.
    gt = "ⓟⓢⓑaⒷⓈⓅ"
    pred = "ⓟⓢⓑaⒷⓈⓢⓑbⒷⓈⓅ"
    e, n = loer_items_read(gt, pred)
    assert (e, n) == (5, 7), (e, n)


def test_loer_double_page_is_pagewise():
    # Two pages, each: P, S, B. DAN resets the level-2 and level-1 "previous" nodes at every
    # page token, so there are NO sequential edges across pages at those levels; only the
    # page level chains (P1->P2).  GT nodes: D,P1,S1,B1,P2,S2,B2 = 7; edges: D->P1, P1->S1,
    # S1->B1, D->P2, P1->P2, P2->S2, S2->B2 = 7 -> normaliser 14.
    # The page-wise GED keeps only nodes with page == p, so D and the P1->P2 edge are outside
    # both per-page subgraphs. Prediction identical on page 1, page 2 missing its body:
    # page-2 subgraph loses node B2 and edge S2->B2 -> edit 2.  LOER = 2/14.
    gt = "ⓟⓢⓑaⒷⓈⓅⓟⓢⓑbⒷⓈⓅ"
    pred = "ⓟⓢⓑaⒷⓈⓅⓟⓢⓈⓅ"
    e, n = loer_items_read(gt, pred)
    assert (e, n) == (2, 14), (e, n)


def test_words_tokenisation_matches_wer_convention():
    assert words_of("ⓟⓑHello, world.ⒷⓅ", LT) == ["Hello", ",", "world", "."]


def test_bwer_hand_cases():
    # gt {a,b,c}, pred {c,a,b}: same bag -> 0 unmatched
    assert bwer_counts(["a", "b", "c"], ["c", "a", "b"]) == (0, 3)
    # gt {a,b}, pred {a,c}: one substitution -> 1
    assert bwer_counts(["a", "b"], ["a", "c"]) == (1, 2)
    # gt {a,b}, pred {a}: one deletion -> 1 ; gt {a}, pred {a,b}: one insertion -> 1
    assert bwer_counts(["a", "b"], ["a"]) == (1, 2)
    assert bwer_counts(["a"], ["a", "b"]) == (1, 1)


def test_hwer_hand_cases():
    # Identical bags in a different order: zero cost, assignment is a permutation.
    cost, n, assign = hwer_counts(["a", "b", "c"], ["c", "a", "b"])
    assert cost == 0 and n == 3 and assign == {0: 1, 1: 2, 2: 0}
    # gt ["abcd"], pred ["abce"]: the pair is assigned (edit cost 1 < dummy costs 2 + 2) and
    # the words differ -> one word error (Eq. 10: δ = 1, D = 0).
    cost, n, _ = hwer_counts(["abcd"], ["abce"])
    assert cost == 1.0 and n == 1
    # gt ["ab"], pred []: one dummy pair, D = 1, b = 1 -> error 1 - (1-1)/2 = 1.
    cost, n, assign = hwer_counts(["ab"], [])
    assert cost == 1.0 and n == 1 and assign == {}
    # gt [a, b], pred [a]: (a,a) matched, (b,λ) dummy: D = 1, b = 1 -> 1 error out of 2.
    cost, n, _ = hwer_counts(["a", "b"], ["a"])
    assert cost == 1.0 and n == 2
    # Insertion + deletion of different words in equal-length sequences counts as one
    # substitution through the (D-b)/2 correction only when dummies are used; here the
    # Hungarian pairs (b, c) directly: gt [a, b], pred [a, c] -> 1 error.
    cost, n, _ = hwer_counts(["a", "b"], ["a", "c"])
    assert cost == 1.0 and n == 2


def test_order_invariant_metrics_reading_order_swap():
    # Two lines swapped in the prediction. Words: gt = [one two three four], pred = [three four one two]
    # WER (Levenshtein on words) = 4 (delete "one two", insert "one two")  -> 4/4 = 1.0
    # bWER = 0, hWER = 0, ΔWER = 1.0.  NSFD (Eq. 5): assigned pairs (j,k) = (0,2),(1,3),(2,0),(3,1)
    # Σ|j-k| = 8 ; ⌊N²/2⌋ = 8 for N = 4 -> 1.0
    gt = "ⓟⓑone two\nthree fourⒷⓅ"
    pred = "ⓟⓑthree four\none twoⒷⓅ"
    m = order_invariant_metrics([(gt, pred)], LT)
    assert abs(m["wer"] - 1.0) < 1e-9, m
    assert m["bwer"] == 0 and m["hwer"] == 0, m
    assert abs(m["delta_wer"] - 1.0) < 1e-9, m
    assert abs(m["nsfd"] - 1.0) < 1e-9, m


def test_nsfd_identity():
    assert nsfd({0: 0, 1: 1, 2: 2}, 3, 3) == 0.0
    # one word displaced by one position among 3: |1-2| + |2-1| = 2 over ⌊9/2⌋ = 4 -> 0.5
    assert abs(nsfd({0: 0, 1: 2, 2: 1}, 3, 3) - 0.5) < 1e-9


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok ", t.__name__)
    print("all {} tests passed".format(len(tests)))
