"""
Query Performance Prediction (QPP) measures for conversational ambiguity.

Hypothesis: ambiguous queries have lower QPP scores — they are harder to
satisfy with retrieved information, so the retrieval signal is weaker.

Implementation tiers:
  Tier 1 — query text only (query_length, query_entropy)
  Tier 2 — query + pseudo-collection from observations
           (avg_idf, max_idf, avg_scq, max_scq, sum_scq, scs, query_scope)
  Tier 3 — ranked list with scores (wig, nqc, smv, sigma_max, n_sigma,
           clarity); requires pyserini index or explicit ranked list

This module implements Tiers 1-2 fully and Tier 3 when a ranked list
is provided (from pyserini or mock).
"""

import math
import re
import random
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Tokenizer
# ══════════════════════════════════════════════════════════════════════════════
STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could and but or nor not "
    "no so if then than that this these those it its i me my we our "
    "you your he him his she her they them their what which who whom "
    "how when where why all each every both few more most other some "
    "such to of in for on with at by from as into through during "
    "before after above below between about against am is are".split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return [t for t in text.split() if len(t) > 1]


def tokenize_no_stop(text: str) -> List[str]:
    """Tokenize and remove stopwords."""
    return [t for t in tokenize(text) if t not in STOPWORDS]


# ══════════════════════════════════════════════════════════════════════════════
# Pseudo-Collection — built from observation fields across the dataset
# ══════════════════════════════════════════════════════════════════════════════
class PseudoCollection:
    """
    Index of all observation texts, providing IDF and collection frequency
    statistics needed for pre-retrieval QPP measures.

    Each observation text is treated as one "document" in the collection.
    """

    def __init__(self):
        self.N = 0                     # total documents
        self.df = Counter()            # term → document frequency
        self.cf = Counter()            # term → total occurrences
        self.total_tokens = 0          # total tokens across all docs
        self._doc_texts: List[str] = []  # raw texts for BM25 scoring

    def add_document(self, text: str):
        """Add one observation text to the collection."""
        tokens = tokenize(text)
        if not tokens:
            return
        self.N += 1
        self._doc_texts.append(text)
        seen = set()
        for t in tokens:
            self.cf[t] += 1
            self.total_tokens += 1
            if t not in seen:
                self.df[t] += 1
                seen.add(t)

    def idf(self, term: str) -> float:
        """log(1 + N / df_t).  Returns 0 for unknown terms."""
        d = self.df.get(term, 0)
        if d == 0:
            return 0.0
        return math.log(1 + self.N / d)

    def collection_prob(self, term: str) -> float:
        """P(t|C) = cf_t / total_tokens."""
        if self.total_tokens == 0:
            return 1e-9
        return self.cf.get(term, 0) / self.total_tokens

    def scq(self, term: str) -> float:
        """SCQ(t) = (1 + ln(cf_t)) · ln(1 + N/df_t)."""
        c = self.cf.get(term, 0)
        d = self.df.get(term, 0)
        if c == 0 or d == 0:
            return 0.0
        return (1 + math.log(c)) * math.log(1 + self.N / d)

    def bm25_score(self, query_tokens: List[str], doc_text: str,
                   k1: float = 1.2, b: float = 0.75) -> float:
        """BM25 score for a single query–document pair."""
        doc_tokens = tokenize(doc_text)
        dl = len(doc_tokens)
        avgdl = self.total_tokens / max(self.N, 1)
        doc_tf = Counter(doc_tokens)
        score = 0.0
        for t in query_tokens:
            tf = doc_tf.get(t, 0)
            if tf == 0:
                continue
            idf_t = self.idf(t)
            num = tf * (k1 + 1)
            den = tf + k1 * (1 - b + b * dl / max(avgdl, 1))
            score += idf_t * num / den
        return score

    def get_random_doc(self, rng: random.Random = None) -> str:
        """Return a random document text (for irrelevant-doc QPP)."""
        if not self._doc_texts:
            return ""
        r = rng or random
        return r.choice(self._doc_texts)

    def __repr__(self):
        return (f"PseudoCollection(N={self.N}, vocab={len(self.df)}, "
                f"tokens={self.total_tokens})")


# ══════════════════════════════════════════════════════════════════════════════
# SIP-format parser — extracts (query, observations, label) per turn
# ══════════════════════════════════════════════════════════════════════════════
def parse_sip_for_qpp(
    raw: Dict, num_classes: int = 2,
) -> List[Dict]:
    """
    Parse a SIP conversation into per-turn records for QPP.

    Returns list of dicts, one per user-system pair:
      {
        "query":        str,    # human utterance (without observation)
        "observations": [str],  # observation texts for this turn
        "system":       str,    # gpt response
        "label":        int,    # remapped ambiguous_type
        "raw_label":    int,
        "turn_idx":     int,
      }
    """
    from ..data.final_loader import remap_label

    convs = raw.get("conversations", raw.get("turns", []))
    records = []
    i = 0
    turn_idx = 0
    while i < len(convs):
        c = convs[i]
        role = c.get("from", c.get("role", ""))
        if role == "function_call":
            i += 1
            continue
        if role == "human":
            query = c.get("value", "")
            observations = []
            i += 1
            while i < len(convs):
                cur = convs[i]
                r = cur.get("from", cur.get("role", ""))
                if r == "function_call":
                    i += 1
                elif r == "observation":
                    observations.append(cur.get("value", ""))
                    i += 1
                elif r in ("gpt", "human"):
                    break
                else:
                    i += 1
            if i < len(convs) and convs[i].get("from", convs[i].get("role", "")) == "gpt":
                gpt = convs[i]
                raw_label = int(gpt.get("ambiguous_type", 0))
                records.append({
                    "query": query,
                    "observations": observations,
                    "system": gpt.get("value", ""),
                    "label": remap_label(raw_label, num_classes),
                    "raw_label": raw_label,
                    "turn_idx": turn_idx,
                })
                turn_idx += 1
                i += 1
        else:
            i += 1
    return records


# ══════════════════════════════════════════════════════════════════════════════
# QPP Scorer — computes all measures for a single turn
# ══════════════════════════════════════════════════════════════════════════════
class QPPScorer:
    """
    Compute QPP features for a query given a PseudoCollection.

    Usage:
        collection = PseudoCollection()
        for obs_text in all_observations:
            collection.add_document(obs_text)

        scorer = QPPScorer(collection)
        features = scorer.score_turn(query, observations)
    """

    def __init__(self, collection: PseudoCollection,
                 rewriter_path: str = None):
        """
        Args:
            collection: PseudoCollection built from all observation texts.
            rewriter_path: local path to a T5 query rewriter model (optional).
                           If provided, enables query-variant QPP (QV-NQC).
        """
        self.col = collection
        self.rewriter = None
        if rewriter_path:
            self._load_rewriter(rewriter_path)

    def _load_rewriter(self, path: str):
        """Load a T5 query rewriter from a local directory."""
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self.rewriter_tok = T5Tokenizer.from_pretrained(path)
            self.rewriter_model = T5ForConditionalGeneration.from_pretrained(path)
            self.rewriter_model.eval()
            self.rewriter = True
            log.info(f"Loaded query rewriter from {path}")
        except Exception as e:
            log.warning(f"Could not load rewriter from {path}: {e}")
            self.rewriter = None

    def rewrite_query(self, query: str, context: str = "") -> str:
        """Rewrite query using T5 model.  Returns original if no rewriter."""
        if not self.rewriter:
            return query
        import torch
        inp = f"rewrite: {context} ||| {query}" if context else f"rewrite: {query}"
        ids = self.rewriter_tok(inp, return_tensors="pt", max_length=256,
                                truncation=True).input_ids
        with torch.no_grad():
            out = self.rewriter_model.generate(ids, max_new_tokens=64)
        return self.rewriter_tok.decode(out[0], skip_special_tokens=True)

    # ── Tier 1: query text only ──────────────────────────────────────────

    def query_length(self, query: str) -> float:
        """Number of non-stopword tokens."""
        return float(len(tokenize_no_stop(query)))

    def query_entropy(self, query: str) -> float:
        """Shannon entropy of term distribution within the query."""
        tokens = tokenize(query)
        if not tokens:
            return 0.0
        tf = Counter(tokens)
        n = len(tokens)
        return -sum((c / n) * math.log2(c / n) for c in tf.values())

    # ── Tier 2: query + pseudo-collection ────────────────────────────────

    def avg_idf(self, query: str) -> float:
        """AvgIDF(Q) = (1/|Q|) · Σ ln(1 + N/df_t)."""
        tokens = tokenize_no_stop(query)
        if not tokens:
            return 0.0
        return sum(self.col.idf(t) for t in tokens) / len(tokens)

    def max_idf(self, query: str) -> float:
        """MaxIDF(Q) = max_t ln(1 + N/df_t)."""
        tokens = tokenize_no_stop(query)
        if not tokens:
            return 0.0
        return max(self.col.idf(t) for t in tokens)

    def avg_scq(self, query: str) -> float:
        """AvgSCQ(Q) = (1/|Q|) · Σ SCQ(t)."""
        tokens = tokenize_no_stop(query)
        if not tokens:
            return 0.0
        return sum(self.col.scq(t) for t in tokens) / len(tokens)

    def max_scq(self, query: str) -> float:
        """MaxSCQ(Q) = max_t SCQ(t)."""
        tokens = tokenize_no_stop(query)
        if not tokens:
            return 0.0
        return max(self.col.scq(t) for t in tokens)

    def sum_scq(self, query: str) -> float:
        """SumSCQ(Q) = Σ SCQ(t)."""
        tokens = tokenize_no_stop(query)
        return sum(self.col.scq(t) for t in tokens)

    def scs(self, query: str) -> float:
        """
        Simplified Clarity Score.
        SCS(Q) = Σ P_ml(t|Q) · log2(P_ml(t|Q) / P(t|C))
        """
        tokens = tokenize(query)
        if not tokens:
            return 0.0
        tf = Counter(tokens)
        n = len(tokens)
        score = 0.0
        for t, c in tf.items():
            p_q = c / n
            p_c = self.col.collection_prob(t)
            if p_c > 0:
                score += p_q * math.log2(p_q / p_c)
        return score

    def query_scope(self, query: str) -> float:
        """ω(Q) = -log(n_Q / N), where n_Q = docs containing ≥1 query term."""
        tokens = set(tokenize_no_stop(query))
        if not tokens or self.col.N == 0:
            return 0.0
        n_q = sum(1 for doc in self.col._doc_texts
                  if any(t in tokenize(doc) for t in tokens))
        n_q = max(n_q, 1)
        return -math.log(n_q / self.col.N) if self.col.N > 0 else 0.0

    # ── Tier 3: post-retrieval (ranked list required) ────────────────────

    def wig(self, query: str, scores: List[float], k: int = 5) -> float:
        """
        Weighted Information Gain.
        WIG(q) = (1/k) · Σ_{i=1..k} (1/√|q|) · (s_i − μ_corpus)
        """
        tokens = tokenize_no_stop(query)
        ql = max(len(tokens), 1)
        if not scores:
            return 0.0
        top_k = scores[:k]
        mu_corpus = np.mean(scores) if scores else 0.0
        return (1 / k) * (1 / math.sqrt(ql)) * sum(s - mu_corpus for s in top_k)

    def nqc(self, query: str, scores: List[float], k: int = 100) -> float:
        """
        Normalized Query Commitment.
        NQC(q) = std(top_k_scores) / μ_corpus
        """
        if not scores:
            return 0.0
        top_k = scores[:k]
        mu_corpus = np.mean(scores) if scores else 1.0
        if mu_corpus == 0:
            mu_corpus = 1e-9
        return float(np.std(top_k) / abs(mu_corpus))

    def smv(self, query: str, scores: List[float], k: int = 100) -> float:
        """
        Score Magnitude and Variance.
        SMV = [Σ s_i · |ln(s_i/μ)|] / (k · μ_corpus)
        """
        if not scores:
            return 0.0
        top_k = scores[:k]
        mu = np.mean(top_k) if top_k else 1e-9
        mu_corpus = np.mean(scores) if scores else 1e-9
        if mu == 0:
            mu = 1e-9
        if mu_corpus == 0:
            mu_corpus = 1e-9
        val = sum(s * abs(math.log(max(s, 1e-9) / mu)) for s in top_k)
        return val / (len(top_k) * abs(mu_corpus))

    def sigma_max(self, scores: List[float], K: int = 100) -> float:
        """
        σ_max: maximum std over all rank prefixes.
        σ_max(q) = max_{k'=1..K} std(s_1, ..., s_{k'})
        """
        if len(scores) < 2:
            return 0.0
        top_K = scores[:K]
        best = 0.0
        for k_prime in range(2, len(top_K) + 1):
            sd = float(np.std(top_K[:k_prime]))
            if sd > best:
                best = sd
        return best

    def n_sigma(self, query: str, scores: List[float],
                x_pct: float = 0.5) -> float:
        """
        n(σ_x%): std of scores above x% of the max score, normalized by √|q|.
        """
        tokens = tokenize_no_stop(query)
        ql = max(len(tokens), 1)
        if not scores:
            return 0.0
        s_max = max(scores)
        thresh = x_pct * s_max
        filtered = [s for s in scores if s >= thresh]
        if len(filtered) < 2:
            return 0.0
        return float(np.std(filtered)) / math.sqrt(ql)

    def clarity(self, query: str, scored_docs: List[Tuple[str, float]],
                k: int = 100, mu: float = 2000.0) -> float:
        """
        Clarity Score (Cronen-Townsend et al.)
        KL-divergence between relevance model P(w|θ_Q) and collection P(w|C).

        scored_docs: list of (doc_text, score) tuples, top-k.
        Uses Dirichlet-smoothed document language models.
        """
        if not scored_docs:
            return 0.0
        top_k = scored_docs[:k]
        total_score = sum(s for _, s in top_k)
        if total_score <= 0:
            return 0.0

        # Build relevance model P(w|θ_Q)
        rel_model = Counter()
        for doc_text, score in top_k:
            doc_tokens = tokenize(doc_text)
            dl = len(doc_tokens)
            doc_tf = Counter(doc_tokens)
            weight = score / total_score
            for t, c in doc_tf.items():
                p_t_c = self.col.collection_prob(t)
                p_t_d = (c + mu * p_t_c) / (dl + mu)
                rel_model[t] += weight * p_t_d

        # KL divergence: Σ P(w|θ_Q) · log2(P(w|θ_Q) / P(w|C))
        kl = 0.0
        for t, p_q in rel_model.items():
            p_c = self.col.collection_prob(t)
            if p_q > 0 and p_c > 0:
                kl += p_q * math.log2(p_q / p_c)
        return kl

    # ── Combined scorer ──────────────────────────────────────────────────

    def score_turn(
        self,
        query: str,
        observations: List[str] = None,
        ranked_list: List[Tuple[str, float]] = None,
        irrelevant_docs: List[str] = None,
        context_queries: List[str] = None,
    ) -> Dict[str, float]:
        """
        Compute all applicable QPP features for one turn.

        Args:
            query: human utterance text
            observations: observation texts for this turn (may be empty)
            ranked_list: [(doc_text, score), ...] if available (Tier 3)
            irrelevant_docs: random observation texts from other conversations
                             used to simulate non-relevant documents for
                             building a mock ranked list
            context_queries: prior queries in the conversation (for
                             conversational QPP features)

        Returns: dict of {measure_name: float_value}
        """
        features = {}

        # Tier 1: query only
        features["query_length"] = self.query_length(query)
        features["query_entropy"] = self.query_entropy(query)

        # Tier 2: query + collection
        features["avg_idf"] = self.avg_idf(query)
        features["max_idf"] = self.max_idf(query)
        features["avg_scq"] = self.avg_scq(query)
        features["max_scq"] = self.max_scq(query)
        features["sum_scq"] = self.sum_scq(query)
        features["scs"] = self.scs(query)
        features["query_scope"] = self.query_scope(query)

        # Build mock ranked list from observation + irrelevant docs
        if ranked_list is None and (observations or irrelevant_docs):
            ranked_list = self._build_mock_ranked_list(
                query, observations or [], irrelevant_docs or [])

        # Tier 3: post-retrieval (if ranked list available)
        if ranked_list:
            scores = [s for _, s in ranked_list]
            features["wig"] = self.wig(query, scores, k=min(5, len(scores)))
            features["nqc"] = self.nqc(query, scores, k=min(100, len(scores)))
            features["smv"] = self.smv(query, scores, k=min(100, len(scores)))
            features["sigma_max"] = self.sigma_max(scores)
            features["n_sigma"] = self.n_sigma(query, scores)
            features["clarity"] = self.clarity(query, ranked_list)

        # Conversational features
        if context_queries:
            features["turn_position"] = float(len(context_queries))
            # Query drift: how different is this query from the first?
            if len(context_queries) >= 1:
                first_tokens = set(tokenize_no_stop(context_queries[0]))
                curr_tokens = set(tokenize_no_stop(query))
                if first_tokens or curr_tokens:
                    jaccard = (len(first_tokens & curr_tokens) /
                               max(len(first_tokens | curr_tokens), 1))
                    features["query_drift"] = 1.0 - jaccard

        return features

    def _build_mock_ranked_list(
        self, query: str,
        observations: List[str],
        irrelevant_docs: List[str],
    ) -> List[Tuple[str, float]]:
        """
        Build a mock ranked list by scoring observations (relevant)
        and random docs (irrelevant) with BM25 against the query.

        Returns: [(doc_text, bm25_score), ...] sorted by score desc.
        """
        query_tokens = tokenize_no_stop(query)
        if not query_tokens:
            return []

        pairs = []
        for doc in observations:
            s = self.col.bm25_score(query_tokens, doc)
            pairs.append((doc, s))
        for doc in irrelevant_docs:
            s = self.col.bm25_score(query_tokens, doc)
            pairs.append((doc, s))

        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Thresholding classifier — converts QPP scores to binary predictions
# ══════════════════════════════════════════════════════════════════════════════
def threshold_classify(
    scores: np.ndarray,
    method: str = "percentile",
    percentile: float = 25.0,
    threshold: float = None,
) -> Tuple[np.ndarray, float]:
    """
    Classify turns as ambiguous (1) or clear (0) based on QPP scores.

    Lower QPP → more likely ambiguous.

    Args:
        scores: array of QPP scores (one per turn)
        method:
          "percentile" — bottom `percentile`% classified as ambiguous
          "fixed"      — scores < threshold are ambiguous
          "otsu"       — automatic threshold maximizing inter-class variance

    Returns: (predictions array, threshold_used)
    """
    if len(scores) == 0:
        return np.array([], dtype=int), 0.0

    if method == "fixed":
        t = threshold if threshold is not None else np.median(scores)
        preds = (scores < t).astype(int)
        return preds, float(t)

    elif method == "percentile":
        t = np.percentile(scores, percentile)
        preds = (scores <= t).astype(int)
        return preds, float(t)

    elif method == "otsu":
        # Otsu's method: find threshold that maximizes between-class variance
        sorted_s = np.sort(scores)
        best_t, best_var = sorted_s[0], -1.0
        for i in range(1, len(sorted_s)):
            c0 = sorted_s[:i]
            c1 = sorted_s[i:]
            w0 = len(c0) / len(sorted_s)
            w1 = len(c1) / len(sorted_s)
            var = w0 * w1 * (np.mean(c0) - np.mean(c1)) ** 2
            if var > best_var:
                best_var = var
                best_t = (sorted_s[i - 1] + sorted_s[i]) / 2
        preds = (scores < best_t).astype(int)
        return preds, float(best_t)

    else:
        raise ValueError(f"Unknown threshold method: {method}")


def find_best_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    num_steps: int = 200,
) -> Tuple[float, float]:
    """
    Search for the threshold on QPP scores that maximizes F1 on the
    given labels.  Lower score → ambiguous (label=1).

    Returns: (best_threshold, best_f1)
    """
    from sklearn.metrics import f1_score

    lo, hi = float(np.min(scores)), float(np.max(scores))
    if lo == hi:
        return lo, 0.0
    best_t, best_f1 = lo, 0.0
    for t in np.linspace(lo, hi, num_steps):
        preds = (scores < t).astype(int)
        f = f1_score(labels, preds, average="macro", zero_division=0)
        if f > best_f1:
            best_f1 = f
            best_t = float(t)
    return best_t, best_f1
