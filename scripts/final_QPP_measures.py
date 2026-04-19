"""
Query Performance Prediction (QPP) measures for conversational ambiguity.

Each measure estimates how well a retrieval system will satisfy a query.
Lower QPP → harder query → more likely ambiguous.

Canonical references and formula validation:
  IDF, SCQ, SCS  — He & Ounis, ECIR 2008 (§3)
  WIG            — Zhou & Croft, SIGIR 2007 (§3.1, Eq. 1)
  NQC            — Shtok et al., TOIS 2012 (§3.2, Eq. 5)
  SMV            — Tao & Wu, CIKM 2014 (§3, Eq. 3)
  σ_max          — Cummins, SIGIR 2014 (§3, Eq. 7)
  n(σ_x%)        — Pérez-Iglesias & Araujo, SPIRE 2010
  Clarity        — Cronen-Townsend et al., SIGIR 2002 (§3, Eq. 4)
  Query Scope    — He & Ounis, ECIR 2008 (§3)
"""

import math
import re
import random
import logging
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Tokenization
# ══════════════════════════════════════════════════════════════════════════════
STOPWORDS = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would shall should may might can could and but or nor not "
    "no so if then than that this these those it its i me my we our "
    "you your he him his she her they them their what which who whom "
    "how when where why all each every both few more most other some "
    "such to of in for on with at by from as into through during "
    "before after above below between about against am".split()
)


def tokenize(text: str) -> List[str]:
    """Lowercase whitespace tokenizer.  Strips non-alphanumeric chars.
    Keeps single-char tokens (important for acronyms like 'x', 'y')."""
    return re.findall(r"[a-z0-9]+", text.lower())


def tokenize_no_stop(text: str) -> List[str]:
    return [t for t in tokenize(text) if t not in STOPWORDS]


_BERT_TOKENIZER = None


def get_bert_tokenizer(path: str):
    """Lazy-load BERT tokenizer from a local directory."""
    global _BERT_TOKENIZER
    if _BERT_TOKENIZER is None:
        from transformers import AutoTokenizer
        _BERT_TOKENIZER = AutoTokenizer.from_pretrained(path)
    return _BERT_TOKENIZER


def tokenize_bert(text: str, tokenizer_path: str) -> List[str]:
    """Tokenize with BERT WordPiece.  Returns subword tokens without
    special tokens, lowercased."""
    tok = get_bert_tokenizer(tokenizer_path)
    ids = tok.encode(text, add_special_tokens=False)
    return [t.lower().lstrip("##") for t in tok.convert_ids_to_tokens(ids)]


# ══════════════════════════════════════════════════════════════════════════════
# PseudoCollection — corpus statistics from observation texts
# ══════════════════════════════════════════════════════════════════════════════
class PseudoCollection:
    """
    Inverted index of observation texts providing df, cf, total docs, and
    total tokens — the four statistics needed by all pre-retrieval QPP.

    Terminology follows Lucene convention used by QPP4CS and Narabzad:
      N           = total number of documents
      df(t)       = number of documents containing term t
      cf(t)       = total occurrences of t across all documents
      total_tokens = Σ |d| over all documents
    """

    def __init__(self):
        self.N = 0
        self.df: Dict[str, int] = {}
        self.cf: Dict[str, int] = {}
        self.total_tokens = 0
        self._doc_texts: List[str] = []
        self._doc_token_sets: List[set] = []  # precomputed for query_scope

    def add_document(self, text: str):
        tokens = tokenize(text)
        if not tokens:
            return
        self.N += 1
        self._doc_texts.append(text)
        seen = set()
        for t in tokens:
            self.cf[t] = self.cf.get(t, 0) + 1
            self.total_tokens += 1
            if t not in seen:
                self.df[t] = self.df.get(t, 0) + 1
                seen.add(t)
        self._doc_token_sets.append(seen)

    def idf(self, term: str) -> float:
        """IDF(t) = log₂(N / df(t))  per He & Ounis 2008.
        Returns 0 for OOV terms (conservative: unknown = no signal)."""
        d = self.df.get(term, 0)
        if d == 0 or self.N == 0:
            return 0.0
        return math.log2(self.N / d)

    def ictf(self, term: str) -> float:
        """Inverse collection term frequency: log₂(total_tokens / cf(t)).
        Measures rarity by occurrence count rather than document count."""
        c = self.cf.get(term, 0)
        if c == 0 or self.total_tokens == 0:
            return 0.0
        return math.log2(self.total_tokens / c)

    def collection_prob(self, term: str) -> float:
        """P(t|C) = cf(t) / total_tokens.  Maximum-likelihood estimate."""
        if self.total_tokens == 0:
            return 0.0
        return self.cf.get(term, 0) / self.total_tokens

    def scq(self, term: str) -> float:
        """SCQ(t) = (1 + ln(cf(t))) · idf(t)   per He & Ounis 2008 §3.
        Note: uses natural log for cf, log₂ for IDF (matching the paper)."""
        c = self.cf.get(term, 0)
        if c == 0:
            return 0.0
        return (1.0 + math.log(c)) * self.idf(term)

    def var(self, term: str) -> float:
        """VAR(t) = Σ_d (tf(t,d) - mean_tf(t))² / N   per He & Ounis 2008.
        Variance of tf across documents.  High variance → term concentrated
        in few documents → more discriminative."""
        c = self.cf.get(term, 0)
        d = self.df.get(term, 0)
        if d == 0 or self.N == 0:
            return 0.0
        mean_tf = c / self.N  # note: over ALL docs, not just containing docs
        # Contribution from docs containing t: Σ(tf_i - mean)²
        # Contribution from docs NOT containing t: (N - df) * mean²
        # We don't store per-doc tf, so we use the approximation
        # VAR ≈ cf/N * (1 - df/N) which is the Bernoulli variance
        # scaled by cf.  This is the standard approximation used when
        # per-doc tf is unavailable (as in Narabzad's implementation).
        return (c / self.N) * (1.0 - d / self.N)

    def n_docs_with_any_term(self, terms: List[str]) -> int:
        """Count documents containing at least one term from the list.
        Used by query_scope.  O(N) but uses precomputed token sets."""
        term_set = set(terms)
        return sum(1 for doc_set in self._doc_token_sets
                   if doc_set & term_set)

    def bm25_score(self, query_tokens: List[str], doc_text: str,
                   k1: float = 0.9, b: float = 0.4) -> float:
        """BM25 with Pyserini default parameters (k1=0.9, b=0.4)."""
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
            den = tf + k1 * (1.0 - b + b * dl / max(avgdl, 1.0))
            score += idf_t * num / den
        return score

    def __repr__(self):
        return (f"PseudoCollection(N={self.N}, vocab={len(self.df)}, "
                f"tokens={self.total_tokens})")


# ══════════════════════════════════════════════════════════════════════════════
# SIP-format parser
# ══════════════════════════════════════════════════════════════════════════════
def parse_sip_for_qpp(raw: Dict, num_classes: int = 2) -> List[Dict]:
    """Parse a SIP conversation into per-turn QPP records."""
    from convqa_eval.data.loader import remap_label
    convs = raw.get("conversations", raw.get("turns", []))
    records, i, turn_idx = [], 0, 0
    while i < len(convs):
        c = convs[i]
        role = c.get("from", c.get("role", ""))
        if role == "function_call":
            i += 1; continue
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
                    "query": query, "observations": observations,
                    "system": gpt.get("value", ""),
                    "label": remap_label(raw_label, num_classes),
                    "raw_label": raw_label, "turn_idx": turn_idx,
                })
                turn_idx += 1; i += 1
        else:
            i += 1
    return records


# ══════════════════════════════════════════════════════════════════════════════
# QPPScorer
# ══════════════════════════════════════════════════════════════════════════════
class QPPScorer:
    """Compute QPP features per query against a PseudoCollection."""

    def __init__(self, collection: PseudoCollection,
                 bert_tokenizer_path: str = None,
                 rewriter_path: str = None):
        self.col = collection
        self.bert_path = bert_tokenizer_path
        self.rewriter = None
        if rewriter_path:
            self._load_rewriter(rewriter_path)

    def _load_rewriter(self, path: str):
        try:
            from transformers import T5ForConditionalGeneration, T5Tokenizer
            self._rw_tok = T5Tokenizer.from_pretrained(path)
            self._rw_model = T5ForConditionalGeneration.from_pretrained(path)
            self._rw_model.eval()
            self.rewriter = True
            log.info(f"Loaded query rewriter from {path}")
        except Exception as e:
            log.warning(f"Could not load rewriter from {path}: {e}")

    def rewrite_query(self, query: str, context: str = "") -> str:
        if not self.rewriter:
            return query
        import torch
        inp = f"rewrite: {context} ||| {query}" if context else f"rewrite: {query}"
        ids = self._rw_tok(inp, return_tensors="pt", max_length=256,
                           truncation=True).input_ids
        with torch.no_grad():
            out = self._rw_model.generate(ids, max_new_tokens=64)
        return self._rw_tok.decode(out[0], skip_special_tokens=True)

    def _q_tokens(self, query: str) -> List[str]:
        """Query tokens for IDF-based measures: stopwords removed."""
        if self.bert_path:
            tokens = tokenize_bert(query, self.bert_path)
            return [t for t in tokens if t not in STOPWORDS and len(t) > 1]
        return tokenize_no_stop(query)

    def _q_tokens_all(self, query: str) -> List[str]:
        """All query tokens including stopwords (for SCS, entropy)."""
        if self.bert_path:
            return tokenize_bert(query, self.bert_path)
        return tokenize(query)

    # ── Tier 1: query text only ──────────────────────────────────────────

    def query_length(self, query: str) -> float:
        """|Q| after stopword removal."""
        return float(len(self._q_tokens(query)))

    def query_entropy(self, query: str) -> float:
        """H(Q) = −Σ P(t|Q) · log₂ P(t|Q) over all tokens (with stops)."""
        tokens = self._q_tokens_all(query)
        if not tokens:
            return 0.0
        tf = Counter(tokens)
        n = len(tokens)
        return -sum((c / n) * math.log2(c / n) for c in tf.values())

    # ── Tier 2: pre-retrieval (query + collection) ───────────────────────
    # Formulas from He & Ounis, ECIR 2008:
    #   AvgIDF(Q) = (1/|Q|) · Σ_{t∈Q} log₂(N / df(t))
    #   MaxIDF(Q) = max_{t∈Q} log₂(N / df(t))
    #   SCQ(t)    = (1 + ln(cf(t))) · log₂(N / df(t))
    #   SCS(Q)    = Σ_{t∈Q} P_ml(t|Q) · log₂(P_ml(t|Q) / P(t|C))
    #   ω(Q)      = −log₂(n_Q / N)

    def avg_idf(self, query: str) -> float:
        tokens = self._q_tokens(query)
        if not tokens:
            return 0.0
        return sum(self.col.idf(t) for t in tokens) / len(tokens)

    def max_idf(self, query: str) -> float:
        tokens = self._q_tokens(query)
        if not tokens:
            return 0.0
        return max(self.col.idf(t) for t in tokens)

    def avg_ictf(self, query: str) -> float:
        """Average inverse collection term frequency."""
        tokens = self._q_tokens(query)
        if not tokens:
            return 0.0
        return sum(self.col.ictf(t) for t in tokens) / len(tokens)

    def avg_scq(self, query: str) -> float:
        tokens = self._q_tokens(query)
        if not tokens:
            return 0.0
        return sum(self.col.scq(t) for t in tokens) / len(tokens)

    def max_scq(self, query: str) -> float:
        tokens = self._q_tokens(query)
        if not tokens:
            return 0.0
        return max(self.col.scq(t) for t in tokens)

    def sum_scq(self, query: str) -> float:
        tokens = self._q_tokens(query)
        return sum(self.col.scq(t) for t in tokens)

    def avg_var(self, query: str) -> float:
        """Average tf-variance across collection per query term."""
        tokens = self._q_tokens(query)
        if not tokens:
            return 0.0
        return sum(self.col.var(t) for t in tokens) / len(tokens)

    def scs(self, query: str) -> float:
        """Simplified Clarity Score — canonical version (He & Ounis 2008 §3).
        SCS(Q) = Σ P_ml(t|Q) · log₂( P_ml(t|Q) / P(t|C) )
        Computed over ALL query tokens (including stopwords).

        WARNING: on domain-specific pseudo-collections (e.g. finance
        observations), stopwords that are rare in the domain inflate SCS
        for vague queries.  Use scs_content() for stopword-free variant."""
        tokens = self._q_tokens_all(query)
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
            elif p_q > 0:
                # OOV term: Laplace smoothing
                score += p_q * math.log2(p_q / (1.0 / max(self.col.total_tokens, 1)))
        return score

    def scs_content(self, query: str) -> float:
        """SCS over content terms only (stopwords removed).
        More reliable than scs() on domain-specific pseudo-collections.
        Returns 0 when no content terms have collection presence (all OOV)."""
        tokens = self._q_tokens(query)
        if not tokens:
            return 0.0
        tf = Counter(tokens)
        n = len(tokens)
        # Check if any term has collection presence
        has_cf = any(self.col.cf.get(t, 0) > 0 for t in tf)
        if not has_cf:
            return 0.0  # no signal: all terms are OOV
        score = 0.0
        for t, c in tf.items():
            p_q = c / n
            p_c = self.col.collection_prob(t)
            if p_c > 0:
                score += p_q * math.log2(p_q / p_c)
            # Skip OOV terms entirely — they inflate scores misleadingly
        return score

    def query_scope(self, query: str) -> float:
        """ω(Q) = −log₂(n_Q / N), n_Q = docs with ≥1 query term.
        Higher → fewer matching docs → more specific query."""
        tokens = self._q_tokens(query)
        if not tokens or self.col.N == 0:
            return 0.0
        n_q = self.col.n_docs_with_any_term(tokens)
        n_q = max(n_q, 1)
        return -math.log2(n_q / self.col.N)

    # ── Tier 3: post-retrieval (ranked list with scores) ─────────────────
    # All methods operate on scores = [s_1, s_2, ..., s_n] sorted desc.
    # μ_C = mean(scores) approximates the corpus-level average score.

    def wig(self, query: str, scores: List[float], k: int = 5) -> float:
        """WIG (Zhou & Croft, SIGIR 2007, Eq. 1):
        WIG(Q) = (1/k) · Σ_{i=1..k} (s_i − μ_C) / √|Q|
        where μ_C = mean of all retrieval scores (corpus avg)."""
        tokens = self._q_tokens(query)
        ql = max(len(tokens), 1)
        if not scores:
            return 0.0
        top_k = scores[:k]
        mu_c = np.mean(scores)
        return float(np.mean([s - mu_c for s in top_k]) / math.sqrt(ql))

    def nqc(self, query: str, scores: List[float], k: int = 100) -> float:
        """NQC (Shtok et al., TOIS 2012, Eq. 5):
        NQC(Q) = σ(s_1..s_k) / μ_C
        Standard deviation of top-k scores normalized by corpus mean."""
        if not scores:
            return 0.0
        top_k = np.array(scores[:k], dtype=np.float64)
        mu_c = np.mean(scores)
        if abs(mu_c) < 1e-12:
            return 0.0
        return float(np.std(top_k) / abs(mu_c))

    def smv(self, query: str, scores: List[float], k: int = 100) -> float:
        """SMV (Tao & Wu, CIKM 2014, Eq. 3):
        SMV(Q) = (1/(k·μ_C)) · Σ_{i=1..k} s_i · |ln(s_i / μ_k)|
        where μ_k = mean of top-k, μ_C = corpus mean."""
        if not scores:
            return 0.0
        top_k = scores[:k]
        mu_k = np.mean(top_k) if top_k else 1e-12
        mu_c = np.mean(scores)
        if abs(mu_c) < 1e-12 or abs(mu_k) < 1e-12:
            return 0.0
        val = sum(s * abs(math.log(max(s, 1e-12) / mu_k)) for s in top_k)
        return float(val / (len(top_k) * abs(mu_c)))

    def sigma_max(self, scores: List[float], K: int = 100) -> float:
        """σ_max (Cummins, SIGIR 2014, Eq. 7):
        σ_max(Q) = max_{k=2..K} σ(s_1..s_k)
        Self-selects the rank cutoff that maximizes score dispersion."""
        if len(scores) < 2:
            return 0.0
        top_K = np.array(scores[:K], dtype=np.float64)
        return float(max(np.std(top_K[:k]) for k in range(2, len(top_K) + 1)))

    def n_sigma(self, query: str, scores: List[float],
                x_pct: float = 0.5) -> float:
        """n(σ_x%) (Pérez-Iglesias & Araujo 2010):
        Filter scores ≥ x% of max, compute σ, normalize by √|Q|."""
        tokens = self._q_tokens(query)
        ql = max(len(tokens), 1)
        if not scores:
            return 0.0
        s_max = max(scores)
        if s_max <= 0:
            return 0.0
        filtered = [s for s in scores if s >= x_pct * s_max]
        if len(filtered) < 2:
            return 0.0
        return float(np.std(filtered) / math.sqrt(ql))

    def clarity(self, query: str, scored_docs: List[Tuple[str, float]],
                k: int = 100, mu_dir: float = 2000.0) -> float:
        """Clarity Score (Cronen-Townsend et al., SIGIR 2002, Eq. 4):
        Clarity = Σ_w P(w|θ_Q) · log₂( P(w|θ_Q) / P(w|C) )
        where P(w|θ_Q) is the relevance model:
          P(w|θ_Q) = Σ_d [ P(d|Q) · P_dir(w|d) ]
          P(d|Q) ∝ Score(Q, d)
          P_dir(w|d) = (tf(w,d) + μ·P(w|C)) / (|d| + μ)
        """
        if not scored_docs:
            return 0.0
        top_k = scored_docs[:k]
        total_score = sum(s for _, s in top_k)
        if total_score <= 0:
            return 0.0

        rel_model: Dict[str, float] = {}
        for doc_text, score in top_k:
            doc_tokens = tokenize(doc_text)
            dl = len(doc_tokens)
            doc_tf = Counter(doc_tokens)
            p_d_q = score / total_score  # P(d|Q)
            for t in set(doc_tokens):
                tf_t = doc_tf[t]
                p_c = self.col.collection_prob(t)
                p_w_d = (tf_t + mu_dir * p_c) / (dl + mu_dir)
                rel_model[t] = rel_model.get(t, 0.0) + p_d_q * p_w_d

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
        """Compute all applicable QPP features for one conversational turn."""
        features = {}

        # Tier 1
        features["query_length"] = self.query_length(query)
        features["query_entropy"] = self.query_entropy(query)

        # Tier 2
        features["avg_idf"] = self.avg_idf(query)
        features["max_idf"] = self.max_idf(query)
        features["avg_ictf"] = self.avg_ictf(query)
        features["avg_scq"] = self.avg_scq(query)
        features["max_scq"] = self.max_scq(query)
        features["sum_scq"] = self.sum_scq(query)
        features["avg_var"] = self.avg_var(query)
        features["scs"] = self.scs(query)
        features["scs_content"] = self.scs_content(query)
        features["query_scope"] = self.query_scope(query)

        # Build mock ranked list if needed
        if ranked_list is None and (observations or irrelevant_docs):
            ranked_list = self._build_mock_ranked_list(
                query, observations or [], irrelevant_docs or [])

        # Tier 3
        if ranked_list:
            scores = [s for _, s in ranked_list]
            features["wig"] = self.wig(query, scores, k=min(5, len(scores)))
            features["nqc"] = self.nqc(query, scores, k=min(100, len(scores)))
            features["smv"] = self.smv(query, scores, k=min(100, len(scores)))
            features["sigma_max"] = self.sigma_max(scores)
            features["n_sigma"] = self.n_sigma(query, scores)
            features["clarity"] = self.clarity(query, ranked_list)

        # Conversational context features
        if context_queries:
            features["turn_position"] = float(len(context_queries))
            if context_queries:
                first_tokens = set(tokenize_no_stop(context_queries[0]))
                curr_tokens = set(tokenize_no_stop(query))
                union = first_tokens | curr_tokens
                features["query_drift"] = (
                    1.0 - len(first_tokens & curr_tokens) / max(len(union), 1))

        return features

    def _build_mock_ranked_list(
        self, query: str,
        observations: List[str],
        irrelevant_docs: List[str],
    ) -> List[Tuple[str, float]]:
        query_tokens = self._q_tokens(query)
        if not query_tokens:
            return []
        pairs = []
        for doc in observations:
            pairs.append((doc, self.col.bm25_score(query_tokens, doc)))
        for doc in irrelevant_docs:
            pairs.append((doc, self.col.bm25_score(query_tokens, doc)))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs


# ══════════════════════════════════════════════════════════════════════════════
# Thresholding
# ══════════════════════════════════════════════════════════════════════════════
def threshold_classify(
    scores: np.ndarray,
    method: str = "percentile",
    percentile: float = 25.0,
    threshold: float = None,
    lower_is_ambiguous: bool = True,
) -> Tuple[np.ndarray, float]:
    """Classify turns based on QPP scores.

    lower_is_ambiguous=True:  score < t → ambiguous (default for IDF, SCQ, etc.)
    lower_is_ambiguous=False: score > t → ambiguous (for query_scope, query_entropy)

    Returns (predictions, threshold).
    """
    if len(scores) == 0:
        return np.array([], dtype=int), 0.0
    if method == "fixed":
        t = threshold if threshold is not None else float(np.median(scores))
    elif method == "percentile":
        if lower_is_ambiguous:
            t = float(np.percentile(scores, percentile))
        else:
            t = float(np.percentile(scores, 100.0 - percentile))
    elif method == "otsu":
        sorted_s = np.sort(scores)
        best_t, best_var = float(sorted_s[0]), -1.0
        for i in range(1, len(sorted_s)):
            w0 = i / len(sorted_s)
            w1 = 1.0 - w0
            m0, m1 = np.mean(sorted_s[:i]), np.mean(sorted_s[i:])
            v = w0 * w1 * (m0 - m1) ** 2
            if v > best_var:
                best_var = v
                best_t = (float(sorted_s[i - 1]) + float(sorted_s[i])) / 2.0
        t = best_t
    else:
        raise ValueError(f"Unknown method: {method}")
    if lower_is_ambiguous:
        preds = (scores < t).astype(int)
    else:
        preds = (scores > t).astype(int)
    return preds, t


def find_best_threshold(
    scores: np.ndarray, labels: np.ndarray, num_steps: int = 200,
) -> Tuple[float, float, bool]:
    """Grid-search threshold maximizing macro-F1.

    Searches BOTH directions (lower-is-ambiguous and higher-is-ambiguous)
    and returns the best.

    Returns: (best_threshold, best_f1, lower_is_ambiguous)
    """
    from sklearn.metrics import f1_score

    lo, hi = float(np.min(scores)), float(np.max(scores))
    if lo == hi:
        return lo, 0.0, True

    best_t, best_f1, best_dir = lo, 0.0, True
    for t in np.linspace(lo, hi, num_steps):
        # Direction 1: lower score → ambiguous
        preds_lo = (scores < t).astype(int)
        f_lo = f1_score(labels, preds_lo, average="macro", zero_division=0)
        if f_lo > best_f1:
            best_f1, best_t, best_dir = f_lo, float(t), True

        # Direction 2: higher score → ambiguous
        preds_hi = (scores > t).astype(int)
        f_hi = f1_score(labels, preds_hi, average="macro", zero_division=0)
        if f_hi > best_f1:
            best_f1, best_t, best_dir = f_hi, float(t), False

    return best_t, best_f1, best_dir
