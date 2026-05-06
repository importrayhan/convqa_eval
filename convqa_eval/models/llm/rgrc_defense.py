"""
Three-Channel Triangulated Defense for Adversarially Robust
Clarification Decisions in Conversational Search.

ARCHITECTURE:
  Channel R — Retrieval QPP features (query-observation statistics)
  Channel L — LLM direction projection (cos/projection on d_clar)
  Channel C — CRF transition consistency (BiLSTM-CRF plausibility
              score via LLM embeddings)

WHY THREE CHANNELS:
  Two-channel RGRC (R+L) achieves AUROC≈0.49 because:
    (1) d_clar cosines cluster in [-0.07, -0.04] with near-zero variance
    (2) QPP features alone cannot distinguish clean from adversarial
        (they measure query quality, not prompt integrity)

  Channel C adds a LEARNED sequential signal. A lightweight BiLSTM-CRF
  projects the flattened conversation through the LLM's own embedding
  layer and learns what label-transition patterns look like in training
  data. An adversarial suffix that flips the LLM's prediction creates an
  INCONSISTENT transition pattern that the CRF finds implausible.

  Three channels triangulate: if all three agree "ambiguous", confidence
  is high. If GCG suppresses Channel L but Channel R and C still say
  "ambiguous", the discrepancy is detectable.

SEPARATOR:
  A kernel SVM (RBF) or gradient-boosted classifier on the concatenated
  [R, L, C] feature vector. The kernel trick captures non-linear
  interactions between channels that linear models miss — critical
  because the clean/adversarial boundary in the joint space is non-linear.

TRAINING:
  All training on TRAIN set only. Test set for evaluation only.
  1. Extract d_clar from contrastive train pairs
  2. Train Channel C (BiLSTM-CRF head on LLM embeddings) on train labels
  3. Generate synthetic adversarial examples on train set
  4. Train separator on [R, L, C] features (clean=0, adversarial=1)

REFERENCES:
  - Meng et al. 2023: MuSIc BiLSTM-CRF for SIP (CIKM)
  - Arditi et al. 2024: refusal direction (NeurIPS)
  - He & Ounis 2008: pre-retrieval QPP (ECIR)
  - Zou et al. 2023: GCG adversarial attacks
"""

import logging
import math
from typing import Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
# Channel R: Retrieval QPP features
# ══════════════════════════════════════════════════════════════════════════════
def compute_channel_r(
    query: str,
    observations: List[str],
    scorer=None,
    irrelevant_docs: List[str] = None,
) -> Dict[str, float]:
    """
    Retrieval-side features immune to prompt suffix attacks.
    Uses QPPScorer for the full feature set.
    """
    from QPP_measures import tokenize, tokenize_no_stop

    features = {}
    q_tokens = set(tokenize_no_stop(query))
    all_obs_tokens = set()
    obs_token_sets = []
    for obs in observations:
        t = set(tokenize(obs))
        obs_token_sets.append(t)
        all_obs_tokens.update(t)

    features["doc_term_overlap"] = (
        len(q_tokens & all_obs_tokens) / max(len(q_tokens | all_obs_tokens), 1)
        if q_tokens and all_obs_tokens else 0.0)
    features["coverage"] = (
        len(q_tokens & all_obs_tokens) / max(len(q_tokens), 1)
        if q_tokens else 0.0)

    if len(obs_token_sets) >= 2:
        jacs = []
        for i in range(len(obs_token_sets)):
            for j in range(i + 1, len(obs_token_sets)):
                u = obs_token_sets[i] | obs_token_sets[j]
                jacs.append(len(obs_token_sets[i] & obs_token_sets[j]) / max(len(u), 1))
        features["obs_coherence"] = float(np.mean(jacs))
    else:
        features["obs_coherence"] = 1.0

    features["query_length"] = float(len(q_tokens))

    if scorer is not None:
        qpp = scorer.score_turn(
            query, observations=observations,
            irrelevant_docs=irrelevant_docs or [])
        for k in ["avg_idf", "max_idf", "avg_ictf", "avg_scq",
                   "scs_content", "query_scope", "avg_var",
                   "wig", "nqc", "smv", "sigma_max", "clarity"]:
            if k in qpp:
                features[k] = qpp[k]

    return features


# ══════════════════════════════════════════════════════════════════════════════
# Channel L: LLM direction features
# ══════════════════════════════════════════════════════════════════════════════
def compute_channel_l(
    hidden_state: np.ndarray,
    direction: np.ndarray,
) -> Dict[str, float]:
    """
    LLM representation features relative to d_clar.
    Multiple features beyond cosine to capture the full geometry.
    """
    d_norm = np.linalg.norm(direction)
    h_norm = np.linalg.norm(hidden_state)
    if d_norm < 1e-8 or h_norm < 1e-8:
        return {"cos_sim": 0.0, "projection": 0.0,
                "residual_norm": 0.0, "h_norm": 0.0,
                "proj_ratio": 0.0}

    d_hat = direction / d_norm
    proj = float(np.dot(hidden_state, d_hat))
    cos = proj / h_norm
    residual = hidden_state - proj * d_hat
    res_norm = float(np.linalg.norm(residual))

    return {
        "cos_sim": float(cos),
        "projection": proj,
        "residual_norm": res_norm,
        "h_norm": float(h_norm),
        "proj_ratio": abs(proj) / (res_norm + 1e-8),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Channel C: CRF transition consistency via LLM embeddings
# ══════════════════════════════════════════════════════════════════════════════
class CRFConsistencyChannel:
    """
    Lightweight BiLSTM-CRF head that operates on LLM embedding
    representations of the flattened conversation.

    The LLM's embedding layer projects each token into a 4096+ dim space.
    We average-pool each utterance, then run a small BiLSTM to capture
    sequential dependencies, and a CRF to produce a label-sequence
    plausibility score.

    The CRF is trained on clean training conversations. At inference,
    the CRF's log-probability of the predicted label sequence is the
    "consistency score" — lower probability means the label pattern is
    implausible (possible adversarial manipulation).

    This channel adds learned transition information that neither QPP
    nor direction projection can provide.
    """

    def __init__(self, hidden_size: int = 64, num_classes: int = 2):
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.model = None
        self.is_trained = False

    def build_model(self, input_dim: int):
        """Build the BiLSTM-CRF head (PyTorch)."""
        import torch
        import torch.nn as nn

        class BiLSTMCRFHead(nn.Module):
            def __init__(self, input_dim, hidden_size, num_classes):
                super().__init__()
                self.lstm = nn.LSTM(input_dim, hidden_size, num_layers=1,
                                    bidirectional=True, batch_first=True)
                self.emission = nn.Linear(hidden_size * 2, num_classes)
                self.transitions = nn.Parameter(
                    torch.randn(num_classes, num_classes) * 0.1)
                self.start = nn.Parameter(torch.randn(num_classes) * 0.1)

            def forward_emissions(self, x):
                """x: [1, T, d] -> emissions [T, C]"""
                out, _ = self.lstm(x)
                return self.emission(out.squeeze(0))

            def log_likelihood(self, emissions, tags):
                """CRF log P(tags | emissions)"""
                T, C = emissions.shape
                # Gold score
                gold = self.start[tags[0]] + emissions[0, tags[0]]
                for t in range(1, T):
                    gold = gold + self.transitions[tags[t-1], tags[t]] + emissions[t, tags[t]]
                # Partition (forward)
                alphas = self.start + emissions[0]
                for t in range(1, T):
                    scores = alphas.unsqueeze(1) + self.transitions + emissions[t].unsqueeze(0)
                    alphas = torch.logsumexp(scores, dim=0)
                log_Z = torch.logsumexp(alphas, dim=0)
                return gold - log_Z  # negative = unlikely

            def viterbi(self, emissions):
                """Decode best path, return (path, log_prob)."""
                import torch
                T, C = emissions.shape
                dp = self.start + emissions[0]
                bp = []
                for t in range(1, T):
                    scores = dp.unsqueeze(1) + self.transitions + emissions[t].unsqueeze(0)
                    dp, idx = scores.max(dim=0)
                    bp.append(idx)
                path = [dp.argmax().item()]
                for ptr in reversed(bp):
                    path.insert(0, ptr[path[0]].item())
                log_prob = dp.max()
                return path, float(log_prob)

        self.model = BiLSTMCRFHead(input_dim, self.hidden_size, self.num_classes)
        return self.model

    def train_on_embeddings(
        self,
        conversations_embeddings: List[np.ndarray],
        conversations_labels: List[np.ndarray],
        epochs: int = 30,
        lr: float = 1e-3,
    ):
        """
        Train the CRF head on LLM-embedded conversations.

        conversations_embeddings: list of [T_i, embed_dim] arrays
        conversations_labels: list of [T_i] label arrays
        """
        import torch
        import torch.optim as optim

        if not conversations_embeddings:
            log.warning("No training data for CRF channel")
            return

        input_dim = conversations_embeddings[0].shape[1]
        model = self.build_model(input_dim)
        optimizer = optim.Adam(model.parameters(), lr=lr)

        model.train()
        for epoch in range(epochs):
            total_loss = 0
            for emb, lab in zip(conversations_embeddings, conversations_labels):
                if len(lab) < 2:
                    continue
                x = torch.tensor(emb, dtype=torch.float32).unsqueeze(0)
                tags = torch.tensor(lab, dtype=torch.long)
                emissions = model.forward_emissions(x)
                loss = -model.log_likelihood(emissions, tags)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            if epoch % 10 == 0:
                log.info(f"  CRF head epoch {epoch}: loss={total_loss/max(len(conversations_embeddings),1):.4f}")

        model.eval()
        self.is_trained = True
        log.info(f"CRF consistency channel trained ({epochs} epochs, "
                 f"{len(conversations_embeddings)} convs)")

    def score(self, utterance_embeddings: np.ndarray,
              predicted_labels: np.ndarray = None) -> Dict[str, float]:
        """
        Score a conversation for CRF consistency.

        utterance_embeddings: [T, d] — one embedding PER UTTERANCE in the
            conversation (not a single classification-prompt embedding).
        predicted_labels: [T] — LLM's predicted labels per turn.

        Also computes topic-diversity transition features:
          topic_drift_mean: mean cosine distance between consecutive turns
          topic_drift_max: max cosine distance between consecutive turns
          embedding_spread: std of utterance embedding norms
        """
        import torch

        features = {}
        T = utterance_embeddings.shape[0]

        # Topic diversity features (always computable, no training needed)
        if T >= 2:
            drifts = []
            for t in range(T - 1):
                a, b = utterance_embeddings[t], utterance_embeddings[t + 1]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 1e-8 and nb > 1e-8:
                    cos = np.dot(a, b) / (na * nb)
                    drifts.append(1.0 - cos)  # cosine distance
            features["topic_drift_mean"] = float(np.mean(drifts))
            features["topic_drift_max"] = float(np.max(drifts))
        else:
            features["topic_drift_mean"] = 0.0
            features["topic_drift_max"] = 0.0

        norms = np.linalg.norm(utterance_embeddings, axis=1)
        features["embedding_spread"] = float(np.std(norms)) if T > 1 else 0.0
        features["embedding_mean_norm"] = float(np.mean(norms))

        # CRF consistency (requires training)
        if not self.is_trained or self.model is None or T < 1:
            features["log_prob"] = 0.0
            features["normalized_log_prob"] = 0.0
            features["viterbi_agreement"] = 0.5
            return features

        with torch.no_grad():
            x = torch.tensor(utterance_embeddings, dtype=torch.float32).unsqueeze(0)
            emissions = self.model.forward_emissions(x)
            crf_path, crf_log_prob = self.model.viterbi(emissions)

            if predicted_labels is not None and len(predicted_labels) == T:
                tags = torch.tensor(predicted_labels, dtype=torch.long)
                log_prob = float(self.model.log_likelihood(emissions, tags))
                agreement = sum(1 for a, b in zip(crf_path, predicted_labels)
                                if a == b) / T
            else:
                log_prob = crf_log_prob
                agreement = 1.0

        features["log_prob"] = log_prob
        features["normalized_log_prob"] = log_prob / max(T, 1)
        features["viterbi_agreement"] = agreement
        return features


# ══════════════════════════════════════════════════════════════════════════════
# Separator model (kernel SVM or gradient boosting)
# ══════════════════════════════════════════════════════════════════════════════
class TriangulatedSeparator:
    """
    Separator that operates on concatenated [R, L, C] features.

    Model options:
      "rbf_svm":   SVM with RBF kernel — captures non-linear boundaries
      "gbm":       gradient-boosted trees — handles feature interactions
      "logistic":  logistic regression — linear baseline

    Trained on training set with clean (label=0) and adversarial (label=1).
    """

    def __init__(self, model_type: str = "rbf_svm"):
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.feature_keys = None

    def fit(self, features_list: List[Dict[str, float]],
            labels: np.ndarray):
        """Train separator on feature dicts."""
        from sklearn.preprocessing import StandardScaler

        self.feature_keys = sorted(features_list[0].keys())
        X = np.array([[f.get(k, 0.0) for k in self.feature_keys]
                       for f in features_list], dtype=np.float64)

        self.scaler = StandardScaler()
        X_s = self.scaler.fit_transform(X)

        if self.model_type == "rbf_svm":
            from sklearn.svm import SVC
            self.model = SVC(kernel="rbf", C=10.0, gamma="scale",
                             probability=True, class_weight="balanced")
        elif self.model_type == "gbm":
            from sklearn.ensemble import GradientBoostingClassifier
            self.model = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                random_state=42)
        elif self.model_type == "logistic":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(
                max_iter=2000, C=1.0, class_weight="balanced")

        self.model.fit(X_s, labels)
        log.info(f"Separator ({self.model_type}) trained: "
                 f"{sum(labels==0)} clean + {sum(labels==1)} adversarial, "
                 f"{len(self.feature_keys)} features")

    def predict(self, features: Dict[str, float]) -> Dict[str, float]:
        """Predict adversarial probability."""
        x = np.array([[features.get(k, 0.0) for k in self.feature_keys]])
        x_s = self.scaler.transform(x)
        prob = self.model.predict_proba(x_s)[0]
        return {
            "is_adversarial": bool(prob[1] > 0.5),
            "adversarial_prob": float(prob[1]),
            "clean_prob": float(prob[0]),
        }

    def decision_function_2d(self, features_list: List[Dict],
                              dim1: str, dim2: str) -> Tuple:
        """
        Compute decision boundary in 2D slice for visualization.
        Returns (xx, yy, Z) meshgrid for contour plotting.
        """
        vals1 = [f[dim1] for f in features_list]
        vals2 = [f[dim2] for f in features_list]
        margin = 0.1
        x_min, x_max = min(vals1) - margin, max(vals1) + margin
        y_min, y_max = min(vals2) - margin, max(vals2) + margin

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200))

        # Build full feature vectors with all other dims at median
        medians = {k: np.median([f.get(k, 0.0) for f in features_list])
                   for k in self.feature_keys}
        grid_features = []
        for i in range(xx.ravel().shape[0]):
            f = dict(medians)
            f[dim1] = xx.ravel()[i]
            f[dim2] = yy.ravel()[i]
            grid_features.append(f)

        X_grid = np.array([[f.get(k, 0.0) for k in self.feature_keys]
                           for f in grid_features])
        X_grid_s = self.scaler.transform(X_grid)
        Z = self.model.predict_proba(X_grid_s)[:, 1].reshape(xx.shape)
        return xx, yy, Z


# ══════════════════════════════════════════════════════════════════════════════
# Full Defense Pipeline — with suffix sensitivity
# ══════════════════════════════════════════════════════════════════════════════
class TriangulatedDefense:
    """
    Defense with suffix-sensitivity features.

    The discriminative signal comes from comparing LLM behavior WITH vs
    WITHOUT the known adversarial suffix on the SAME input.

    Channel S (suffix sensitivity): measures how much the suffix changes
    the LLM's hidden state, logits, and prediction.

    Plus Channel R (QPP) and Channel L (direction) for grounding.
    """

    def __init__(self, direction: np.ndarray, layer: int,
                 scorer=None, crf_channel: CRFConsistencyChannel = None,
                 separator_type: str = "rbf_svm",
                 known_suffixes: List[str] = None):
        self.direction = direction
        self.layer = layer
        self.scorer = scorer
        self.crf = crf_channel or CRFConsistencyChannel()
        self.separator = TriangulatedSeparator(separator_type)
        self.known_suffixes = known_suffixes or []

    def _merge_all(self, r: Dict, l: Dict, c: Dict,
                   s: Dict = None) -> Dict[str, float]:
        merged = {}
        for k, v in r.items():
            if isinstance(v, (int, float)):
                merged[f"r_{k}"] = float(v)
        for k, v in l.items():
            if isinstance(v, (int, float)):
                merged[f"l_{k}"] = float(v)
        for k, v in c.items():
            if isinstance(v, (int, float)):
                merged[f"c_{k}"] = float(v)
        if s:
            for k, v in s.items():
                if isinstance(v, (int, float)):
                    merged[f"s_{k}"] = float(v)
        return merged

    def compute_suffix_sensitivity(
        self,
        h_clean: np.ndarray,
        h_suffix: np.ndarray,
        logits_clean: Dict[str, float],
        logits_suffix: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Measure how much the suffix changes the LLM's behavior.
        Large delta = suffix is actively manipulating the model.
        """
        d_hat = self.direction / max(np.linalg.norm(self.direction), 1e-8)
        proj_c = float(np.dot(h_clean, d_hat))
        proj_s = float(np.dot(h_suffix, d_hat))

        h_nc = np.linalg.norm(h_clean)
        h_ns = np.linalg.norm(h_suffix)
        cos_c = proj_c / max(h_nc, 1e-8)
        cos_s = proj_s / max(h_ns, 1e-8)

        # How much did h change overall?
        h_cos = float(np.dot(h_clean, h_suffix) / (max(h_nc, 1e-8) * max(h_ns, 1e-8)))

        # Label logit KL divergence
        n_lab = 4
        cp = np.clip([logits_clean.get(f"label_{i}_prob", 0.25) for i in range(n_lab)], 1e-8, 1)
        sp = np.clip([logits_suffix.get(f"label_{i}_prob", 0.25) for i in range(n_lab)], 1e-8, 1)
        cp, sp = cp / cp.sum(), sp / sp.sum()
        kl = float(np.sum(cp * np.log(cp / sp)))

        flipped = 1.0 if int(np.argmax(cp)) != int(np.argmax(sp)) else 0.0

        return {
            "cos_delta": cos_c - cos_s,
            "proj_delta": proj_c - proj_s,
            "logit_kl": kl,
            "pred_flipped": flipped,
            "entropy_delta": logits_clean.get("label_entropy", 0) - logits_suffix.get("label_entropy", 0),
            "h_cos": h_cos,
            "h_norm_delta": float(h_nc - h_ns),
        }

    def train(self, clean_features: List[Dict], adv_features: List[Dict]):
        """Train separator on merged clean (0) and adversarial (1) features."""
        all_f = clean_features + adv_features
        labels = np.array([0]*len(clean_features) + [1]*len(adv_features))
        self.separator.fit(all_f, labels)

    def detect(self, merged_features: Dict[str, float]) -> Dict:
        result = self.separator.predict(merged_features)
        result["merged"] = merged_features
        return result


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_defense(clean_results, adversarial_results):
    """Compute detection metrics."""
    from sklearn.metrics import (roc_auc_score, f1_score,
                                  precision_score, recall_score)

    labels = [0]*len(clean_results) + [1]*len(adversarial_results)
    preds = ([int(r["is_adversarial"]) for r in clean_results] +
             [int(r["is_adversarial"]) for r in adversarial_results])
    scores = ([r["adversarial_prob"] for r in clean_results] +
              [r["adversarial_prob"] for r in adversarial_results])

    fpr = sum(preds[:len(clean_results)]) / max(len(clean_results), 1)
    tpr = sum(preds[len(clean_results):]) / max(len(adversarial_results), 1)

    return {
        "tpr": float(tpr), "fpr": float(fpr),
        "f1": f1_score(labels, preds, zero_division=0),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "auroc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.5,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Visualization
# ══════════════════════════════════════════════════════════════════════════════
def plot_triangulated_defense(
    clean_results, adversarial_results,
    separator: TriangulatedSeparator,
    output_dir: str = "llm_results",
    model_name: str = "",
    comparison_metrics: Dict = None,
):
    """
    Publication-quality 4-panel figure:
      1. Decision boundary (logit_kl vs cos_delta with contour)
      2. ASR / Utility / Latency bar chart (multi-method comparison)
      3. ROC curve
      4. Per-method metrics table
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    all_merged = [r["merged"] for r in clean_results + adversarial_results]
    clean_scores = [r.get("adversarial_prob", 0) for r in clean_results]
    adv_scores = [r.get("adversarial_prob", 0) for r in adversarial_results]
    labels = [0]*len(clean_results) + [1]*len(adversarial_results)
    scores = clean_scores + adv_scores

    fig, axes = plt.subplots(2, 2, figsize=(13, 11))

    # ── Panel 1: Decision boundary ───────────────────────────────────
    ax = axes[0, 0]
    x_key = "logit_kl" if "logit_kl" in all_merged[0] else sorted(all_merged[0].keys())[0]
    y_key = "cos_delta" if "cos_delta" in all_merged[0] else sorted(all_merged[0].keys())[1]

    clean_x = [r["merged"].get(x_key, 0) for r in clean_results]
    clean_y = [r["merged"].get(y_key, 0) for r in clean_results]
    adv_x = [r["merged"].get(x_key, 0) for r in adversarial_results]
    adv_y = [r["merged"].get(y_key, 0) for r in adversarial_results]

    ax.scatter(clean_x, clean_y, c="#4CAF50", s=20, alpha=0.5,
               label="not flipped", edgecolors="none", zorder=3)
    ax.scatter(adv_x, adv_y, c="#F44336", s=20, alpha=0.5,
               label="flipped by suffix", marker="x", zorder=3)
    ax.set_xlabel(x_key, fontsize=10)
    ax.set_ylabel(y_key, fontsize=10)
    ax.set_title("Decision Boundary", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    # ── Panel 2: ASR and F1 comparison ──────────────────────────────
    ax = axes[0, 1]
    if comparison_metrics:
        methods = list(comparison_metrics.keys())
        asr_vals = [comparison_metrics[m].get("asr", 0) * 100 for m in methods]
        f1_vals = [comparison_metrics[m].get("f1", 0) * 100 for m in methods]

        x_pos = np.arange(len(methods))
        w = 0.35
        bars1 = ax.bar(x_pos - w/2, asr_vals, w, label="ASR (%) ↓",
                        color="#F44336", alpha=0.8)
        bars2 = ax.bar(x_pos + w/2, f1_vals, w, label="F1 (%) ↑",
                        color="#4CAF50", alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(methods, fontsize=9, rotation=15)
        ax.set_ylabel("Percentage", fontsize=11)
        ax.set_title("ASR vs F1", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.set_ylim(0, 105)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}", ha="center", fontsize=8)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{bar.get_height():.1f}", ha="center", fontsize=8)
    else:
        ax.text(0.5, 0.5, "Run with --compare\nfor multi-method comparison",
                ha="center", va="center", transform=ax.transAxes, fontsize=11)
        ax.set_title("ASR vs Utility", fontsize=12, fontweight="bold")

    # ── Panel 3: ROC curve ───────────────────────────────────────────
    ax = axes[1, 0]
    if len(set(labels)) > 1 and len(set(scores)) > 1:
        fpr_arr, tpr_arr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr_arr, tpr_arr)
        ax.plot(fpr_arr, tpr_arr, color="#3F51B5", lw=2,
                label=f"AUROC={roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("FPR", fontsize=11)
    ax.set_ylabel("TPR", fontsize=11)
    ax.set_title("ROC — Adversarial Detection", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.2)

    # ── Panel 4: Metrics table ───────────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    if comparison_metrics:
        col_labels = ["Method", "ASR↓", "FP", "Prec↑", "Rec↑", "F1↑", "Acc↑"]
        table_data = []
        for m in methods:
            d = comparison_metrics[m]
            table_data.append([
                m,
                f"{d.get('asr',0)*100:.1f}%",
                f"{d.get('fp',0)}",
                f"{d.get('precision',0)*100:.1f}%",
                f"{d.get('recall',0)*100:.1f}%",
                f"{d.get('f1',0)*100:.1f}%",
                f"{d.get('accuracy',0)*100:.1f}%",
            ])
        table = ax.table(cellText=table_data, colLabels=col_labels,
                         loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        # Color header
        for j in range(len(col_labels)):
            table[0, j].set_facecolor("#3F51B5")
            table[0, j].set_text_props(color="white", fontweight="bold")
    ax.set_title("Comparison", fontsize=12, fontweight="bold")

    fig.suptitle(f"Defense Evaluation — {model_name}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    p = out / "triangulated_defense.png"
    fig.savefig(p, dpi=200, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Plot saved: {p}")
    return [str(p)]
