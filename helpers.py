"""
Helper / utility functions for the Stylometric API.

Includes: language detection, sentence splitting, tonal classification,
JSD scoring, numpy conversion, NTP visualization generation.
"""

import re
from urllib.parse import urlparse
from difflib import SequenceMatcher
import os
import logging
from typing import Dict, List
from datetime import datetime
from collections import Counter

import numpy as np
from langdetect import detect, LangDetectException

logger = logging.getLogger("stylometry_api")

VISUALIZATION_DIR = "visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# ── Tonal labels used by zero-shot classifier ────────────────────────────────

TONAL_LABELS = [
    "formal or professional",
    "casual or friendly",
    "cautious or hedging",
    "directive or assertive",
    "supportive or encouraging",
    "analytical or neutral",
    "emotional or personal",
    "humorous or sarcastic",
    "urgent or alarming",
    "transactional or procedural",
]


# ── Language detection ────────────────────────────────────────────────────────

def detect_language(text: str, fallback: str = "en") -> str:
    """Detect language with fallback."""
    try:
        lang = detect(text)
        lang_mapping = {"en": "en", "es": "es", "ru": "ru", "sv": "sv"}
        return lang_mapping.get(lang, fallback)
    except (LangDetectException, Exception):
        return fallback


# ── Sentence splitting ────────────────────────────────────────────────────────

def split_sentences(text: str) -> List[str]:
    """Split text into sentences, respecting email structure."""
    lines = text.strip().split("\n")
    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = re.split(r"(?<=[.!?])\s+", line)
        for part in parts:
            part = part.strip()
            if len(part) > 2:
                sentences.append(part)
    return sentences

# ── Piece from parent words ────────────────────────────────────────────────────

def reconstruct_parent_words(detailed_results):
    """
    Post-process token-level NTP results to reconstruct the full parent word.
    Uses token_raw (pre-strip) to detect word boundaries via leading space.
    """
    if not detailed_results:
        return detailed_results

    word_groups = []
    current_group = []

    for i, result in enumerate(detailed_results):
        raw = result.get("token_raw", result.get("token", ""))

        is_word_start = (
            i == 0
            or raw.startswith(" ")
            or raw.startswith("\n")
            or raw.startswith("\t")
        )

        if is_word_start and current_group:
            word_groups.append(current_group)
            current_group = []

        current_group.append(i)

    if current_group:
        word_groups.append(current_group)

    for group in word_groups:
        full_word = "".join(
            detailed_results[idx].get("token_raw", detailed_results[idx].get("token", ""))
            for idx in group
        ).strip()

        for idx in group:
            token_stripped = detailed_results[idx].get("token", "").strip()
            detailed_results[idx]["parent_word"] = full_word if full_word != token_stripped else None

    return detailed_results


def add_parent_words_to_anomalous_tokens(anomalous_tokens, detailed_results):
    """
    Add parent_word field to the anomalous_tokens list by looking up
    each token's position in the detailed_results (which should already
    have parent_word set by reconstruct_parent_words).
    """
    if not detailed_results or not anomalous_tokens:
        return anomalous_tokens

    # Build position -> parent_word lookup
    pos_to_parent = {}
    for r in detailed_results:
        pos = r.get("position", -1)
        parent = r.get("parent_word", "")
        if pos >= 0:
            pos_to_parent[pos] = parent

    for tok in anomalous_tokens:
        pos = tok.get("position", -1)
        token_text = tok.get("token", "").strip()
        parent = pos_to_parent.get(pos, token_text)

        # Only set parent_word if it differs from the token itself
        if parent and parent != token_text:
            tok["parent_word"] = parent
        else:
            tok["parent_word"] = None

    return anomalous_tokens

# ── Numpy type conversion ────────────────────────────────────────────────────

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


# ── Tonal classification ─────────────────────────────────────────────────────

def classify_tone(sentences: List[str], tonal_classifier) -> List[Dict[str, float]]:
    """
    Run zero-shot classification on each sentence.
    Returns a list of {label: score} dicts, one per sentence.
    """
    if tonal_classifier is None:
        return [{} for _ in sentences]

    try:
        results = tonal_classifier(
            sentences,
            candidate_labels=TONAL_LABELS,
            multi_label=False,
        )
        if not isinstance(results, list):
            results = [results]

        output = []
        for r in results:
            scores = dict(zip(r["labels"], r["scores"]))
            output.append(scores)
        return output

    except Exception as e:
        logger.warning(f"Tonal classification failed: {e}")
        return [{} for _ in sentences]


def compute_jsd_tonal_scores(
    tonal_scores_per_sentence: List[Dict[str, float]],
) -> List[Dict]:
    """Compute JSD-based tonal anomaly scores for each sentence."""
    from scipy.spatial.distance import jensenshannon

    label_order = TONAL_LABELS
    n = len(tonal_scores_per_sentence)

    dist_matrix = np.array(
        [[s.get(label, 0.0) for label in label_order] for s in tonal_scores_per_sentence],
        dtype=float,
    )

    row_sums = dist_matrix.sum(axis=1, keepdims=True)
    zero_rows = (row_sums == 0).flatten()
    dist_matrix[zero_rows] = 1.0 / len(label_order)
    dist_matrix = dist_matrix / dist_matrix.sum(axis=1, keepdims=True)

    email_mean = dist_matrix.mean(axis=0)
    email_mean = email_mean / (email_mean.sum() + 1e-8)

    jsd_scores = []
    for row in dist_matrix:
        jsd = float(jensenshannon(row, email_mean))
        if np.isnan(jsd):
            jsd = 0.0
        jsd_scores.append(jsd)

    jsd_mean = float(np.mean(jsd_scores))
    jsd_std = float(np.std(jsd_scores))

    if jsd_std < 1e-6:
        jsd_z_scores = [0.0] * n
    else:
        jsd_z_scores = [(s - jsd_mean) / jsd_std for s in jsd_scores]

    dominant_tones = [
        max(s, key=s.get) if s else "informational"
        for s in tonal_scores_per_sentence
    ]

    tone_counts = Counter(dominant_tones)
    modal_tone = tone_counts.most_common(1)[0][0] if tone_counts else "informational"

    results = []
    for i in range(n):
        z = float(jsd_z_scores[i])
        is_anomaly = bool(z > 2.0)
        flag = None
        if is_anomaly:
            dom = dominant_tones[i]
            flag = (
                f"Tone shift: '{dom}' diverges from email's "
                f"'{modal_tone}' register (JSD z={z:.1f})"
            )
        results.append(
            {
                "jsd_score": round(jsd_scores[i], 4),
                "tonal_z_score": round(z, 2),
                "dominant_tone": dominant_tones[i],
                "modal_tone": modal_tone,
                "tonal_is_anomaly": is_anomaly,
                "tonal_flag": flag,
            }
        )

    return results


# ── Contact info helper ───────────────────────────────────────────────────────

def get_contact_info(contact_id: str) -> dict:
    """Return basic contact info stub."""
    return {"id": contact_id, "name": f"Contact {contact_id}", "email": None}


# ── 2-D cluster boundaries ───────────────────────────────────────────────────

def compute_2d_cluster_boundaries(
    projected_data: np.ndarray, y_train: np.ndarray, label_encoder: Dict
):
    """Compute 2D boundaries for each author cluster."""
    reverse_encoder = {v: k for k, v in label_encoder.items()}
    clusters = {}

    for author_id in np.unique(y_train):
        author_name = reverse_encoder[author_id]
        author_mask = y_train == author_id
        author_points = projected_data[author_mask]
        center = np.mean(author_points, axis=0)
        distances = np.linalg.norm(author_points - center, axis=1)
        radius_95 = np.percentile(distances, 95)
        radius_threshold = radius_95 * 1.5

        clusters[str(author_name)] = {
            "center": {"x": float(center[0]), "y": float(center[1])},
            "radius_95": float(radius_95),
            "radius_threshold": float(radius_threshold),
            "n_samples": len(author_points),
            "samples": [
                {"x": float(p[0]), "y": float(p[1])} for p in author_points[:100]
            ],
        }

    return clusters


# ── NTP visualization (matplotlib) ───────────────────────────────────────────

def generate_ntp_visualization(detailed_results, anomalous_tokens, content_preview, suspicion_threshold=7.0):
    """
    Generate comprehensive matplotlib visualization for NTP analysis.
    
    Uses suspicion scoring system:
    - Raw suspicion (0-10) combines probability, ratio, rank, capitalisation
    - Context weight scales suspicion by position in text
    - Weighted suspicion > threshold flags anomalies
    
    Args:
        detailed_results: Token-level analysis results from _analyze_tokens
        anomalous_tokens: List of flagged anomalous tokens
        content_preview: First ~100 chars of content for filename
        suspicion_threshold: Weighted suspicion threshold used (default 7.0)
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    try:
        positions, probabilities, ranks = [], [], []
        raw_suspicions, weighted_suspicions = [], []
        context_weights, is_anomaly, contains_emoji = [], [], []
        token_display = []

        for result in detailed_results:
            token = result.get("token", "").strip()
            if not token:
                continue

            positions.append(int(result.get("position", 0)))
            token_display.append(token if token.isprintable() else repr(token))
            probabilities.append(float(result.get("probability", 0)))
            ranks.append(int(result.get("rank", 0)))
            raw_suspicions.append(float(result.get("raw_suspicion", 0)))
            weighted_suspicions.append(float(result.get("weighted_suspicion", 0)))
            context_weights.append(float(result.get("context_weight", 0)))
            is_anomaly.append(bool(result.get("is_anomaly", False)))
            contains_emoji.append(bool(result.get("contains_emoji", False)))

        fig, axes = plt.subplots(4, 1, figsize=(16, 16))

        # ── Plot 1: Context Weight Progression ──
        ax1 = axes[0]
        ax1.plot(positions, context_weights, "b-", linewidth=2.5, label="Context Weight", zorder=3)
        ax1.fill_between(positions, 0, context_weights, alpha=0.2, color="blue")
        ax1.axhline(y=0.5, color="orange", linestyle="--", alpha=0.4, linewidth=1.5, label="25% of text (0.5)")
        ax1.axhline(y=1.0, color="green", linestyle="--", alpha=0.4, linewidth=1.5, label="50% of text (1.0)")
        ax1.set_xlabel("Token Position", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Context Weight", fontsize=12, fontweight="bold")
        ax1.set_title("Context Weight Progression Over Email", fontsize=14, fontweight="bold")
        ax1.legend(loc="lower right", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])

        # ── Plot 2: Raw vs Weighted Suspicion Scores with token labels ──
        ax2 = axes[1]
        ax2.plot(positions, raw_suspicions, "r--", alpha=0.4, linewidth=1.5, label="Raw Suspicion")
        ax2.plot(positions, weighted_suspicions, "b-", linewidth=2.5, label="Weighted Suspicion")
        ax2.axhline(y=suspicion_threshold, color="red", linestyle=":", alpha=0.6, linewidth=2, label=f"Weighted Threshold ({suspicion_threshold}/10)")
        ax2.axhline(y=suspicion_threshold, color="darkred", linestyle="--", alpha=0.3, linewidth=1.5)
        
        # Annotate tokens that breach the weighted threshold
        annotated_count = 0
        for i in range(len(positions)):
            if is_anomaly[i] and annotated_count < 10:
                ax2.annotate(
                    token_display[i],
                    xy=(positions[i], weighted_suspicions[i]),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=7, fontweight="bold", color="blue",
                    arrowprops=dict(arrowstyle="-", color="blue", alpha=0.5),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", edgecolor="blue", alpha=0.8),
                    zorder=15
                )
                annotated_count += 1
        
        # Annotate tokens that breach raw threshold but NOT weighted
        # (high raw suspicion dampened by low context weight)
        raw_only_count = 0
        for i in range(len(positions)):
            if (not is_anomaly[i] and raw_suspicions[i] > suspicion_threshold 
                    and raw_only_count < 8):
                ax2.annotate(
                    token_display[i],
                    xy=(positions[i], raw_suspicions[i]),
                    xytext=(5, -12), textcoords="offset points",
                    fontsize=6, fontstyle="italic", color="red", alpha=0.7,
                    arrowprops=dict(arrowstyle="-", color="red", alpha=0.3),
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="mistyrose", edgecolor="red", alpha=0.6),
                    zorder=12
                )
                raw_only_count += 1
        
        ax2.set_xlabel("Token Position", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Suspicion Score (0-10)", fontsize=12, fontweight="bold")
        ax2.set_title("Raw vs Context-Weighted Suspicion Scores", fontsize=14, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 10.5])

        # ── Plot 3: Token Probabilities by Context Quality ──
        ax3 = axes[2]
        high_mask = [w >= 0.6 for w in context_weights]
        low_mask = [w < 0.6 for w in context_weights]
        low_pos = [positions[i] for i in range(len(positions)) if low_mask[i]]
        low_prob = [probabilities[i] for i in range(len(probabilities)) if low_mask[i]]
        ax3.scatter(low_pos, low_prob, alpha=0.3, s=20, c="gray", label="Low Context (w<0.6)")

        high_pos = [positions[i] for i in range(len(positions)) if high_mask[i]]
        high_prob = [probabilities[i] for i in range(len(probabilities)) if high_mask[i]]
        high_ws = [weighted_suspicions[i] for i in range(len(weighted_suspicions)) if high_mask[i]]
        scatter = ax3.scatter(
            high_pos, high_prob, alpha=0.7, s=50, c=high_ws,
            cmap="RdYlGn_r", vmin=0, vmax=10, edgecolors="black", linewidth=0.5,
            label="High Context (w≥0.6)",
        )
        anom_pos = [positions[i] for i in range(len(positions)) if is_anomaly[i]]
        anom_prob = [probabilities[i] for i in range(len(probabilities)) if is_anomaly[i]]
        anom_tokens = [token_display[i] for i in range(len(positions)) if is_anomaly[i]]
        ax3.scatter(anom_pos, anom_prob, alpha=0.9, s=100, c="red", marker="x", linewidths=2.5, label="Anomalies", zorder=10)
        
        # Label anomalous tokens on the scatter plot
        for j, (ap, aprob, atk) in enumerate(zip(anom_pos, anom_prob, anom_tokens)):
            if j < 10:  # Limit labels to avoid clutter
                ax3.annotate(
                    atk,
                    xy=(ap, aprob),
                    xytext=(5, 8), textcoords="offset points",
                    fontsize=7, fontweight="bold", color="darkred",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="lightyellow", edgecolor="red", alpha=0.8),
                    zorder=15
                )
        ax3.set_yscale("log")
        ax3.set_xlabel("Token Position", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Token Probability (log)", fontsize=12, fontweight="bold")
        ax3.set_title("Token Probabilities by Context Quality", fontsize=14, fontweight="bold")
        ax3.legend(loc="lower right", fontsize=9)
        ax3.grid(True, alpha=0.3, which="both")
        plt.colorbar(scatter, ax=ax3).set_label("Weighted Suspicion (0-10)", fontsize=10)

        # ── Plot 4: Top Suspicious Tokens (by weighted suspicion) ──
        ax4 = axes[3]
        suspicious = sorted(
            [
                (positions[i], token_display[i], weighted_suspicions[i], context_weights[i], raw_suspicions[i])
                for i in range(len(positions))
                if weighted_suspicions[i] > 3.0  # Show medium+ suspicion
            ],
            key=lambda x: x[2],
            reverse=True,
        )[:15]

        if suspicious:
            labels = [f"{t[1]} @{t[0]}" for t in suspicious]
            scores = [t[2] for t in suspicious]
            weights = [t[3] for t in suspicious]
            colors = [
                "darkred" if w >= 0.9 else "red" if w >= 0.6 else "orange" if w >= 0.3 else "gold"
                for w in weights
            ]
            y_pos = np.arange(len(labels))
            ax4.barh(y_pos, scores, color=colors, alpha=0.7, edgecolor="black", linewidth=1)
            ax4.axvline(x=suspicion_threshold, color="red", linestyle="--", linewidth=2, alpha=0.6, label=f"Threshold ({suspicion_threshold})")
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(labels, fontsize=8)
            ax4.set_xlabel("Weighted Suspicion Score (0-10)", fontsize=12, fontweight="bold")
            ax4.set_title("Top Suspicious Tokens (by weighted suspicion, colored by context weight)", fontsize=14, fontweight="bold")
            ax4.set_xlim([0, 10.5])
            ax4.grid(True, alpha=0.3, axis="x")
            legend_elements = [
                Patch(facecolor="darkred", alpha=0.7, edgecolor="black", label="Full Context (≥0.9)"),
                Patch(facecolor="red", alpha=0.7, edgecolor="black", label="Good Context (0.6-0.9)"),
                Patch(facecolor="orange", alpha=0.7, edgecolor="black", label="Partial Context (0.3-0.6)"),
                Patch(facecolor="gold", alpha=0.7, edgecolor="black", label="Minimal Context (<0.3)"),
            ]
            ax4.legend(handles=legend_elements, loc="lower right", fontsize=8)
        else:
            ax4.text(0.5, 0.5, "✅ No Suspicious Tokens Detected", ha="center", va="center", fontsize=14, fontweight="bold", transform=ax4.transAxes)

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_preview = "".join(c for c in content_preview if c.isalnum() or c in (" ", "-", "_"))[:50]
        filename = f"ntp_suspicion_{timestamp}_{safe_preview}.png"
        filepath = os.path.join(VISUALIZATION_DIR, filename)
        plt.savefig(filepath, dpi=150, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        logger.info(f"✅ Saved NTP visualization: {filepath}")
        return filename

    except Exception as e:
        logger.error(f"Failed to generate NTP visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

# Phishing persuasion labels (Cialdini-based, for social engineering detection)
PHISHING_PERSUASION_LABELS = [
    "creating urgency or scarcity",
    "claiming authority or official status",
    "requesting credentials or sensitive data",
    "threatening negative consequences",
    "offering a reward or incentive",
    "impersonating a trusted person or organization",
    "neutral or informational",
]

# ── Known brands for typosquat detection ─────────────────────────────────────
_KNOWN_BRANDS = [
    "google", "microsoft", "apple", "amazon", "paypal", "netflix", "facebook",
    "instagram", "linkedin", "dropbox", "chase", "wellsfargo", "bankofamerica",
    "americanexpress", "citibank", "hsbc", "barclays", "outlook", "office365",
    "icloud", "yahoo", "twitter", "github", "slack", "zoom", "docusign",
    "adobe", "salesforce", "shopify", "stripe", "coinbase", "binance",
    "sberbank", "tinkoff", "yandex", "vkontakte",  # Russian
    "santander", "bbva", "mercadolibre",             # Spanish
    "klarna", "swish", "handelsbanken",              # Swedish
]

# ── Multilingual phishy domain keywords ──────────────────────────────────────
_PHISHY_DOMAIN_KEYWORDS = [
    # English
    "login", "logon", "signin", "signup", "secure", "security", "verify",
    "verification", "account", "update", "confirm", "auth", "password",
    "support", "helpdesk", "alert", "suspend", "recover", "reset",
    # Russian
    "вход", "авторизация", "подтверждение", "безопасность", "аккаунт",
    "пароль", "поддержка", "обновление", "проверка",
    # Spanish
    "inicio", "sesion", "seguridad", "verificacion", "cuenta",
    "contrasena", "soporte", "actualizar", "confirmar",
    # Swedish
    "logga", "inloggning", "sakerhet", "verifiera", "konto",
    "losenord", "support", "uppdatera", "bekrafta",
]

# ── Multilingual heuristic patterns ──────────────────────────────────────────

_HEURISTIC_PATTERNS = {
    "credential_request": re.compile(
        r"(?i)\b("
        r"password|passcode|pin\b|credentials?|log\s*in|sign\s*in|verify your"
        r"|пароль|логин|войти|подтвердите"
        r"|contraseña|iniciar sesión|verificar"
        r"|lösenord|logga in|verifiera"
        r")\b"
    ),
    "urgency_pressure": re.compile(
        r"(?i)\b("
        r"immediately|urgent|right away|within \d+ hours?|act now|expires?\b|deadline|asap"
        r"|срочно|немедленно|истекает"
        r"|inmediatamente|urgente|de inmediato"
        r"|omedelbart|brådskande|genast"
        r")\b"
    ),
    "threat_consequence": re.compile(
        r"(?i)\b("
        r"suspend|terminat|delet|block|frozen|locked out|lose access|will be closed"
        r"|заблокирован|удалён|приостановлен"
        r"|suspendid|eliminad|bloquead"
        r"|avstäng|rader|blockera"
        r")\b"
    ),
    "financial_lure": re.compile(
        r"(?i)\b("
        r"(?:transfer|send|wire|deposit)\s+(?:money|funds|\$|€|£|₽)"
        r"|bank account|credit card|payment details|invoice attached"
        r"|банковский счёт|перевод средств"
        r"|cuenta bancaria|transferencia|tarjeta de crédito"
        r"|bankkonto|överföring|kreditkort"
        r")\b"
    ),
    "suspicious_link_ref": re.compile(
        r"(?i)\b("
        r"click (?:here|this|the link|below)|visit this link|open the attachment"
        r"|нажмите (?:здесь|ссылку)|откройте вложение"
        r"|haga clic|abra el enlace|adjunto"
        r"|klicka här|öppna länken|bilaga"
        r")\b"
    ),
    "impersonation_cue": re.compile(
        r"(?i)\b("
        r"(?:this is|i am|speaking on behalf of)\s+(?:the |your )?(?:IT|HR|CEO|director|admin|support|security|bank)"
        r"|от имени|служба безопасности|техническая поддержка"
        r"|en nombre de|soporte técnico|departamento"
        r"|på uppdrag av|IT-avdelningen|säkerhetsavdelningen"
        r")\b"
    ),
}

# Pre-compile the keyword pattern (ASCII-folded versions in the domain)
_PHISHY_KEYWORD_RE = re.compile(
    r"(?i)(?:" + "|".join(re.escape(k) for k in _PHISHY_DOMAIN_KEYWORDS) + r")"
)

# ── URL extraction ───────────────────────────────────────────────────────────
_URL_RE = re.compile(r'https?://[^\s\]\)>\"\']+', re.IGNORECASE)

def _levenshtein_distance(s1: str, s2: str) -> int:
    """Standard DP Levenshtein."""
    if len(s1) < len(s2):
        return _levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1]
        for j, c2 in enumerate(s2):
            curr.append(min(prev[j + 1] + 1, curr[j] + 1, prev[j] + (c1 != c2)))
        prev = curr
    return prev[-1]


def _extract_domain_parts(url: str) -> tuple[str, list[str]]:
    """Return (full_domain, list_of_segments) from a URL."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        host = ""
    # Strip www.
    host = re.sub(r"^www\.", "", host.lower())
    # Split on dots and hyphens for segment analysis
    segments = re.split(r"[.\-]", host)
    return host, segments


def _check_typosquat(domain: str, segments: list[str], threshold: int = 2) -> list[str]:
    """
    Check if any domain segment is within `threshold` Levenshtein distance
    of a known brand but NOT an exact match on the full domain
    (e.g. 'g00gle' in 'g00gle-login.com' → typosquat of 'google').
    """
    hits = []
    for seg in segments:
        if len(seg) < 4:  # skip tiny segments like 'com', 'co', 'it'
            continue
        for brand in _KNOWN_BRANDS:
            if seg == brand:
                # Exact segment match — could be legit subdomain or phishing;
                # flag only if the TLD+1 isn't the brand's real domain
                if not domain.endswith(f"{brand}.com") and not domain.endswith(f"{brand}.org"):
                    hits.append(brand)
                break
            dist = _levenshtein_distance(seg, brand)
            if 0 < dist <= threshold and len(seg) >= len(brand) - 1:
                hits.append(brand)
    return hits


def detect_url_heuristic_flags(sentence: str) -> list[str]:
    """
    Analyse URLs found in a sentence and return heuristic flag names.
    """
    urls = _URL_RE.findall(sentence)
    if not urls:
        return []

    flags = set()
    flags.add("suspicious_link_ref")  # Any raw URL in email body is notable

    for url in urls:
        domain, segments = _extract_domain_parts(url)

        # ── HTTP without TLS ─────────────────────────────────────────────
        if url.lower().startswith("http://"):
            flags.add("suspicious_link_ref")

        # ── Phishy keywords in domain ────────────────────────────────────
        if _PHISHY_KEYWORD_RE.search(domain):
            flags.add("credential_request")

        # ── Typosquatting / brand impersonation ──────────────────────────
        typo_hits = _check_typosquat(domain, segments)
        if typo_hits:
            flags.add("impersonation_cue")

        # ── IP-address URL (http://192.168.1.1/...) ─────────────────────
        if re.match(r"https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", url):
            flags.add("suspicious_link_ref")
            flags.add("impersonation_cue")

        # ── Excessive subdomains (login.secure.microsoft.fakesite.com) ──
        if domain.count(".") >= 3:
            flags.add("impersonation_cue")

    return sorted(flags)

def detect_heuristic_flags(text: str) -> list:
    """Run regex heuristic checks and return list of triggered flag names."""
    flags = []
    for flag_name, pattern in _HEURISTIC_PATTERNS.items():
        if pattern.search(text):
            flags.append(flag_name)
    return flags


def detect_sentence_heuristic_flags(sentence: str) -> list[str]:
    """Run heuristic checks on a single sentence (regex patterns + URL analysis)."""
    flags = set(detect_heuristic_flags(sentence))
    flags.update(detect_url_heuristic_flags(sentence))
    return sorted(flags)


def classify_phishing_persuasion(sentences, classifier):
    """
    Run zero-shot classification with phishing persuasion labels.
    Reuses the SAME tonal_classifier (mDeBERTa) — no new model loaded.
    """
    if classifier is None:
        return [{} for _ in sentences]

    try:
        results = classifier(
            sentences,
            candidate_labels=PHISHING_PERSUASION_LABELS,
            multi_label=False,
        )
        if not isinstance(results, list):
            results = [results]

        output = []
        for r in results:
            scores = dict(zip(r["labels"], r["scores"]))
            output.append(scores)
        return output

    except Exception as e:
        logger.warning(f"Phishing classification failed: {e}")
        return [{} for _ in sentences]


def compute_phishing_scores(persuasion_scores, sentences):
    """
    Compute per-sentence phishing scores combining NLI persuasion + heuristics.
    """
    n = len(sentences)
    neutral_label = "neutral or informational"

    manipulation_strengths = []
    dominant_tactics = []
    for scores in persuasion_scores:
        non_neutral = {k: v for k, v in scores.items() if k != neutral_label}
        if non_neutral:
            best_tactic = max(non_neutral, key=non_neutral.get)
            best_score = non_neutral[best_tactic]
            dominant_tactics.append(best_tactic)
            manipulation_strengths.append(best_score)
        else:
            dominant_tactics.append(neutral_label)
            manipulation_strengths.append(0.0)

    strengths = np.array(manipulation_strengths)
    mean_s = float(np.mean(strengths))
    std_s = float(np.std(strengths))

    results = []
    for i in range(n):
        z = (manipulation_strengths[i] - mean_s) / (std_s + 1e-8) if std_s > 1e-6 else 0.0

        h_flags = detect_sentence_heuristic_flags(sentences[i])

        nli_score = manipulation_strengths[i]
        heuristic_boost = min(len(h_flags) * 0.15, 0.4)
        combined = min(nli_score + heuristic_boost, 1.0)

        has_heuristics = len(h_flags) > 0
        is_manipulative = (
            (combined > 0.5 and has_heuristics and dominant_tactics[i] != neutral_label)
            or (combined > 0.8 and dominant_tactics[i] != neutral_label)
        )

        results.append({
            "persuasion_scores": [
                {"label": k, "score": round(v, 4)}
                for k, v in sorted(
                    persuasion_scores[i].items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            ],
            "dominant_tactic": dominant_tactics[i],
            "persuasion_z_score": round(float(z), 2),
            "heuristic_flags": h_flags,
            "combined_phishing_score": round(combined, 4),
            "is_manipulative": is_manipulative,
        })

    return results