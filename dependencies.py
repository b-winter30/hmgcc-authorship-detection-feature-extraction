"""
Application state: global singletons, ML model loading, startup initialisation.
"""

import os
import pickle
import logging

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from sklearn.manifold import TSNE

from feature_extractor import StylometricExtractor, get_feature_vector
from next_token_prediction import get_ntp_feature_names

logger = logging.getLogger("stylometry_api")

# ── Optional imports ──────────────────────────────────────────────────────────

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# ── Singleton objects ─────────────────────────────────────────────────────────

extractor = StylometricExtractor()
sentence_model = SentenceTransformer("./models/paraphrase-multilingual-MiniLM")

# ── Mutable global state ─────────────────────────────────────────────────────

MODELS_LOADED = False
classifier = None
scaler = None
open_set_detector = None
extractor_with_ntp = None
tonal_classifier = None
feature_weights: dict = {}
author_feature_stats: dict = {}
label_encoder: dict = {}
stylometric_feature_names: list = []
total_feature_names: list = []

# Visualization projection state
projection_model = None
projected_training_data = None
projection_method = None
X_train_stored = None
y_train_stored = None

tokenizer = None
model = None


# ── Model loading ─────────────────────────────────────────────────────────────

def load_models(models_dir: str = "models"):
    """Load trained ML models — pure stylometry version."""
    global MODELS_LOADED, classifier, scaler
    global feature_weights, label_encoder
    global stylometric_feature_names, total_feature_names
    global open_set_detector, author_feature_stats

    try:
        logger.info(f"📦 Loading ML models from {models_dir}/ ...")

        with open(f"{models_dir}/classifier_latest.pkl", "rb") as f:
            classifier = pickle.load(f)
        with open(f"{models_dir}/open_set_detector_latest.pkl", "rb") as f:
            open_set_detector = pickle.load(f)
        with open(f"{models_dir}/scaler_latest.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(f"{models_dir}/feature_weights_latest.pkl", "rb") as f:
            feature_weights = pickle.load(f)
        with open(f"{models_dir}/label_encoder_latest.pkl", "rb") as f:
            label_encoder = pickle.load(f)
        with open(f"{models_dir}/stylometric_feature_names_latest.pkl", "rb") as f:
            stylometric_feature_names = pickle.load(f)
        with open(f"{models_dir}/total_feature_names_latest.pkl", "rb") as f:
            total_feature_names = pickle.load(f)

        try:
            with open(f"{models_dir}/author_feature_stats_latest.pkl", "rb") as f:
                author_feature_stats = pickle.load(f)
            logger.info("   ✅ Loaded author feature statistics")
        except FileNotFoundError:
            logger.warning("   ⚠️  Author feature stats not found")
            author_feature_stats = {}

        MODELS_LOADED = True
        logger.info("✅ All ML models loaded successfully!")
        logger.info(f"   Known contacts: {list(label_encoder.keys())}")
        logger.info(f"   Total features: {len(total_feature_names)} (pure stylometry)")

    except FileNotFoundError as e:
        logger.error(f"❌ Models not found: {e}")
        logger.error("   Train models first using: python train.py")
        MODELS_LOADED = False
    except Exception as e:
        logger.error(f"❌ Error loading models: {e}")
        import traceback
        traceback.print_exc()
        MODELS_LOADED = False


def initialize_extractor_with_ntp():
    """Initialise the feature extractor with NTP support."""
    global extractor_with_ntp, tokenizer, model

    try:
        extractor_with_ntp = StylometricExtractor(
            enable_ntp=True,
            ntp_config={
                "model_name": "./models/qwen3-8b",
                "suspicion_threshold": 7.0,
                "device": "cuda",
                "cache_dir": "./models",
            },
        )
        logger.info("✅ Feature extractor with NTP initialized")

        # Re-export the Qwen model/tokenizer for use in /summarize-analysis
        if extractor_with_ntp.enable_ntp:
            tokenizer = extractor_with_ntp.ntp_detector.tokenizer
            model = extractor_with_ntp.ntp_detector.model
            logger.info("✅ Qwen tokenizer/model re-exported to deps")

        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize NTP: {e}")
        logger.warning("   Falling back to non-NTP extractor")
        extractor_with_ntp = StylometricExtractor(enable_ntp=False)
        return False


def initialize_tonal_classifier():
    """Load zero-shot tonal classifier."""
    global tonal_classifier

    try:
        from transformers import pipeline as hf_pipeline

        device = 0 if torch.cuda.is_available() else -1
        tonal_classifier = hf_pipeline(
            "zero-shot-classification",
            model="./models/mdeberta-xnli-multilingual",
            device=device,
        )
        logger.info("✅ Tonal classifier loaded")
    except Exception as e:
        logger.warning(f"⚠️  Tonal classifier not available: {e}")
        tonal_classifier = None


# ── Projection fitting (for /visualize-prediction) ───────────────────────────

def fit_projection_model(X_train_scaled: np.ndarray, y_train: np.ndarray, method: str = "tsne"):
    """Fit dimensionality-reduction model on training data."""
    global projection_model, projected_training_data, projection_method

    logger.info(f"🎨 Fitting {method.upper()} projection …")

    if method == "umap" and UMAP_AVAILABLE:
        projection_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric="euclidean",
            random_state=42,
        )
        projected_training_data = projection_model.fit_transform(X_train_scaled)
    else:
        if method == "umap" and not UMAP_AVAILABLE:
            logger.warning("   ⚠️  UMAP not available, falling back to t-SNE")
        projection_model = TSNE(
            n_components=2,
            perplexity=min(30, len(X_train_scaled) - 1),
            random_state=42,
            max_iter=1000,
        )
        method = "tsne"
        projected_training_data = projection_model.fit_transform(X_train_scaled)

    projection_method = method
    logger.info(f"✅ {method.upper()} projection fitted — shape {projected_training_data.shape}")
    return projected_training_data