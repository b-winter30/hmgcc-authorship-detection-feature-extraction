"""
FastAPI service for multilingual stylometric feature extraction
WITH AUTHORSHIP PREDICTION - PURE STYLOMETRY VERSION
"""

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
import uuid
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
import time
import re
from sentence_transformers import SentenceTransformer
from langdetect import detect, LangDetectException
import pickle
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("api_activity.log")
    ]
)
logger = logging.getLogger("stylometry_api") 

from feature_extractor import StylometricExtractor, get_feature_vector
from next_token_prediction import get_ntp_feature_names

VISUALIZATION_DIR = "visualizations"
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        start_time = time.time()
        
        logger.info(f"ID:{request_id} | Req: {request.method} {request.url.path}")
        
        try:
            response = await call_next(request)
            process_time = (time.time() - start_time) * 1000
            
            logger.info(
                f"ID:{request_id} | Status: {response.status_code} | Time: {process_time:.2f}ms"
            )
            return response
        except Exception as e:
            logger.error(f"ID:{request_id} | Request failed: {str(e)}")
            raise

# Initialize FastAPI app
app = FastAPI(
    title="Stylometric Feature Extraction & Authorship Detection API",
    description="Extract multilingual stylometric features and predict authorship (Pure Stylometry)",
    version="3.0.0"
)

# CORS configuration
app.add_middleware(LoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:3001",
        "https://your-nextjs-app.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize feature extractor
extractor = StylometricExtractor()

sentence_model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

# Global variables for ML models
MODELS_LOADED = False
classifier = None
scaler = None
open_set_detector = None
extractor_with_ntp = None
# ngram_vectorizer = None  # NO LONGER USED
feature_weights = {}
author_feature_stats = {}
label_encoder = {}
stylometric_feature_names = []
total_feature_names = []
# embedding_model = None  # NO LONGER USED
X_train_stored = None
y_train_stored = None

def load_models(models_dir: str = "models"):
    """Load trained ML models at startup - PURE STYLOMETRY VERSION"""
    global MODELS_LOADED, classifier, scaler
    global feature_weights, label_encoder
    global stylometric_feature_names, total_feature_names
    global open_set_detector, author_feature_stats
    
    try:
        logger.info(f"\n📦 Loading ML models from {models_dir}/...")
        
        # Load core components
        with open(f'{models_dir}/classifier_latest.pkl', 'rb') as f:
            classifier = pickle.load(f)
        
        with open(f'{models_dir}/open_set_detector_latest.pkl', 'rb') as f:
            open_set_detector = pickle.load(f)

        with open(f'{models_dir}/scaler_latest.pkl', 'rb') as f:
            scaler = pickle.load(f)
        
        # N-gram vectorizer NO LONGER NEEDED
        # with open(f'{models_dir}/ngram_vectorizer_latest.pkl', 'rb') as f:
        #     ngram_vectorizer = pickle.load(f)
        
        with open(f'{models_dir}/feature_weights_latest.pkl', 'rb') as f:
            feature_weights = pickle.load(f)
        
        with open(f'{models_dir}/label_encoder_latest.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        
        with open(f'{models_dir}/stylometric_feature_names_latest.pkl', 'rb') as f:
            stylometric_feature_names = pickle.load(f)
        
        with open(f'{models_dir}/total_feature_names_latest.pkl', 'rb') as f:
            total_feature_names = pickle.load(f)
        
        try:
            with open(f'{models_dir}/author_feature_stats_latest.pkl', 'rb') as f:
                author_feature_stats = pickle.load(f)
            logger.info("   ✅ Loaded author feature statistics")
        except FileNotFoundError:
            logger.warning("   ⚠️  Author feature stats not found")
            author_feature_stats = {}

        # NO EMBEDDING MODEL
        
        MODELS_LOADED = True
        logger.info("✅ All ML models loaded successfully!")
        
        reverse_encoder = {v: k for k, v in label_encoder.items()}
        logger.info(f"\n📊 Model info:")
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

def convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types"""
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
    else:
        return obj

def initialize_extractor_with_ntp():
    """Initialize the feature extractor with NTP on startup"""
    global extractor_with_ntp
    
    try:
        extractor_with_ntp = StylometricExtractor(
            enable_ntp=True,
            ntp_config={
                'model_name': 'Qwen/Qwen3-8B',  # Smaller model for API
                'anomaly_threshold': 0.05,
                'device': 'cuda',  # or 'cpu'
                'cache_dir': './model_cache'  # Cache models
            }
        )
        logger.info("✅ Feature extractor with NTP initialized")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to initialize NTP: {e}")
        logger.warning("   Falling back to non-NTP extractor")
        
        # Fallback to regular extractor
        extractor_with_ntp = StylometricExtractor(enable_ntp=False)
        return False

# Load models on startup
@app.on_event("startup")
async def startup_event():
    """Load ML models when FastAPI starts"""
    load_models("models")
    initialize_extractor_with_ntp()

from sklearn.manifold import TSNE

# UMAP is optional
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

# Global variables for visualization
projection_model = None
projected_training_data = None
projection_method = None


class VisualizationRequest(BaseModel):
    content: str
    language: str = "auto"


class VisualizationResponse(BaseModel):
    new_email_position: Dict[str, float]
    predicted_author: str
    confidence: float
    decision: str
    author_clusters: Dict[str, Dict]
    training_samples: List[Dict]
    projection_method: str
    explanation: str

class ProfileCompareRequest(BaseModel):
    content: str = Field(..., min_length=1)
    contact_id: str
    language: str = Field(default="auto")
    ntp_stats: Optional[Dict] = None


class ProfileCompareResponse(BaseModel):
    contact_id: str
    contact_name: str
    profile_sample_count: int
    total_features: int
    unusual_count: int
    unusual_features: List[Dict]
    overall_deviation: float
    match_percentage: float
    summary: str
    ntp_baseline_comparison: Optional[Dict] = None

# Request/Response Models
class EmailInput(BaseModel):
    content: str = Field(..., description="Email content to analyze", min_length=1)
    language: Optional[str] = Field(
        default="auto",
        description="Language code (en, es, ru, sv) or 'auto' for detection"
    )
    include_raw_features: bool = Field(
        default=True,
        description="Include raw feature dictionary in response"
    )

class SentenceOutlierRequest(BaseModel):
    content: str = Field(..., min_length=1)
    min_sentences: int = Field(default=5, ge=3)
    language: str = Field(default="auto")


class SentenceOutlierResponse(BaseModel):
    total_sentences: int
    outlier_count: int
    sentences: List[Dict]  # All sentences with their influence scores
    skipped: bool  # True if email was too short
    skip_reason: Optional[str] = None
    summary: str

class FeatureResponse(BaseModel):
    features: Optional[Dict[str, float]] = None
    feature_vector: List[float]
    feature_names: List[str]
    dimension: int
    detected_language: str
    text_length: int


class PredictRequest(BaseModel):
    content: str = Field(..., min_length=1)
    language: str = Field(default="auto")
    suspected_author_id: Optional[str] = None

class FeatureExplanation(BaseModel):
    """Individual feature explanation"""
    feature: str
    value: float
    author_typical: float
    closeness: float
    importance_for_author: float
    match_score: float

class PredictResponse(BaseModel):
    predicted_author: str
    contact_info: Optional[Dict[str, Optional[str]]] = None
    confidence: float
    all_probabilities: Dict[str, float]
    weighted_score: Optional[float] = None
    is_anomaly: bool
    anomaly_score: float
    top_features: List[Dict]
    explanation: Optional[List[FeatureExplanation]] = None
    method: str
    message: str


class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool


class BatchEmailInput(BaseModel):
    emails: List[EmailInput] = Field(..., max_length=100)


class BatchFeatureResponse(BaseModel):
    results: List[FeatureResponse]
    total_processed: int


class NTPVisualizationRequest(BaseModel):
    content: str
    max_length: int = 512
    language: str = "auto"

# Helper Functions
def detect_language(text: str, fallback: str = "en") -> str:
    """Detect language with fallback"""
    try:
        lang = detect(text)
        lang_mapping = {'en': 'en', 'es': 'es', 'ru': 'ru', 'sv': 'sv'}
        return lang_mapping.get(lang, fallback)
    except (LangDetectException, Exception):
        return fallback

def split_sentences(text: str) -> List[str]:
    """
    Split text into sentences. Handles common email patterns
    like greetings, sign-offs, and line breaks.
    """
    # First split on line breaks to respect email structure
    lines = text.strip().split("\n")

    sentences = []
    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Split on sentence-ending punctuation
        # But keep the punctuation attached to the sentence
        parts = re.split(r'(?<=[.!?])\s+', line)
        for part in parts:
            part = part.strip()
            if len(part) > 2:  # Skip very short fragments
                sentences.append(part)

    return sentences

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        models_loaded=MODELS_LOADED
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check"""
    return HealthResponse(
        status="healthy",
        version="3.0.0",
        models_loaded=MODELS_LOADED
    )

@app.post("/extract-features")
async def extract_features(email: EmailInput):
    """Extract stylometric features ONLY - no NTP"""
    try:
        # Detect language
        if email.language == "auto":
            detected_lang = detect_language(email.content)
        else:
            detected_lang = email.language.lower()
        
        # ✅ Use regular extractor ONLY (no NTP)
        features = extractor.extract_all_features(
            email.content,
            detected_lang
        )
        
        from feature_extractor import get_feature_vector
        feature_names, feature_vector = get_feature_vector(features)
        
        response_data = {
            "features": features if email.include_raw_features else None,
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "dimension": len(feature_vector),
            "detected_language": detected_lang,
            "text_length": len(email.content)
        }
        
        return response_data
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Feature extraction failed: {str(e)}"
        )

def get_contact_info(contact_id: str) -> dict:
    """Return basic contact info"""
    return {
        'id': contact_id,
        'name': f'Contact {contact_id}',
        'email': None
    }
        
@app.post("/predict-author", response_model=PredictResponse)
async def predict_author(request: PredictRequest):
    """
    Predict the author of an email - PURE STYLOMETRY VERSION
    
    Uses ONLY:
    1. Stylometric features (72 features)
    """
    
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded. Train models first."
        )
    
    try:
        if request.suspected_author_id:
            logger.info(f"Suspected author ID: {request.suspected_author_id}")
        else:
            logger.info("No suspected author ID provided")
            
        # Step 1: Detect language
        if request.language == "auto":
            detected_lang = detect_language(request.content)
        else:
            detected_lang = request.language.lower()
        
        logger.info(f"Detected language: {detected_lang}")
        
        # Step 2: Extract ONLY stylometric features
        logger.info("Extracting stylometric features...")
        features = extractor.extract_all_features(request.content, detected_lang)
        feature_names, stylometric_vector = get_feature_vector(features)
        stylometric_vector = np.array(stylometric_vector)
        
        logger.info(f"Extracted {len(stylometric_vector)} features")
        
        # NO N-GRAMS
        # NO EMBEDDINGS
        
        # Step 3: Use only stylometric features
        combined_vector = stylometric_vector
        
        logger.info(f"Combined vector shape: {combined_vector.shape}")
        logger.info(f"Total feature names: {len(total_feature_names)}")
        
        # Validate dimension
        if len(combined_vector) != len(total_feature_names):
            error_msg = f"Feature dimension mismatch: got {len(combined_vector)} features, expected {len(total_feature_names)}"
            logger.error(error_msg)
            raise HTTPException(
                status_code=500,
                detail=error_msg
            )
        
        # Step 4: Normalize
        logger.info("Normalizing features...")
        combined_scaled = scaler.transform([combined_vector])
        
        # Step 5: Open-set detection
        logger.info("Running open-set detection...")
        open_set_result = open_set_detector.predict(
            combined_scaled[0],
            return_distances=False
        )
        
        logger.info(f"Open-set result: {open_set_result['decision']}")

        if open_set_result['decision'] == 'rejected':
            # Unknown author detected!
            logger.info("Author rejected as UNKNOWN")
            return PredictResponse(
                predicted_author="UNKNOWN",
                confidence=0.0,
                all_probabilities={},
                weighted_score=None,
                is_anomaly=False,
                anomaly_score=0.0,
                top_features=[],
                explanation=None,
                method="open_set_rejection",
                message=f"❌ Unknown author: {open_set_result['reason']}"
            )

        # Step 6: Predict
        logger.info("Running Random Forest prediction...")
        prediction_idx = classifier.predict(combined_scaled)[0]
        probabilities = classifier.predict_proba(combined_scaled)[0]
        
        # Get author name
        reverse_encoder = {v: k for k, v in label_encoder.items()}
        predicted_author = reverse_encoder[prediction_idx]
        confidence = float(max(probabilities))
        
        logger.info(f"Predicted author: {predicted_author}, confidence: {confidence}")
        
        contact_info = get_contact_info(predicted_author)
        
        # All probabilities
        all_probs = {
            reverse_encoder[i]: float(prob)
            for i, prob in enumerate(probabilities)
        }
        
        # Step 7: Calculate weighted score (if suspected author provided)
        weighted_score = None
        if request.suspected_author_id and request.suspected_author_id in feature_weights:
            logger.info(f"Calculating weighted score for suspected author {request.suspected_author_id}")
            weights = feature_weights[request.suspected_author_id]
            weighted_diff = np.abs(combined_vector * weights)
            weighted_score = float(1.0 / (1.0 + np.mean(weighted_diff)))
        
        # Step 8: Use Random Forest confidence as anomaly indicator
        is_anomaly = confidence < 0.5
        anomaly_score = float(1.0 - confidence)
        
        # Step 9: Get top contributing features
        logger.info("Computing feature contributions...")
        feature_importance = classifier.feature_importances_
        feature_contributions = [
            {
                'feature': total_feature_names[i],
                'value': float(combined_vector[i]),
                'importance': float(feature_importance[i]),
                'contribution': float(combined_vector[i] * feature_importance[i])
            }
            for i in range(len(combined_vector))
        ]
        
        feature_contributions.sort(key=lambda x: abs(x['contribution']), reverse=True)
        top_features = feature_contributions[:10]
        
        # Step 10: Generate message
        if is_anomaly:
            message = f"⚠️ Unusual writing pattern - low confidence ({confidence:.0%})"
        elif confidence > 0.9:
            message = f"✅ High confidence: Contact {predicted_author}"
        elif confidence > 0.7:
            message = f"✅ Likely: Contact {predicted_author}"
        else:
            message = f"⚠️ Low-medium confidence: Contact {predicted_author}"
        
        logger.info(f"Prediction complete: {message}")
        
        return PredictResponse(
            predicted_author=predicted_author,
            contact_info=contact_info,
            confidence=confidence,
            all_probabilities=all_probs,
            weighted_score=weighted_score,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            top_features=top_features,
            method="pure_stylometry",
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # Log the full error with traceback
        logger.error(f"ERROR in predict_author: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/sentence-outlier-detection", response_model=SentenceOutlierResponse)
async def sentence_outlier_detection(request: SentenceOutlierRequest):
    """
    Detect semantically anomalous sentences using leave-one-out embedding analysis.

    For each sentence, we compute how much removing it shifts the overall
    email embedding. Sentences that cause a large shift when removed are
    semantically distant from the rest of the email — potential injections.
    """
    try:
        sentences = split_sentences(request.content)

        # Check minimum sentence count
        if len(sentences) < request.min_sentences:
            return SentenceOutlierResponse(
                total_sentences=len(sentences),
                outlier_count=0,
                sentences=[],
                skipped=True,
                skip_reason=f"Email has {len(sentences)} sentences, minimum is {request.min_sentences}",
                summary=f"Skipped — email too short ({len(sentences)} sentences, need {request.min_sentences})",
            )

        # Encode the full email as a single embedding
        full_text = " ".join(sentences)
        full_embedding = sentence_model.encode(full_text, normalize_embeddings=True)

        # For each sentence, encode the email WITHOUT that sentence
        results = []
        for i, sentence in enumerate(sentences):
            remaining = sentences[:i] + sentences[i + 1:]
            remaining_text = " ".join(remaining)
            remaining_embedding = sentence_model.encode(remaining_text, normalize_embeddings=True)

            # Also encode the sentence alone for bidirectional comparison
            sentence_embedding = sentence_model.encode(sentence, normalize_embeddings=True)

            # Euclidean distance: how much does removing this sentence shift the embedding?
            influence_distance = float(np.linalg.norm(full_embedding - remaining_embedding))

            # Cosine distance between the sentence alone and the rest of the email
            cos_sim = float(np.dot(sentence_embedding, remaining_embedding))
            context_distance = 1.0 - cos_sim  # Convert similarity to distance

            # Combined score: average of influence and context distance
            # Influence = how much it shifts the whole; context = how different it is from the rest
            combined_score = (influence_distance + context_distance) / 2.0

            results.append({
                "index": i,
                "sentence": sentence,
                "influence_distance": round(influence_distance, 4),
                "context_distance": round(context_distance, 4),
                "combined_score": round(combined_score, 4),
            })

        # Compute outlier threshold using IQR on combined scores
        scores = [r["combined_score"] for r in results]
        q1 = float(np.percentile(scores, 25))
        q3 = float(np.percentile(scores, 75))
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr

        # Mark outliers and compute z-scores relative to the email's own sentences
        mean_score = float(np.mean(scores))
        std_score = float(np.std(scores))

        for r in results:
            z = (r["combined_score"] - mean_score) / (std_score + 1e-8)
            r["z_score"] = round(float(z), 2)
            r["is_outlier"] = bool(r["combined_score"] > outlier_threshold)

        # Sort by combined score descending (most anomalous first)
        results.sort(key=lambda x: x["combined_score"], reverse=True)

        outlier_count = sum(1 for r in results if r["is_outlier"])

        # Generate summary
        if outlier_count == 0:
            summary = "All sentences are semantically consistent with each other."
        elif outlier_count == 1:
            top = results[0]
            preview = top["sentence"][:60] + "..." if len(top["sentence"]) > 60 else top["sentence"]
            summary = f"1 sentence appears semantically inconsistent: \"{preview}\""
        else:
            summary = f"{outlier_count} sentences appear semantically inconsistent with the rest of the email."

        logger.info(f"Sentence outlier detection: {len(sentences)} sentences, "
                     f"{outlier_count} outliers (threshold={outlier_threshold:.4f})")

        return SentenceOutlierResponse(
            total_sentences=len(sentences),
            outlier_count=outlier_count,
            sentences=results,
            skipped=False,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Sentence outlier detection error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Sentence outlier detection failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about loaded models"""
    
    info = {
        "loaded": MODELS_LOADED,
        "ntp_available": extractor_with_ntp.enable_ntp if extractor_with_ntp else False,
    }
    
    if MODELS_LOADED:
        info.update({
            "known_contacts": list(label_encoder.keys()),
            "n_contacts": len(label_encoder),
            "n_stylometric_features": len(stylometric_feature_names),
            "n_total_features": len(total_feature_names),
            "feature_type": "pure_stylometry_with_ntp" if extractor_with_ntp.enable_ntp else "pure_stylometry"
        })
    
    if extractor_with_ntp and extractor_with_ntp.enable_ntp:
        info.update({
            "ntp_model": extractor_with_ntp.ntp_detector.model_name,
            "ntp_threshold": extractor_with_ntp.ntp_detector.anomaly_threshold,
            "ntp_feature_names": get_ntp_feature_names()
        })
    
    return info

@app.post("/build-contact-profile/{contact_id}")
async def build_contact_profile(contact_id: int):
    """Compute mean/std of all features across a contact's training emails"""
    # 1. Fetch all training emails for this contact from DB
    # 2. Extract features for each
    # 3. Compute mean + std per feature
    # 4. Store as JSON in contacts.stylometric_profile
    
    all_vectors = []
    for email in training_emails:
        features = extractor.extract_all_features(email.content, email.language)
        _, vector = get_feature_vector(features)
        all_vectors.append(vector)
    
    all_vectors = np.array(all_vectors)
    profile = {}
    for i, name in enumerate(feature_names):
        profile[name] = {
            "mean": float(np.mean(all_vectors[:, i])),
            "std": float(np.std(all_vectors[:, i])),
        }
    
    # Save to DB
    return {"contact_id": contact_id, "features_profiled": len(profile)}

@app.post("/compare-to-profile", response_model=ProfileCompareResponse)
async def compare_to_profile(request: ProfileCompareRequest):
    """
    Compare a new email's features against a contact's stored stylometric profile.
    Also compares NTP stats against the contact's NTP baseline if both are available.
    """
    import psycopg2
    import json

    try:
        # Step 1: Extract features from the new email
        if request.language == "auto":
            detected_lang = detect_language(request.content)
        else:
            detected_lang = request.language.lower()

        features = extractor.extract_all_features(request.content, detected_lang)
        feature_names, feature_vector = get_feature_vector(features)

        # Step 2: Load contact profile from database
        db_host = os.environ.get("DB_HOST", "postgres")
        db_port = os.environ.get("DB_PORT", "5432")
        db_name = os.environ.get("DB_NAME", "authorship-detection")
        db_user = os.environ.get("DB_USER", "postgres")
        db_password = os.environ.get("DB_PASSWORD", "postgres")

        conn = psycopg2.connect(
            host=db_host,
            port=db_port,
            dbname=db_name,
            user=db_user,
            password=db_password
        )
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name, stylometric_profile, profile_sample_count
            FROM contacts
            WHERE contact_id = %s
        """, (int(request.contact_id),))

        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail=f"Contact {request.contact_id} not found")

        contact_name = row[0]
        profile_json = row[1]
        sample_count = row[2] or 0

        if not profile_json:
            raise HTTPException(
                status_code=404,
                detail=f"No stylometric profile for {contact_name}. Run training first."
            )

        raw_profile = profile_json if isinstance(profile_json, dict) else json.loads(profile_json)

        # Support both old flat format and new nested format
        stylometric_profile = raw_profile.get("stylometric", raw_profile)
        ntp_baseline = raw_profile.get("ntp_baseline", None)

        # Step 3: Compute z-scores for each stylometric feature
        deviations = []
        for i, name in enumerate(feature_names):
            if name not in stylometric_profile:
                continue

            new_val = feature_vector[i]
            prof = stylometric_profile[name]
            prof_mean = prof["mean"]
            prof_std = prof["std"]

            z_score = (new_val - prof_mean) / (prof_std + 1e-8)

            deviations.append({
                "feature": name,
                "new_value": float(new_val),
                "profile_mean": float(prof_mean),
                "profile_std": float(prof_std),
                "z_score": float(z_score),
                "is_unusual": bool(abs(z_score) > 2.0),
            })

        # Step 4: Sort and summarise stylometric deviations
        unusual = [d for d in deviations if d["is_unusual"]]
        unusual.sort(key=lambda x: abs(x["z_score"]), reverse=True)

        all_z = [abs(d["z_score"]) for d in deviations]
        overall_deviation = float(np.mean(all_z)) if all_z else 0.0

        normal_count = len(deviations) - len(unusual)
        match_pct = (normal_count / len(deviations) * 100) if deviations else 0.0

        # Adjust match percentage for extreme outliers
        max_z = max(abs(d["z_score"]) for d in unusual) if unusual else 0
        extreme_outlier = max_z > 100
        notable_outlier = max_z > 10

        if extreme_outlier:
            match_pct = min(match_pct, 50.0)
        elif notable_outlier:
            match_pct = min(match_pct, 70.0)

        # Step 5: NTP baseline comparison
        ntp_comparison = None

        if ntp_baseline and request.ntp_stats:
            # Full comparison: we have both baseline and new NTP results
            ntp_stats = request.ntp_stats
            ntp_comparison = {
                "n_baseline_samples": ntp_baseline.get("n_samples", 0),
                "baseline_only": False,
            }

            for metric in ["mean_anomaly_score", "anomaly_ratio", "mean_probability", "median_probability"]:
                if metric in ntp_baseline and metric in ntp_stats:
                    baseline = ntp_baseline[metric]
                    new_val = float(ntp_stats[metric])
                    b_mean = float(baseline["mean"])
                    b_std = float(baseline["std"])

                    z = (new_val - b_mean) / (b_std + 1e-8)

                    ntp_comparison[metric] = {
                        "new_value": new_val,
                        "baseline_mean": b_mean,
                        "baseline_std": b_std,
                        "z_score": float(z),
                        "is_unusual": bool(abs(z) > 2.0),
                    }

        elif ntp_baseline:
            # Baseline exists but no NTP stats yet — return baseline for frontend to use later
            ntp_comparison = {
                "n_baseline_samples": ntp_baseline.get("n_samples", 0),
                "baseline_only": True,
                "mean_anomaly_score": {
                    "baseline_mean": float(ntp_baseline["mean_anomaly_score"]["mean"]),
                    "baseline_std": float(ntp_baseline["mean_anomaly_score"]["std"]),
                },
                "anomaly_ratio": {
                    "baseline_mean": float(ntp_baseline["anomaly_ratio"]["mean"]),
                    "baseline_std": float(ntp_baseline["anomaly_ratio"]["std"]),
                },
                "mean_probability": {
                    "baseline_mean": float(ntp_baseline["mean_probability"]["mean"]),
                    "baseline_std": float(ntp_baseline["mean_probability"]["std"]),
                },
                "median_probability": {
                    "baseline_mean": float(ntp_baseline["median_probability"]["mean"]),
                    "baseline_std": float(ntp_baseline["median_probability"]["std"]),
                },
            }

        # Step 6: Generate summary incorporating NTP if available
        ntp_flag = ""
        if ntp_comparison and not ntp_comparison.get("baseline_only"):
            mas = ntp_comparison.get("mean_anomaly_score", {})
            anomaly_z = mas.get("z_score", 0)
            if abs(anomaly_z) > 2.0:
                direction = "more predictable than normal" if anomaly_z < 0 else "more unpredictable than normal"
                ntp_flag = f" NTP anomaly is {abs(anomaly_z):.1f}\u03c3 {direction} for this contact."

        if extreme_outlier:
            summary = f"Suspicious — most features match {contact_name} but extreme deviations detected (max z={max_z:.0f}).{ntp_flag}"
        elif notable_outlier:
            summary = f"Mixed signals — broadly consistent with {contact_name} but notable outliers present.{ntp_flag}"
        elif match_pct >= 90:
            summary = f"Strong match — writing style is consistent with {contact_name}'s profile.{ntp_flag}"
        elif match_pct >= 70:
            summary = f"Moderate match — mostly consistent with {contact_name}, some deviations.{ntp_flag}"
        elif match_pct >= 50:
            summary = f"Weak match — significant deviations from {contact_name}'s typical style.{ntp_flag}"
        else:
            summary = f"Poor match — writing style differs substantially from {contact_name}'s profile.{ntp_flag}"

        logger.info(f"Profile comparison for contact {request.contact_id} ({contact_name}): "
                     f"{len(unusual)}/{len(deviations)} unusual features, "
                     f"match={match_pct:.1f}%"
                     f"{', NTP z-scores included' if ntp_comparison and not ntp_comparison.get('baseline_only') else ''}")

        return ProfileCompareResponse(
            contact_id=request.contact_id,
            contact_name=contact_name,
            profile_sample_count=sample_count,
            total_features=len(deviations),
            unusual_count=len(unusual),
            unusual_features=unusual[:20],
            overall_deviation=overall_deviation,
            match_percentage=round(match_pct, 1),
            summary=summary,
            ntp_baseline_comparison=ntp_comparison,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Profile comparison error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Profile comparison failed: {str(e)}")

@app.post("/extract-features/batch", response_model=BatchFeatureResponse)
async def extract_features_batch(batch: BatchEmailInput):
    """Extract features from multiple emails"""
    try:
        results = []
        
        for email_input in batch.emails:
            if email_input.language == "auto":
                detected_lang = detect_language(email_input.content)
            else:
                detected_lang = email_input.language.lower()
            
            features = extractor.extract_all_features(email_input.content, detected_lang)
            feature_names, feature_vector = get_feature_vector(features)
            
            response_data = {
                "feature_vector": feature_vector,
                "feature_names": feature_names,
                "dimension": len(feature_vector),
                "detected_language": detected_lang,
                "text_length": len(email_input.content)
            }
            
            if email_input.include_raw_features:
                response_data["features"] = features
            
            results.append(FeatureResponse(**response_data))
        
        return BatchFeatureResponse(
            results=results,
            total_processed=len(results)
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}"
        )

@app.post("/visualize-ntp-anomalies")
async def visualize_ntp_anomalies(request: NTPVisualizationRequest):
    """
    Visualize next-token prediction anomalies with comprehensive charts
    """
    if not extractor_with_ntp.enable_ntp:
        raise HTTPException(
            status_code=503,
            detail="NTP not available on this server"
        )
    
    try:
        # Extract features with detailed results
        features = extractor_with_ntp.ntp_detector.extract_anomaly_features(
            request.content,
            max_length=request.max_length,
            return_detailed=True
        )
        
        # Get detailed results
        detailed_results = features.get('_detailed_results', [])
        
        # Remove internal field
        if '_detailed_results' in features:
            del features['_detailed_results']
        
        # Calculate statistics
        probabilities = [float(r['probability']) for r in detailed_results if 'probability' in r]
        mean_probability = float(np.mean(probabilities)) if probabilities else 0.0
        median_probability = float(np.median(probabilities)) if probabilities else 0.0
        min_probability = float(np.min(probabilities)) if probabilities else 0.0
        max_probability = float(np.max(probabilities)) if probabilities else 0.0
        
        # Prepare anomalous tokens
        anomalous_tokens = [
            {
                'position': int(r['position']),
                'token': r['token'],
                'probability': float(r['probability']),
                'rank': int(r['rank']),
                'anomaly_score': float(r['anomaly_score']),
                'is_capitalized': bool(r.get('is_capitalized', False))
            }
            for r in detailed_results
            if r.get('is_anomaly', False)
        ]
        
        # ✅ Generate visualization
        visualization_path = generate_ntp_visualization(
            detailed_results,
            anomalous_tokens,
            request.content[:100]  # First 100 chars as identifier
        )
        
        # Build response
        response_data = {
            'aggregate_features': convert_numpy_types(features),
            'total_tokens': int(len(detailed_results)),
            'anomalous_tokens': anomalous_tokens,
            'anomaly_count': int(len(anomalous_tokens)),
            'anomaly_ratio': float(len(anomalous_tokens) / len(detailed_results) if detailed_results else 0),
            'mean_probability': mean_probability,
            'median_probability': median_probability,
            'min_probability': min_probability,
            'max_probability': max_probability,
            'mean_anomaly_score': float(np.mean([r['anomaly_score'] for r in detailed_results])) if detailed_results else 0.0,
            'mean_rank': float(np.mean([r['rank'] for r in detailed_results])) if detailed_results else 0.0,
            'visualization_path': visualization_path  # ✅ Add this
        }
        
        return response_data
        
    except Exception as e:
        import traceback
        logger.error(f"NTP visualization error: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"NTP visualization failed: {str(e)}"
        )

def generate_ntp_visualization(detailed_results, anomalous_tokens, content_preview):
    """
    Generate comprehensive matplotlib visualization for NTP analysis
    WITH CONTEXT WEIGHTING VISUALIZATION AND EMOJI SUPPORT
    """
    try:
        # Configure matplotlib for emoji support
        import matplotlib.font_manager as fm
        
        # Try to find a font that supports emoji
        emoji_fonts = [
            'Apple Color Emoji',      # macOS
            'Segoe UI Emoji',         # Windows
            'Noto Color Emoji',       # Linux
            'Symbola',                # Linux fallback
            'DejaVu Sans'             # Universal fallback
        ]
        
        available_fonts = [f.name for f in fm.fontManager.ttflist]
        emoji_font = None
        
        for font in emoji_fonts:
            if font in available_fonts:
                emoji_font = font
                logger.info(f"Using font for emoji support: {font}")
                break
        
        if emoji_font:
            plt.rcParams['font.family'] = emoji_font
        
        # Extract data
        positions = []
        tokens = []
        token_display = []  # For display (handle emojis specially)
        probabilities = []
        ranks = []
        anomaly_scores = []
        weighted_anomaly_scores = []
        context_weights = []
        is_anomaly = []
        contains_emoji = []
        
        for result in detailed_results:
            pos = result.get('position', 0)
            token = result.get('token', '').strip()
            
            if not token or len(token) < 1:
                continue
            
            positions.append(int(pos))
            tokens.append(token)
            
            # Create display version - escape non-printable but keep emoji
            try:
                # Test if token is displayable
                display_token = token
                if not self._is_displayable(token):
                    display_token = repr(token)
                token_display.append(display_token)
            except:
                token_display.append(repr(token))
            
            probabilities.append(float(result.get('probability', 0)))
            ranks.append(int(result.get('rank', 0)))
            anomaly_scores.append(float(result.get('anomaly_score', 0)))
            weighted_anomaly_scores.append(float(result.get('weighted_anomaly_score', 0)))
            context_weights.append(float(result.get('context_weight', 0)))
            is_anomaly.append(bool(result.get('is_anomaly', False)))
            contains_emoji.append(bool(result.get('contains_emoji', False)))
        
        # Create figure with 4 subplots
        fig, axes = plt.subplots(4, 1, figsize=(16, 16))
        
        # ============ Plot 1: Context Weight Progression ============
        ax1 = axes[0]
        
        ax1.plot(positions, context_weights, 'b-', linewidth=2.5, label='Context Weight', zorder=3)
        ax1.fill_between(positions, 0, context_weights, alpha=0.2, color='blue')
        
        # Mark positions with emojis
        emoji_positions = [positions[i] for i in range(len(positions)) if contains_emoji[i]]
        emoji_weights = [context_weights[i] for i in range(len(context_weights)) if contains_emoji[i]]
        
        if emoji_positions:
            ax1.scatter(emoji_positions, emoji_weights, color='purple', s=100, 
                       marker='*', zorder=10, label='Emoji Tokens', edgecolors='black', linewidth=1)
        
        # Add threshold lines for context levels
        ax1.axhline(y=0.3, color='red', linestyle='--', alpha=0.4, linewidth=1.5, label='Minimal→Partial (0.3)')
        ax1.axhline(y=0.6, color='orange', linestyle='--', alpha=0.4, linewidth=1.5, label='Partial→Good (0.6)')
        ax1.axhline(y=0.9, color='green', linestyle='--', alpha=0.4, linewidth=1.5, label='Good→Full (0.9)')
        
        # Shade background by context level
        ax1.axhspan(0.0, 0.3, alpha=0.1, color='red', label='Minimal Context')
        ax1.axhspan(0.3, 0.6, alpha=0.1, color='orange', label='Partial Context')
        ax1.axhspan(0.6, 0.9, alpha=0.1, color='yellow', label='Good Context')
        ax1.axhspan(0.9, 1.0, alpha=0.1, color='green', label='Full Context')
        
        ax1.set_xlabel('Token Position', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Context Weight', fontsize=12, fontweight='bold')
        ax1.set_title('Context Weight Progression Over Email', fontsize=14, fontweight='bold')
        ax1.legend(loc='lower right', fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim([0, 1.05])
        
        # ============ Plot 2: Raw vs Weighted Anomaly Scores ============
        ax2 = axes[1]
        
        # Plot raw anomaly scores (light)
        ax2.plot(positions, anomaly_scores, 'r--', alpha=0.4, linewidth=1.5, label='Raw Anomaly Score')
        
        # Plot weighted anomaly scores (bold)
        ax2.plot(positions, weighted_anomaly_scores, 'b-', linewidth=2.5, label='Context-Weighted Anomaly Score')
        
        # Mark emoji positions
        emoji_weighted_scores = [weighted_anomaly_scores[i] for i in range(len(weighted_anomaly_scores)) if contains_emoji[i]]
        
        if emoji_positions:
            ax2.scatter(emoji_positions, emoji_weighted_scores, color='purple', s=100, 
                       marker='*', zorder=10, alpha=0.8, edgecolors='black', linewidth=1)
        
        # Add high suspicion threshold
        ax2.axhline(y=0.95, color='red', linestyle=':', alpha=0.6, linewidth=2, 
                   label='High Suspicion Threshold (weighted > 0.95)')
        
        # Highlight areas where weighting made a difference
        for i in range(len(positions)):
            raw = anomaly_scores[i]
            weighted = weighted_anomaly_scores[i]
            
            # If raw was high but weighted is low (early in email)
            if raw > 0.8 and weighted < 0.3:
                ax2.scatter(positions[i], raw, color='orange', s=50, alpha=0.6, zorder=5)
        
        ax2.set_xlabel('Token Position', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Anomaly Score', fontsize=12, fontweight='bold')
        ax2.set_title('Raw vs Context-Weighted Anomaly Scores', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([0, 1.05])
        
        # Annotate high weighted anomalies (use display tokens)
        high_weighted = [(positions[i], weighted_anomaly_scores[i], token_display[i], contains_emoji[i]) 
                        for i in range(len(positions)) 
                        if weighted_anomaly_scores[i] > 0.95]
        
        if high_weighted:
            high_weighted.sort(key=lambda x: x[1], reverse=True)
            for pos, score, token, is_emoji in high_weighted[:5]:  # Top 5
                # Use special formatting for emoji
                display_text = token if not is_emoji else f"😀 {token}"
                
                ax2.annotate(display_text, 
                           xy=(pos, score), 
                           xytext=(0, 10),
                           textcoords='offset points',
                           ha='center',
                           fontsize=9,
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='plum' if is_emoji else 'yellow', 
                                   alpha=0.8),
                           arrowprops=dict(arrowstyle='->', color='red', lw=1.5))
        
        # ============ Plot 3: Token Probabilities (colored by context weight) ============
        ax3 = axes[2]
        
        # Separate by context quality
        high_context_mask = [context_weights[i] >= 0.6 for i in range(len(positions))]
        low_context_mask = [context_weights[i] < 0.6 for i in range(len(positions))]
        
        # Plot low-context tokens (less reliable)
        low_positions = [positions[i] for i in range(len(positions)) if low_context_mask[i]]
        low_probs = [probabilities[i] for i in range(len(probabilities)) if low_context_mask[i]]
        
        ax3.scatter(low_positions, low_probs, alpha=0.3, s=20, c='gray', label='Low Context (weight < 0.6)')
        
        # Plot high-context tokens (more reliable) - colored by weighted anomaly score
        high_positions = [positions[i] for i in range(len(positions)) if high_context_mask[i]]
        high_probs = [probabilities[i] for i in range(len(probabilities)) if high_context_mask[i]]
        high_weighted_scores = [weighted_anomaly_scores[i] for i in range(len(weighted_anomaly_scores)) if high_context_mask[i]]
        
        scatter = ax3.scatter(high_positions, high_probs, alpha=0.7, s=50, 
                            c=high_weighted_scores, cmap='RdYlGn_r', 
                            vmin=0, vmax=1, edgecolors='black', linewidth=0.5,
                            label='High Context (weight ≥ 0.6)')
        
        # Mark anomalies
        anomaly_positions = [positions[i] for i in range(len(positions)) if is_anomaly[i]]
        anomaly_probs = [probabilities[i] for i in range(len(probabilities)) if is_anomaly[i]]
        
        ax3.scatter(anomaly_positions, anomaly_probs, alpha=0.9, s=100, c='red', 
                   marker='x', linewidths=2.5, label='Detected Anomalies', zorder=10)
        
        # Mark emoji tokens specially
        emoji_probs = [probabilities[i] for i in range(len(probabilities)) if contains_emoji[i]]
        
        if emoji_positions:
            ax3.scatter(emoji_positions, emoji_probs, alpha=0.8, s=150, 
                       facecolors='none', edgecolors='purple', linewidths=2.5,
                       marker='o', label='Emoji Tokens', zorder=9)
        
        ax3.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, linewidth=1.5,
                   label='Base Anomaly Threshold (0.05)')
        
        ax3.set_xlabel('Token Position', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Token Probability (log scale)', fontsize=12, fontweight='bold')
        ax3.set_title('Token Probabilities (High-Context tokens colored by weighted suspicion)', 
                     fontsize=14, fontweight='bold')
        ax3.legend(loc='lower right', fontsize=9)
        ax3.grid(True, alpha=0.3, which='both')
        ax3.set_yscale('log')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax3)
        cbar.set_label('Weighted Anomaly Score', fontsize=10)
        
        # ============ Plot 4: Suspicious Tokens Summary (Bar Chart) ============
        ax4 = axes[3]
        
        # Get top suspicious tokens (by weighted score)
        suspicious_data = [
            (positions[i], token_display[i], weighted_anomaly_scores[i], 
             context_weights[i], probabilities[i], contains_emoji[i])
            for i in range(len(positions))
            if weighted_anomaly_scores[i] > 0.5  # Show tokens with notable weighted suspicion
        ]
        
        suspicious_data.sort(key=lambda x: x[2], reverse=True)
        top_suspicious = suspicious_data[:15]  # Top 15
        
        if top_suspicious:
            sus_tokens = [f"{t[1]} @{t[0]}" for t in top_suspicious]
            sus_scores = [t[2] for t in top_suspicious]
            sus_weights = [t[3] for t in top_suspicious]
            sus_probs = [t[4] for t in top_suspicious]
            sus_emoji = [t[5] for t in top_suspicious]
            
            y_pos = np.arange(len(sus_tokens))
            
            # Color bars by context weight and emoji status
            colors = []
            for weight, is_emoji in zip(sus_weights, sus_emoji):
                if is_emoji:
                    colors.append('plum')  # Purple for emojis
                elif weight >= 0.9:
                    colors.append('darkred')
                elif weight >= 0.6:
                    colors.append('red')
                elif weight >= 0.3:
                    colors.append('orange')
                else:
                    colors.append('yellow')
            
            bars = ax4.barh(y_pos, sus_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1)
            
            # Add vertical line for high suspicion threshold
            ax4.axvline(x=0.95, color='red', linestyle='--', linewidth=2, alpha=0.6,
                       label='High Suspicion (0.95)')
            
            ax4.set_yticks(y_pos)
            ax4.set_yticklabels(sus_tokens, fontsize=8)
            ax4.set_xlabel('Weighted Anomaly Score', fontsize=12, fontweight='bold')
            ax4.set_title('Top Suspicious Tokens (Ranked by Context-Weighted Score)', 
                         fontsize=14, fontweight='bold')
            ax4.set_xlim([0, 1.05])
            ax4.grid(True, alpha=0.3, axis='x')
            
            # Add score labels
            for i, (bar, score, weight, prob, is_emoji) in enumerate(zip(bars, sus_scores, sus_weights, sus_probs, sus_emoji)):
                width = bar.get_width()
                emoji_marker = "😀" if is_emoji else ""
                ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                        f'{emoji_marker}{score:.2f}\n(w:{weight:.2f}, p:{prob:.4f})',
                        ha='left', va='center', fontsize=7, fontweight='bold')
            
            # Create custom legend for context levels
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='plum', alpha=0.7, edgecolor='black', label='Emoji Token'),
                Patch(facecolor='darkred', alpha=0.7, edgecolor='black', label='Full Context (≥0.9)'),
                Patch(facecolor='red', alpha=0.7, edgecolor='black', label='Good Context (0.6-0.9)'),
                Patch(facecolor='orange', alpha=0.7, edgecolor='black', label='Partial Context (0.3-0.6)'),
                Patch(facecolor='yellow', alpha=0.7, edgecolor='black', label='Minimal Context (<0.3)')
            ]
            ax4.legend(handles=legend_elements, loc='lower right', fontsize=8)
        else:
            ax4.text(0.5, 0.5, '✅ No Suspicious Tokens Detected', 
                    ha='center', va='center', fontsize=14, fontweight='bold',
                    transform=ax4.transAxes)
            ax4.set_xlim([0, 1])
            ax4.set_ylim([0, 1])
        
        plt.tight_layout()
        
        # Save figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_preview = "".join(c for c in content_preview if c.isalnum() or c in (' ', '-', '_'))[:50]
        filename = f"ntp_context_weighted_{timestamp}_{safe_preview}.png"
        filepath = os.path.join(VISUALIZATION_DIR, filename)
        
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        logger.info(f"✅ Saved NTP context-weighted visualization: {filepath}")
        
        return filename
        
    except Exception as e:
        logger.error(f"Failed to generate NTP visualization: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None

def _is_displayable(self, text: str) -> bool:
    """Check if text is displayable (printable or emoji)"""
    if not text:
        return False
    
    # Check if it's printable ASCII or contains emoji
    for char in text:
        if char.isprintable() or self._is_emoji_char(char):
            continue
        else:
            return False
    return True

def _is_emoji_char(self, char: str) -> bool:
    """Check if a character is an emoji"""
    code_point = ord(char)
    emoji_ranges = [
        (0x1F600, 0x1F64F), (0x1F300, 0x1F5FF), (0x1F680, 0x1F6FF),
        (0x1F1E0, 0x1F1FF), (0x2600, 0x26FF), (0x2700, 0x27BF),
        (0xFE00, 0xFE0F), (0x1F900, 0x1F9FF), (0x1FA00, 0x1FA6F),
        (0x1FA70, 0x1FAFF),
    ]
    
    for start, end in emoji_ranges:
        if start <= code_point <= end:
            return True
    return False

@app.get("/ntp-visualization/{filename}")
async def get_ntp_visualization(filename: str):
    """Serve NTP visualization images"""
    from fastapi.responses import FileResponse
    
    filepath = os.path.join(VISUALIZATION_DIR, filename)
    
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Visualization not found")
    
    return FileResponse(filepath, media_type="image/png")

@app.post("/visualize-prediction", response_model=VisualizationResponse)
async def visualize_prediction(request: VisualizationRequest):
    """
    Predict author and return 2D visualization data - PURE STYLOMETRY VERSION
    """
    
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded"
        )
    
    if projection_model is None:
        raise HTTPException(
            status_code=503,
            detail="Projection model not initialized. Call /initialize-visualization first"
        )
    
    try:
        # Step 1: Extract features
        if request.language == "auto":
            detected_lang = detect_language(request.content)
        else:
            detected_lang = request.language.lower()
        
        features = extractor.extract_all_features(request.content, detected_lang)
        feature_names, stylometric_vector = get_feature_vector(features)
        stylometric_vector = np.array(stylometric_vector)
        
        # NO N-GRAMS
        # NO EMBEDDINGS
        
        combined_vector = stylometric_vector
        
        # Step 2: Normalize
        combined_scaled = scaler.transform([combined_vector])
        
        # Step 3: Open-set detection
        open_set_result = open_set_detector.predict(
            combined_scaled[0],
            return_distances=True
        )
        
        predicted_author = open_set_result['predicted_author']
        decision = open_set_result['decision']
        confidence = open_set_result['confidence']

        reverse_encoder = {v: k for k, v in label_encoder.items()}

        if decision == 'rejected' or open_set_result['predicted_author'] == 'UNKNOWN':
            predicted_author = 'UNKNOWN'
        else:
            encoded_author_id = open_set_result['predicted_author']
            predicted_author = reverse_encoder[encoded_author_id]

        logger.info(f"Encoded prediction: {open_set_result['predicted_author']}")
        logger.info(f"Decoded prediction: {predicted_author}")
        
        # Step 4: Project to 2D
        if projection_method == 'umap' and UMAP_AVAILABLE:
            new_point_2d = projection_model.transform(combined_scaled)[0]
        else:
            from scipy.spatial.distance import cdist
            
            k = 5
            distances = cdist(combined_scaled, open_set_detector.X_train)[0]
            nearest_indices = np.argsort(distances)[:k]
            
            nearest_distances = distances[nearest_indices]
            weights = 1.0 / (nearest_distances + 1e-6)
            weights = weights / weights.sum()
            
            nearest_2d_points = projected_training_data[nearest_indices]
            new_point_2d = np.average(nearest_2d_points, axis=0, weights=weights)
        
        # Step 5: Get cluster boundaries
        clusters = compute_2d_cluster_boundaries(
            projected_training_data,
            y_train_stored,
            label_encoder
        )
        
        # Step 6: Build training samples for visualization
        training_samples = []
        for i, (point, author_id) in enumerate(zip(projected_training_data, y_train_stored)):
            if i % 5 == 0:
                training_samples.append({
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'author': reverse_encoder[author_id],
                    'is_prototype': False
                })
        
        for author, cluster_info in clusters.items():
            training_samples.append({
                'x': cluster_info['center']['x'],
                'y': cluster_info['center']['y'],
                'author': author,
                'is_prototype': True
            })
        
        # Step 7: Generate explanation
        if decision == 'rejected' or predicted_author == 'UNKNOWN':
            distance_ratio = open_set_result.get('distance_ratio', 0)
            explanation = f"Email falls outside all author clusters (distance ratio: {distance_ratio:.2f}x normal range)"
        else:
            predicted_author_str = str(predicted_author)
            cluster = clusters.get(predicted_author_str)
    
            if cluster and 'center' in cluster:
                distance_from_center = np.linalg.norm(
                    new_point_2d - np.array([cluster['center']['x'], cluster['center']['y']])
                )
                radius_threshold = cluster.get('radius_threshold', 0)
                explanation = f"Email is {distance_from_center:.2f} units from Contact {predicted_author_str}'s center (threshold: {radius_threshold:.2f})"
            else:
                explanation = f"Contact {predicted_author_str} exists in training but cluster missing"
        
        return VisualizationResponse(
            new_email_position={'x': float(new_point_2d[0]), 'y': float(new_point_2d[1])},
            predicted_author=str(predicted_author),
            confidence=confidence,
            decision=decision,
            author_clusters=clusters,
            training_samples=training_samples,
            projection_method=projection_method,
            explanation=explanation
        )
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def fit_projection_model(X_train_scaled: np.ndarray, y_train: np.ndarray, method: str = 'tsne'):
    """Fit dimensionality reduction model on training data"""
    global projection_model, projected_training_data, projection_method
    
    print(f"\n🎨 Fitting {method.upper()} projection for visualization...")
    
    if method == 'umap' and UMAP_AVAILABLE:
        projection_model = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42
        )
        projected_training_data = projection_model.fit_transform(X_train_scaled)
    else:
        if method == 'umap' and not UMAP_AVAILABLE:
            print("   ⚠️  UMAP not available, using t-SNE instead")
        
        projection_model = TSNE(
            n_components=2,
            perplexity=min(30, len(X_train_scaled) - 1),
            random_state=42,
            max_iter=1000
        )
        method = 'tsne'
        
        projected_training_data = projection_model.fit_transform(X_train_scaled)
    
    projection_method = method
    
    print(f"✅ {method.upper()} projection fitted!")
    print(f"   Training data shape: {X_train_scaled.shape}")
    print(f"   Projected shape: {projected_training_data.shape}")
    
    return projected_training_data

@app.post("/initialize-visualization")
async def initialize_visualization(method: str = "tsne"):
    """Initialize the 2D projection model"""
    global projection_model, projected_training_data, X_train_stored, y_train_stored
    
    if not MODELS_LOADED:
        raise HTTPException(
            status_code=503,
            detail="ML models not loaded"
        )
    
    X_train_stored = open_set_detector.X_train
    y_train_stored = open_set_detector.y_train
    
    if X_train_stored is None or y_train_stored is None:
        raise HTTPException(
            status_code=503,
            detail="Training data not available in open_set_detector"
        )
    
    try:
        fit_projection_model(X_train_stored, y_train_stored, method)
        
        clusters = compute_2d_cluster_boundaries(
            projected_training_data,
            y_train_stored,
            label_encoder
        )
        
        logger.info(f"Unique authors in y_train_stored: {np.unique(y_train_stored)}")
        logger.info(f"Cluster keys created: {list(clusters.keys())}")
        logger.info(f"Label encoder: {label_encoder}")

        return {
            "status": "success",
            "method": projection_method,
            "training_samples": len(projected_training_data),
            "authors": list(clusters.keys()),
            "message": f"✅ Visualization initialized with {projection_method.upper()}"
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/visualization-data")
async def get_visualization_data():
    """Get pre-computed visualization data for the frontend"""
    
    if projection_model is None:
        raise HTTPException(
            status_code=503,
            detail="Visualization not initialized. Call /initialize-visualization first"
        )
    
    try:
        reverse_encoder = {v: k for k, v in label_encoder.items()}
        
        clusters = compute_2d_cluster_boundaries(
            projected_training_data,
            y_train_stored,
            label_encoder
        )
        
        training_samples = []
        for i, (point, author_id) in enumerate(zip(projected_training_data, y_train_stored)):
            if i % 5 == 0:
                training_samples.append({
                    'x': float(point[0]),
                    'y': float(point[1]),
                    'author': reverse_encoder[author_id]
                })
        
        return {
            "clusters": clusters,
            "training_samples": training_samples,
            "projection_method": projection_method,
            "n_samples": len(projected_training_data),
            "n_authors": len(clusters)
        }
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def compute_2d_cluster_boundaries(projected_data: np.ndarray, y_train: np.ndarray, label_encoder: Dict):
    """Compute 2D boundaries for each author cluster"""
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
        
        author_key = str(author_name)
        
        clusters[author_key] = {
            'center': {'x': float(center[0]), 'y': float(center[1])},
            'radius_95': float(radius_95),
            'radius_threshold': float(radius_threshold),
            'n_samples': len(author_points),
            'samples': [
                {'x': float(point[0]), 'y': float(point[1])} 
                for point in author_points[:100]
            ]
        }
    
    return clusters

@app.get("/feature-info")
async def get_feature_info():
    """Get information about available features"""
    return {
        "feature_categories": {
            "lexical": [
                "avg_sentence_length", "median_sentence_length", "std_sentence_length",
                "avg_word_length", "median_word_length", "std_word_length",
                "ttr", "hapax_ratio", "dis_legomena_ratio", "yules_k"
            ],
            "punctuation": [
                "punct_period_per_100", "punct_comma_per_100", "punct_exclamation_per_100",
                "punct_question_per_100", "punct_total_density", "all_caps_ratio", "title_case_ratio"
            ],
            "structural": [
                "paragraph_count", "avg_sentences_per_paragraph",
                "whitespace_ratio", "indentation_features"
            ],
            "email_specific": [
                "has_greeting", "has_signoff", "char_count", "word_count", "sentence_count"
            ],
            "readability": [
                "avg_words_per_sentence", "avg_syllables_per_word", "flesch_kincaid_grade"
            ],
            "sociolinguistic": [
                "modal_verb_density", "hedge_word_density"
            ],
            "ai_detection": [
                "burstiness_sentence_length", "burstiness_coefficient_variation",
                "lexical_diversity_variance"
            ],
            "encoding": [
                "non_ascii_density", "cyrillic_char_ratio", "latin_char_ratio"
            ],
            "errors": [
                "repeated_char_ratio", "all_caps_word_ratio", "mixed_case_ratio"
            ]
        },
        "supported_languages": ["en", "es", "ru", "sv"],
        "total_stylometric_features": "72 (pure stylometry)",
        "note": "No n-grams, no embeddings - content-independent authorship detection"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)