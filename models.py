"""
Pydantic request/response models for the Stylometric API.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, TypedDict, Any


# ── TypedDicts (internal use) ────────────────────────────────────────────────

class TonalScore(TypedDict):
    label: str
    score: float


# ── Health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    models_loaded: bool


# ── Feature extraction ───────────────────────────────────────────────────────

class EmailInput(BaseModel):
    content: str = Field(..., description="Email content to analyze", min_length=1)
    language: Optional[str] = Field(
        default="auto",
        description="Language code (en, es, ru, sv) or 'auto' for detection",
    )
    include_raw_features: bool = Field(
        default=True,
        description="Include raw feature dictionary in response",
    )


class FeatureResponse(BaseModel):
    features: Optional[Dict[str, float]] = None
    feature_vector: List[float]
    feature_names: List[str]
    dimension: int
    detected_language: str
    text_length: int


class BatchEmailInput(BaseModel):
    emails: List[EmailInput] = Field(..., max_length=100)


class BatchFeatureResponse(BaseModel):
    results: List[FeatureResponse]
    total_processed: int


# ── Prediction ───────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    content: str = Field(..., min_length=1)
    language: str = Field(default="auto")
    suspected_author_id: Optional[str] = None


class FeatureExplanation(BaseModel):
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


# ── Semantic outlier detection ───────────────────────────────────────────────

class SentenceAnalysisRequest(BaseModel):
    content: str = Field(..., min_length=1)
    min_sentences: int = Field(default=5, ge=3)
    language: str = Field(default="auto")


class SemanticOutlierResponse(BaseModel):
    total_sentences: int
    outlier_count: int
    threshold: float
    sentences: List[Dict]
    skipped: bool
    skip_reason: Optional[str] = None
    summary: str


# ── Tonal outlier detection ──────────────────────────────────────────────────

class TonalOutlierResponse(BaseModel):
    total_sentences: int
    anomaly_count: int
    modal_tone: str
    sentences: List[Dict]
    skipped: bool
    skip_reason: Optional[str] = None
    summary: str


# ── Profile comparison ───────────────────────────────────────────────────────

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


# ── Visualization ────────────────────────────────────────────────────────────

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


class NTPVisualizationRequest(BaseModel):
    content: str
    max_length: int = 512
    language: str = "auto"

class PhishingDetectionRequest(BaseModel):
    content: str
    language: str = "auto"
    min_sentences: int = 3

class PhishingSentenceResult(BaseModel):
    index: int
    sentence: str
    persuasion_scores: List[Dict[str, Any]] = []
    dominant_tactic: str
    persuasion_z_score: float
    is_manipulative: bool
    heuristic_flags: List[str] = []
    combined_phishing_score: float

class PhishingDetectionResponse(BaseModel):
    total_sentences: int
    manipulative_count: int
    heuristic_flag_count: int
    overall_phishing_score: float
    risk_level: str
    sentences: List[PhishingSentenceResult] = []
    heuristic_summary: List[str] = []
    skipped: bool = False
    skip_reason: Optional[str] = None
    summary: str

class SummaryRequest(BaseModel):
    content: str
    claimed_sender: str = ""
    prediction: Optional[Dict] = None
    profile_comparison: Optional[Dict] = None
    ntp: Optional[Dict] = None
    semantic_outliers: Optional[Dict] = None
    tonal_outliers: Optional[Dict] = None
    phishing: Optional[Dict] = None
    contact_names: Optional[Dict] = None

class SummaryResponse(BaseModel):
    summary: str