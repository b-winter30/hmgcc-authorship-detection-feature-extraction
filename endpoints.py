"""
API endpoint handlers for the Stylometric API.
"""

import os
import json
import logging
import traceback

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from scipy.spatial.distance import cdist

from feature_extractor import get_feature_vector
from next_token_prediction import get_ntp_feature_names

from models import (
    BatchEmailInput,
    BatchFeatureResponse,
    EmailInput,
    FeatureResponse,
    HealthResponse,
    NTPVisualizationRequest,
    PredictRequest,
    PredictResponse,
    ProfileCompareRequest,
    ProfileCompareResponse,
    SemanticOutlierResponse,
    SentenceAnalysisRequest,
    TonalOutlierResponse,
    VisualizationRequest,
    VisualizationResponse,
    PhishingDetectionRequest,
    PhishingDetectionResponse,
    SummaryRequest,
    SummaryResponse,
)
from helpers import (
    VISUALIZATION_DIR,
    classify_tone,
    compute_2d_cluster_boundaries,
    compute_jsd_tonal_scores,
    convert_numpy_types,
    detect_language,
    generate_ntp_visualization,
    get_contact_info,
    split_sentences,
    detect_heuristic_flags,
    classify_phishing_persuasion,
    compute_phishing_scores,
    reconstruct_parent_words,
    add_parent_words_to_anomalous_tokens,
)
import dependencies as deps

logger = logging.getLogger("stylometry_api")

router = APIRouter()

# ── Health ────────────────────────────────────────────────────────────────────

@router.get("/", response_model=HealthResponse)
async def root():
    return HealthResponse(status="healthy", version="3.0.0", models_loaded=deps.MODELS_LOADED)


@router.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="healthy", version="3.0.0", models_loaded=deps.MODELS_LOADED)


# ── Feature extraction ────────────────────────────────────────────────────────

@router.post("/extract-features")
async def extract_features(email: EmailInput):
    """Extract stylometric features (no NTP)."""
    try:
        detected_lang = (
            detect_language(email.content) if email.language == "auto" else email.language.lower()
        )
        features = deps.extractor.extract_all_features(email.content, detected_lang)
        feature_names, feature_vector = get_feature_vector(features)

        return {
            "features": features if email.include_raw_features else None,
            "feature_vector": feature_vector,
            "feature_names": feature_names,
            "dimension": len(feature_vector),
            "detected_language": detected_lang,
            "text_length": len(email.content),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature extraction failed: {e}")


@router.post("/extract-features/batch", response_model=BatchFeatureResponse)
async def extract_features_batch(batch: BatchEmailInput):
    """Extract features from multiple emails."""
    try:
        results = []
        for email_input in batch.emails:
            detected_lang = (
                detect_language(email_input.content)
                if email_input.language == "auto"
                else email_input.language.lower()
            )
            features = deps.extractor.extract_all_features(email_input.content, detected_lang)
            feature_names, feature_vector = get_feature_vector(features)

            data = {
                "feature_vector": feature_vector,
                "feature_names": feature_names,
                "dimension": len(feature_vector),
                "detected_language": detected_lang,
                "text_length": len(email_input.content),
            }
            if email_input.include_raw_features:
                data["features"] = features
            results.append(FeatureResponse(**data))

        return BatchFeatureResponse(results=results, total_processed=len(results))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {e}")


# ── Prediction ────────────────────────────────────────────────────────────────

@router.post("/predict-author", response_model=PredictResponse)
async def predict_author(request: PredictRequest):
    """Predict the author of an email — pure stylometry."""
    if not deps.MODELS_LOADED:
        raise HTTPException(status_code=503, detail="ML models not loaded. Train models first.")

    try:
        if request.suspected_author_id:
            logger.info(f"Suspected author ID: {request.suspected_author_id}")

        detected_lang = (
            detect_language(request.content) if request.language == "auto" else request.language.lower()
        )
        logger.info(f"Detected language: {detected_lang}")

        # Extract stylometric features
        features = deps.extractor.extract_all_features(request.content, detected_lang)
        feature_names, stylometric_vector = get_feature_vector(features)
        combined_vector = np.array(stylometric_vector)
        logger.info(f"Extracted {len(combined_vector)} features")

        if len(combined_vector) != len(deps.total_feature_names):
            raise HTTPException(
                status_code=500,
                detail=(
                    f"Feature dimension mismatch: got {len(combined_vector)}, "
                    f"expected {len(deps.total_feature_names)}"
                ),
            )

        # Normalise
        combined_scaled = deps.scaler.transform([combined_vector])

        # Open-set detection
        open_set_result = deps.open_set_detector.predict(combined_scaled[0], return_distances=False)
        logger.info(f"Open-set result: {open_set_result['decision']}")

        if open_set_result["decision"] == "rejected":
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
                message=f"❌ Unknown author: {open_set_result['reason']}",
            )

        # Random Forest prediction
        prediction_idx = deps.classifier.predict(combined_scaled)[0]
        probabilities = deps.classifier.predict_proba(combined_scaled)[0]

        reverse_encoder = {v: k for k, v in deps.label_encoder.items()}
        predicted_author = reverse_encoder[prediction_idx]
        confidence = float(max(probabilities))
        logger.info(f"Predicted: {predicted_author}, confidence: {confidence}")

        contact_info = get_contact_info(predicted_author)
        all_probs = {reverse_encoder[i]: float(p) for i, p in enumerate(probabilities)}

        # Weighted score (optional)
        weighted_score = None
        if request.suspected_author_id and request.suspected_author_id in deps.feature_weights:
            weights = deps.feature_weights[request.suspected_author_id]
            weighted_score = float(1.0 / (1.0 + np.mean(np.abs(combined_vector * weights))))

        is_anomaly = confidence < 0.5
        anomaly_score = float(1.0 - confidence)

        # Feature contributions
        importance = deps.classifier.feature_importances_
        contributions = sorted(
            [
                {
                    "feature": deps.total_feature_names[i],
                    "value": float(combined_vector[i]),
                    "importance": float(importance[i]),
                    "contribution": float(combined_vector[i] * importance[i]),
                }
                for i in range(len(combined_vector))
            ],
            key=lambda x: abs(x["contribution"]),
            reverse=True,
        )

        # Message
        if is_anomaly:
            message = f"⚠️ Unusual writing pattern — low confidence ({confidence:.0%})"
        elif confidence > 0.9:
            message = f"✅ High confidence: Contact {predicted_author}"
        elif confidence > 0.7:
            message = f"✅ Likely: Contact {predicted_author}"
        else:
            message = f"⚠️ Low-medium confidence: Contact {predicted_author}"

        return PredictResponse(
            predicted_author=predicted_author,
            contact_info=contact_info,
            confidence=confidence,
            all_probabilities=all_probs,
            weighted_score=weighted_score,
            is_anomaly=is_anomaly,
            anomaly_score=anomaly_score,
            top_features=contributions[:10],
            method="pure_stylometry",
            message=message,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"predict_author error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


# ── Semantic outlier detection ────────────────────────────────────────────────

@router.post("/semantic-outlier-detection", response_model=SemanticOutlierResponse)
async def semantic_outlier_detection(request: SentenceAnalysisRequest):
    """Leave-one-out semantic analysis."""
    try:
        sentences = split_sentences(request.content)

        if len(sentences) < request.min_sentences:
            return SemanticOutlierResponse(
                total_sentences=len(sentences),
                outlier_count=0,
                threshold=0.0,
                sentences=[],
                skipped=True,
                skip_reason=f"Email has {len(sentences)} sentences, minimum is {request.min_sentences}",
                summary=f"Skipped — email too short ({len(sentences)} sentences, need {request.min_sentences})",
            )

        full_text = " ".join(sentences)
        full_emb = deps.sentence_model.encode(full_text, normalize_embeddings=True)

        rows = []
        for i, sentence in enumerate(sentences):
            remaining = sentences[:i] + sentences[i + 1:]
            remaining_emb = deps.sentence_model.encode(" ".join(remaining), normalize_embeddings=True)
            sentence_emb = deps.sentence_model.encode(sentence, normalize_embeddings=True)

            influence_distance = float(np.linalg.norm(full_emb - remaining_emb))
            context_distance = 1.0 - float(np.dot(sentence_emb, remaining_emb))
            combined_score = (influence_distance + context_distance) / 2.0

            rows.append({
                "index": i,
                "sentence": sentence,
                "influence_distance": round(influence_distance, 4),
                "context_distance": round(context_distance, 4),
                "combined_score": round(combined_score, 4),
            })

        scores = [r["combined_score"] for r in rows]
        q1, q3 = float(np.percentile(scores, 25)), float(np.percentile(scores, 75))
        outlier_threshold = q3 + 1.5 * (q3 - q1)
        mean_s, std_s = float(np.mean(scores)), float(np.std(scores))

        for r in rows:
            z = (r["combined_score"] - mean_s) / (std_s + 1e-8)
            r["z_score"] = round(float(z), 2)
            r["is_outlier"] = bool(r["combined_score"] > outlier_threshold)
            r["suspicion_score"] = round(min(max(z / 4.0 if z > 0 else 0.0, 0.0), 1.0), 4)

        outlier_count = sum(1 for r in rows if r["is_outlier"])
        rows.sort(key=lambda x: x["suspicion_score"], reverse=True)

        if outlier_count == 0:
            summary = "No semantic outliers — all sentences are topically consistent."
        elif outlier_count == 1:
            top = next(r for r in rows if r["is_outlier"])
            preview = top["sentence"][:60] + ("..." if len(top["sentence"]) > 60 else "")
            summary = f'1 semantic outlier: "{preview}" (z={top["z_score"]:.1f})'
        else:
            summary = f"{outlier_count} semantic outliers detected."

        return SemanticOutlierResponse(
            total_sentences=len(sentences),
            outlier_count=outlier_count,
            threshold=round(outlier_threshold, 4),
            sentences=rows,
            skipped=False,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Semantic outlier error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Semantic outlier detection failed: {e}")


# ── Tonal outlier detection ───────────────────────────────────────────────────

@router.post("/tonal-outlier-detection", response_model=TonalOutlierResponse)
async def tonal_outlier_detection(request: SentenceAnalysisRequest):
    """JSD-based tonal consistency analysis."""
    try:
        sentences = split_sentences(request.content)

        if len(sentences) < request.min_sentences:
            return TonalOutlierResponse(
                total_sentences=len(sentences),
                anomaly_count=0,
                modal_tone="unknown",
                sentences=[],
                skipped=True,
                skip_reason=f"Email has {len(sentences)} sentences, minimum is {request.min_sentences}",
                summary=f"Skipped — email too short ({len(sentences)} sentences, need {request.min_sentences})",
            )

        tonal_scores = classify_tone(sentences, deps.tonal_classifier)
        jsd_results = compute_jsd_tonal_scores(tonal_scores)

        rows = []
        for i, (scores_dict, jsd) in enumerate(zip(tonal_scores, jsd_results)):
            z = jsd["tonal_z_score"]
            rows.append({
                "index": i,
                "sentence": sentences[i],
                "tonal_scores": [
                    {"label": k, "score": round(v, 4)}
                    for k, v in sorted(scores_dict.items(), key=lambda x: x[1], reverse=True)
                ],
                "dominant_tone": jsd["dominant_tone"],
                "modal_tone": jsd["modal_tone"],
                "jsd_score": jsd["jsd_score"],
                "tonal_z_score": jsd["tonal_z_score"],
                "tonal_is_anomaly": jsd["tonal_is_anomaly"],
                "tonal_flag": jsd["tonal_flag"],
                "suspicion_score": round(min(max(z / 4.0 if z > 0 else 0.0, 0.0), 1.0), 4),
            })

        modal_tone = jsd_results[0]["modal_tone"] if jsd_results else "informational"
        anomaly_count = sum(1 for r in rows if r["tonal_is_anomaly"])
        rows.sort(key=lambda x: x["suspicion_score"], reverse=True)

        anomalies = [r for r in rows if r["tonal_is_anomaly"]]
        if anomalies:
            worst = max(anomalies, key=lambda r: r["tonal_z_score"])
            preview = worst["sentence"][:60] + ("..." if len(worst["sentence"]) > 60 else "")
            summary = (
                f"Dominant tone: '{modal_tone}'. "
                f"{anomaly_count} sentence(s) break register. "
                f'Most anomalous: "{preview}"'
            )
        else:
            summary = f"Dominant tone: '{modal_tone}'. All sentences tonally consistent."

        return TonalOutlierResponse(
            total_sentences=len(sentences),
            anomaly_count=anomaly_count,
            modal_tone=modal_tone,
            sentences=rows,
            skipped=False,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Tonal outlier error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Tonal outlier detection failed: {e}")

# ── Profile comparison ────────────────────────────────────────────────────────

@router.post("/compare-to-profile", response_model=ProfileCompareResponse)
async def compare_to_profile(request: ProfileCompareRequest):
    """Compare email features against a contact's stored profile."""
    import psycopg2

    try:
        detected_lang = (
            detect_language(request.content) if request.language == "auto" else request.language.lower()
        )
        features = deps.extractor.extract_all_features(request.content, detected_lang)
        feature_names, feature_vector = get_feature_vector(features)

        # Load contact profile from DB
        conn = psycopg2.connect(
            host=os.environ.get("DB_HOST", "postgres"),
            port=os.environ.get("DB_PORT", "5432"),
            dbname=os.environ.get("DB_NAME", "authorship-detection"),
            user=os.environ.get("DB_USER", "postgres"),
            password=os.environ.get("DB_PASSWORD", "postgres"),
        )
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name, stylometric_profile, profile_sample_count FROM contacts WHERE contact_id = %s",
            (int(request.contact_id),),
        )
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if not row:
            raise HTTPException(status_code=404, detail=f"Contact {request.contact_id} not found")

        contact_name, profile_json, sample_count = row[0], row[1], row[2] or 0

        if not profile_json:
            raise HTTPException(
                status_code=404,
                detail=f"No stylometric profile for {contact_name}. Run training first.",
            )

        raw_profile = profile_json if isinstance(profile_json, dict) else json.loads(profile_json)
        stylometric_profile = raw_profile.get("stylometric", raw_profile)
        ntp_baseline = raw_profile.get("ntp_baseline", None)

        # Z-scores
        deviations = []
        for i, name in enumerate(feature_names):
            if name not in stylometric_profile:
                continue
            prof = stylometric_profile[name]
            z_score = (feature_vector[i] - prof["mean"]) / (prof["std"] + 1e-8)
            deviations.append({
                "feature": name,
                "new_value": float(feature_vector[i]),
                "profile_mean": float(prof["mean"]),
                "profile_std": float(prof["std"]),
                "z_score": float(z_score),
                "is_unusual": bool(abs(z_score) > 2.0),
            })

        unusual = sorted(
            [d for d in deviations if d["is_unusual"]],
            key=lambda x: abs(x["z_score"]),
            reverse=True,
        )
        all_z = [abs(d["z_score"]) for d in deviations]
        overall_deviation = float(np.mean(all_z)) if all_z else 0.0
        normal_count = len(deviations) - len(unusual)
        match_pct = (normal_count / len(deviations) * 100) if deviations else 0.0

        max_z = max((abs(d["z_score"]) for d in unusual), default=0)
        if max_z > 100:
            match_pct = min(match_pct, 50.0)
        elif max_z > 10:
            match_pct = min(match_pct, 70.0)

        # NTP baseline comparison
        ntp_comparison = _build_ntp_comparison(ntp_baseline, request.ntp_stats)

        # Summary
        ntp_flag = ""
        if ntp_comparison and not ntp_comparison.get("baseline_only"):
            anomaly_z = ntp_comparison.get("mean_anomaly_score", {}).get("z_score", 0)
            if abs(anomaly_z) > 2.0:
                direction = "more predictable" if anomaly_z < 0 else "more unpredictable"
                ntp_flag = f" NTP anomaly is {abs(anomaly_z):.1f}σ {direction} than normal for this contact."

        if max_z > 100:
            summary = f"Suspicious — extreme deviations detected (max z={max_z:.0f}).{ntp_flag}"
        elif max_z > 10:
            summary = f"Mixed signals — notable outliers present.{ntp_flag}"
        elif match_pct >= 90:
            summary = f"Strong match — consistent with {contact_name}'s profile.{ntp_flag}"
        elif match_pct >= 70:
            summary = f"Moderate match — mostly consistent with {contact_name}.{ntp_flag}"
        elif match_pct >= 50:
            summary = f"Weak match — significant deviations from {contact_name}.{ntp_flag}"
        else:
            summary = f"Poor match — differs substantially from {contact_name}.{ntp_flag}"

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
        logger.error(f"Profile comparison error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Profile comparison failed: {e}")


def _build_ntp_comparison(ntp_baseline, ntp_stats):
    """Build NTP baseline comparison dict."""
    if not ntp_baseline:
        return None

    metrics = ["mean_anomaly_score", "anomaly_ratio", "mean_probability", "median_probability"]

    if ntp_stats:
        comparison = {"n_baseline_samples": ntp_baseline.get("n_samples", 0), "baseline_only": False}
        for metric in metrics:
            if metric in ntp_baseline and metric in ntp_stats:
                b = ntp_baseline[metric]
                new_val = float(ntp_stats[metric])
                b_mean, b_std = float(b["mean"]), float(b["std"])
                z = (new_val - b_mean) / (b_std + 1e-8)
                comparison[metric] = {
                    "new_value": new_val,
                    "baseline_mean": b_mean,
                    "baseline_std": b_std,
                    "z_score": float(z),
                    "is_unusual": bool(abs(z) > 2.0),
                }
        return comparison

    # Baseline only (no new NTP stats)
    comparison = {"n_baseline_samples": ntp_baseline.get("n_samples", 0), "baseline_only": True}
    for metric in metrics:
        if metric in ntp_baseline:
            comparison[metric] = {
                "baseline_mean": float(ntp_baseline[metric]["mean"]),
                "baseline_std": float(ntp_baseline[metric]["std"]),
            }
    return comparison


# ── NTP visualization ─────────────────────────────────────────────────────────

@router.post("/visualize-ntp-anomalies")
async def visualize_ntp_anomalies(request: NTPVisualizationRequest):
    """Visualize next-token prediction anomalies."""
    if not deps.extractor_with_ntp or not deps.extractor_with_ntp.enable_ntp:
        raise HTTPException(status_code=503, detail="NTP not available on this server")

    try:
        features = deps.extractor_with_ntp.ntp_detector.extract_anomaly_features(
            request.content, max_length=request.max_length, return_detailed=True
        )
        detailed_results = features.pop("_detailed_results", [])
        detailed_results = reconstruct_parent_words(detailed_results)

        probabilities = [float(r["probability"]) for r in detailed_results if "probability" in r]

        anomalous_tokens = [
            {
                "position": int(r["position"]),
                "token": r["token"],
                "probability": float(r["probability"]),
                "rank": int(r["rank"]),
                "anomaly_score": float(r["anomaly_score"]),
                "is_capitalized": bool(r.get("is_capitalized", False)),
            }
            for r in detailed_results
            if r.get("is_anomaly", False)
        ]
        anomalous_tokens = add_parent_words_to_anomalous_tokens(anomalous_tokens, detailed_results)

        viz_path = generate_ntp_visualization(
            detailed_results, anomalous_tokens, request.content[:100]
        )

        return {
            "aggregate_features": convert_numpy_types(features),
            "total_tokens": len(detailed_results),
            "anomalous_tokens": anomalous_tokens,
            "anomaly_count": len(anomalous_tokens),
            "anomaly_ratio": len(anomalous_tokens) / len(detailed_results) if detailed_results else 0,
            "mean_probability": float(np.mean(probabilities)) if probabilities else 0.0,
            "median_probability": float(np.median(probabilities)) if probabilities else 0.0,
            "min_probability": float(np.min(probabilities)) if probabilities else 0.0,
            "max_probability": float(np.max(probabilities)) if probabilities else 0.0,
            "mean_anomaly_score": float(np.mean([r["anomaly_score"] for r in detailed_results])) if detailed_results else 0.0,
            "mean_rank": float(np.mean([r["rank"] for r in detailed_results])) if detailed_results else 0.0,
            "visualization_path": viz_path,
        }

    except Exception as e:
        logger.error(f"NTP visualization error: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"NTP visualization failed: {e}")


@router.get("/ntp-visualization/{filename}")
async def get_ntp_visualization(filename: str):
    """Serve NTP visualization images."""
    filepath = os.path.join(VISUALIZATION_DIR, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Visualization not found")
    return FileResponse(filepath, media_type="image/png")


# ── 2-D visualization ────────────────────────────────────────────────────────

@router.post("/initialize-visualization")
async def initialize_visualization(method: str = "tsne"):
    """Initialize the 2D projection model."""
    if not deps.MODELS_LOADED:
        raise HTTPException(status_code=503, detail="ML models not loaded")

    deps.X_train_stored = deps.open_set_detector.X_train
    deps.y_train_stored = deps.open_set_detector.y_train

    if deps.X_train_stored is None or deps.y_train_stored is None:
        raise HTTPException(status_code=503, detail="Training data not available in open_set_detector")

    try:
        deps.fit_projection_model(deps.X_train_stored, deps.y_train_stored, method)

        clusters = compute_2d_cluster_boundaries(
            deps.projected_training_data, deps.y_train_stored, deps.label_encoder
        )

        return {
            "status": "success",
            "method": deps.projection_method,
            "training_samples": len(deps.projected_training_data),
            "authors": list(clusters.keys()),
            "message": f"✅ Visualization initialized with {deps.projection_method.upper()}",
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/visualization-data")
async def get_visualization_data():
    """Get pre-computed visualization data."""
    if deps.projection_model is None:
        raise HTTPException(status_code=503, detail="Call /initialize-visualization first")

    try:
        reverse_encoder = {v: k for k, v in deps.label_encoder.items()}
        clusters = compute_2d_cluster_boundaries(
            deps.projected_training_data, deps.y_train_stored, deps.label_encoder
        )

        training_samples = [
            {"x": float(point[0]), "y": float(point[1]), "author": reverse_encoder[aid]}
            for i, (point, aid) in enumerate(zip(deps.projected_training_data, deps.y_train_stored))
            if i % 5 == 0
        ]

        return {
            "clusters": clusters,
            "training_samples": training_samples,
            "projection_method": deps.projection_method,
            "n_samples": len(deps.projected_training_data),
            "n_authors": len(clusters),
        }
    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/visualize-prediction", response_model=VisualizationResponse)
async def visualize_prediction(request: VisualizationRequest):
    """Predict author and return 2D visualization data."""
    if not deps.MODELS_LOADED:
        raise HTTPException(status_code=503, detail="ML models not loaded")
    if deps.projection_model is None:
        raise HTTPException(status_code=503, detail="Call /initialize-visualization first")

    try:
        detected_lang = (
            detect_language(request.content) if request.language == "auto" else request.language.lower()
        )
        features = deps.extractor.extract_all_features(request.content, detected_lang)
        _, stylometric_vector = get_feature_vector(features)
        combined_vector = np.array(stylometric_vector)
        combined_scaled = deps.scaler.transform([combined_vector])

        open_set_result = deps.open_set_detector.predict(combined_scaled[0], return_distances=True)
        reverse_encoder = {v: k for k, v in deps.label_encoder.items()}

        decision = open_set_result["decision"]
        confidence = open_set_result["confidence"]

        if decision == "rejected" or open_set_result["predicted_author"] == "UNKNOWN":
            predicted_author = "UNKNOWN"
        else:
            predicted_author = reverse_encoder[open_set_result["predicted_author"]]

        # Project to 2D
        if deps.projection_method == "umap" and deps.UMAP_AVAILABLE:
            new_point_2d = deps.projection_model.transform(combined_scaled)[0]
        else:
            distances = cdist(combined_scaled, deps.open_set_detector.X_train)[0]
            nearest_idx = np.argsort(distances)[:5]
            weights = 1.0 / (distances[nearest_idx] + 1e-6)
            weights /= weights.sum()
            new_point_2d = np.average(deps.projected_training_data[nearest_idx], axis=0, weights=weights)

        clusters = compute_2d_cluster_boundaries(
            deps.projected_training_data, deps.y_train_stored, deps.label_encoder
        )

        training_samples = [
            {"x": float(pt[0]), "y": float(pt[1]), "author": reverse_encoder[aid], "is_prototype": False}
            for i, (pt, aid) in enumerate(zip(deps.projected_training_data, deps.y_train_stored))
            if i % 5 == 0
        ]
        for author, info in clusters.items():
            training_samples.append({
                "x": info["center"]["x"],
                "y": info["center"]["y"],
                "author": author,
                "is_prototype": True,
            })

        # Explanation
        if predicted_author == "UNKNOWN":
            dr = open_set_result.get("distance_ratio", 0)
            explanation = f"Email falls outside all author clusters (distance ratio: {dr:.2f}x)"
        else:
            cluster = clusters.get(str(predicted_author))
            if cluster and "center" in cluster:
                dist = np.linalg.norm(
                    new_point_2d - np.array([cluster["center"]["x"], cluster["center"]["y"]])
                )
                explanation = (
                    f"Email is {dist:.2f} units from Contact {predicted_author}'s center "
                    f"(threshold: {cluster.get('radius_threshold', 0):.2f})"
                )
            else:
                explanation = f"Contact {predicted_author} in training but cluster missing"

        return VisualizationResponse(
            new_email_position={"x": float(new_point_2d[0]), "y": float(new_point_2d[1])},
            predicted_author=str(predicted_author),
            confidence=confidence,
            decision=decision,
            author_clusters=clusters,
            training_samples=training_samples,
            projection_method=deps.projection_method,
            explanation=explanation,
        )

    except Exception as e:
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/phishing-detection", response_model=PhishingDetectionResponse)
async def phishing_detection(request: PhishingDetectionRequest):
    """Detect phishing/social engineering tactics via persuasion NLI + heuristics."""
    try:
        sentences = split_sentences(request.content)

        if len(sentences) < request.min_sentences:
            return PhishingDetectionResponse(
                total_sentences=len(sentences),
                manipulative_count=0,
                heuristic_flag_count=0,
                overall_phishing_score=0.0,
                risk_level="low",
                sentences=[],
                heuristic_summary=[],
                skipped=True,
                skip_reason=f"Email has {len(sentences)} sentences, minimum is {request.min_sentences}",
                summary=f"Skipped — email too short ({len(sentences)} sentences, need {request.min_sentences})",
            )

        # Full-email heuristic scan
        full_heuristic_flags = detect_heuristic_flags(request.content)

        # NLI persuasion classification — reuses deps.tonal_classifier (no new model)
        persuasion_scores = classify_phishing_persuasion(
            sentences, deps.tonal_classifier
        )

        # Compute combined scores
        scored = compute_phishing_scores(persuasion_scores, sentences)

        # Debug: log all sentence scores
        for i, s in enumerate(scored):
            logger.info(f"[Phishing] Sentence {i}: score={s['combined_phishing_score']:.4f} manipulative={s['is_manipulative']} tactic={s['dominant_tactic']} heuristics={s['heuristic_flags']} | {sentences[i][:80]}")

        # Build response rows
        rows = []
        for i, s_result in enumerate(scored):
            rows.append({
                "index": i,
                "sentence": sentences[i],
                **s_result,
            })

        manipulative_count = sum(1 for r in rows if r["is_manipulative"])
        all_heuristic_flags = set(full_heuristic_flags)
        for r in rows:
            all_heuristic_flags.update(r["heuristic_flags"])

        # Overall phishing score: weighted combination
        if rows:
            max_sentence_score = max(r["combined_phishing_score"] for r in rows)
            avg_sentence_score = sum(r["combined_phishing_score"] for r in rows) / len(rows)
            heuristic_weight = min(len(all_heuristic_flags) * 0.1, 0.3)
            overall = 0.5 * max_sentence_score + 0.3 * avg_sentence_score + 0.2 * heuristic_weight
            overall = min(overall, 1.0)
        else:
            overall = 0.0

        # Risk level
        if overall >= 0.9:
            risk_level = "critical"
        elif overall >= 0.8:
            risk_level = "high"
        elif overall >= 0.5:
            risk_level = "medium"
        elif overall >= 0.3:
            risk_level = "low"
        else:
            risk_level = "very low"

        if manipulative_count == 0 and len(all_heuristic_flags) == 0:
            risk_level = "very low"
            overall = min(overall, 0.29)
            
        # Sort by combined score descending
        rows.sort(key=lambda x: x["combined_phishing_score"], reverse=True)

        # Readable heuristic summary
        flag_descriptions = {
            "credential_request": "Requests for passwords or login credentials",
            "urgency_pressure": "Urgent time pressure language",
            "threat_consequence": "Threats of account suspension or data loss",
            "financial_lure": "Financial transaction or payment references",
            "suspicious_link_ref": "Prompts to click links or open attachments",
            "impersonation_cue": "Claims of authority or official status",
        }
        heuristic_summary = [
            flag_descriptions.get(f, f) for f in sorted(all_heuristic_flags)
        ]

        # Summary text
        if risk_level == "very low":
            summary = "No significant phishing indicators detected."
        elif risk_level == "low":
            summary = f"Minor persuasion tactics detected ({manipulative_count} sentence(s)). {', '.join(heuristic_summary[:2]) + '.' if heuristic_summary else ''}"
        elif risk_level == "medium":
            summary = f"Moderate phishing indicators detected ({manipulative_count} sentence(s)). {', '.join(heuristic_summary[:3]) + '.' if heuristic_summary else ''}"
        elif risk_level == 'high':
            summary = f"HIGH RISK: {manipulative_count} manipulative sentence(s) with {len(all_heuristic_flags)} heuristic flag(s). Likely social engineering attempt."
        else:
            summary = f"CRITICAL RISK: {manipulative_count} manipulative sentence(s) with {len(all_heuristic_flags)} heuristic flag(s). Likely social engineering attempt."

        return PhishingDetectionResponse(
            total_sentences=len(sentences),
            manipulative_count=manipulative_count,
            heuristic_flag_count=len(all_heuristic_flags),
            overall_phishing_score=round(overall, 4),
            risk_level=risk_level,
            sentences=rows,
            heuristic_summary=heuristic_summary,
            skipped=False,
            summary=summary,
        )

    except Exception as e:
        logger.error(f"Phishing detection error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Phishing detection failed: {e}")

# ── Model / feature info ─────────────────────────────────────────────────────

@router.get("/model-info")
async def get_model_info():
    info = {
        "loaded": deps.MODELS_LOADED,
        "ntp_available": deps.extractor_with_ntp.enable_ntp if deps.extractor_with_ntp else False,
    }
    if deps.MODELS_LOADED:
        info.update({
            "known_contacts": list(deps.label_encoder.keys()),
            "n_contacts": len(deps.label_encoder),
            "n_stylometric_features": len(deps.stylometric_feature_names),
            "n_total_features": len(deps.total_feature_names),
            "feature_type": (
                "pure_stylometry_with_ntp"
                if deps.extractor_with_ntp and deps.extractor_with_ntp.enable_ntp
                else "pure_stylometry"
            ),
        })
    if deps.extractor_with_ntp and deps.extractor_with_ntp.enable_ntp:
        info.update({
            "ntp_model": deps.extractor_with_ntp.ntp_detector.model_name,
            "ntp_threshold": deps.extractor_with_ntp.ntp_detector.anomaly_threshold,
            "ntp_feature_names": get_ntp_feature_names(),
        })
    return info


@router.get("/feature-info")
async def get_feature_info():
    return {
        "feature_categories": {
            "lexical": [
                "avg_sentence_length", "median_sentence_length", "std_sentence_length",
                "avg_word_length", "median_word_length", "std_word_length",
                "ttr", "hapax_ratio", "dis_legomena_ratio", "yules_k",
            ],
            "punctuation": [
                "punct_period_per_100", "punct_comma_per_100", "punct_exclamation_per_100",
                "punct_question_per_100", "punct_total_density", "all_caps_ratio", "title_case_ratio",
            ],
            "structural": [
                "paragraph_count", "avg_sentences_per_paragraph",
                "whitespace_ratio", "indentation_features",
            ],
            "email_specific": [
                "has_greeting", "has_signoff", "char_count", "word_count", "sentence_count",
            ],
            "readability": [
                "avg_words_per_sentence", "avg_syllables_per_word", "flesch_kincaid_grade",
            ],
            "sociolinguistic": ["modal_verb_density", "hedge_word_density"],
            "ai_detection": [
                "burstiness_sentence_length", "burstiness_coefficient_variation",
                "lexical_diversity_variance",
            ],
            "encoding": ["non_ascii_density", "cyrillic_char_ratio", "latin_char_ratio"],
            "errors": ["repeated_char_ratio", "all_caps_word_ratio", "mixed_case_ratio"],
        },
        "supported_languages": ["en", "es", "ru", "sv"],
        "total_stylometric_features": "72 (pure stylometry)",
        "note": "No n-grams, no embeddings — content-independent authorship detection",
    }

@router.post("/summarize-analysis", response_model=SummaryResponse)
async def summarize_analysis(request: SummaryRequest):
    """Generate a human-readable summary of all pipeline stages using Qwen."""
    if not deps.extractor_with_ntp or not deps.extractor_with_ntp.enable_ntp:
        raise HTTPException(status_code=503, detail="Qwen model not available on this server")

    try:
        # Build a structured context from all stages
        sections = []
        sections.append(f"EMAIL FROM: {request.claimed_sender}")
        sections.append(f"EMAIL PREVIEW: {request.content[:300]}")

        # Stage 1+2: Prediction
        if request.prediction:
            p = request.prediction
            predicted = p.get("predicted_author", "UNKNOWN")
            conf = p.get("confidence", 0)
            method = p.get("method", "")
            message = p.get("message", "")
            probs = p.get("all_probabilities", {})
            top_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)[:3]
            top_str = ", ".join([f"{k}: {v*100:.0f}%" for k, v in top_probs])
            sections.append(
                f"AUTHORSHIP PREDICTION: predicted={predicted}, confidence={conf*100:.0f}%, "
                f"method={method}. {message}. Top candidates: {top_str}"
            )

        # Stage 3: Profile comparison
        if request.profile_comparison:
            pc = request.profile_comparison
            sections.append(
                f"PROFILE COMPARISON: match={pc.get('match_percentage', 0):.0f}% with {pc.get('contact_name', '?')} "
                f"({pc.get('profile_sample_count', 0)} training samples). "
                f"Unusual features: {pc.get('unusual_count', 0)}/{pc.get('total_features', 0)}. "
                f"{pc.get('summary', '')}"
            )
            # NTP baseline
            ntp_bl = pc.get("ntp_baseline_comparison")
            if ntp_bl and not ntp_bl.get("baseline_only"):
                mas = ntp_bl.get("mean_anomaly_score", {})
                if isinstance(mas, dict) and "z_score" in mas:
                    sections.append(
                        f"NTP BASELINE: anomaly z-score={mas['z_score']:.1f} "
                        f"(value={mas.get('new_value', 0):.3f}, baseline={mas.get('baseline_mean', 0):.3f}±{mas.get('baseline_std', 0):.3f})"
                    )

        # Stage 4: NTP
        if request.ntp:
            n = request.ntp
            sections.append(
                f"NTP ANOMALY: {n.get('anomaly_count', 0)} anomalous tokens out of {n.get('total_tokens', 0)} "
                f"({n.get('anomaly_count', 0) / max(n.get('total_tokens', 1), 1) * 100:.1f}%). "
                f"Mean probability: {n.get('mean_probability', 0)*100:.1f}%, "
                f"mean anomaly score: {n.get('mean_anomaly_score', 0)*100:.1f}%"
            )
            anom_tokens = n.get("anomalous_tokens", [])
            if anom_tokens:
                token_str = ", ".join([f"'{t.get('token', '').strip()}'" for t in anom_tokens[:8]])
                sections.append(f"STRANGE TOKENS: {token_str}")

        # Stage 5: Semantic outliers
        if request.semantic_outliers:
            so = request.semantic_outliers
            if so.get("skipped"):
                sections.append(f"SEMANTIC OUTLIERS: skipped — {so.get('skip_reason', 'too short')}")
            else:
                sections.append(
                    f"SEMANTIC OUTLIERS: {so.get('outlier_count', 0)} outlier(s) in {so.get('total_sentences', 0)} sentences. "
                    f"{so.get('summary', '')}"
                )

        # Stage 6: Tonal outliers
        if request.tonal_outliers:
            to = request.tonal_outliers
            if to.get("skipped"):
                sections.append(f"TONAL OUTLIERS: skipped — {to.get('skip_reason', 'too short')}")
            else:
                sections.append(
                    f"TONAL OUTLIERS: {to.get('anomaly_count', 0)} anomaly(ies), modal tone: '{to.get('modal_tone', '?')}'. "
                    f"{to.get('summary', '')}"
                )

        # Stage 7: Phishing
        if request.phishing:
            ph = request.phishing
            if ph.get("skipped"):
                sections.append(f"PHISHING DETECTION: skipped — {ph.get('skip_reason', 'too short')}")
            else:
                sections.append(
                    f"PHISHING DETECTION: risk={ph.get('risk_level', '?').upper()}, "
                    f"score={ph.get('overall_phishing_score', 0)*100:.0f}%, "
                    f"{ph.get('manipulative_count', 0)} manipulative sentence(s), "
                    f"{ph.get('heuristic_flag_count', 0)} heuristic flag(s). "
                    f"{ph.get('summary', '')}"
                )
                if ph.get("heuristic_summary"):
                    sections.append(f"HEURISTIC FLAGS: {', '.join(ph['heuristic_summary'])}")

        contact_names = request.contact_names or {}
        if contact_names:
            sections.append(f"CONTACT NAMES: {', '.join(f'{k}={v}' for k, v in contact_names.items())}")
        context = "\n".join(sections)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an email security analyst. You write concise bullet-point security reports. "
                    "You always interpret findings meaningfully rather than restating raw numbers. "
                    "You never add commentary, reasoning, or text after your final bullet point."
                )
            },
            {
                "role": "user",
                "content": (
                    "Below are the results of a 7-stage authorship and phishing detection pipeline.\n\n"
                    "Write a security report using 4–7 bullet points. Each bullet must interpret what the "
                    "finding implies — not just restate numbers. Use plain English. "
                    "When referring to contacts, always use their name from the CONTACT NAMES mapping, never their ID.\n\n"
                    "Format each bullet exactly like:\n"
                    "• [Category] Interpretation of the finding.\n\n"
                    "Valid categories: Profile Match, Writing Style, Anomalies, Tone, Phishing\n\n"
                    "Output only the bullet points. Stop after the last bullet. No commentary after.\n\n"
                    f"{context}"
                )
            }
        ]
        if not deps.tokenizer or not deps.model:
            raise HTTPException(status_code=503, detail="Qwen model/tokenizer not available")
        # Apply Qwen3 chat template — non-thinking mode for clean output
        text = deps.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,  # disables <think>...</think> blocks entirely
        )

        inputs = deps.tokenizer(text, return_tensors="pt", truncation=True, max_length=2048)
        inputs = {k: v.to(deps.model.device) for k, v in inputs.items()}

        # Official Qwen3 non-thinking mode sampling params (from model card)
        outputs = deps.model.generate(
            **inputs,
            max_new_tokens=400,
            do_sample=True,
            temperature=0.7,
            top_p=0.8,
            top_k=20,
            min_p=0.0,
            repetition_penalty=1.1,
            pad_token_id=deps.tokenizer.eos_token_id,
        )

        # Decode only new tokens
        input_len = inputs["input_ids"].shape[1]
        summary_text = deps.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True).strip()

        # Strip any post-bullet rambling — keep only bullet lines
        lines = summary_text.splitlines()
        bullet_lines = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith("•"):
                bullet_lines.append(stripped)
            else:
                break  # first non-bullet line = model went off-script, stop here

        summary_text = "\n".join(bullet_lines)

        return SummaryResponse(summary=summary_text)

    except Exception as e:
        logger.error(f"Summary generation error: {e}\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Summary generation failed: {e}")