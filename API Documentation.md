# Stylometric Feature Extraction & Authorship Detection API

**Version:** 3.0.0

Extract multilingual stylometric features and predict authorship using pure stylometry. This API provides a comprehensive pipeline for email analysis, including feature extraction, author prediction, outlier detection, phishing analysis, and visualisation.

---

## Base URL

All endpoints are relative to your deployed API base URL.

---

## Endpoints

### Health & Status

#### `GET /`

Returns the current health status of the API.

**Response** (`200 OK`):

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | Current service status |
| `version` | string | API version |
| `models_loaded` | boolean | Whether ML models are loaded and ready |

#### `GET /health`

Identical to the root endpoint. Provides a dedicated health-check route for monitoring and load balancers.

---

### Feature Extraction

#### `POST /extract-features`

Extract stylometric features from a single piece of email content. No next-token prediction (NTP) is performed.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | — | Email content to analyse (minimum 1 character) |
| `language` | string \| null | No | `"auto"` | Language code (`en`, `es`, `ru`, `sv`) or `"auto"` for automatic detection |
| `include_raw_features` | boolean | No | `true` | Include the raw feature dictionary in the response |

**Response** (`200 OK`):

| Field | Type | Description |
|-------|------|-------------|
| `features` | object \| null | Dictionary of feature names to numeric values (if `include_raw_features` is `true`) |
| `feature_vector` | number[] | Numeric feature vector for the input |
| `feature_names` | string[] | Ordered list of feature names corresponding to the vector |
| `dimension` | integer | Dimensionality of the feature vector |
| `detected_language` | string | Language detected or confirmed |
| `text_length` | integer | Character length of the input text |

**Example Request:**

```json
{
  "content": "Dear colleague, I wanted to follow up on our earlier discussion regarding the quarterly figures.",
  "language": "en",
  "include_raw_features": true
}
```

---

#### `POST /extract-features/batch`

Extract features from multiple emails in a single request (up to 100).

**Request Body:**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `emails` | EmailInput[] | Yes | Array of email objects (maximum 100 items) |

Each item in `emails` follows the same schema as the single `/extract-features` request body.

**Response** (`200 OK`):

| Field | Type | Description |
|-------|------|-------------|
| `results` | FeatureResponse[] | Array of feature extraction results |
| `total_processed` | integer | Number of emails successfully processed |

---

### Author Prediction

#### `POST /predict-author`

Predict the author of an email using pure stylometry.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | — | Email content to analyse (minimum 1 character) |
| `language` | string | No | `"auto"` | Language code or `"auto"` |
| `suspected_author_id` | string \| null | No | `null` | Optional author ID to compare against |

**Response** (`200 OK`):

| Field | Type | Description |
|-------|------|-------------|
| `predicted_author` | string | Identifier of the predicted author |
| `contact_info` | object \| null | Contact details for the predicted author (if available) |
| `confidence` | number | Confidence score for the prediction |
| `all_probabilities` | object | Map of author IDs to probability scores |
| `weighted_score` | number \| null | Weighted confidence score |
| `is_anomaly` | boolean | Whether the email is flagged as anomalous (no strong author match) |
| `anomaly_score` | number | Numeric anomaly score |
| `top_features` | object[] | Most influential features driving the prediction |
| `explanation` | FeatureExplanation[] \| null | Detailed per-feature breakdown |
| `method` | string | Classification method used |
| `message` | string | Human-readable summary of the result |

**FeatureExplanation Object:**

| Field | Type | Description |
|-------|------|-------------|
| `feature` | string | Feature name |
| `value` | number | Feature value in the analysed email |
| `author_typical` | number | Typical value for the predicted author |
| `closeness` | number | How close the value is to the author's norm |
| `importance_for_author` | number | Weight of this feature for the predicted author |
| `match_score` | number | Combined match score |

**Example Request:**

```json
{
  "content": "Hi team, just checking in on the deliverables. Let me know if there are any blockers.",
  "language": "auto",
  "suspected_author_id": "contact_042"
}
```

---

### Profile Comparison

#### `POST /compare-to-profile`

Compare an email's stylometric features against a stored contact profile.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | — | Email content to analyse (minimum 1 character) |
| `contact_id` | string | Yes | — | Identifier of the contact profile to compare against |
| `language` | string | No | `"auto"` | Language code or `"auto"` |
| `ntp_stats` | object \| null | No | `null` | Optional NTP statistics for baseline comparison |

**Response** (`200 OK`):

| Field | Type | Description |
|-------|------|-------------|
| `contact_id` | string | The contact ID that was compared |
| `contact_name` | string | Display name of the contact |
| `profile_sample_count` | integer | Number of samples in the stored profile |
| `total_features` | integer | Total features compared |
| `unusual_count` | integer | Number of features flagged as unusual |
| `unusual_features` | object[] | Details of each unusual feature |
| `overall_deviation` | number | Aggregate deviation score |
| `match_percentage` | number | Overall match percentage (0–100) |
| `summary` | string | Human-readable summary |
| `ntp_baseline_comparison` | object \| null | NTP baseline comparison results (if `ntp_stats` provided) |

---

### Outlier Detection

#### `POST /semantic-outlier-detection`

Perform leave-one-out semantic analysis to identify sentences that are semantically inconsistent with the rest of the email.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | — | Email content to analyse (minimum 1 character) |
| `min_sentences` | integer | No | `5` | Minimum number of sentences required (must be ≥ 3) |
| `language` | string | No | `"auto"` | Language code or `"auto"` |

**Response** (`200 OK`):

| Field | Type | Description |
|-------|------|-------------|
| `total_sentences` | integer | Total sentences found in the content |
| `outlier_count` | integer | Number of outlier sentences detected |
| `threshold` | number | Threshold used for outlier detection |
| `sentences` | object[] | Per-sentence analysis results |
| `skipped` | boolean | Whether analysis was skipped (e.g. too few sentences) |
| `skip_reason` | string \| null | Reason for skipping, if applicable |
| `summary` | string | Human-readable summary |

---

#### `POST /tonal-outlier-detection`

Analyse tonal consistency across sentences using Jensen–Shannon divergence (JSD).

**Request Body:** Same as `/semantic-outlier-detection`.

**Response** (`200 OK`):

| Field | Type | Description |
|-------|------|-------------|
| `total_sentences` | integer | Total sentences found |
| `anomaly_count` | integer | Number of tonal anomalies detected |
| `modal_tone` | string | The predominant tone of the email |
| `sentences` | object[] | Per-sentence tonal analysis |
| `skipped` | boolean | Whether analysis was skipped |
| `skip_reason` | string \| null | Reason for skipping, if applicable |
| `summary` | string | Human-readable summary |

---

### Phishing Detection

#### `POST /phishing-detection`

Detect phishing and social engineering tactics using persuasion NLI and heuristic analysis.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | — | Email content to analyse |
| `language` | string | No | `"auto"` | Language code or `"auto"` |
| `min_sentences` | integer | No | `3` | Minimum sentences required for analysis |

**Response** (`200 OK`):

| Field | Type | Description |
|-------|------|-------------|
| `total_sentences` | integer | Total sentences analysed |
| `manipulative_count` | integer | Number of sentences flagged as manipulative |
| `heuristic_flag_count` | integer | Number of heuristic-based flags |
| `overall_phishing_score` | number | Aggregate phishing risk score |
| `risk_level` | string | Risk classification (e.g. `"low"`, `"medium"`, `"high"`) |
| `sentences` | PhishingSentenceResult[] | Per-sentence breakdown |
| `heuristic_summary` | string[] | List of triggered heuristic rules |
| `skipped` | boolean | Whether analysis was skipped |
| `skip_reason` | string \| null | Reason for skipping, if applicable |
| `summary` | string | Human-readable summary |

**PhishingSentenceResult Object:**

| Field | Type | Description |
|-------|------|-------------|
| `index` | integer | Sentence index |
| `sentence` | string | The sentence text |
| `persuasion_scores` | object[] | Scores for each persuasion tactic |
| `dominant_tactic` | string | The most prominent persuasion tactic |
| `persuasion_z_score` | number | Z-score indicating persuasion intensity |
| `is_manipulative` | boolean | Whether the sentence is classified as manipulative |
| `heuristic_flags` | string[] | Heuristic rules triggered by this sentence |
| `combined_phishing_score` | number | Combined risk score for this sentence |

---

### Visualisation

#### `POST /initialize-visualization`

Initialise the 2D projection model used for author cluster visualisation.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `method` | string | No | `"tsne"` | Projection method (e.g. `"tsne"`) |

**Response** (`200 OK`): Confirmation of initialisation.

---

#### `GET /visualization-data`

Retrieve pre-computed 2D visualisation data for all known authors.

**Response** (`200 OK`): Visualisation data object containing author clusters and sample positions.

---

#### `POST /visualize-prediction`

Predict the author of an email and return 2D visualisation data showing where the email sits relative to known author clusters.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | — | Email content to analyse |
| `language` | string | No | `"auto"` | Language code or `"auto"` |

**Response** (`200 OK`):

| Field | Type | Description |
|-------|------|-------------|
| `new_email_position` | object | `{ x, y }` coordinates of the email in 2D space |
| `predicted_author` | string | Predicted author identifier |
| `confidence` | number | Prediction confidence |
| `decision` | string | Classification decision label |
| `author_clusters` | object | Per-author cluster centroids and metadata |
| `training_samples` | object[] | Positions of training samples |
| `projection_method` | string | Projection method used |
| `explanation` | string | Human-readable explanation of the result |

---

#### `POST /visualize-ntp-anomalies`

Generate a visualisation of next-token prediction anomalies within the email text.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | — | Email content to analyse |
| `max_length` | integer | No | `512` | Maximum token length to process |
| `language` | string | No | `"auto"` | Language code or `"auto"` |

**Response** (`200 OK`): Visualisation metadata including the filename of the generated image.

---

#### `GET /ntp-visualization/{filename}`

Serve a previously generated NTP visualisation image.

**Path Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `filename` | string | Filename returned by `/visualize-ntp-anomalies` |

**Response** (`200 OK`): The visualisation image file.

---

### Summary

#### `POST /summarize-analysis`

Generate a human-readable summary of all pipeline stages, powered by the Qwen language model.

**Request Body:**

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `content` | string | Yes | — | Original email content |
| `claimed_sender` | string | No | `""` | The purported sender of the email |
| `prediction` | object \| null | No | `null` | Output from `/predict-author` |
| `profile_comparison` | object \| null | No | `null` | Output from `/compare-to-profile` |
| `ntp` | object \| null | No | `null` | NTP analysis results |
| `semantic_outliers` | object \| null | No | `null` | Output from `/semantic-outlier-detection` |
| `tonal_outliers` | object \| null | No | `null` | Output from `/tonal-outlier-detection` |
| `phishing` | object \| null | No | `null` | Output from `/phishing-detection` |
| `contact_names` | object \| null | No | `null` | Mapping of contact IDs to display names |

**Response** (`200 OK`):

| Field | Type | Description |
|-------|------|-------------|
| `summary` | string | Complete human-readable analysis summary |

---

### Model & Feature Information

#### `GET /model-info`

Retrieve metadata about the currently loaded ML models.

#### `GET /feature-info`

Retrieve descriptions and metadata for all supported stylometric features.

---

## Error Handling

All endpoints return a `422 Unprocessable Entity` response for validation errors, with the following structure:

| Field | Type | Description |
|-------|------|-------------|
| `detail` | ValidationError[] | Array of validation error objects |

**ValidationError Object:**

| Field | Type | Description |
|-------|------|-------------|
| `loc` | (string \| integer)[] | Path to the field that caused the error |
| `msg` | string | Human-readable error message |
| `type` | string | Error type identifier |

**Example:**

```json
{
  "detail": [
    {
      "loc": ["body", "content"],
      "msg": "String should have at least 1 character",
      "type": "string_too_short"
    }
  ]
}
```

---

## Supported Languages

The API currently supports the following language codes:

| Code | Language |
|------|----------|
| `en` | English |
| `es` | Spanish |
| `ru` | Russian |
| `sv` | Swedish |
| `auto` | Automatic detection (default) |