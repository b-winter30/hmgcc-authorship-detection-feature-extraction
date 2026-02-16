"""
Next-Token Prediction Anomaly Detection (Stage 2)
Detects token-level anomalies using language model predictions
WITH HYBRID CONTEXT WEIGHTING (Position + TF-IDF Novelty)
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Optional, Tuple
from colorama import Fore, Style
import logging

logger = logging.getLogger(__name__)


class NextTokenAnomalyDetector:
    """
    Detect token-level anomalies using next-token prediction
    WITH HYBRID CONTEXT WEIGHTING (Position + TF-IDF Novelty)
    
    This is Stage 2 anomaly detection - operates at the token level
    to detect unusual word choices, potential name substitutions, etc.
    
    Context weighting ensures that anomalies detected with more context
    (later in the text) are weighted more heavily than anomalies detected
    with minimal context (early in the text).
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        anomaly_threshold: float = 0.05,
        use_flash_attention: bool = False,
        cache_dir: Optional[str] = None,
        use_context_weighting: bool = True,
        context_weighting_mode: str = "hybrid"  # "position", "tfidf", or "hybrid"
    ):
        """
        Initialize the next-token anomaly detector
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cuda' or 'cpu'
            anomaly_threshold: Probability threshold for flagging anomalies
            use_flash_attention: Use Flash Attention 2 if available
            cache_dir: Directory to cache models
            use_context_weighting: Enable context-aware weighting (default: True)
            context_weighting_mode: "position" (position-based), "tfidf" (novelty-based), 
                                   or "hybrid" (both combined)
        """
        self.device = device
        self.anomaly_threshold = anomaly_threshold
        self.model_name = model_name
        self.use_context_weighting = use_context_weighting
        self.context_weighting_mode = context_weighting_mode
        
        logger.info(f"\n📦 Loading {model_name}...")
        logger.info("   This may take a minute on first run (downloading model)...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True,
            cache_dir=cache_dir
        )
        
        # Load model with optimizations
        model_kwargs = {
            "trust_remote_code": True,
            "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
        }
        
        if use_flash_attention and device == "cuda":
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("   Using Flash Attention 2 for faster inference")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **model_kwargs
        ).to(device)
        
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"✅ Model loaded on {device}")
        n_params = sum(p.numel() for p in self.model.parameters()) / 1e6
        logger.info(f"   Model size: ~{n_params:.0f}M parameters")
        
        if device == "cuda":
            mem_allocated = torch.cuda.memory_allocated() / 1e9
            logger.info(f"   Memory usage: ~{mem_allocated:.2f}GB")
        
        if use_context_weighting:
            logger.info(f"   Context weighting: ENABLED ({context_weighting_mode} mode)")
            if context_weighting_mode in ["position", "hybrid"]:
                logger.info(f"     Position-based: peaks at 50% of text length")
            if context_weighting_mode in ["tfidf", "hybrid"]:
                logger.info(f"     TF-IDF novelty: weights unseen/rare tokens higher")
    
    def calculate_positional_weight(self, position: int, total_tokens: int) -> float:
        """
        Calculate position-based context weight
        
        Peaks at 50% of text length (middle of document)
        
        Logic:
        - 0% to 25%: weight 0.0 → 0.5 (building context)
        - 25% to 50%: weight 0.5 → 1.0 (peak context)
        - 50% to 100%: weight 1.0 (maintain full context)
        """
        if not self.use_context_weighting:
            return 1.0
        
        # Calculate percentage through text
        pct = position / total_tokens if total_tokens > 0 else 0.0
        
        if pct < 0.25:
            # First quarter: linear ramp 0.0 → 0.5
            return pct / 0.25 * 0.5
        elif pct < 0.50:
            # Second quarter: linear ramp 0.5 → 1.0
            return 0.5 + ((pct - 0.25) / 0.25) * 0.5
        else:
            # Second half: full context
            return 1.0
    
    def calculate_tfidf_novelty_weight(
        self, 
        token: str, 
        token_freq: Dict[str, int],
        total_tokens_seen: int,
        is_first_occurrence: bool
    ) -> float:
        """
        Calculate TF-IDF based novelty weight
        
        Higher weight for:
        - Tokens never seen before (first occurrence)
        - Rare tokens in the document
        - Important/content words
        
        Args:
            token: The decoded token string
            token_freq: Dictionary of token -> frequency in document so far
            total_tokens_seen: Total tokens processed so far
            is_first_occurrence: Whether this is the first time we've seen this token
        
        Returns:
            Weight from 0.5 (common/repeated) to 1.0 (novel/rare)
        """
        if not self.use_context_weighting:
            return 1.0
        
        # Clean token
        token_clean = token.strip().lower()
        
        if not token_clean:
            return 0.5  # Empty tokens get low weight
        
        # Get current frequency
        freq = token_freq.get(token_clean, 0)
        
        # Calculate term frequency (TF)
        # More frequent = lower novelty
        if total_tokens_seen > 0:
            tf = freq / total_tokens_seen
        else:
            tf = 0.0
        
        # Base weight calculation
        if is_first_occurrence:
            # First occurrence: HIGH novelty
            base_weight = 1.0
        elif freq == 1:
            # Second occurrence: still novel
            base_weight = 0.95
        elif freq <= 3:
            # Rare (2-3 occurrences)
            base_weight = 0.85
        elif freq <= 5:
            # Uncommon (4-5 occurrences)
            base_weight = 0.75
        else:
            # Common (6+ occurrences)
            # Exponential decay
            base_weight = max(0.5, 0.75 * np.exp(-0.1 * (freq - 5)))
        
        # Boost for capitalized words (likely names/entities)
        if token.strip() and token.strip()[0].isupper() and len(token.strip()) > 1:
            base_weight = min(1.0, base_weight * 1.1)
        
        # Penalize very short tokens (likely function words)
        if len(token_clean) <= 2:
            base_weight *= 0.8
        
        return base_weight
    
    def calculate_hybrid_weight(
        self,
        position: int,
        total_tokens: int,
        token: str,
        token_freq: Dict[str, int],
        total_tokens_seen: int,
        is_first_occurrence: bool
    ) -> float:
        """
        Calculate hybrid context weight combining position and TF-IDF novelty
        
        Formula:
        hybrid_weight = positional_weight * novelty_weight
        
        This means:
        - Early in text + common word = very low weight
        - Early in text + novel word = low-medium weight
        - Middle of text + common word = medium weight
        - Middle of text + novel word = HIGH weight (best signal!)
        """
        if self.context_weighting_mode == "position":
            return self.calculate_positional_weight(position, total_tokens)
        
        elif self.context_weighting_mode == "tfidf":
            return self.calculate_tfidf_novelty_weight(
                token, token_freq, total_tokens_seen, is_first_occurrence
            )
        
        elif self.context_weighting_mode == "hybrid":
            pos_weight = self.calculate_positional_weight(position, total_tokens)
            novelty_weight = self.calculate_tfidf_novelty_weight(
                token, token_freq, total_tokens_seen, is_first_occurrence
            )
            
            # Combine: both must be high for maximum weight
            return pos_weight * novelty_weight
        
        else:
            return 1.0
    
    def extract_anomaly_features(
        self,
        text: str,
        max_length: int = 512,
        return_detailed: bool = False
    ) -> Dict[str, float]:
        """
        Extract anomaly detection features from text
        
        This is the main method called by feature_extractor.py
        
        Args:
            text: Input text to analyze
            max_length: Maximum sequence length
            return_detailed: Return detailed token-level analysis
        
        Returns:
            Dictionary of anomaly features suitable for ML pipeline
        """
        # Tokenize with truncation
        tokens = self.tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self.device)
        
        if len(tokens[0]) < 2:
            return self._empty_features()
        
        # Analyze tokens
        results = self._analyze_tokens(tokens)
        
        # Extract aggregate features
        features = self._compute_aggregate_features(results)
        
        if return_detailed:
            features['_detailed_results'] = results
        
        return features
    
    def _analyze_tokens(self, tokens: torch.Tensor) -> List[Dict]:
        """
        Analyze each token in the sequence WITH HYBRID CONTEXT WEIGHTING
        
        Returns:
            List of per-token analysis results
        """
        results = []
        total_tokens = len(tokens[0])
        
        # Track token frequencies for TF-IDF
        token_freq = {}
        total_tokens_seen = 0
        
        with torch.no_grad():
            for i in range(1, total_tokens):
                # Get context
                context = tokens[:, :i]
                
                # Predict next token
                outputs = self.model(context)
                next_token_logits = outputs.logits[0, -1, :]
                probs = torch.softmax(next_token_logits, dim=-1)
                
                # Get actual token
                actual_token_id = tokens[0, i].item()
                actual_token_prob = probs[actual_token_id].item()
                
                # Get top predictions
                top_k_probs, top_k_indices = torch.topk(probs, k=10)
                
                # Decode token
                try:
                    actual_token = self.tokenizer.decode([actual_token_id], skip_special_tokens=False)
                    actual_token = actual_token.strip() if actual_token.strip() else actual_token
                except Exception as e:
                    logger.warning(f"Token decode error at position {i}: {e}")
                    actual_token = f"<TOKEN_{actual_token_id}>"
                
                # Update token frequency tracking
                token_clean = actual_token.strip().lower()
                is_first_occurrence = token_clean not in token_freq
                
                if token_clean:
                    token_freq[token_clean] = token_freq.get(token_clean, 0) + 1
                    total_tokens_seen += 1
                
                # Calculate hybrid context weight
                context_weight = self.calculate_hybrid_weight(
                    position=i,
                    total_tokens=total_tokens,
                    token=actual_token,
                    token_freq=token_freq,
                    total_tokens_seen=total_tokens_seen,
                    is_first_occurrence=is_first_occurrence
                )
                
                # Also calculate individual components for analysis
                positional_weight = self.calculate_positional_weight(i, total_tokens)
                novelty_weight = self.calculate_tfidf_novelty_weight(
                    actual_token, token_freq, total_tokens_seen, is_first_occurrence
                )
                
                # Calculate rank of actual token
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                actual_rank = (sorted_indices == actual_token_id).nonzero(as_tuple=True)[0].item() + 1
                
                # Calculate anomaly scores
                base_anomaly_score = 1.0 - actual_token_prob
                weighted_anomaly_score = base_anomaly_score * context_weight
                
                # Determine if anomaly using weighted score
                is_anomaly = weighted_anomaly_score > (1.0 - self.anomaly_threshold)
                
                # Check if token contains emoji
                contains_emoji = self._contains_emoji(actual_token)
                
                results.append({
                    'position': i,
                    'token': actual_token,
                    'probability': actual_token_prob,
                    'rank': actual_rank,
                    'anomaly_score': base_anomaly_score,
                    'weighted_anomaly_score': weighted_anomaly_score,
                    'context_weight': context_weight,
                    'positional_weight': positional_weight,
                    'novelty_weight': novelty_weight,
                    'is_first_occurrence': is_first_occurrence,
                    'token_frequency': token_freq.get(token_clean, 0),
                    'is_anomaly': is_anomaly,
                    'is_capitalized': actual_token.strip() and actual_token.strip()[0].isupper(),
                    'token_length': len(actual_token.strip()),
                    'contains_emoji': contains_emoji
                })
        
        return results
    
    def _contains_emoji(self, text: str) -> bool:
        """Check if text contains emoji"""
        try:
            import emoji
            return emoji.emoji_count(text) > 0
        except ImportError:
            # Fallback: check Unicode ranges for common emoji blocks
            emoji_ranges = [
                (0x1F600, 0x1F64F),  # Emoticons
                (0x1F300, 0x1F5FF),  # Misc Symbols and Pictographs
                (0x1F680, 0x1F6FF),  # Transport and Map
                (0x1F1E0, 0x1F1FF),  # Flags
                (0x2600, 0x26FF),    # Misc symbols
                (0x2700, 0x27BF),    # Dingbats
                (0xFE00, 0xFE0F),    # Variation Selectors
                (0x1F900, 0x1F9FF),  # Supplemental Symbols and Pictographs
                (0x1FA00, 0x1FA6F),  # Chess Symbols
                (0x1FA70, 0x1FAFF),  # Symbols and Pictographs Extended-A
            ]
            
            for char in text:
                code_point = ord(char)
                for start, end in emoji_ranges:
                    if start <= code_point <= end:
                        return True
            return False
    
    def _compute_aggregate_features(self, results: List[Dict]) -> Dict[str, float]:
        """
        Compute aggregate features from token-level analysis WITH HYBRID WEIGHTING
        
        These features are designed to integrate into the ML pipeline
        """
        if not results:
            return self._empty_features()
        
        probs = [r['probability'] for r in results]
        anomaly_scores = [r['anomaly_score'] for r in results]
        weighted_anomaly_scores = [r['weighted_anomaly_score'] for r in results]
        context_weights = [r['context_weight'] for r in results]
        positional_weights = [r['positional_weight'] for r in results]
        novelty_weights = [r['novelty_weight'] for r in results]
        ranks = [r['rank'] for r in results]
        
        # Count anomalies (using weighted scores)
        anomalies = [r for r in results if r['is_anomaly']]
        capitalized_anomalies = [r for r in anomalies if r['is_capitalized']]
        first_occurrence_anomalies = [r for r in anomalies if r['is_first_occurrence']]
        
        features = {
            # Core statistics
            'ntp_mean_probability': np.mean(probs),
            'ntp_median_probability': np.median(probs),
            'ntp_min_probability': np.min(probs),
            'ntp_std_probability': np.std(probs),
            
            # Anomaly counts (now using weighted scores)
            'ntp_anomaly_ratio': len(anomalies) / len(results),
            'ntp_anomaly_count': len(anomalies),
            'ntp_capitalized_anomaly_ratio': len(capitalized_anomalies) / len(results),
            'ntp_first_occurrence_anomaly_ratio': len(first_occurrence_anomalies) / len(results),
            
            # Rank statistics
            'ntp_mean_rank': np.mean(ranks),
            'ntp_median_rank': np.median(ranks),
            'ntp_max_rank': np.max(ranks),
            
            # Anomaly score statistics - both raw and weighted
            'ntp_mean_anomaly_score': np.mean(anomaly_scores),
            'ntp_max_anomaly_score': np.max(anomaly_scores),
            'ntp_mean_weighted_anomaly_score': np.mean(weighted_anomaly_scores),
            'ntp_max_weighted_anomaly_score': np.max(weighted_anomaly_scores),
            
            # Context weight statistics
            'ntp_mean_context_weight': np.mean(context_weights),
            'ntp_min_context_weight': np.min(context_weights),
            'ntp_max_context_weight': np.max(context_weights),
            
            # Positional weight statistics
            'ntp_mean_positional_weight': np.mean(positional_weights),
            
            # Novelty weight statistics
            'ntp_mean_novelty_weight': np.mean(novelty_weights),
            'ntp_min_novelty_weight': np.min(novelty_weights),
            
            # Low probability token density
            'ntp_very_low_prob_ratio': sum(1 for p in probs if p < 0.01) / len(probs),
            'ntp_low_prob_ratio': sum(1 for p in probs if p < 0.05) / len(probs),
            
            # High rank token density
            'ntp_high_rank_ratio': sum(1 for r in ranks if r > 100) / len(ranks),
            'ntp_very_high_rank_ratio': sum(1 for r in ranks if r > 1000) / len(ranks),
        }
        
        # Context-aware features - split by POSITIONAL context quality
        first_quarter = [r for r in results if r['position'] / len(results) < 0.25]
        second_quarter = [r for r in results if 0.25 <= r['position'] / len(results) < 0.5]
        second_half = [r for r in results if r['position'] / len(results) >= 0.5]
        
        # Statistics for each positional segment
        for name, subset in [
            ('first_quarter', first_quarter),
            ('second_quarter', second_quarter),
            ('second_half', second_half)
        ]:
            if subset:
                subset_probs = [r['probability'] for r in subset]
                subset_anomalies = [r for r in subset if r['is_anomaly']]
                features[f'ntp_{name}_mean_prob'] = np.mean(subset_probs)
                features[f'ntp_{name}_anomaly_ratio'] = len(subset_anomalies) / len(subset)
            else:
                features[f'ntp_{name}_mean_prob'] = 1.0
                features[f'ntp_{name}_anomaly_ratio'] = 0.0
        
        # High-weight anomaly features (most reliable indicators)
        # Tokens with BOTH high positional weight AND high novelty
        high_weight_tokens = [r for r in results if r['context_weight'] >= 0.7]
        if high_weight_tokens:
            high_weight_anomalies = [r for r in high_weight_tokens if r['is_anomaly']]
            features['ntp_high_weight_anomaly_ratio'] = len(high_weight_anomalies) / len(high_weight_tokens)
            features['ntp_high_weight_mean_weighted_score'] = np.mean([r['weighted_anomaly_score'] for r in high_weight_tokens])
        else:
            features['ntp_high_weight_anomaly_ratio'] = 0.0
            features['ntp_high_weight_mean_weighted_score'] = 0.0
        
        # Novel token statistics (first occurrences)
        novel_tokens = [r for r in results if r['is_first_occurrence']]
        if novel_tokens:
            novel_anomalies = [r for r in novel_tokens if r['is_anomaly']]
            features['ntp_novel_token_anomaly_ratio'] = len(novel_anomalies) / len(novel_tokens)
            features['ntp_novel_token_mean_prob'] = np.mean([r['probability'] for r in novel_tokens])
        else:
            features['ntp_novel_token_anomaly_ratio'] = 0.0
            features['ntp_novel_token_mean_prob'] = 1.0
        
        return features
    
    def _empty_features(self) -> Dict[str, float]:
        """Return empty/default features when text is too short"""
        base_features = {
            'ntp_mean_probability': 1.0,
            'ntp_median_probability': 1.0,
            'ntp_min_probability': 1.0,
            'ntp_std_probability': 0.0,
            'ntp_anomaly_ratio': 0.0,
            'ntp_anomaly_count': 0.0,
            'ntp_capitalized_anomaly_ratio': 0.0,
            'ntp_first_occurrence_anomaly_ratio': 0.0,
            'ntp_mean_rank': 1.0,
            'ntp_median_rank': 1.0,
            'ntp_max_rank': 1.0,
            'ntp_mean_anomaly_score': 0.0,
            'ntp_max_anomaly_score': 0.0,
            'ntp_mean_weighted_anomaly_score': 0.0,
            'ntp_max_weighted_anomaly_score': 0.0,
            'ntp_mean_context_weight': 0.0,
            'ntp_min_context_weight': 0.0,
            'ntp_max_context_weight': 0.0,
            'ntp_mean_positional_weight': 0.0,
            'ntp_mean_novelty_weight': 0.0,
            'ntp_min_novelty_weight': 0.0,
            'ntp_very_low_prob_ratio': 0.0,
            'ntp_low_prob_ratio': 0.0,
            'ntp_high_rank_ratio': 0.0,
            'ntp_very_high_rank_ratio': 0.0,
        }
        
        # Add positional segment features
        for name in ['first_quarter', 'second_quarter', 'second_half']:
            base_features[f'ntp_{name}_mean_prob'] = 1.0
            base_features[f'ntp_{name}_anomaly_ratio'] = 0.0
        
        base_features['ntp_high_weight_anomaly_ratio'] = 0.0
        base_features['ntp_high_weight_mean_weighted_score'] = 0.0
        base_features['ntp_novel_token_anomaly_ratio'] = 0.0
        base_features['ntp_novel_token_mean_prob'] = 1.0
        
        return base_features
    
    def visualize_anomalies(
        self,
        text: str,
        max_length: int = 512
    ) -> None:
        """
        Print colored visualization of token-level anomalies
        
        Useful for debugging and understanding what the model detects
        """
        # Tokenize
        tokens = self.tokenizer.encode(
            text,
            return_tensors="pt",
            max_length=max_length,
            truncation=True
        ).to(self.device)
        
        if len(tokens[0]) < 2:
            print("❌ Text too short for analysis")
            return
        
        results = self._analyze_tokens(tokens)
        
        print("\n" + "="*80)
        print("NEXT-TOKEN PREDICTION ANOMALY DETECTION (HYBRID CONTEXT-WEIGHTED)")
        print("="*80)
        
        print("\nColor Legend:")
        print(f"{Fore.GREEN}Green{Style.RESET_ALL}: High confidence (p > 0.5)")
        print(f"{Fore.YELLOW}Yellow{Style.RESET_ALL}: Medium confidence (0.1 < p < 0.5)")
        print(f"{Fore.RED}Red{Style.RESET_ALL}: Low confidence / ANOMALY (p < 0.1)")
        
        if self.use_context_weighting:
            print(f"\n{Fore.CYAN}Context Weighting: {self.context_weighting_mode.upper()}{Style.RESET_ALL}")
            print("  Anomalies with high positional + novelty weight are most reliable")
        
        print("\n" + "-"*80)
        
        # Print tokens with color coding
        line_length = 0
        for r in results:
            prob = r['probability']
            token = r['token']
            context_weight = r['context_weight']
            
            # Color based on probability
            if prob > 0.5:
                color = Fore.GREEN
            elif prob > 0.1:
                color = Fore.YELLOW
            else:
                color = Fore.RED
            
            # Add context weight indicator for anomalies
            if r['is_anomaly'] and self.use_context_weighting:
                weight_indicator = f"[cw:{context_weight:.2f}]"
                print(f"{color}{token}{weight_indicator}{Style.RESET_ALL}", end="")
            else:
                print(f"{color}{token}{Style.RESET_ALL}", end="")
            
            # Word wrap at 80 chars
            line_length += len(token)
            if line_length > 80:
                print()
                line_length = 0
        
        print("\n" + "-"*80)
        
        # Print flagged tokens with context awareness
        flagged = [r for r in results if r['is_anomaly']]
        if flagged:
            print(f"\n🚨 {len(flagged)} ANOMALOUS TOKENS DETECTED (hybrid context-weighted):")
            for r in flagged:
                print(f"\n  Position {r['position']}: '{r['token'].strip()}'")
                print(f"    Probability: {r['probability']:.6f} (Rank: #{r['rank']})")
                print(f"    Base anomaly score: {r['anomaly_score']:.4f}")
                if self.use_context_weighting:
                    print(f"    Positional weight: {r['positional_weight']:.4f}")
                    print(f"    Novelty weight: {r['novelty_weight']:.4f}")
                    print(f"    Combined context weight: {r['context_weight']:.4f}")
                    print(f"    Weighted anomaly score: {r['weighted_anomaly_score']:.4f}")
                    if r['is_first_occurrence']:
                        print(f"    ⭐ First occurrence (novel token!)")
        else:
            print("\n✅ No anomalies detected")
        
        # Statistics
        probs = [r['probability'] for r in results]
        weighted_scores = [r['weighted_anomaly_score'] for r in results]
        context_weights = [r['context_weight'] for r in results]
        novelty_weights = [r['novelty_weight'] for r in results]
        
        print(f"\n📊 Statistics:")
        print(f"   Mean probability: {np.mean(probs):.4f}")
        print(f"   Median probability: {np.median(probs):.4f}")
        print(f"   Min probability: {np.min(probs):.4f}")
        print(f"   Tokens flagged: {len(flagged)} / {len(results)} ({len(flagged)/len(results)*100:.1f}%)")
        
        if self.use_context_weighting:
            print(f"\n   Hybrid context weighting:")
            print(f"     Mean context weight: {np.mean(context_weights):.4f}")
            print(f"     Mean novelty weight: {np.mean(novelty_weights):.4f}")
            print(f"     Mean weighted anomaly score: {np.mean(weighted_scores):.4f}")
            print(f"     Max weighted anomaly score: {np.max(weighted_scores):.4f}")


def get_ntp_feature_names() -> List[str]:
    """
    Get list of feature names produced by NextTokenAnomalyDetector
    WITH HYBRID CONTEXT WEIGHTING
    
    Useful for feature engineering pipelines
    """
    return [
        # Core features
        'ntp_mean_probability',
        'ntp_median_probability',
        'ntp_min_probability',
        'ntp_std_probability',
        'ntp_anomaly_ratio',
        'ntp_anomaly_count',
        'ntp_capitalized_anomaly_ratio',
        'ntp_first_occurrence_anomaly_ratio',
        'ntp_mean_rank',
        'ntp_median_rank',
        'ntp_max_rank',
        # Anomaly scores
        'ntp_mean_anomaly_score',
        'ntp_max_anomaly_score',
        'ntp_mean_weighted_anomaly_score',
        'ntp_max_weighted_anomaly_score',
        # Context weights
        'ntp_mean_context_weight',
        'ntp_min_context_weight',
        'ntp_max_context_weight',
        'ntp_mean_positional_weight',
        'ntp_mean_novelty_weight',
        'ntp_min_novelty_weight',
        # Density features
        'ntp_very_low_prob_ratio',
        'ntp_low_prob_ratio',
        'ntp_high_rank_ratio',
        'ntp_very_high_rank_ratio',
        # Positional segment features
        'ntp_first_quarter_mean_prob',
        'ntp_first_quarter_anomaly_ratio',
        'ntp_second_quarter_mean_prob',
        'ntp_second_quarter_anomaly_ratio',
        'ntp_second_half_mean_prob',
        'ntp_second_half_anomaly_ratio',
        # High-weight features
        'ntp_high_weight_anomaly_ratio',
        'ntp_high_weight_mean_weighted_score',
        # Novel token features
        'ntp_novel_token_anomaly_ratio',
        'ntp_novel_token_mean_prob',
    ]


# Example usage
if __name__ == "__main__":
    import torch
    from colorama import init
    
    init(autoreset=True)
    
    print("🔧 Setup")
    print(f"   PyTorch: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    
    # Initialize detector with hybrid context weighting
    detector = NextTokenAnomalyDetector(
        model_name="Qwen/Qwen3-8B",
        anomaly_threshold=0.01,
        use_context_weighting=True,
        context_weighting_mode="hybrid"  # Use hybrid mode
    )
    
    # Test text
    test_text = """
    Dear John,
    
    I wanted to follow up on our discussion about the Q3 budget. The HR department
    has approved the new hiring plan, and we should be able to start interviews
    next week.
    
    Best regards,
    Sarah
    """
    
    print("\n" + "="*80)
    print("TEST: Extracting Features with Hybrid Context Weighting")
    print("="*80)
    
    # Extract features (for ML pipeline)
    features = detector.extract_anomaly_features(test_text)
    
    print("\nExtracted Features:")
    for name, value in features.items():
        if not name.startswith('_'):
            print(f"  {name}: {value:.4f}")
    
    # Visualize anomalies (for debugging)
    detector.visualize_anomalies(test_text)
    
    print("\n✅ Test complete!")