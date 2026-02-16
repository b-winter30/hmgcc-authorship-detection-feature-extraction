"""
Multilingual Stylometric Feature Extractor with Next-Token Prediction
Supports English, Russian, Spanish, Swedish
"""

import re
import numpy as np
from collections import Counter
from typing import Dict, List, Tuple, Optional
import unicodedata
import logging

logger = logging.getLogger(__name__)


class StylometricExtractor:
    """Extract stylometric features from text for authorship attribution"""
    
    def __init__(self, enable_ntp: bool = False, ntp_config: Optional[Dict] = None):
        """
        Initialize the feature extractor
        
        Args:
            enable_ntp: Enable next-token prediction anomaly detection (Stage 2)
            ntp_config: Configuration for NTP detector (model_name, device, etc.)
        """
        # Language-specific modal verbs (authority markers)
        self.modal_verbs = {
            'en': ['must', 'should', 'will', 'would', 'can', 'could', 'may', 'might', 'shall'],
            'es': ['debe', 'debería', 'puede', 'podría', 'tiene que', 'tengo que', 'hay que'],
            'ru': ['должен', 'должна', 'должны', 'может', 'могу', 'нужно', 'надо', 'следует'],
            'sv': ['måste', 'ska', 'skulle', 'kan', 'kunde', 'bör', 'får', 'vill']
        }
        
        # Language-specific hedge words (mitigation markers)
        self.hedge_words = {
            'en': ['maybe', 'perhaps', 'possibly', 'probably', 'somewhat', 'rather', 'quite', 
                   'seem', 'appears', 'might', 'could', 'may'],
            'es': ['quizás', 'tal vez', 'posiblemente', 'probablemente', 'parece', 'puede ser'],
            'ru': ['возможно', 'вероятно', 'может быть', 'наверное', 'кажется', 'пожалуй'],
            'sv': ['kanske', 'möjligen', 'troligen', 'förmodligen', 'tydligen', 'verkar']
        }
        
        # Next-Token Prediction (Stage 2 Anomaly Detection)
        self.enable_ntp = enable_ntp
        self.ntp_detector = None
        
        if enable_ntp:
            try:
                from next_token_prediction import NextTokenAnomalyDetector
                
                ntp_config = ntp_config or {}
                self.ntp_detector = NextTokenAnomalyDetector(**ntp_config)
                logger.info("✅ Next-Token Prediction anomaly detection enabled")
            except ImportError as e:
                logger.warning(f"⚠️ Could not load NTP detector: {e}")
                logger.warning("   Install: pip install transformers torch")
                self.enable_ntp = False
            except Exception as e:
                logger.warning(f"⚠️ NTP detector initialization failed: {e}")
                self.enable_ntp = False
        
    def extract_all_features(self, text: str, language: str = 'en') -> Dict[str, float]:
        """Extract all stylometric features from text"""
        features = {}
        
        # A. Lexical & Syntactic Features
        features.update(self._extract_lexical_features(text))
        features.update(self._extract_punctuation_features(text))
        
        # B. Structural Features
        features.update(self._extract_structural_features(text))
        features.update(self._extract_email_specific_features(text))
        
        # C. Readability Metrics
        features.update(self._extract_readability_features(text))
        
        # D. Sociolinguistic Features
        features.update(self._extract_sociolinguistic_features(text, language))
        
        # E. AI Detection Features
        features.update(self._extract_ai_detection_features(text))
        
        # F. Encoding Features
        features.update(self._extract_encoding_features(text))
        
        # G. Error/Spelling Features
        features.update(self._extract_error_features(text))
        
        # H. Emoji Features
        features.update(self._extract_emoji_features(text))
        
        # I. STAGE 2: Next-Token Prediction Anomaly Detection
        if self.enable_ntp and self.ntp_detector:
            try:
                ntp_features = self.ntp_detector.extract_anomaly_features(text)
                features.update(ntp_features)
            except Exception as e:
                logger.warning(f"⚠️ NTP feature extraction failed: {e}")
                # Add default NTP features
                from next_token_prediction import get_ntp_feature_names
                for feat_name in get_ntp_feature_names():
                    features[feat_name] = 0.0
        
        return features
    
    # All existing methods remain the same...
    # (Copy all methods from your original feature_extractor.py)
    
    def _extract_emoji_features(self, text: str) -> Dict[str, float]:
        """Extract emoji usage patterns"""
        features = {}
        
        try:
            import emoji
        except ImportError:
            return {
                'emoji_density': 0,
                'emoji_burst': 0,
                'avg_emojis_per_sentence': 0
            }
        
        sentences = self._tokenize_sentences(text)
        total_chars = len(text)
        
        emoji_count = emoji.emoji_count(text)
        features['emoji_density'] = emoji_count / total_chars if total_chars > 0 else 0
        
        if sentences:
            emojis_per_sent = [emoji.emoji_count(s) for s in sentences]
            features['emoji_burst'] = np.std(emojis_per_sent) if len(sentences) > 1 else 0
            features['avg_emojis_per_sentence'] = np.mean(emojis_per_sent)
        else:
            features['emoji_burst'] = 0
            features['avg_emojis_per_sentence'] = 0
        
        return features
    
    def _extract_whitespace_features(self, text: str) -> Dict[str, float]:
        """Extract detailed whitespace and formatting features"""
        features = {}
        
        original_text = text
        lines = text.split('\n')
        
        total_chars = len(text)
        if total_chars == 0:
            return {'whitespace_error': 1.0}
        
        whitespace_count = sum(1 for c in text if c.isspace())
        features['whitespace_ratio'] = whitespace_count / total_chars
        
        # Space runs (consecutive spaces)
        space_runs = re.findall(r' +', text)
        if space_runs:
            space_run_lengths = [len(run) for run in space_runs]
            features['avg_space_run_length'] = np.mean(space_run_lengths)
            features['space_run_variance'] = np.var(space_run_lengths)
            features['space_run_uniformity'] = 1 - (np.std(space_run_lengths) / (np.mean(space_run_lengths) + 1))
            
            single_spaces = sum(1 for length in space_run_lengths if length == 1)
            features['single_space_ratio'] = single_spaces / len(space_run_lengths)
        else:
            features['avg_space_run_length'] = 0
            features['space_run_variance'] = 0
            features['space_run_uniformity'] = 1
            features['single_space_ratio'] = 0
        
        # Indentation Analysis
        indentation_pattern = []
        for line in lines:
            if line.strip():
                leading_spaces = len(line) - len(line.lstrip(' '))
                indentation_pattern.append(leading_spaces)
        
        if indentation_pattern:
            indent_counts = Counter(indentation_pattern)
            features['indent_uniformity'] = max(indent_counts.values()) / len(indentation_pattern)
            features['unique_indent_levels'] = len(indent_counts)
            features['avg_indentation'] = np.mean(indentation_pattern)
            features['indent_variance'] = np.var(indentation_pattern)
            
            tab_lines = sum(1 for line in lines if line.startswith('\t'))
            features['uses_tabs'] = float(tab_lines > 0)
            features['tab_ratio'] = tab_lines / len(lines) if lines else 0
        else:
            features['indent_uniformity'] = 1
            features['unique_indent_levels'] = 0
            features['avg_indentation'] = 0
            features['indent_variance'] = 0
            features['uses_tabs'] = 0
            features['tab_ratio'] = 0
        
        # Line Break Patterns
        features['total_lines'] = len(lines)
        
        empty_lines = [i for i, line in enumerate(lines) if not line.strip()]
        features['empty_line_count'] = len(empty_lines)
        features['empty_line_ratio'] = len(empty_lines) / len(lines) if lines else 0
        
        if len(empty_lines) > 1:
            empty_line_gaps = [empty_lines[i+1] - empty_lines[i] for i in range(len(empty_lines)-1)]
            features['empty_line_clustering'] = np.var(empty_line_gaps) if empty_line_gaps else 0
        else:
            features['empty_line_clustering'] = 0
        
        # Trailing Whitespace
        trailing_space_lines = sum(1 for line in lines if line.rstrip() != line and line.strip())
        features['trailing_whitespace_ratio'] = trailing_space_lines / len(lines) if lines else 0
        
        # Line Length Uniformity
        line_lengths = [len(line) for line in lines if line.strip()]
        if line_lengths:
            features['avg_line_length'] = np.mean(line_lengths)
            features['line_length_variance'] = np.var(line_lengths)
            features['line_length_uniformity'] = 1 - (np.std(line_lengths) / (np.mean(line_lengths) + 1))
        else:
            features['avg_line_length'] = 0
            features['line_length_variance'] = 0
            features['line_length_uniformity'] = 1
        
        # Paragraph Spacing Patterns
        paragraphs = text.split('\n\n')
        paragraphs = [p for p in paragraphs if p.strip()]
        
        features['paragraph_count'] = len(paragraphs)
        
        if paragraphs:
            para_lengths = [len(p.split('\n')) for p in paragraphs]
            features['avg_paragraph_lines'] = np.mean(para_lengths)
            features['paragraph_uniformity'] = 1 - (np.std(para_lengths) / (np.mean(para_lengths) + 1))
        else:
            features['avg_paragraph_lines'] = 0
            features['paragraph_uniformity'] = 1
        
        # Space After Punctuation
        space_after_period = len(re.findall(r'\. [A-Z]', text))
        space_after_comma = len(re.findall(r', \w', text))
        total_periods = text.count('.')
        total_commas = text.count(',')
        
        features['space_after_period_ratio'] = space_after_period / total_periods if total_periods > 0 else 0
        features['space_after_comma_ratio'] = space_after_comma / total_commas if total_commas > 0 else 0
        
        # Multiple Spaces Detection
        double_spaces = text.count('  ')
        features['double_space_count'] = double_spaces
        features['double_space_ratio'] = double_spaces / total_chars
        
        # Overall Whitespace Consistency Score
        uniformity_scores = [
            features.get('space_run_uniformity', 0),
            features.get('indent_uniformity', 0),
            features.get('line_length_uniformity', 0),
            features.get('paragraph_uniformity', 0)
        ]
        features['overall_whitespace_uniformity'] = np.mean(uniformity_scores)
        
        return features

    def _tokenize_sentences(self, text: str) -> List[str]:
        """Simple sentence tokenization using punctuation"""
        sentences = re.split(r'[.!?]+[\s\n]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _tokenize_words(self, text: str) -> List[str]:
        """Simple word tokenization"""
        words = re.findall(r'\b\w+\b', text.lower())
        return words
    
    def _extract_lexical_features(self, text: str) -> Dict[str, float]:
        """Extract token statistics and vocabulary richness"""
        features = {}
        
        sentences = self._tokenize_sentences(text)
        words = self._tokenize_words(text)
        
        if not words:
            return {'lexical_error': 1.0}
        
        if sentences:
            sentence_lengths = [len(self._tokenize_words(s)) for s in sentences]
            features['avg_sentence_length'] = np.mean(sentence_lengths)
            features['median_sentence_length'] = np.median(sentence_lengths)
            features['std_sentence_length'] = np.std(sentence_lengths)
        else:
            features['avg_sentence_length'] = 0
            features['median_sentence_length'] = 0
            features['std_sentence_length'] = 0
        
        word_lengths = [len(w) for w in words]
        features['avg_word_length'] = np.mean(word_lengths)
        features['median_word_length'] = np.median(word_lengths)
        features['std_word_length'] = np.std(word_lengths)
        
        unique_words = set(words)
        features['ttr'] = len(unique_words) / len(words)
        
        word_freq = Counter(words)
        hapax_legomena = sum(1 for count in word_freq.values() if count == 1)
        dis_legomena = sum(1 for count in word_freq.values() if count == 2)
        
        features['hapax_ratio'] = hapax_legomena / len(words)
        features['dis_legomena_ratio'] = dis_legomena / len(words)
        
        M1 = len(words)
        M2 = sum([freq ** 2 for freq in word_freq.values()])
        features['yules_k'] = 10000 * (M2 - M1) / (M1 * M1) if M1 > 0 else 0
        
        text_nospaces = text.replace(" ", "").lower()
        if len(text_nospaces) >= 3:
            trigrams = [text_nospaces[i:i+3] for i in range(len(text_nospaces) - 2)]
            features['char_trigram_richness'] = len(set(trigrams)) / len(trigrams) if trigrams else 0
        else:
            features['char_trigram_richness'] = 0
        
        return features
    
    def _extract_punctuation_features(self, text: str) -> Dict[str, float]:
        """Extract punctuation usage patterns"""
        features = {}
        
        total_chars = len(text)
        if total_chars == 0:
            return features
        
        punctuation_marks = {
            'period': '.',
            'comma': ',',
            'exclamation': '!',
            'question': '?',
            'semicolon': ';',
            'colon': ':',
            'dash': '-',
            'quote_single': "'",
            'quote_double': '"',
            'ellipsis': '...'
        }
        
        for name, mark in punctuation_marks.items():
            count = text.count(mark)
            features[f'punct_{name}_per_100'] = (count / total_chars) * 100
        
        all_punct = sum(1 for c in text if c in '.,!?;:\'"()-')
        features['punct_total_density'] = (all_punct / total_chars) * 100
        
        sentences = self._tokenize_sentences(text)
        if sentences:
            puncs_per_sent = [sum(1 for c in s if c in '.,!?;:\'"()-') for s in sentences]
            features['punctuation_burst'] = np.std(puncs_per_sent) if len(sentences) > 1 else 0
        else:
            features['punctuation_burst'] = 0
        
        words = text.split()
        if words:
            all_caps_words = sum(1 for w in words if w.isupper() and len(w) > 1)
            title_case_words = sum(1 for w in words if w.istitle())
            
            features['all_caps_ratio'] = all_caps_words / len(words)
            features['title_case_ratio'] = title_case_words / len(words)
        
        return features
    
    def _extract_structural_features(self, text: str) -> Dict[str, float]:
        """Extract formatting and structural patterns with enhanced whitespace analysis"""
        features = {}
        
        whitespace_features = self._extract_whitespace_features(text)
        features.update(whitespace_features)
        
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        sentences = self._tokenize_sentences(text)
        
        if paragraphs:
            sentences_per_para = [len(self._tokenize_sentences(p)) for p in paragraphs]
            features['avg_sentences_per_paragraph'] = np.mean(sentences_per_para)
        else:
            features['avg_sentences_per_paragraph'] = 0
        
        return features
    
    def _extract_email_specific_features(self, text: str) -> Dict[str, float]:
        """Extract email-specific structural features"""
        features = {}
        
        greetings = [
            'dear', 'hello', 'hi', 'hey', 'greetings',
            'estimado', 'hola', 'saludos',
            'здравствуйте', 'привет', 'добрый день',
            'hej', 'hallå', 'goddag'
        ]
        
        text_lower = text.lower()
        features['has_greeting'] = float(any(g in text_lower[:200] for g in greetings))
        
        signoffs = [
            'regards', 'sincerely', 'best', 'thanks', 'cheers',
            'saludos', 'atentamente', 'gracias',
            'с уважением', 'всего доброго', 'спасибо',
            'hälsningar', 'tack', 'mvh'
        ]
        
        features['has_signoff'] = float(any(s in text_lower[-200:] for s in signoffs))
        
        features['char_count'] = len(text)
        features['word_count'] = len(self._tokenize_words(text))
        features['sentence_count'] = len(self._tokenize_sentences(text))
        
        return features
    
    def _extract_readability_features(self, text: str) -> Dict[str, float]:
        """Extract readability metrics"""
        features = {}
        
        sentences = self._tokenize_sentences(text)
        words = self._tokenize_words(text)
        
        if not sentences or not words:
            return features
        
        features['avg_words_per_sentence'] = len(words) / len(sentences)
        
        def count_syllables(word):
            word = word.lower()
            vowels = 'aeiouy'
            syllable_count = 0
            previous_was_vowel = False
            
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not previous_was_vowel:
                    syllable_count += 1
                previous_was_vowel = is_vowel
            
            if word.endswith('e'):
                syllable_count -= 1
            
            return max(1, syllable_count)
        
        total_syllables = sum(count_syllables(w) for w in words)
        features['avg_syllables_per_word'] = total_syllables / len(words) if words else 0
        
        if len(sentences) > 0 and len(words) > 0:
            asl = len(words) / len(sentences)
            asw = total_syllables / len(words)
            features['flesch_kincaid_grade'] = 0.39 * asl + 11.8 * asw - 15.59
        else:
            features['flesch_kincaid_grade'] = 0
        
        return features
    
    def _extract_sociolinguistic_features(self, text: str, language: str) -> Dict[str, float]:
        """Extract modal verbs and hedge words"""
        features = {}
        
        text_lower = text.lower()
        words = self._tokenize_words(text)
        
        if not words:
            return features
        
        modal_verbs = self.modal_verbs.get(language, [])
        modal_count = sum(1 for word in words if word in modal_verbs)
        features['modal_verb_density'] = modal_count / len(words)
        
        hedge_words = self.hedge_words.get(language, [])
        hedge_count = sum(1 for word in words if word in hedge_words)
        features['hedge_word_density'] = hedge_count / len(words)
        
        return features
    
    def _extract_ai_detection_features(self, text: str) -> Dict[str, float]:
        """Extract features for AI-generated text detection"""
        features = {}
        
        sentences = self._tokenize_sentences(text)
        
        if not sentences:
            return features
        
        sentence_lengths = [len(self._tokenize_words(s)) for s in sentences]
        
        if len(sentence_lengths) > 1:
            features['burstiness_sentence_length'] = np.var(sentence_lengths)
            features['burstiness_coefficient_variation'] = np.std(sentence_lengths) / np.mean(sentence_lengths) if np.mean(sentence_lengths) > 0 else 0
        else:
            features['burstiness_sentence_length'] = 0
            features['burstiness_coefficient_variation'] = 0
        
        if len(sentences) >= 3:
            segment_size = len(sentences) // 3
            segments = [
                sentences[:segment_size],
                sentences[segment_size:2*segment_size],
                sentences[2*segment_size:]
            ]
            
            ttrs = []
            for segment in segments:
                segment_text = ' '.join(segment)
                segment_words = self._tokenize_words(segment_text)
                if segment_words:
                    ttr = len(set(segment_words)) / len(segment_words)
                    ttrs.append(ttr)
            
            features['lexical_diversity_variance'] = np.var(ttrs) if ttrs else 0
        else:
            features['lexical_diversity_variance'] = 0
        
        return features
    
    def _extract_encoding_features(self, text: str) -> Dict[str, float]:
        """Extract UTF-8 byte distribution features"""
        features = {}
        
        text_bytes = text.encode('utf-8')
        byte_freq = Counter(text_bytes)
        total_bytes = len(text_bytes)
        
        if total_bytes == 0:
            return features
        
        non_ascii_bytes = sum(1 for b in text_bytes if b > 127)
        features['non_ascii_density'] = non_ascii_bytes / total_bytes
        
        cyrillic_chars = sum(1 for c in text if '\u0400' <= c <= '\u04FF')
        latin_chars = sum(1 for c in text if c.isalpha() and c.isascii())
        
        features['cyrillic_char_ratio'] = cyrillic_chars / len(text) if text else 0
        features['latin_char_ratio'] = latin_chars / len(text) if text else 0
        
        special_chars = sum(1 for c in text if unicodedata.category(c).startswith('P'))
        features['special_char_density'] = special_chars / len(text) if text else 0
        
        return features
    
    def _extract_error_features(self, text: str) -> Dict[str, float]:
        """Extract spelling and typo features"""
        features = {}
        
        words = self._tokenize_words(text)
        
        if not words:
            return features
        
        repeated_chars = sum(1 for w in words if re.search(r'(.)\1{2,}', w))
        features['repeated_char_ratio'] = repeated_chars / len(words)
        
        all_caps = sum(1 for w in words if w.isupper() and len(w) > 1)
        features['all_caps_word_ratio'] = all_caps / len(words)
        
        mixed_case = sum(1 for w in words if not (w.islower() or w.isupper() or w.istitle()) and len(w) > 1)
        features['mixed_case_ratio'] = mixed_case / len(words)
        
        mixed_alnum = sum(1 for w in words if any(c.isdigit() for c in w) and any(c.isalpha() for c in w))
        features['mixed_alnum_ratio'] = mixed_alnum / len(words)
        
        return features


def get_feature_vector(features: Dict[str, float]) -> Tuple[List[str], List[float]]:
    """Convert feature dictionary to ordered lists for ML"""
    feature_names = sorted(features.keys())
    feature_values = [features[name] for name in feature_names]
    return feature_names, feature_values