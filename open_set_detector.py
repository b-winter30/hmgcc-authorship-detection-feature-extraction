"""
Open-Set Authorship Recognition
Handles both known and unknown authors
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, euclidean
from typing import Dict, List, Tuple, Optional
import pickle

class OpenSetAuthorshipDetector:
    """
    Open-set recognition for authorship detection
    
    Can identify:
    1. Known authors (from training set)
    2. Unknown authors (not in training set)
    3. Anomalous emails from known authors
    """
    
    def __init__(
        self,
        n_neighbors: int = 5,
        distance_metric: str = 'euclidean',
        reject_threshold_multiplier: float = 1.1,
        radius_percentile: int = 80
    ):
        """
        Args:
            n_neighbors: Number of neighbors for KNN
            distance_metric: 'euclidean', 'cosine', 'manhattan'
            reject_threshold_multiplier: How far outside normal range to reject
                                        (1.5 = 150% of normal radius)
        """
        self.n_neighbors = n_neighbors
        self.distance_metric = distance_metric
        self.reject_threshold_multiplier = reject_threshold_multiplier
        self.radius_percentile = radius_percentile
        
        # Core models
        self.knn = None
        self.scaler = StandardScaler()
        
        # Per-author statistics
        self.author_prototypes: Dict[str, np.ndarray] = {}  # Mean vector
        self.author_radiuses: Dict[str, float] = {}  # Normal variation radius
        self.author_covariances: Dict[str, np.ndarray] = {}  # For Mahalanobis
        
        # Training data (stored for retraining)
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None
        
        self.feature_names: List[str] = []
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: List[str]):
        """
        Train the open-set detector
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Author labels
            feature_names: Names of features
        """
        self.feature_names = feature_names
        
        print("🎓 Training Open-Set Authorship Detector...")
        print(f"   Samples: {X.shape[0]}")
        print(f"   Features: {X.shape[1]}")
        print(f"   Known Authors: {len(np.unique(y))}")
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Store training data
        self.X_train = X_scaled.copy()
        self.y_train = y.copy()
        
        # Train KNN classifier
        print("\n1️⃣ Training KNN classifier...")
        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            metric=self.distance_metric,
            weights='distance'  # Closer neighbors have more weight
        )
        self.knn.fit(X_scaled, y)
        
        # Compute per-author statistics
        print("\n2️⃣ Computing per-author statistics...")
        self._compute_author_statistics(X_scaled, y)
        
        print("\n✅ Open-set detector trained!")
        self._print_author_statistics()
    
    def _compute_author_statistics(self, X: np.ndarray, y: np.ndarray):
        """
        Compute prototype, radius, and covariance for each author
        """
        unique_authors = np.unique(y)
        
        for author in unique_authors:
            # Get all samples for this author
            author_mask = y == author
            author_samples = X[author_mask]
            n_samples = len(author_samples)
            
            # 1. Prototype = centroid (mean vector)
            prototype = np.mean(author_samples, axis=0)
            self.author_prototypes[author] = prototype
            
            # 2. Radius = typical distance from prototype
            distances = cdist([prototype], author_samples, metric=self.distance_metric)[0]
            
            # Use 95th percentile as "normal" radius
            # This means 95% of author's emails fall within this distance
            radius = np.percentile(distances, self.radius_percentile)
            self.author_radiuses[author] = radius
            
            # 3. Covariance (for Mahalanobis distance - optional but better)
            if n_samples > X.shape[1]:  # Need more samples than features
                try:
                    cov = np.cov(author_samples.T)
                    # Add small regularization for numerical stability
                    cov += np.eye(cov.shape[0]) * 1e-6
                    self.author_covariances[author] = cov
                except np.linalg.LinAlgError:
                    # Fallback to identity if covariance is singular
                    self.author_covariances[author] = np.eye(X.shape[1])
            else:
                self.author_covariances[author] = np.eye(X.shape[1])
            
            print(f"   Author '{author}': {n_samples} samples, radius={radius:.3f}")
    
    def predict(
        self,
        features: np.ndarray,
        return_distances: bool = False
    ) -> Dict:
        """
        Predict author with open-set recognition
        
        Args:
            features: Feature vector
            return_distances: Include distances to all author prototypes
        
        Returns:
            Dictionary with prediction, confidence, and open-set decision
        """
        # Normalize features
        features_scaled = self.scaler.transform([features])[0]
        
        # Step 1: Get KNN prediction
        knn_prediction = self.knn.predict([features_scaled])[0]
        knn_probabilities = self.knn.predict_proba([features_scaled])[0]
        knn_confidence = max(knn_probabilities)
        
        # Step 2: Compute distance to predicted author's prototype
        prototype = self.author_prototypes[knn_prediction]
        
        if self.distance_metric == 'euclidean':
            distance = euclidean(features_scaled, prototype)
        elif self.distance_metric == 'cosine':
            from scipy.spatial.distance import cosine
            distance = cosine(features_scaled, prototype)
        elif self.distance_metric == 'manhattan':
            from scipy.spatial.distance import cityblock
            distance = cityblock(features_scaled, prototype)
        else:
            distance = euclidean(features_scaled, prototype)
        
        # Step 3: Check if within normal range
        normal_radius = self.author_radiuses[knn_prediction]
        rejection_threshold = normal_radius * self.reject_threshold_multiplier
        
        distance_ratio = distance / normal_radius
        is_within_range = distance <= rejection_threshold
        
        # Step 4: Make open-set decision
        if not is_within_range:
            # REJECTED - likely unknown author
            final_prediction = "UNKNOWN"
            confidence = 0.0
            decision = "rejected"
            reason = f"Distance {distance:.3f} exceeds threshold {rejection_threshold:.3f} for {knn_prediction}"
        else:
            # ACCEPTED - likely the predicted author
            final_prediction = knn_prediction
            # Confidence decreases as distance increases
            confidence = max(0.0, 1.0 - distance_ratio)
            decision = "accepted"
            reason = f"Within normal range for {knn_prediction}"
        
        # Step 5: Compute distances to all author prototypes (optional)
        all_distances = {}
        if return_distances:
            for author, proto in self.author_prototypes.items():
                dist = euclidean(features_scaled, proto)
                all_distances[author] = {
                    'distance': float(dist),
                    'normalized': float(dist / self.author_radiuses[author]),
                    'within_range': dist <= self.author_radiuses[author] * self.reject_threshold_multiplier
                }
        
        return {
            'predicted_author': final_prediction,
            'confidence': float(confidence),
            'decision': decision,
            'reason': reason,
            'distance_to_prototype': float(distance),
            'normal_radius': float(normal_radius),
            'distance_ratio': float(distance_ratio),
            'knn_prediction': knn_prediction,
            'knn_confidence': float(knn_confidence),
            'all_author_distances': all_distances if return_distances else None,
            'recommendation': self._generate_recommendation(
                final_prediction, confidence, decision, distance_ratio
            )
        }
    
    def predict_top_k(
        self,
        features: np.ndarray,
        k: int = 3
    ) -> List[Dict]:
        """
        Get top-k author predictions with open-set scores
        
        Useful when you want to see "could be A, B, or C"
        """
        features_scaled = self.scaler.transform([features])[0]
        
        # Compute distance to all author prototypes
        results = []
        for author, prototype in self.author_prototypes.items():
            distance = euclidean(features_scaled, prototype)
            normal_radius = self.author_radiuses[author]
            distance_ratio = distance / normal_radius
            
            is_within_range = distance <= normal_radius * self.reject_threshold_multiplier
            confidence = max(0.0, 1.0 - distance_ratio) if is_within_range else 0.0
            
            results.append({
                'author': author,
                'distance': float(distance),
                'distance_ratio': float(distance_ratio),
                'confidence': float(confidence),
                'within_range': is_within_range
            })
        
        # Sort by distance (closest first)
        results.sort(key=lambda x: x['distance'])
        
        return results[:k]
    
    def add_new_author(
        self,
        new_samples: np.ndarray,
        author_name: str
    ):
        """
        Add a new author to the system WITHOUT full retraining
        
        Args:
            new_samples: Feature vectors for new author (n_samples, n_features)
            author_name: Name of new author
        """
        print(f"\n➕ Adding new author: {author_name}")
        print(f"   Samples: {len(new_samples)}")
        
        # Normalize new samples
        new_samples_scaled = self.scaler.transform(new_samples)
        
        # Compute statistics for new author
        prototype = np.mean(new_samples_scaled, axis=0)
        distances = cdist([prototype], new_samples_scaled, metric=self.distance_metric)[0]
        radius = np.percentile(distances, 95)
        
        self.author_prototypes[author_name] = prototype
        self.author_radiuses[author_name] = radius
        
        # Add to training data
        self.X_train = np.vstack([self.X_train, new_samples_scaled])
        self.y_train = np.concatenate([self.y_train, [author_name] * len(new_samples)])
        
        # Retrain KNN (fast)
        self.knn.fit(self.X_train, self.y_train)
        
        print(f"   ✅ Added {author_name} with radius={radius:.3f}")
        print(f"   Total authors now: {len(self.author_prototypes)}")
    
    def _generate_recommendation(
        self,
        prediction: str,
        confidence: float,
        decision: str,
        distance_ratio: float
    ) -> str:
        """Generate human-readable recommendation"""
        
        if decision == "rejected":
            if distance_ratio > 3.0:
                return f"❌ STRONGLY REJECTED - Very far from any known author (distance ratio: {distance_ratio:.1f}x)"
            else:
                return f"⚠️ REJECTED - Outside normal range for known authors (distance ratio: {distance_ratio:.1f}x)"
        
        # Accepted cases
        if confidence > 0.9:
            return f"✅ VERY CONFIDENT - Strongly matches {prediction}"
        elif confidence > 0.7:
            return f"✅ CONFIDENT - Likely {prediction}"
        elif confidence > 0.5:
            return f"⚠️ MODERATE - Probably {prediction} but borderline"
        else:
            return f"⚠️ WEAK - Barely within range for {prediction}"
    
    def _print_author_statistics(self):
        """Print summary of author statistics"""
        print("\n" + "="*60)
        print("AUTHOR STATISTICS")
        print("="*60)
        
        for author in sorted(self.author_prototypes.keys()):
            n_samples = np.sum(self.y_train == author)
            radius = self.author_radiuses[author]
            threshold = radius * self.reject_threshold_multiplier
            
            print(f"\n{author}:")
            print(f"  Samples: {n_samples}")
            print(f"  Normal radius: {radius:.4f}")
            print(f"  Rejection threshold: {threshold:.4f}")
    
    def get_calibration_plot_data(self) -> Dict:
        """
        Get data for plotting calibration curves
        
        Shows distribution of distances for each author's samples
        """
        calibration_data = {}
        
        for author in self.author_prototypes:
            author_samples = self.X_train[self.y_train == author]
            prototype = self.author_prototypes[author]
            
            distances = cdist([prototype], author_samples, metric=self.distance_metric)[0]
            
            calibration_data[author] = {
                'distances': distances.tolist(),
                'mean_distance': float(np.mean(distances)),
                'std_distance': float(np.std(distances)),
                'radius_95': float(self.author_radiuses[author]),
                'rejection_threshold': float(self.author_radiuses[author] * self.reject_threshold_multiplier)
            }
        
        return calibration_data
    
    def evaluate_on_unknowns(
        self,
        unknown_samples: np.ndarray,
        unknown_labels: np.ndarray
    ) -> Dict:
        """
        Evaluate how well the system rejects truly unknown authors
        
        Args:
            unknown_samples: Samples from authors NOT in training
            unknown_labels: Labels (just for tracking, not used for prediction)
        
        Returns:
            Evaluation metrics
        """
        print("\n🧪 Evaluating on unknown authors...")
        print(f"   Unknown samples: {len(unknown_samples)}")
        print(f"   Unknown authors: {len(np.unique(unknown_labels))}")
        
        unknown_samples_scaled = self.scaler.transform(unknown_samples)
        
        correctly_rejected = 0
        incorrectly_accepted = 0
        accepted_authors = []
        
        for sample, true_label in zip(unknown_samples_scaled, unknown_labels):
            result = self.predict(sample)
            
            if result['decision'] == 'rejected':
                correctly_rejected += 1
            else:
                incorrectly_accepted += 1
                accepted_authors.append({
                    'true_author': true_label,
                    'predicted_as': result['predicted_author'],
                    'confidence': result['confidence']
                })
        
        rejection_rate = correctly_rejected / len(unknown_samples)
        
        print(f"\n   ✅ Correctly rejected: {correctly_rejected}/{len(unknown_samples)} ({rejection_rate:.1%})")
        print(f"   ❌ Incorrectly accepted: {incorrectly_accepted}/{len(unknown_samples)}")
        
        return {
            'total_unknown': len(unknown_samples),
            'correctly_rejected': correctly_rejected,
            'incorrectly_accepted': incorrectly_accepted,
            'rejection_rate': rejection_rate,
            'false_acceptances': accepted_authors
        }
    
    def save_model(self, path: str):
        """Save the open-set detector"""
        import os
        os.makedirs(path, exist_ok=True)
        
        with open(f'{path}/open_set_detector.pkl', 'wb') as f:
            pickle.dump({
                'knn': self.knn,
                'scaler': self.scaler,
                'author_prototypes': self.author_prototypes,
                'author_radiuses': self.author_radiuses,
                'author_covariances': self.author_covariances,
                'X_train': self.X_train,
                'y_train': self.y_train,
                'feature_names': self.feature_names,
                'n_neighbors': self.n_neighbors,
                'distance_metric': self.distance_metric,
                'reject_threshold_multiplier': self.reject_threshold_multiplier
            }, f)
        
        print(f"💾 Open-set detector saved to {path}")
    
    @classmethod
    def load_model(cls, path: str) -> 'OpenSetAuthorshipDetector':
        """Load saved detector"""
        with open(f'{path}/open_set_detector.pkl', 'rb') as f:
            data = pickle.load(f)
        
        detector = cls(
            n_neighbors=data['n_neighbors'],
            distance_metric=data['distance_metric'],
            reject_threshold_multiplier=data['reject_threshold_multiplier']
        )
        
        detector.knn = data['knn']
        detector.scaler = data['scaler']
        detector.author_prototypes = data['author_prototypes']
        detector.author_radiuses = data['author_radiuses']
        detector.author_covariances = data['author_covariances']
        detector.X_train = data['X_train']
        detector.y_train = data['y_train']
        detector.feature_names = data['feature_names']
        
        print(f"✅ Open-set detector loaded from {path}")
        return detector


# Example usage and testing
if __name__ == "__main__":
    print("="*70)
    print("OPEN-SET AUTHORSHIP DETECTION - DEMO")
    print("="*70)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # 3 known authors
    n_samples_per_author = 100
    n_features = 50
    
    # Author A: high mean, low variance
    X_A = np.random.randn(n_samples_per_author, n_features) * 0.5 + 2.0
    y_A = np.array(['Alice'] * n_samples_per_author)
    
    # Author B: medium mean, medium variance
    X_B = np.random.randn(n_samples_per_author, n_features) * 1.0 + 0.0
    y_B = np.array(['Bob'] * n_samples_per_author)
    
    # Author C: low mean, high variance
    X_C = np.random.randn(n_samples_per_author, n_features) * 1.5 - 2.0
    y_C = np.array(['Carol'] * n_samples_per_author)
    
    # Combine training data
    X_train = np.vstack([X_A, X_B, X_C])
    y_train = np.concatenate([y_A, y_B, y_C])
    
    feature_names = [f'feature_{i}' for i in range(n_features)]
    
    # Train detector
    detector = OpenSetAuthorshipDetector(
        n_neighbors=5,
        reject_threshold_multiplier=1.5
    )
    detector.fit(X_train, y_train, feature_names)
    
    # Test 1: Known author (should accept)
    print("\n" + "="*70)
    print("TEST 1: Known Author (Alice)")
    print("="*70)
    test_alice = X_A[0]  # Take one of Alice's actual emails
    result = detector.predict(test_alice, return_distances=True)
    print(f"Prediction: {result['predicted_author']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Decision: {result['decision']}")
    print(f"Recommendation: {result['recommendation']}")
    
    # Test 2: Unknown author (should reject)
    print("\n" + "="*70)
    print("TEST 2: Unknown Author (David)")
    print("="*70)
    X_david = np.random.randn(1, n_features) * 2.0 + 5.0  # Very different pattern
    result = detector.predict(X_david[0], return_distances=True)
    print(f"Prediction: {result['predicted_author']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Decision: {result['decision']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"\nDistances to all authors:")
    for author, info in result['all_author_distances'].items():
        print(f"  {author}: {info['distance']:.3f} (normalized: {info['normalized']:.2f}x)")
    
    # Test 3: Top-k predictions
    print("\n" + "="*70)
    print("TEST 3: Top-3 Predictions")
    print("="*70)
    top_k = detector.predict_top_k(test_alice, k=3)
    for rank, item in enumerate(top_k, 1):
        print(f"{rank}. {item['author']}: distance={item['distance']:.3f}, " 
              f"confidence={item['confidence']:.1%}, "
              f"within_range={item['within_range']}")
    
    # Test 4: Add new author
    print("\n" + "="*70)
    print("TEST 4: Adding New Author (David)")
    print("="*70)
    X_david_samples = np.random.randn(50, n_features) * 2.0 + 5.0
    detector.add_new_author(X_david_samples, 'David')
    
    # Now test David's email again
    result = detector.predict(X_david[0])
    print(f"\nAfter adding David:")
    print(f"Prediction: {result['predicted_author']}")
    print(f"Confidence: {result['confidence']:.1%}")
    print(f"Decision: {result['decision']}")
    
    # Test 5: Evaluate on unknowns
    print("\n" + "="*70)
    print("TEST 5: Evaluation on Unknown Authors")
    print("="*70)
    
    # Create samples from 2 unknown authors
    X_unknown_1 = np.random.randn(30, n_features) * 1.0 + 10.0
    y_unknown_1 = np.array(['Unknown_1'] * 30)
    
    X_unknown_2 = np.random.randn(30, n_features) * 1.0 - 10.0
    y_unknown_2 = np.array(['Unknown_2'] * 30)
    
    X_unknown = np.vstack([X_unknown_1, X_unknown_2])
    y_unknown = np.concatenate([y_unknown_1, y_unknown_2])
    
    eval_results = detector.evaluate_on_unknowns(X_unknown, y_unknown)
    
    print("\n✅ Demo complete!")