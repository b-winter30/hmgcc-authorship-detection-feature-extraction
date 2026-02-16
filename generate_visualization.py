"""
Standalone script to generate 2D authorship visualization
Can be run after training to create visualization plots

Usage:
    python generate_visualization.py
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import os
from sklearn.manifold import TSNE

# Optional: Use UMAP if available (requires Python <3.10)
try:
    import umap
    USE_UMAP = True
except ImportError:
    USE_UMAP = False
    print("ℹ️  Using t-SNE (UMAP not available, requires Python <3.10)")


def load_models(models_dir="models"):
    """Load trained models"""
    print(f"\n📦 Loading models from {models_dir}/...")
    
    models = {}
    required = [
        'open_set_detector',
        'label_encoder',
        'scaler'
    ]
    
    for name in required:
        with open(f'{models_dir}/{name}_latest.pkl', 'rb') as f:
            models[name] = pickle.load(f)
    
    print("✅ Models loaded!")
    return models


def create_2d_projection(X_train, y_train, method='tsne'):
    """
    Project high-dimensional features to 2D
    
    Args:
        X_train: Training features (already scaled)
        y_train: Training labels
        method: 'tsne' (default, built into scikit-learn) or 'umap' (requires umap-learn)
    
    Returns:
        projected_data: 2D coordinates
    """
    print(f"\n🎨 Creating 2D projection using {method.upper()}...")
    print(f"   Input shape: {X_train.shape}")
    
    if method == 'umap' and USE_UMAP:
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=2,
            metric='euclidean',
            random_state=42,
            verbose=True
        )
    else:
        if method == 'umap' and not USE_UMAP:
            print("   ⚠️  UMAP not available, using t-SNE instead")
        
        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(X_train) - 1),  # Perplexity must be < n_samples
            random_state=42,
            max_iter=1000,  # Changed from n_iter to max_iter
            verbose=1
        )
        method = 'tsne'
    
    projected = reducer.fit_transform(X_train)
    print(f"✅ Projection complete! Shape: {projected.shape}")
    
    return projected, reducer


def compute_cluster_stats(projected_data, y_train, label_encoder):
    """Compute cluster centers and boundaries"""
    reverse_encoder = {v: k for k, v in label_encoder.items()}
    
    stats = {}
    
    for author_id in np.unique(y_train):
        author_name = reverse_encoder[author_id]
        
        # Get all points for this author
        mask = y_train == author_id
        points = projected_data[mask]
        
        # Center
        center = np.mean(points, axis=0)
        
        # Radii
        distances = np.linalg.norm(points - center, axis=1)
        radius_95 = np.percentile(distances, 95)
        radius_threshold = radius_95 * 1.5  # Rejection threshold
        
        stats[author_name] = {
            'center': center,
            'radius_95': radius_95,
            'radius_threshold': radius_threshold,
            'points': points,
            'n_samples': len(points)
        }
    
    return stats


def plot_visualization(projected_data, y_train, cluster_stats, label_encoder, 
                       new_email_pos=None, predicted_author=None, decision=None,
                       output_path='visualization.png'):
    """
    Create beautiful 2D visualization plot
    
    Args:
        projected_data: 2D coordinates of training samples
        y_train: Training labels
        cluster_stats: Cluster statistics from compute_cluster_stats
        label_encoder: Label encoder dictionary
        new_email_pos: Optional (x, y) position of new email
        predicted_author: Optional predicted author for new email
        decision: Optional 'accepted' or 'rejected'
        output_path: Where to save the plot
    """
    reverse_encoder = {v: k for k, v in label_encoder.items()}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Color palette
    colors = plt.cm.Set3(np.linspace(0, 1, len(cluster_stats)))
    
    # Plot each author's cluster
    for idx, (author, stats) in enumerate(sorted(cluster_stats.items(), key=lambda x: int(x[0]))):
        color = colors[idx]
        
        # Plot training samples
        mask = y_train == label_encoder[author]
        points = projected_data[mask]
        
        ax.scatter(
            points[:, 0],
            points[:, 1],
            c=[color],
            alpha=0.4,
            s=20,
            label=f'Contact {author} ({stats["n_samples"]} emails)',
            edgecolors='none'
        )
        
        # Plot center (prototype)
        ax.scatter(
            stats['center'][0],
            stats['center'][1],
            c=[color],
            s=300,
            marker='*',
            edgecolors='black',
            linewidths=2,
            zorder=100
        )
        
        # Plot 95% confidence boundary
        circle_95 = Circle(
            stats['center'],
            stats['radius_95'],
            fill=False,
            edgecolor=color,
            linewidth=2,
            linestyle='--',
            alpha=0.6,
            label=f'Contact {author} 95% boundary'
        )
        ax.add_patch(circle_95)
        
        # Plot rejection threshold
        circle_threshold = Circle(
            stats['center'],
            stats['radius_threshold'],
            fill=False,
            edgecolor=color,
            linewidth=2,
            linestyle=':',
            alpha=0.4,
            label=f'Contact {author} rejection threshold'
        )
        ax.add_patch(circle_threshold)
        
        # Add author label
        ax.text(
            stats['center'][0],
            stats['center'][1] + stats['radius_threshold'] + 0.5,
            f'Contact {author}',
            fontsize=12,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7)
        )
    
    # Plot new email if provided
    if new_email_pos is not None:
        marker_color = 'green' if decision == 'accepted' else 'red'
        marker_size = 500
        
        ax.scatter(
            new_email_pos[0],
            new_email_pos[1],
            c=marker_color,
            s=marker_size,
            marker='*',
            edgecolors='black',
            linewidths=3,
            zorder=200,
            label=f'New Email ({decision})'
        )
        
        # Add arrow and label
        ax.annotate(
            f'New Email\n({decision})\nPredicted: {predicted_author}',
            xy=new_email_pos,
            xytext=(new_email_pos[0] + 2, new_email_pos[1] + 2),
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=marker_color, alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=2, color=marker_color)
        )
    
    # Styling
    ax.set_xlabel('Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('Dimension 2', fontsize=14, fontweight='bold')
    ax.set_title('Authorship Detection - 2D Visualization\n' +
                 'Training Samples, Cluster Boundaries, and Predictions',
                 fontsize=16, fontweight='bold', pad=20)
    
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)
    
    # Legend (only show unique items)
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Filter to show only author names and new email
    filtered = {k: v for k, v in by_label.items() if 'Contact' in k and 'boundary' not in k and 'threshold' not in k}
    if new_email_pos is not None:
        filtered['New Email'] = by_label.get(f'New Email ({decision})', None)
    
    ax.legend(
        filtered.values(),
        filtered.keys(),
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        framealpha=0.9
    )
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Visualization saved to: {output_path}")
    
    return fig, ax


def visualize_new_email(email_features, models, projected_data, y_train, 
                        cluster_stats, projection_reducer, method='tsne'):
    """
    Project and visualize a new email
    
    Args:
        email_features: Feature vector for new email (already scaled)
        models: Dict of loaded models
        projected_data: 2D training data
        y_train: Training labels
        cluster_stats: Cluster statistics
        projection_reducer: Fitted UMAP/t-SNE model
        method: 'tsne' (default) or 'umap'
    
    Returns:
        new_pos: (x, y) coordinates
        prediction: Prediction result from open_set_detector
    """
    # Get prediction from open-set detector
    open_set_detector = models['open_set_detector']
    prediction = open_set_detector.predict(email_features, return_distances=True)
    
    # Project to 2D
    if method == 'umap' and USE_UMAP:
        # UMAP can transform new points
        new_pos = projection_reducer.transform([email_features])[0]
    else:
        # t-SNE cannot transform new points
        # Use k-nearest neighbor approximation with weighted average
        from scipy.spatial.distance import cdist
        X_train = open_set_detector.X_train
        
        # Find k nearest neighbors
        k = 5
        distances = cdist([email_features], X_train)[0]
        nearest_indices = np.argsort(distances)[:k]
        
        # Weight by inverse distance
        nearest_distances = distances[nearest_indices]
        weights = 1.0 / (nearest_distances + 1e-6)
        weights = weights / weights.sum()
        
        # Weighted average of nearest neighbors' 2D positions
        nearest_2d_points = projected_data[nearest_indices]
        new_pos = np.average(nearest_2d_points, axis=0, weights=weights)
        
        print(f"   ℹ️  Using k-NN approximation (k={k}) for t-SNE projection")
    
    return new_pos, prediction


def main():
    """Generate visualization from saved models"""
    print("="*60)
    print("🎨 AUTHORSHIP DETECTION VISUALIZATION")
    print("="*60)
    
    # Load models
    models = load_models("models")
    
    # Extract training data from open_set_detector
    open_set_detector = models['open_set_detector']
    X_train = open_set_detector.X_train
    y_train = open_set_detector.y_train
    label_encoder = models['label_encoder']
    
    if X_train is None or y_train is None:
        print("❌ Training data not found in open_set_detector")
        print("   Make sure you're using the updated training script")
        return
    
    print(f"\n📊 Training data:")
    print(f"   Samples: {len(X_train)}")
    print(f"   Features: {X_train.shape[1]}")
    print(f"   Authors: {len(np.unique(y_train))}")
    
    # Create 2D projection
    method = 'tsne'  # Default to t-SNE (built into scikit-learn)
    projected_data, projection_reducer = create_2d_projection(X_train, y_train, method)
    
    # Compute cluster statistics
    print("\n📍 Computing cluster boundaries...")
    cluster_stats = compute_cluster_stats(projected_data, y_train, label_encoder)
    
    for author, stats in sorted(cluster_stats.items(), key=lambda x: int(x[0])):
        print(f"   Contact {author}:")
        print(f"      Samples: {stats['n_samples']}")
        print(f"      95% radius: {stats['radius_95']:.2f}")
        print(f"      Rejection threshold: {stats['radius_threshold']:.2f}")
    
    # Plot base visualization (without new email)
    print("\n🎨 Creating visualization...")
    plot_visualization(
        projected_data,
        y_train,
        cluster_stats,
        label_encoder,
        output_path='authorship_visualization_base.png'
    )
    
    print("\n✅ Done!")
    print("\n💡 To visualize a new email:")
    print("   1. Extract features and normalize them")
    print("   2. Call visualize_new_email()")
    print("   3. Pass result to plot_visualization()")
    
    # Save projection model and data for later use
    output_data = {
        'projection_reducer': projection_reducer,
        'projected_data': projected_data,
        'y_train': y_train,
        'cluster_stats': cluster_stats,
        'method': method
    }
    
    with open('models/visualization_data.pkl', 'wb') as f:
        pickle.dump(output_data, f)
    
    print("\n💾 Saved visualization data to: models/visualization_data.pkl")


if __name__ == "__main__":
    main()