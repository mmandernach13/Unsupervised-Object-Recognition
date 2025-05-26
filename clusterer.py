import types
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from skimage.feature import hog
from sklearn.preprocessing import StandardScaler
import pickle
import torch
import torchvision.transforms as transforms
from tqdm import tqdm

class Clusterer:
    def __init__(self, num_clusters=10, feature_extraction_method='moco',
                 clustering_method='gmm', output_dir=None):
        """
        Initialize the clusterer with options for different feature extraction and clustering methods

        Parameters:
        - num_clusters: Number of clusters for clustering method
        - feature_extraction_method: 'hog', 'dino', 'moco', or 'simclr'
        - clustering_method: 'kmeans' or 'gmm'
        """
        self.clustering_method = clustering_method
        self.num_clusters = num_clusters
        self.cluster_centers = None
        self.cluster_labels = None

        self.feature_extraction_method = feature_extraction_method

        self.pca = None
        self.scaler = None
        self.pca_whitener = None

        self.output_dir = output_dir

        self.last_processed_images = None

        # Initialize appropriate model based on method
        if feature_extraction_method == 'dino':
            self._initialize_dino_model()
        elif feature_extraction_method == 'moco':
            self._initialize_moco_model()

        # aliases to match requested function names
        self.processImagery = self.process_imagery
        self.getClusterCenters = self.get_cluster_centers

    def _initialize_dino_model(self):
        """Initialize DINO ViT model for feature extraction"""

        # Load pre-trained DINO model
        try:
            # Using TIMM if available
            import timm
            self.model = timm.create_model('vit_small_patch16_224.dino', pretrained=True)
            self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        except ImportError:
            # Fallback to torch hub
            self.model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')

            def forward_features(x):
                # Extract features before classification head
                x = self.model.prepare_tokens(x)
                for blk in self.model.blocks:
                    x = blk(x)
                x = self.model.norm(x)
                return x[:, 0]  # Return CLS token features

            # Monkey patch the forward method
            self.model.forward_features = types.MethodType(forward_features, self.model)
            self.feature_extractor = self.model

        self.feature_extractor.eval()

        # Set up image transformation for ViT
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def _initialize_moco_model(self):
        """Initialize MoCo v3 model for feature extraction"""
        import torch
        import torchvision.models as models

        # Load pre-trained MoCo model
        try:
            # Try to load from torch hub
            self.model = torch.hub.load('facebookresearch/moco-v3', 'resnet50_mocov3')
            self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])
        except:
            # Fallback to torchvision if available
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
            self.feature_extractor = torch.nn.Sequential(*list(self.model.children())[:-1])

        self.feature_extractor.eval()

        # Set up image transformation
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def process_imagery(self, images):
        """
        Process images using the selected feature extraction method
        """
        if self.feature_extraction_method == 'hog':
            return self._process_imagery_hog(images)
        elif self.feature_extraction_method in ['simclr', 'dino', 'moco']:
            return self._process_imagery_ssl(images)
        else:
            raise ValueError(f"Unknown feature extraction method: {self.feature_extraction_method}")

    def _process_imagery_hog(self, images):
        """
        Enhanced HOG feature extraction with multi-scale and color features
        """
        # Normalize the images to the range [0, 1]
        images = images / 255.0

        # Initialize feature storage
        all_features = []
        with tqdm(total= len(images)) as pbar:
            for image in images:

                # Reshape image from 3072 to 32x32x3
                image = image.reshape(3, 32, 32).transpose(1, 2, 0)

                # Extract multiple HOG features with different parameters for better representation
                fd1, _ = hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                             visualize=True, channel_axis=2)

                fd2, _ = hog(image, pixels_per_cell=(4, 4), cells_per_block=(2, 2),
                             visualize=True, channel_axis=2)

                # Extract color histogram features (captures color information HOG misses)
                hist_features = []
                for channel in range(3):
                    hist, _ = np.histogram(image[:, :, channel], bins=32, range=(0, 1))
                    hist_features.extend(hist / np.sum(hist))  # normalize

                # Combine all features
                combined_features = np.concatenate([fd1, fd2, hist_features])
                all_features.append(combined_features)

                pbar.update(1)

        all_features = np.array(all_features)

        # Apply feature normalization before PCA
        if self.scaler is not None and all_features.shape[1] != self.scaler.n_features_in_:
            self.scaler = None
        if self.scaler is None:
            self.scaler = StandardScaler()
            all_features = self.scaler.fit_transform(all_features)
        else:
            all_features = self.scaler.transform(all_features)

        # Apply PCA with variance retention instead of fixed components
        if self.pca is None:
            self.pca = PCA(n_components=0.95)  # Keep 95% of variance
            reduced_features = self.pca.fit_transform(all_features)
            print(f"PCA reduced dimensions from {all_features.shape[1]} to {reduced_features.shape[1]}")
            print(f"Variance retained: {sum(self.pca.explained_variance_ratio_):.4f}")
        else:
            reduced_features = self.pca.transform(all_features)

        return reduced_features

    def _process_imagery_ssl(self, images):
        """
        Extract features using a self-supervised learning model
        with PCA dimensionality reduction
        """

        # Normalize images
        images = images / 255.0

        features = []
        with tqdm(total=len(images)) as pbar:
            for image in images:
                # Reshape image from 3072 to 32x32x3
                image = image.reshape(3, 32, 32).transpose(1, 2, 0)

                # Convert to PIL Image, apply transformations
                img_tensor = self.transform(image)
                img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

                # Extract features
                with torch.no_grad():
                    if self.feature_extraction_method == 'dino':
                        # DINO models have a different extraction pattern
                        feature = self.feature_extractor(img_tensor)
                    else:
                        feature = self.feature_extractor(img_tensor)

                # Flatten and convert to numpy
                feature = feature.squeeze().flatten().numpy()
                features.append(feature)

                pbar.update(1)

        features = np.array(features)
        print(f"Raw feature shape: {features.shape}")

        # Apply standardization
        if self.scaler is not None and features.shape[1] != self.scaler.n_features_in_:
            self.scaler = None
        if self.scaler is None:
            self.scaler = StandardScaler()
            features = self.scaler.fit_transform(features)
        else:
            features = self.scaler.transform(features)

        # Apply PCA with whitening
        if self.pca_whitener is None:
            # Determine number of components to retain at least 85% variance
            # First, find how many components we need with a temporary PCA
            temp_pca = PCA(n_components=min(1000, features.shape[1]),
                           random_state=42).fit(features)
            cumsum = np.cumsum(temp_pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= 0.85) + 1
            n_components = max(min(n_components, features.shape[1]), 100)

            print(f"Applying PCA with whitening to reduce from {features.shape[1]} to {n_components} dimensions")
            self.pca_whitener = PCA(
                n_components=n_components,
                whiten=True,  # Normalize variance in each direction
                random_state=42
            )
            features = self.pca_whitener.fit_transform(features)
            print(f"Variance retained: {self.pca_whitener.explained_variance_ratio_.sum():.4f}")
        else:
            features = self.pca_whitener.transform(features)

        return features

    def get_cluster_centers(self, batch_filename):
        """
        Perform clustering on images in the specified batch file.
        """
        print(f"Loading data from {batch_filename}...")
        with open(batch_filename, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        images = batch[b"data"]

        print(f"Extracting features using {self.feature_extraction_method} method...")
        processed_images = self.process_imagery(images)
        self.last_processed_images = processed_images

        print(f"Processed images shape: {processed_images.shape}")

        # Select clustering algorithm based on method
        if self.clustering_method == 'kmeans':
            print(f"Starting K-means clustering with {self.num_clusters} clusters...")
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, max_iter=100)
            self.cluster_labels = kmeans.fit_predict(processed_images)
            self.cluster_centers = kmeans.cluster_centers_

        elif self.clustering_method == 'gmm':
            print(f"Starting GMM clustering with {self.num_clusters} components...")
            from sklearn.mixture import GaussianMixture
            gmm = GaussianMixture(n_components=self.num_clusters, covariance_type='diag',
                                  random_state=42, max_iter=100, n_init=3)
            gmm.fit(processed_images)
            self.cluster_labels = gmm.predict(processed_images)
            self.cluster_centers = gmm.means_

        print("Clustering complete")

        return self.cluster_centers, self.cluster_labels

    def visualize_clusters(self, batch_filename=None, save_path=None):
        """
        Visualize the clusters in 2D using t-SNE for dimensionality reduction.
        Shows cluster centers with 'x' markers and original images as points.

        Parameters:
        - batch_filename: Optional. If provided, will load and process this batch file.
                          If None, uses the most recently processed data.
        - save_path: Optional. If provided, saves the visualization to this path.
        """
        if self.cluster_centers is None:
            raise ValueError("No cluster centers available. Run get_cluster_centers first.")

        # Process data if batch_filename is provided
        if batch_filename is not None:
            print(f"Loading data from {batch_filename}...")
            with open(batch_filename, 'rb') as f:
                batch = pickle.load(f, encoding='bytes')
            images = batch[b"data"]

            print("Processing images for visualization...")
            processed_images = self.process_imagery(images)

            # Assign each point to a cluster
            print("Assigning points to clusters...")
            dist_to_centers = np.linalg.norm(processed_images[:, np.newaxis] - self.cluster_centers, axis=2)
            point_labels = np.argmin(dist_to_centers, axis=1)
        else:
            # Use the last processed images and cluster assignments if available
            if not hasattr(self, 'last_processed_images') or self.last_processed_images is None:
                raise ValueError(
                    "No processed images available. Provide a batch_filename or run get_cluster_centers first.")
            processed_images = self.last_processed_images
            point_labels = self.cluster_labels

        # Apply t-SNE for dimensionality reduction
        print("Computing t-SNE embedding for visualization")
        # Adjust perplexity for dataset size
        perplexity = min(30, max(5, processed_images.shape[0] // 100))
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)

        # Project both data points and cluster centers
        all_data = np.vstack([processed_images, self.cluster_centers])
        embedded_data = tsne.fit_transform(all_data)

        # Split the results
        embedded_points = embedded_data[:len(processed_images)]
        embedded_centers = embedded_data[len(processed_images):]

        # Create the plot
        plt.figure(figsize=(12, 10))

        cmap = plt.cm.get_cmap('tab10', self.num_clusters)

        # Plot data points colored by their cluster assignment
        scatter_points = plt.scatter(
            embedded_points[:, 0],
            embedded_points[:, 1],
            c=point_labels,
            cmap=cmap,
            alpha=0.5,
            s=30,
            marker='o',
            label='Images'
        )

        # Plot cluster centers with 'x' markers
        for i in range(self.num_clusters):
            plt.scatter(
                embedded_centers[i, 0],
                embedded_centers[i, 1],
                color=cmap(i),
                s=250,
                marker='X',  # Capital X is larger and more distinct
                edgecolors='white',
                linewidths=2,
                zorder=10  # Ensure centers are drawn on top of points
            )

        plt.colorbar(label='Cluster')
        plt.title('Image Clusters Visualization')
        plt.xlabel('t-SNE dimension 1')
        plt.ylabel('t-SNE dimension 2')
        plt.legend()
        plt.grid(alpha=0.3)

        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()

        return embedded_points, embedded_centers, point_labels

    def visualize_cluster_examples(self, batch_filename, num_examples=5, save_path=None):
        """
        Visualize example images from each cluster.

        Parameters:
        - batch_filename: Path to batch file with images
        - num_examples: Number of example images to show per cluster
        - save_path: Optional path to save the visualization
        - save_images: If True, save the first 10 images per cluster to separate directories
        """
        if self.cluster_centers is None:
            raise ValueError("No cluster centers available. Run get_cluster_centers first.")

        # Load images
        with open(batch_filename, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        images = batch[b"data"]

        # Process images and assign to clusters if not already done
        if not hasattr(self, 'last_processed_images') or self.last_processed_images is None:
            processed_images = self.process_imagery(images)
            dist_to_centers = np.linalg.norm(processed_images[:, np.newaxis] - self.cluster_centers, axis=2)
            cluster_assignments = np.argmin(dist_to_centers, axis=1)
        else:
            processed_images = self.last_processed_images
            cluster_assignments = self.cluster_labels

        # Number of clusters to visualize
        n_clusters = len(self.cluster_centers)

        # Create figure with subplots
        fig, axes = plt.subplots(n_clusters, num_examples, figsize=(num_examples * 2, n_clusters * 2))

        # For each cluster, show example images
        for i in range(n_clusters):
            # Get indices of images in this cluster
            cluster_indices = np.where(cluster_assignments == i)[0]

            if len(cluster_indices) == 0:
                print(f"Warning: Cluster {i} is empty")
                continue

            # Select random examples (or all if fewer than requested)
            num_to_show = min(num_examples, len(cluster_indices))
            example_indices = np.random.choice(cluster_indices, num_to_show, replace=False)

            # Plot each example
            for j, idx in enumerate(example_indices):
                if n_clusters == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]

                # Convert image to displayable format (32x32x3)
                img = images[idx].reshape(3, 32, 32).transpose(1, 2, 0) / 255.0
                ax.imshow(img)
                ax.axis('off')

                if j == 0:
                    ax.set_title(f"Cluster {i}\n({len(cluster_indices)} images)")

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path)
            print(f"Cluster examples visualization saved to {save_path}")
        else:
            plt.show()

    def calculate_cluster_purity(self, batch_filename, save_path=None, evaluate_rank_k=True, max_k=10):
        """
        Calculate cluster purity and rank-k classification accuracy.

        Parameters:
        - batch_filename: Path to batch file with labeled images
        - evaluate_rank_k: Whether to evaluate rank-k classification accuracy
        - max_k: Maximum k value for rank-k evaluation

        Returns:
        - Dictionary with cluster IDs as keys and (purity score, dominant label) tuples as values
        - Overall average purity score
        - Dictionary with rank-k accuracies if evaluate_rank_k is True
        """
        # Load images and their true labels
        with open(batch_filename, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')

        images = batch[b"data"]
        true_labels = batch[b"labels"]

        # Get the label names if available
        label_names = None
        if b"label_names" in batch:
            label_names = [name.decode('utf-8') for name in batch[b"label_names"]]

        # Process images and assign to clusters
        processed_images = self.last_processed_images
        dist_to_centers = np.linalg.norm(processed_images[:, np.newaxis] - self.cluster_centers, axis=2)
        cluster_assignments = np.argmin(dist_to_centers, axis=1)

        # Calculate purity for each cluster
        cluster_purity = {}
        for cluster_id in range(self.num_clusters):
            # Get indices of images assigned to this cluster
            cluster_indices = np.where(cluster_assignments == cluster_id)[0]

            if len(cluster_indices) == 0:
                cluster_purity[cluster_id] = (0.0, None, {})
                continue

            # Get true labels of images in this cluster
            cluster_true_labels = [true_labels[i] for i in cluster_indices]

            # Count occurrences of each label
            label_counts = {}
            for label in cluster_true_labels:
                label_counts[label] = label_counts.get(label, 0) + 1

            # Find the most common label and its count
            most_common_label = max(label_counts, key=label_counts.get)
            most_common_count = label_counts[most_common_label]

            # Calculate purity (percentage of most common label)
            purity = most_common_count / len(cluster_indices) * 100
            cluster_purity[cluster_id] = (purity, most_common_label, label_counts)

        # Calculate overall average purity
        average_purity = sum(purity for purity, _, _ in cluster_purity.values()) / len(cluster_purity)

        # Group clusters by dominant label
        label_to_clusters = {}
        for cluster_id, (purity, dominant_label, _) in cluster_purity.items():
            if dominant_label not in label_to_clusters:
                label_to_clusters[dominant_label] = []
            label_to_clusters[dominant_label].append((cluster_id, purity))

        # Sort clusters within each dominant label group by purity (descending)
        for label in label_to_clusters:
            label_to_clusters[label].sort(key=lambda x: x[1], reverse=True)

        # Print results
        print("\nCluster Purity Analysis (Grouped by Dominant Class):")
        print("----------------------------------------------------")

        for dominant_label in sorted(label_to_clusters.keys()):
            if dominant_label is None:
                print("\nClusters with no dominant class:")
                continue

            label_name = label_names[dominant_label] if label_names else f"Class {dominant_label}"
            print(f"\nDominant Class: {dominant_label} ({label_name})")
            print("-" * 40)

            for cluster_id, purity in label_to_clusters[dominant_label]:
                # Get the distribution of labels in this cluster for more details
                _, _, label_counts = cluster_purity[cluster_id]
                total_images = sum(label_counts.values())

                # Format the label distribution
                if label_names:
                    distribution = ", ".join([f"{label_names[label]}: {count / total_images * 100:.1f}%"
                                              for label, count in sorted(label_counts.items(),
                                                                         key=lambda x: x[1],
                                                                         reverse=True)[:3]])  # Show top 3 labels
                else:
                    distribution = ", ".join([f"Class {label}: {count / total_images * 100:.1f}%"
                                              for label, count in sorted(label_counts.items(),
                                                                         key=lambda x: x[1],
                                                                         reverse=True)[:3]])  # Show top 3 labels

                print(f"Cluster {cluster_id}: {purity:.2f}% purity, {total_images} images")
                print(f"   Top labels: {distribution}")

        print("\nOverall Statistics:")
        print(f"Average cluster purity: {average_purity:.2f}%")
        print(f"Number of clusters: {len(cluster_purity)}")
        print(f"Number of unique dominant classes: {len([k for k in label_to_clusters.keys() if k is not None])}")

        # Print duplicate classes (classes that are dominant in multiple clusters)
        duplicate_classes = [label for label, clusters in label_to_clusters.items()
                             if label is not None and len(clusters) > 1]

        if duplicate_classes:
            print("\nDuplicate dominant classes (appearing in multiple clusters):")
            for label in sorted(duplicate_classes):
                clusters = label_to_clusters[label]
                label_name = label_names[label] if label_names else f"Class {label}"
                print(f"Class {label} ({label_name}) appears in {len(clusters)} clusters: " +
                      ", ".join([f"Cluster {c_id} ({purity:.1f}%)" for c_id, purity in clusters]))

        # Implement rank-k classification accuracy
        rank_k_accuracies = {}
        if evaluate_rank_k:
            print("\nRank-k Classification Accuracy:")
            print("------------------------------")

            # Create a mapping from cluster ID to its dominant class label
            cluster_to_label = {cluster_id: dominant_label for cluster_id, (_, dominant_label, _)
                                in cluster_purity.items() if dominant_label is not None}

            # Calculate similarity scores for each image to all clusters
            # (using negative distance as similarity)
            similarity_scores = -dist_to_centers

            # For each k value, calculate the rank-k accuracy
            for k in range(1, min(max_k + 1, self.num_clusters + 1)):
                # Get the top-k cluster predictions for each image
                top_k_clusters = np.argsort(-similarity_scores, axis=1)[:, :k]

                # Convert cluster predictions to label predictions
                correct_count = 0
                for i, (image_top_k_clusters, true_label) in enumerate(zip(top_k_clusters, true_labels)):
                    # Get the dominant labels for the top-k clusters
                    predicted_labels = [cluster_to_label.get(cluster_id) for cluster_id in image_top_k_clusters
                                        if cluster_id in cluster_to_label]

                    # Check if the true label is in the predicted labels
                    if true_label in predicted_labels:
                        correct_count += 1

                # Calculate accuracy
                accuracy = correct_count / len(true_labels) * 100
                rank_k_accuracies[k] = accuracy
                print(f"Rank-{k} accuracy: {accuracy:.2f}%")

            # Plot rank-k curve
            plt.figure(figsize=(10, 6))
            plt.plot(list(rank_k_accuracies.keys()), list(rank_k_accuracies.values()), 'o-', linewidth=2)
            plt.xlabel('k')
            plt.ylabel('Accuracy (%)')
            plt.title('Rank-k Classification Accuracy')
            plt.grid(True)
            plt.xticks(list(rank_k_accuracies.keys()))
            plt.ylim(0, 105)

            # Save the plot if output directory is specified
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Rank-k plot saved to {save_path}")
            else:
                plt.show()

        return cluster_purity, average_purity, rank_k_accuracies if evaluate_rank_k else None