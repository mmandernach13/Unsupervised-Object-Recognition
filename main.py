import os
import sys
import argparse
from clusterer import Clusterer

def main():
    parser = argparse.ArgumentParser(description='Unsupervised Image Clustering')
    parser.add_argument('batch_file', help='Path to the batch file')
    parser.add_argument('--m', choices=['hog', 'simclr', 'dino', 'moco'], default='moco',
                        help='Feature extraction method to use (default: moco)')
    parser.add_argument('--cm', choices=['kmeans', 'gmm'],
                        default='gmm', help='Clustering method to use (default: gmm)')
    parser.add_argument('--cn', type=int, default=10,
                        help='Number of clusters (default: 10)')
    parser.add_argument('--a', action='store_true',
                        help='Analyze cluster purity')
    parser.add_argument('--o', default='results',
                        help='Directory to save output files (default: results)')
    args = parser.parse_args()


    print(f"Initializing clusterer using {args.cn} {args.cm} clusters and {args.m} feature extraction")
    clusterer = Clusterer(num_clusters=args.cn, feature_extraction_method=args.m,
                          clustering_method=args.cm, output_dir=args.o)

    # Check if the provided file exists
    if not os.path.exists(args.batch_file):
        print(f"Error: The file {args.batch_file} does not exist.")
        sys.exit(1)

    # Create output directory if it doesn't exist
    if not os.path.exists(clusterer.output_dir):
        os.makedirs(clusterer.output_dir)
        print(f"Created output directory: {clusterer.output_dir}")

    # Perform clustering on the provided batch file
    print(f"Performing clustering on {args.batch_file}...")
    cluster_centers, cluster_labels = clusterer.get_cluster_centers(args.batch_file)

    # Visualize the clusters
    print("Visualizing clusters...")
    visualization_path = os.path.join(clusterer.output_dir, f'clusters_{clusterer.feature_extraction_method}_{clusterer.clustering_method}.png')
    clusterer.visualize_clusters(save_path=visualization_path)

    # Visualize example images from each cluster
    print("Visualizing cluster examples...")
    examples_path = os.path.join(clusterer.output_dir, f'cluster_examples_{clusterer.feature_extraction_method}_{clusterer.clustering_method}.png')
    clusterer.visualize_cluster_examples(args.batch_file, num_examples=10, save_path=examples_path)

    # Calculate and display cluster purity if requested
    if args.a:
        print("Analyzing cluster purity...")
        purity_dict, avg_purity, rank_k_accuracies = clusterer.calculate_cluster_purity(args.batch_file)

        # Save purity results to a file
        purity_path = os.path.join(clusterer.output_dir,
                                   f'cluster_purity_{clusterer.feature_extraction_method}_{clusterer.clustering_method}.txt')
        with open(purity_path, 'w') as f:
            # Group clusters by dominant label
            label_to_clusters = {}
            for cluster_id, (purity, dominant_label, _) in purity_dict.items():
                if dominant_label not in label_to_clusters:
                    label_to_clusters[dominant_label] = []
                label_to_clusters[dominant_label].append((cluster_id, purity))

            # Sort clusters within each dominant label group by purity
            for label in label_to_clusters:
                label_to_clusters[label].sort(key=lambda x: x[1], reverse=True)

            f.write("Cluster Purity Analysis (Grouped by Dominant Class):\n")
            f.write("----------------------------------------------------\n")

            for dominant_label in sorted(label_to_clusters.keys()):
                if dominant_label is None:
                    f.write("\nClusters with no dominant class:\n")
                    continue

                f.write(f"\nDominant Class: {dominant_label}\n")
                f.write("-" * 40 + "\n")

                for cluster_id, purity in label_to_clusters[dominant_label]:
                    f.write(f"Cluster {cluster_id}: {purity:.2f}% purity\n")

            f.write("\nOverall Statistics:\n")
            f.write(f"Average cluster purity: {avg_purity:.2f}%\n")
            f.write(f"Number of clusters: {len(purity_dict)}\n")

            # Print duplicate classes (classes that are dominant in multiple clusters)
            duplicate_classes = [label for label, clusters in label_to_clusters.items()
                                 if label is not None and len(clusters) > 1]

            if duplicate_classes:
                f.write("\nDuplicate dominant classes (appearing in multiple clusters):\n")
                for label in sorted(duplicate_classes):
                    clusters = label_to_clusters[label]
                    f.write(f"Class {label} appears in {len(clusters)} clusters: " +
                            ", ".join([f"Cluster {c_id} ({purity:.1f}%)" for c_id, purity in clusters]) + "\n")

            # Save rank-k accuracies if available
            if rank_k_accuracies:
                f.write("\nRank-k Classification Accuracy:\n")
                f.write("------------------------------\n")
                for k, accuracy in rank_k_accuracies.items():
                    f.write(f"Rank-{k} accuracy: {accuracy:.2f}%\n")

        print(f"Purity analysis saved to {purity_path}")

    print("All processing completed successfully!")

if __name__ == "__main__":
    main()