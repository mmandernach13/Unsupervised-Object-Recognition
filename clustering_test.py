import os
from clusterer import Clusterer

def main():
    num_clusters = [10, 20, 30]
    feature_types = ['hog', 'dino', 'moco']
    clustering_types = ['kmeans', 'gmm']
    batch_file = 'cifar-10/data_batch_2'

    for method in feature_types:
        for num in num_clusters:
            for ctype in clustering_types:
                c = Clusterer(feature_extraction_method=method, num_clusters=num, clustering_method=ctype, output_dir='test/')

                c.get_cluster_centers(batch_file)

                visualization_path = os.path.join(c.output_dir,
                                                  f'{c.feature_extraction_method}/clusters_{c.clustering_method}_{c.num_clusters}.png')
                c.visualize_clusters(save_path=visualization_path)
                examples_path = os.path.join(c.output_dir,
                                                  f'{c.feature_extraction_method}/examples_{c.clustering_method}_{c.num_clusters}.png')
                c.visualize_cluster_examples(batch_filename=batch_file, save_path=examples_path)

                kmeans_plt_path = os.path.join(c.output_dir,
                                           f'{c.feature_extraction_method}/kmeans_plt_{c.clustering_method}_{c.num_clusters}.png')
                purity_dict, avg_purity, rank_k_accuracies = c.calculate_cluster_purity(batch_file, save_path=kmeans_plt_path)

                # Save purity results to a file
                purity_path = os.path.join(c.output_dir,
                                           f'{c.feature_extraction_method}/cluster_purity_{c.clustering_method}_{c.num_clusters}.txt')
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

if __name__ == "__main__":
    main()