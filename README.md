# Unsupervised Image Clustering

This project implements an unsupervised image clustering pipeline using various feature extraction and clustering methods. It processes image datasets (e.g., CIFAR-10) to extract features using techniques like Histogram of Oriented Gradients (HOG), DINO (Vision Transformer), or MoCo (ResNet50-based), and clusters them using K-means or Gaussian Mixture Models (GMM). The project includes visualization tools to display cluster distributions and example images, as well as metrics like cluster purity and rank-k classification accuracy to evaluate clustering performance.

## Features

- **Feature Extraction**:
  - Histogram of Oriented Gradients (HOG) with multi-scale and color histogram features for robust image representation.
  - Self-supervised learning models: DINO (Vision Transformer) and MoCo (ResNet50-based) for advanced feature extraction.
- **Clustering Algorithms**:
  - K-means for simple, centroid-based clustering.
  - Gaussian Mixture Models (GMM) for probabilistic clustering with diagonal covariance.
- **Visualization**:
  - 2D t-SNE plots showing cluster distributions and centers.
  - Example images from each cluster to visually inspect clustering results.
- **Evaluation**:
  - Cluster purity analysis to measure the dominance of a single class per cluster.
  - Rank-k classification accuracy to assess clustering performance as a pseudo-classifier.
- **Command-Line Interface**:
  - Flexible script for single-run clustering with customizable parameters.
  - Batch testing script to evaluate multiple configurations.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. **Install Dependencies**: Ensure you have Python 3.8+ installed. Install the required packages using:

   ```bash
   pip install -r requirements.txt
   ```

3. **Download CIFAR-10 Dataset**: Download the CIFAR-10 dataset (Python version) from https://www.cs.toronto.edu/\~kriz/cifar.html and extract it. Place the batch files (e.g., `data_batch_1`, `data_batch_2`) in a `cifar-10` folder within the project directory:

   ```
   <your-repo-name>/cifar-10/data_batch_1
   ```

## Usage

The project provides two main scripts to run clustering experiments:

### 1. Main Script (`main.py`)

Run clustering on a single batch file with customizable options:

```bash
python main.py <batch_file> --m <method> --cm <clustering_method> --cn <num_clusters> --a --o <output_dir>
```

- `<batch_file>`: Path to the CIFAR-10 batch file (e.g., `cifar-10/data_batch_1`).
- `--m`: Feature extraction method (`hog`, `dino`, `moco`; default: `moco`).
- `--cm`: Clustering method (`kmeans`, `gmm`; default: `gmm`).
- `--cn`: Number of clusters (default: 10).
- `--a`: Analyze cluster purity (optional; includes rank-k accuracy).
- `--o`: Output directory for results (default: `results`).

**Example**:

```bash
python main.py cifar-10/data_batch_1 --m hog --cm kmeans --cn 10 --a --o results
```

This command processes `data_batch_1` using HOG features and K-means clustering with 10 clusters, saving visualizations and purity analysis to the `results` folder.

### 2. Test Script (`clustering_test.py`)

Run clustering experiments with multiple configurations:

```bash
python clustering_test.py
```

This script tests all combinations of:

- Feature extraction methods: `hog`, `dino`, `moco`.
- Clustering methods: `kmeans`, `gmm`.
- Number of clusters: 10, 20, 30.

Results are saved in the `test` directory, organized by feature extraction method (e.g., `test/hog/`, `test/dino/`).

## Output

The scripts generate the following outputs in the specified output directory (`results` or `test`):

- **Cluster Visualizations**: 2D t-SNE plots showing image clusters and centers (e.g., `clusters_hog_kmeans.png`).
- **Cluster Examples**: Grids of example images from each cluster (e.g., `examples_hog_kmeans.png`).
- **Purity Analysis**: Text files with cluster purity, dominant classes, and rank-k accuracies (e.g., `cluster_purity_hog_kmeans.txt`).
- **Rank-k Accuracy Plots**: Graphs of rank-k classification accuracy when purity analysis is enabled (e.g., `kmeans_plt_hog_kmeans.png`).

## Project Structure

```
<your-repo-name>/
│
├── clusterer.py              # Core clustering and feature extraction logic
├── clustering_test.py        # Script to test multiple clustering configurations
├── main.py                   # Main script for single-run clustering
├── cifar-10/                 # Directory for CIFAR-10 batch files (user-provided)
├── results/                  # Output directory for single-run results
├── test/                     # Output directory for test script results
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

## Dependencies

See `requirements.txt` for the full list of dependencies. Key libraries include:

- `numpy`: Numerical computations and array handling.
- `matplotlib`: Visualization of clusters and rank-k plots.
- `scikit-learn`: Clustering (K-means, GMM) and PCA.
- `scikit-image`: HOG feature extraction.
- `torch` and `torchvision`: Self-supervised model loading and image preprocessing.
- `timm`: Optional, for DINO model loading.
- `tqdm`: Progress bars for feature extraction.

## Notes

- **Dataset Compatibility**: The code is designed for CIFAR-10 (32x32 RGB images stored as 3072-dimensional vectors). To use other datasets, ensure images are preprocessed to match this format.
- **DINO and MoCo Models**: The project uses pre-trained DINO (via TIMM or Torch Hub) and MoCo (via Torch Hub or torchvision’s ResNet50). A GPU is recommended for these methods to improve performance.
- **Troubleshooting**:
  - **Dimension Mismatch Errors**: If you encounter errors with StandardScaler or PCA, ensure input data matches the expected format. The code automatically resets the scaler if feature dimensions change.
  - **Model Loading**: If TIMM is unavailable, the code falls back to Torch Hub or torchvision for model loading. Ensure internet access for Torch Hub or install TIMM for DINO.
- **Performance**: HOG is lightweight and CPU-friendly, while DINO and MoCo benefit from CUDA-enabled GPUs.
- **Extensibility**: The `simclr` feature extraction method is referenced but not implemented. You can extend `clusterer.py` to add support for SimCLR or other methods.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue on GitHub to suggest improvements, report bugs, or add new feature extraction/clustering methods.

## License

This project is licensed under the MIT License.

## Acknowledgments

- Built with inspiration from self-supervised learning research, including DINO and MoCo.
- Uses the CIFAR-10 dataset for testing and evaluation.
- Leverages open-source libraries like PyTorch, scikit-learn, and scikit-image.