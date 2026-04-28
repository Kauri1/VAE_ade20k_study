# ADE20K Latent Space Interpretability Specification

## 1. Project Overview & Goals
This project investigates the continuous representation and interpretability of highly complex visual scenes using deep generative models, specifically evaluated on the ADE20K dataset. The central objective is to train variants of Variational Autoencoders ($\beta$-VAEs) to learn compressed, meaningful latent dimensions, and systematically map these encoded vectors to human-understandable scene concepts.

Secondary benchmarking is achieved by pipelining these extracted latent continuous representations into downstream classification boundaries (using CNNs and MLPs) to rigorously evaluate disentanglement, spatial separation, and concept retention.

---

## 2. Data Pipeline & Augmentation
The data loading framework is localized within `ade20k_dataset.py`, providing a highly configurable pipeline for querying the `ADEData2016` dataset.

### 2.1 Dynamic Subset Filtration
The pipeline supports extensive filtering, ensuring targeted scenario analysis:
*   **Targeted Labels (`n_common_labels`)**: Restricts the dataset iterators to only sample from the top $N$ most frequently occurring visual scene labels.
*   **Concept Exclusion (`exclude_concepts`)**: A dynamic masking feature discarding particular scenes conceptually irrelevant to the target distribution.

### 2.2 Preprocessing and Augmentations
The preprocessing workflow implements substantial augmentations for training robustness (as outlined in `show_augmentations.py` and standard subsets):
*   **Standard Processing**: Dynamic resizing (configurable spatial mappings, standard defaulting to $256 \times 256$) and Tensor transformation.
*   **Augmentation (Train split only)**: Employs padding (reflect mode), rotational affine bounds (e.g., $5^{\circ}$), random resized cropping (scales $0.8 \rightarrow 1$), horizontal flipping, color jittering (brightness, contrast, saturation), and stochastic sharpness adjustments.

### 2.3 Dual-Input Modality
The `get_dataloaders` engine utilizes a dual-pathway design: 
*   **Standard Mode**: Yields raw normalized $[B, C, H, W]$ dimensional tensors.
*   **Latent Mode (`latent_dir`)**: Bypasses image extraction entirely, instead loading directly from pre-computed `.pt` latent tensors. This optimization enables exceedingly fast supervised training iterating strictly over learned latent boundaries for downstream classifiers.

---

## 3. Model Architectures
The project explores the encoding structures via generative networks (`vae_model.py`) and evaluates the encoded distributions against decision-boundary networks (`cnn_model.py`).

### 3.1 Generative Variants (VAEs)
The generative codebase allows dynamic instantiations of the core VAE, leveraging distinct architectural paradigms:
*   **Core `VAE`**: Parameterized symmetrical CNN encoder-decoders. Utilizes configurable bounds (`max_channels`, `min_channels`, `bottleneck_spatial`) condensing into separate $\mu$ and $\log(\sigma^2)$ dense mapping layers to permit the reparameterization trick ($z = \mu + \sigma \odot \epsilon$).
*   **`original_BVAE` / `SimpleVAE`**: Specific architectural ablations designed mapped against Higgins et al., ICLR 2017 structures for comparative metric gathering.

**Loss Function Formalization:**
The total VAE objective is computed utilizing an aggregated function incorporating parameterized weights:
$$ \mathcal{L}_{total} = \lambda_{recon}\mathcal{L}_{recon} + \beta \mathcal{L}_{KL} + \lambda_{ssim}\mathcal{L}_{SSIM} + \lambda_{ld}\mathcal{L}_{LabelDistance} $$
Where $\mathcal{L}_{KL}$ evaluates the distributional drift:
$$ \mathcal{L}_{KL} = -0.5 \sum \left( 1 + \log(\sigma^2) - \mu^2 - \sigma^2 \right) $$

### 3.2 Downstream Classifiers
To evaluate spatial entanglement within the extracted latent spaces, the project standardizes three evaluation networks (`cnn_model.py`):
*   **`CNN` (2D)**: A trifold configuration (Conv2D -> BatchNorm2D -> MaxPool2D -> ReLU) reducing dimensional hierarchies to a bounded Linear layer calculating classification logits.
*   **`CNN_1D`**: Exploits `Conv1d` groupings for evaluating spatially flattened image sequences or specifically structured 1D array abstractions derived from intermediate representations.
*   **`MLP`**: A multi-layer perceptron comprising sequential configurable linear mappings heavily parameterized by structural dropout and batch normalizations. Often ingests directly derived latent embeddings.

---

## 4. Latent Space Analysis & Interpretability
Interpretability is the core focus of the evaluation pipeline, encapsulated purely by the `latent_space_analysis.py` module. It explores structural mapping without requiring explicit ground-truth alignments during initial representation abstractions.

### 4.1 Foundational Sampling
The `LatentSpaceSampler` engine manages all continuous dimensional mapping:
*   **Continuous Interpolation**: Spherically and linearly iterating vectors between $z_1 \rightarrow z_2$ parameters, evaluating transition smoothness.
*   **Directional Traversals**: Sequentially offsetting specific continuous boundaries derived heuristically (e.g., evaluating feature derivations over a $3\sigma$ shifting variance step).

### 4.2 Concept Extraction
The `ConceptSampler` aims to calculate interpretable semantic axes across abstract dimensions:
*   Extracts distinct directional vectors mapped against localized spatial clusters.
*   Evaluates boundaries employing threshold estimators ensuring $z$-directional movement actively correlates with label mappings.

### 4.3 Dimensional Mapping Visualizations
Uses the `LatentSpaceVisualizer` structurally tied to PCA and computationally heavy t-SNE mapping graphs for condensing the $N$-dimensional semantic spaces back down into perceptible 2D grids, allowing qualitative evaluation of dataset groupings.

---

## 5. Evaluation Metrics & Benchmarking
The evaluation suite handles exhaustive logging through tensorboard (`train_vae.py`) alongside JSON-exported final benchmarking outputs utilizing scikit-learn (`run_cnn_study.py`).

### 5.1 Generative Metrics
Tracked throughout `VaeTrainer`:
*   Granular tracking of comparative KL divergence rates scaled against $\beta$-warmup variables.
*   Reconstruction pixel-wise calculations (MSE, standard Bernoulli configurations) combined incrementally with Structural Similarity (SSIM).

### 5.2 Classification Benchmarks
Extracted comprehensively via the test suites mapped in `run_cnn_study.py`:
*   **Global Accuracy**: Validating overall metric thresholds.
*   **Class-Level Extraction**: Granular mapping separating representations using localized parameters for True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).
*   **Derived Analysis**: The metrics natively assemble confusion matrices alongside structured statistical tracking containing categorical Precision, Recall, F1 Scores, and individual class Support constraints for the $N$ common labels parameter.