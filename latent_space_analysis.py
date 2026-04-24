import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from vae_model import VAE, reparameterize

class LatentSpaceSampler:
    """
    Class for sampling and analyzing the latent space of a trained VAE model.
    """
    
    def __init__(self, model: VAE, device: str = 'cpu'):
        self.model = model
        self.model.eval()  # Set model to evaluation mode
        self.device = device
        self.latent_dim = model.latent_dim

    @torch.no_grad()
    def sample_N(self, num_samples: int) -> torch.Tensor:
        """
        Sample N random points from the latent space.

        Args:
            num_samples: Number of random samples to generate
        Returns:
            Tensor of shape (num_samples, latent_dim) containing the sampled latent vectors
        """
        # Sample from standard normal distribution
        samples = torch.randn(num_samples, self.latent_dim).to(self.device)
        return samples
    
    @torch.no_grad()
    def sample_images_from_N(self, num_samples: int) -> torch.Tensor:
        """
        Sample N random points from the latent space and decode them into images.

        Args:
            num_samples: Number of random samples to generate
        Returns:
            Tensor of shape (num_samples, 3, img_size, img_size) containing the decoded images
        """
        latent_samples = self.sample_N(num_samples)
        decoded_images = self.model.decode(latent_samples)
        return decoded_images
    
    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of images into the latent space.

        Args:
            images: Tensor of shape (batch_size, 3, img_size, img_size) containing the input images
        Returns:
            Tensor of shape (batch_size, latent_dim) containing the encoded latent vectors
        """
        mu, log_var = self.model.encode(images.to(self.device))
        return mu, log_var
    
    @torch.no_grad()
    def decode_latent_vectors(self, latent_vectors: torch.Tensor) -> torch.Tensor:
        """
        Decode a batch of latent vectors into images.

        Args:
            latent_vectors: Tensor of shape (batch_size, latent_dim) containing the latent vectors to decode
        Returns:
            Tensor of shape (batch_size, 3, img_size, img_size) containing the decoded images
        """
        decoded_images = self.model.decode(latent_vectors.to(self.device))
        return decoded_images
    
    @torch.no_grad()
    def reconstruct_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Reconstruct a batch of images by encoding and then decoding them.

        Args:
            images: Tensor of shape (batch_size, 3, img_size, img_size) containing the input images
        Returns:
            Tensor of shape (batch_size, 3, img_size, img_size) containing the reconstructed images
        """        
        mu, log_var = self.encode_images(images)
        z = reparameterize(mu, log_var)
        reconstructed_images = self.decode_latent_vectors(z)
        return reconstructed_images
    
    @torch.no_grad()
    def interpolate_in_latent_space(self,
                                    z1: torch.Tensor,
                                    z2: torch.Tensor,
                                    num_steps: int = 10) -> torch.Tensor:
        """
        Interpolate between two latent vectors z1 and z2.
        Args:
            z1: Tensor of shape (latent_dim,) representing the first latent vector
            z2: Tensor of shape (latent_dim,) representing the second latent vector
            num_steps: Number of interpolation steps (including endpoints)
        Returns:
            Tensor of shape (num_steps, latent_dim) containing the interpolated latent vectors
        """

        # Linear interpolation
        t = torch.linspace(0, 1, num_steps).to(self.device)
        interpolated = (1 - t.unsqueeze(1)) * z1.unsqueeze(0) + t.unsqueeze(1) * z2.unsqueeze(0)
        return interpolated
    
    @torch.no_grad()
    def traverse_latent_direction(self,
                                 base_z: torch.Tensor,
                                 direction: torch.Tensor,
                                 num_steps: int = 10,
                                 step_size: float = 1.0) -> torch.Tensor:
        """
        Traverse in a specific direction in the latent space starting from a latent vector z.

        Args:
            z: Tensor of shape (latent_dim,) representing the starting latent vector
            direction: Tensor of shape (latent_dim,) representing the direction to traverse
            num_steps: Number of steps to take in the specified direction (including starting point)
            step_size: Size of each step in the specified direction
        Returns:
            Tensor of shape (num_steps, latent_dim) containing the traversed latent vectors
        """

        # Normalize the direction vector
        direction = direction / (torch.norm(direction) + 1e-8)

        # Generate steps in the specified direction
        t = torch.linspace(-step_size * (num_steps // 2), step_size * (num_steps // 2), num_steps).to(self.device)
        traversed = base_z.unsqueeze(0) + t.unsqueeze(1) * direction.unsqueeze(0)
        return traversed

    def collect_latent_samples(self,
                               dataloader: torch.utils.data.DataLoader,
                               max_samples: Optional[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Collect latent representations for a batch of images from the dataloader.

        Args:
            dataloader: DataLoader providing batches of images to encode
            max_samples: Maximum number of samples to collect (for memory efficiency)
        Returns:
            Tuple of tensors (images, mus, log_vars) where:
                images: Tensor of shape (num_collected_samples, 3, img_size, img_size) containing the input images
                mus: Tensor of shape (num_collected_samples, latent_dim) containing the mean vectors of the latent representations
                log_vars: Tensor of shape (num_collected_samples, latent_dim) containing the log variance vectors of the latent representations
        """
        all_images = []
        all_mus = []
        all_log_vars = []
        all_labels = []
        total_collected = 0
        for images, labels in dataloader:
            images = images.to(self.device)
            mu, log_var = self.encode_images(images)

            all_images.append(images.cpu())
            all_mus.append(mu.cpu())
            all_log_vars.append(log_var.cpu())
            #print(labels)
            all_labels.extend(labels)

            total_collected += images.size(0)
            if max_samples is not None and total_collected >= max_samples:
                break

        return torch.cat(all_images, dim=0), torch.cat(all_mus, dim=0), torch.cat(all_log_vars, dim=0), all_labels

    @torch.no_grad()
    def find_similar_latent_vectors(self, target_z: torch.Tensor, 
                                    candidate_zs: torch.Tensor, 
                                    top_k: int = 5) -> torch.Tensor:
        """
        Find the top-k most similar latent vectors to a target latent vector based on cosine similarity.

        Args:
            target_z: Tensor of shape (latent_dim,) representing the target latent vector
            candidate_zs: Tensor of shape (num_candidates, latent_dim) containing the candidate latent vectors to compare against
            top_k: Number of most similar latent vectors to return
        Returns:
            Tensor of shape (top_k, latent_dim) containing the top-k most similar latent vectors
        """
        # Normalize the target and candidate latent vectors
        target_norm = target_z / (torch.norm(target_z) + 1e-8)
        candidate_norms = candidate_zs / (torch.norm(candidate_zs, dim=1, keepdim=True) + 1e-8)

        # Compute cosine similarity
        similarities = torch.matmul(candidate_norms, target_norm.unsqueeze(1)).squeeze(1)

        # Get indices of top-k most similar latent vectors
        top_k_indices = torch.topk(similarities, k=top_k).indices

        return candidate_zs[top_k_indices]
    
    def distance_between_latent_vectors(self, z1: torch.Tensor, z2: torch.Tensor) -> float:
        """
        Compute the distance between two latent vectors.

        Args:
            z1: Tensor of shape (latent_dim,) representing the first latent vector
            z2: Tensor of shape (latent_dim,) representing the second latent vector
        Returns:
            Distance between the two latent vectors (using Euclidean distance)
        """
        #return np.linalg.norm(z1.cpu().numpy() - z2.cpu().numpy())
        return torch.norm(z1 - z2).item()
    
    def concept_distances(self, mus: torch.Tensor, all_labels: list, concepts: list) -> dict:
        """
        Compute distances from a target latent vector to multiple concept vectors.

        Args:
            mus: Tensor of shape (num_concepts, latent_dim) containing the latent vectors for each concept
            all_labels: List of labels for each latent vector in mus (should correspond to the order of mus)
            concepts: List of selected concept names

        Returns:
            concept_distances[(concept1, concept2)] = avg_distance_value
        """
        concept_distances = {}
        concept_lens = {}

        for i in range(len(all_labels)):
            for j in range(i + 1, len(all_labels)):
                concept_a = all_labels[i]
                concept_b = all_labels[j]
                if concept_a not in concepts or concept_b not in concepts:
                    continue
                distance = self.distance_between_latent_vectors(mus[i], mus[j])
                concept_distances[(concept_a, concept_b)] = concept_distances.get((concept_a, concept_b), 0) + distance
                concept_lens[(concept_a, concept_b)] = concept_lens.get((concept_a, concept_b), 0) + 1

        # Compute average distances
        for (concept_a, concept_b), total_distance in concept_distances.items():
            concept_distances[(concept_a, concept_b)] = total_distance / concept_lens[(concept_a, concept_b)]

        return concept_distances

class LatentSpaceVisualizer:

    def __init__(self, sampler: LatentSpaceSampler, save_dir: str = './visualizations'):
        self.sampler = sampler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def visualize_reconstructions(self,
                                  images: torch.Tensor,
                                  num_samples: int = 16,
                                  errors: bool = True,
                                  labels: list = None,
                                  filename: str = "reconstructions.png"):
        
        """Visualize original images, reconstructions."""

        print(f"Visualizing reconstructions for {num_samples} samples...")

        num_samples = min(num_samples, images.size(0))

        reconstructed = self.sampler.reconstruct_images(images[:num_samples])

        fig, axes = plt.subplots(2 + int(errors), num_samples, figsize=(num_samples * 2, 4 + 2 * int(errors)))
        
        for i in range(num_samples):
            # Original image
            axes[0, i].imshow(images[i].cpu().permute(1, 2, 0))
            axes[0, i].set_title("Original" + (f"\n{labels[i]}" if labels is not None else ""))
            axes[0, i].axis('off')
            # Reconstructed image
            axes[1, i].imshow(reconstructed[i].cpu().permute(1, 2, 0))
            axes[1, i].set_title("Reconstruction")
            axes[1, i].axis('off')

            if errors:
                # Absolute error
                error = torch.abs(images[i] - reconstructed[i])
                error_image = error.cpu().permute(1, 2, 0)
                axes[2, i].imshow(error_image)
                axes[2, i].set_title("Absolute Error")
                axes[2, i].axis('off')
                
        plt.tight_layout()
        plt.savefig(self.save_dir / filename)
        plt.close()


    def visualize_images(self, images: torch.Tensor, 
                         in_row: int = 6, 
                         filename: str = "samples.png",
                         image_labels: list = None) -> None:
        """
        Visualize a batch of images and save to disk.

        Args:
            images: Tensor of shape (B, C, H, W) containing the images to visualize
            in_row: Number of images to display in each row of the grid, if 0 then all images will be in one row
            filename: Name of the file to save the visualization (will be saved in self.save_dir)
            image_labels: List of labels for each image (if provided, should be the same length as the number of images)
        """
        print(f"Visualizing {images.size(0)} images with {in_row} images per row...")


        images = images.cpu()

        rows = (images.size(0) + in_row - 1) // in_row if in_row > 0 else 1

        fig, axes = plt.subplots(rows, in_row if in_row > 0 else images.size(0), figsize=(in_row * 1.5 if in_row > 0 else images.size(0) * 1.5, rows * 1.5), squeeze=False)

        for i, img in enumerate(images):
            row = i // in_row if in_row > 0 else 0
            col = i % in_row if in_row > 0 else i
            axes[row, col].imshow(img.permute(1, 2, 0))
            axes[row, col].axis('off')
            if image_labels is not None and i < len(image_labels):
                axes[row, col].set_title(image_labels[i], fontsize=8)

        plt.subplots_adjust(wspace=0.0, hspace=0.0)

        plt.tight_layout()

        plt.savefig(self.save_dir / filename, bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def visualize_latent_interpolation(self,
                                     z1: torch.Tensor,
                                     z2: torch.Tensor,
                                     num_steps: int = 10,
                                     filename: str = "interpolation.png") -> None:
        """
        Visualize interpolation between two latent vectors.

        Args:
            z1: Tensor of shape (latent_dim,) representing the first latent vector
            z2: Tensor of shape (latent_dim,) representing the second latent vector
            num_steps: Number of interpolation steps (including endpoints)
            filename: Name of the file to save the visualization (will be saved in self.save_dir)
        """
        print(f"Visualizing interpolation between two latent vectors with {num_steps} steps...")

        interpolated_zs = self.sampler.interpolate_in_latent_space(z1, z2, num_steps)
        interpolated_images = self.sampler.decode_latent_vectors(interpolated_zs)

        self.visualize_images(interpolated_images, filename=filename)

    def visualize_latent_traversal(
            self,
            base_z: torch.Tensor,
            dims_to_traverse: Optional[List[int]] = None,
            num_steps: int = 10,
            n_sigma: float = 3.0,
            dataset_mus: torch.Tensor = None,
            num_top_dims: int = 7,
            filename: str = "traversal.png") -> None:
        """
        Visualize traversal in specified dimensions of the latent space.
        Args:
            base_z: Tensor of shape (latent_dim,) representing the starting latent vector
            dims_to_traverse: List of dimension indices to traverse
            num_steps: Number of steps to take in each direction (including starting point)
            n_sigma: Number of standard deviations to traverse in each direction
            dataset_mus: Tensor of shape (num_samples, latent_dim) containing the mean vectors of the latent representations for the dataset (used to compute variance for dimension selection)
            num_top_dims: Number of top dimensions to visualize (if dims_to_traverse is None)
            filename: Name of the file to save the visualization (will be saved in self.save_dir)
        """

        print(f"Visualizing latent traversal for dimensions {dims_to_traverse if dims_to_traverse is not None else 'top ' + str(num_top_dims)} with {num_steps} steps and n_sigma={n_sigma}...")

        if dims_to_traverse is None:
            if dataset_mus is not None:
                var_per_dim = dataset_mus.var(dim=0)  # [latent_dim]
                dims_to_traverse = var_per_dim.argsort(descending=True)[:num_top_dims].tolist()
            else:
                dims_to_traverse = list(range(self.sampler.latent_dim))[:num_top_dims]

        dim_means = dataset_mus.mean(dim=0)   # [latent_dim]
        dim_stds  = dataset_mus.std(dim=0)    # [latent_dim]
        ranges = {
            dim: np.linspace(
                (dim_means[dim] - n_sigma * dim_stds[dim]).item(),
                (dim_means[dim] + n_sigma * dim_stds[dim]).item(),
                num_steps
            )
            for dim in dims_to_traverse
        }

        n_dims = len(dims_to_traverse)
        fig, axes = plt.subplots(n_dims, num_steps, figsize=(num_steps * 2, n_dims * 2))

        for i, dim in enumerate(dims_to_traverse):
            for j, val in enumerate(ranges[dim]):
                modified_z = base_z.clone()
                #print(modified_z)
                modified_z[dim] = val
                decoded_image = self.sampler.decode_latent_vectors(modified_z.unsqueeze(0))[0]
                axes[i, j].imshow(decoded_image.cpu().permute(1, 2, 0))
                axes[i, j].axis('off')
                if i == 0:
                    axes[i, j].set_title(f"Val {val:.2f}", fontsize=8)
        plt.tight_layout()
        plt.savefig(self.save_dir / filename)
        plt.close()
    
    def visualize_directional_traversal(self,
                                   base_z: torch.Tensor,
                                   direction: torch.Tensor,
                                   num_steps: int = 12,
                                   n_sigma: float = 3.0,
                                   dataset_mus: torch.Tensor = None,
                                   filename: str = "directional_traversal.png") -> None:
        """
        Visualize traversal in a specific direction in the latent space.

        Args:
            base_z: Tensor of shape (latent_dim,) representing the starting latent vector
            direction: Tensor of shape (latent_dim,) representing the direction to traverse
            num_steps: Number of steps to take in the specified direction (including starting point)
            n_sigma: Number of standard deviations to traverse in the specified direction
            dataset_mus: Tensor of shape (num_samples, latent_dim) containing the mean vectors of the latent representations for the dataset (used to compute variance for dimension selection)
            filename: Name of the file to save the visualization (will be saved in self.save_dir)
        """

        print(f"Visualizing directional traversal with {num_steps} steps and n_sigma={n_sigma}...")

        step_size = n_sigma * torch.norm(direction) / (num_steps // 2)

        traversed_zs = self.sampler.traverse_latent_direction(base_z, direction, num_steps, step_size)
        traversed_images = self.sampler.decode_latent_vectors(traversed_zs)

        self.visualize_images(traversed_images, filename=filename, in_row=0)
    
    def visualize_latent_distribution(self, dataset_mus: torch.Tensor,
                                      filename: str = "latent_distribution.png",
                                      labels: list = None,
                                      num_top_concepts: int = 30) -> None:
        """
        Visualize the distribution of latent representations in 2D using PCA and t-SNE.

        Args:
            dataset_mus: Tensor of shape (num_samples, latent_dim) containing the mean vectors of the latent representations for the dataset
            filename: Name of the file to save the visualization (will be saved in self.save_dir)
            labels: List of labels corresponding to the samples in dataset_mus
            num_top_concepts: Number of top concepts to display in the visualization
        """

        print(f"Visualizing latent distribution with PCA and t-SNE...")

        mus_np = dataset_mus.cpu().numpy()

        if labels is not None:
            unique_concepts, concept_counts = np.unique(labels, return_counts=True)
            if len(unique_concepts) > num_top_concepts:
                print(f"Warning: More than {num_top_concepts} concepts provided, choosing top {num_top_concepts} for visualization.")
                top_concept_indices = np.argsort(concept_counts)[-num_top_concepts:]
                top_concepts = set(unique_concepts[top_concept_indices])
                new_labels = [c if c in top_concepts else 'Other' for c in labels]
            else:
                new_labels = [c if c != "" else 'Other' for c in labels]
                #print(new_concepts)


        # PCA
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(mus_np)

        # t-SNE
        tsne = TSNE(n_components=2, perplexity=30)
        tsne_result = tsne.fit_transform(mus_np)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))

        
        sns.scatterplot(x=pca_result[:, 0], 
                        y=pca_result[:, 1], 
                        ax=axes[0], 
                        hue=new_labels, 
                        palette='husl', 
                        legend=False, 
                        alpha=0.7,
                        s=30)
        
        axes[0].set_title("PCA of Latent Representations")

        sns.scatterplot(x=tsne_result[:, 0], 
                        y=tsne_result[:, 1], 
                        ax=axes[1], 
                        hue=new_labels, 
                        palette='husl', 
                        legend='auto' if labels is not None else False, 
                        alpha=0.7,
                        s=30)
        
        if labels is not None:
            axes[1].legend(loc='center left', bbox_to_anchor=(1.02, 0.5), title="Labels", borderaxespad=0.)
        
        axes[1].set_title("t-SNE of Latent Representations")

        plt.tight_layout()
        plt.savefig(self.save_dir / filename, bbox_inches='tight')
        plt.close()

    def visualize_confusion_matrix(self, true_positives: int, false_positives: int, true_negatives: int, false_negatives: int, filename: str = "confusion_matrix.png") -> None:
        """
        Visualize a confusion matrix for concept prediction results.

        Args:
            true_positives: Number of true positive predictions
            false_positives: Number of false positive predictions
            true_negatives: Number of true negative predictions
            false_negatives: Number of false negative predictions
            filename: Name of the file to save the visualization
        """

        print(f"Visualizing confusion matrix...")

        confusion_matrix = np.array([[true_positives, false_negatives],
                                     [false_positives, true_negatives]])

        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted Positive', 'Predicted Negative'],
                    yticklabels=['Actual Positive', 'Actual Negative'])
        plt.title("Confusion Matrix")
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, bbox_inches='tight')
        plt.close()
    
    def visualize_concept_distance(self, concept_distances: dict, filename: str = "concept_distances.png") -> None:
        """
        Visualize the distances between latent vectors and concept directions.

        Args:
            concept_distances: concept_distances[concept_name] = distance_value
            filename: Name of the file to save the visualization
        """

        print(f"Visualizing concept distances...")

        def _format_concept_label(concept_key):
            if isinstance(concept_key, (tuple, list)):
                return " vs ".join(str(item) for item in concept_key)
            return str(concept_key)

        def _to_scalar_distance(distance_value):
            if isinstance(distance_value, torch.Tensor):
                return float(distance_value.detach().cpu().item())
            if isinstance(distance_value, np.ndarray):
                return float(np.asarray(distance_value).squeeze().item())
            return float(distance_value)

        concepts = [_format_concept_label(key) for key in concept_distances.keys()]
        distances = [_to_scalar_distance(value) for value in concept_distances.values()]

        plt.figure(figsize=(10, 6))
        sns.barplot(x=concepts, y=distances, palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title("Distances to Concept Directions")
        plt.ylabel("Distance")
        plt.xlabel("Concept")
        plt.tight_layout()
        plt.savefig(self.save_dir / filename, bbox_inches='tight')
        plt.close()


class ConceptSampler:
    def __init__(self, sampler: LatentSpaceSampler, save_dir: str = './visualizations'):
        self.sampler = sampler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
    

    def find_concept_directions(self,
                            dataset_mus: torch.Tensor,
                            labels: list,
                            concepts: list = None) -> dict:
        """
        Find latent directions corresponding to specific concepts by computing the mean latent vector for each concept and then finding the direction from the overall mean to each concept mean.

        Args:
            dataset_mus: Tensor of shape (num_samples, latent_dim) containing the mean vectors of the latent representations for the dataset
            labels: List of labels corresponding to each sample in dataset_mus
            concepts: List of concepts to analyze (if None, no concepts will be analyzed)
        Returns:
            Dictionary mapping concept names to their corresponding latent direction vectors
        """

        print(f"Finding concept directions for {len(concepts) if concepts is not None else 0} concepts...")

        concept_directions = {}
        overall_mean = dataset_mus.mean(dim=0)

        for concept in concepts:
            concept_indices = [i for i, label in enumerate(labels) if label == concept]
            if len(concept_indices) == 0:
                print(f"Warning: No samples found for concept '{concept}', skipping.")
                continue
            concept_mean = dataset_mus[concept_indices].mean(dim=0)
            direction = concept_mean - overall_mean
            concept_directions[concept] = direction
            print(f"Concept '{concept}': Found direction with norm {torch.norm(direction):.4f}")
        
        return concept_directions
    
    def predict_concept_labels(
            self,
            latent_vectors: torch.Tensor,
            concept_directions: dict,
            threshold: float = 0.5) -> str:
        """
        Predict the concept based on the difference between the latent vector and the concept directions.

        Args:
            latent_vectors: Tensor of shape (num_vectors, latent_dim) representing the latent vectors to classify
            concept_directions: Dictionary mapping concept names to their corresponding latent direction vectors
            threshold: Minimum projection value required to assign a concept (if no concept exceeds this threshold, "Unknown" will be returned)
        Returns:
            List of predicted concepts for each latent vector
        """

        predictions = []

        for z in latent_vectors:
            best_concept = "Unknown"
            best_projection = -float('inf')

            for concept, direction in concept_directions.items():
                projection = torch.dot(z, direction) / (torch.norm(direction) + 1e-8)
                if projection > best_projection and projection > threshold:
                    best_projection = projection
                    best_concept = concept
            
            predictions.append(best_concept)
        
        return predictions 
    
    def evaluate_concept_predictions(self, true_labels: list, predicted_labels: list, label: str) -> Tuple[int, int, int, int]:
        """
        Evaluate the concept prediction results by computing true positives, false positives, true negatives, and false negatives.

        Args:
            true_labels: List of true concept labels for each sample
            predicted_labels: List of predicted concept labels for each sample
            label: The specific concept label to evaluate
        Returns:
            Tuple containing counts of (true_positives, false_positives, true_negatives, false_negatives)
        """

        true_positives = sum(1 for t, p in zip(true_labels, predicted_labels) if t == p and p == label)
        false_positives = sum(1 for t, p in zip(true_labels, predicted_labels) if t != p and p == label)
        true_negatives = sum(1 for t, p in zip(true_labels, predicted_labels) if p != label and t != label)
        false_negatives = sum(1 for t, p in zip(true_labels, predicted_labels) if p != label and t == label)

        return true_positives, false_positives, true_negatives, false_negatives

    def tune_threshold_on_val(self, concept_sampler, val_mus, val_labels, concept_directions, concepts, thresholds_to_test):
        """
        Finds the threshold that yields the highest average F1-score across all concepts 
        using the validation set.
        """
        print("Tuning threshold on validation set...")
        best_threshold = 0.5
        best_avg_f1 = -1

        for t in thresholds_to_test:
            # Predict on validation data
            val_predictions = concept_sampler.predict_concept_labels(
                latent_vectors=val_mus,
                concept_directions=concept_directions,
                threshold=t
            )
            
            f1_scores = []
            for concept in concepts:
                if concept not in concept_directions:
                    continue
                    
                tp, fp, tn, fn = concept_sampler.evaluate_concept_predictions(
                    true_labels=val_labels,
                    predicted_labels=val_predictions,
                    label=concept
                )
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                f1_scores.append(f1)
                
            avg_f1 = np.mean(f1_scores) if f1_scores else 0
            print(f"  Threshold {t:.2f} -> Validation Avg F1: {avg_f1:.4f}")
            
            if avg_f1 > best_avg_f1:
                best_avg_f1 = avg_f1
                best_threshold = t
                
        print(f"Selected Optimal Threshold: {best_threshold:.2f} (Val F1: {best_avg_f1:.4f})")
        return best_threshold