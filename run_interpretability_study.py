from xml.parsers.expat import model

import torch
import argparse
import json
import numpy as np
from itertools import combinations
from pathlib import Path
from torch.utils.data import DataLoader

from tqdm import tqdm

from vae_model import VAE
from ade20k_dataset import get_dataloaders
from train_vae import VaeTrainer
from latent_space_analysis import ConceptSampler, LatentSpaceSampler, LatentSpaceVisualizer
#from nevanlinna_pick import LogisticDirectionFinder, NevanlinnaPickConceptDiscovery, MultiConceptNP
#from ade20k_concept_labels import load_concept_labels_for_dataset


def train_beta_vae(config: dict):
    """
    Train a Beta-VAE on the ADE20K dataset using the provided configuration.

    Args:
        config: Dictionary containing training configuration parameters
    """

    print("\n" + "=" * 60)
    print("Step 1: Train β-VAE")
    print("=" * 60)
    print(f"β hyperparameter: {config['beta']}")
    print(f"Latent dimension: {config['latent_dim']}")

    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers'],
        train_augmentation=config['train_augmentation'],
        pin_memory=config['device'] == 'cuda',
        n_common_labels=config['n_common_labels'],
        exclude_concepts=config['exclude_concepts']
    )

    # Get config from config.json
    if config['checkpoint_path']:
        print(f"Checkpoint path provided: {config['checkpoint_path']}")
        conf_file = Path(config['checkpoint_path']).parent / 'config.json'
        checkpoint_experiment = Path(config['checkpoint_path']).parent.name
        config['experiment_name'] = checkpoint_experiment
        if conf_file.exists():
            print(f"Loading training configuration from {conf_file}")
            with open(conf_file, 'r') as f:
                loaded_config = json.load(f)
            print("Loaded configuration:")
            for key, value in loaded_config.items():
                print(f"  {key}: {value}")
                # Update config with loaded values (except for checkpoint_path)
                if key != 'checkpoint_path':
                    config[key] = value
        else:
            print(f"Warning: Configuration file {conf_file} not found. Using default configuration values.")
    else:
        print("No checkpoint path provided. Training from scratch.")

    model = VAE(
        input_channels=3,
        latent_dim=config['latent_dim'],
        img_size=config['img_size'],
        max_channels=config['max_channels'],
        min_channels=config['min_channels'],
        bottleneck_spatial=config['bottleneck_spatial']
    )

    model.print_architecture()

    trainer = VaeTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=config['learning_rate'],
        device=config['device'],
        save_dir=config['save_dir'],
        experiment_name=config['experiment_name'],
        use_amp=config['use_amp'],
        use_channels_last=config['use_channels_last'],
        img_size=config['img_size'],
        recon_weight=config['recon_weight'],
        ssim_weight=config['ssim_weight'],
        beta=config['beta'],
        beta_start=config['beta_start'],
        beta_warmup_epochs=config['beta_warmup_epochs'],
        label_distance_loss_weight=config['label_distance_loss_weight'],
        n_common_labels=config['n_common_labels'],
        exclude_concepts=config['exclude_concepts']
    )


    if config['checkpoint_path']:
        print(f"Loading model weights from checkpoint: {config['checkpoint_path']}")
        trainer.load_checkpoint(config['checkpoint_path'])
    else:
        print("No checkpoint provided. Starting training from scratch.")

    if not config['skip_training'] and config['num_epochs'] > 0:
        trainer.train(num_epochs=config['num_epochs'], visualize_every=config['visualize_every'])
        trainer.visualize_reconstructions(num_images=16)
    else:
        print("Skipping training. Proceeding to interpretability analysis.")

    return model, train_loader, val_loader, test_loader, trainer


def analyze_latent_space(model: VAE, val_loader: DataLoader, train_loader: DataLoader, test_loader: DataLoader, config: dict):
    """
    Analyze the latent space of the trained VAE using Nevanlinna-Pick interpolation to discover concepts.

    Args:
        model: Trained VAE model
        val_loader: DataLoader for the validation set (used for sampling images for analysis)
        train_loader: DataLoader for the training set (used for finding concept vectors)
        test_loader: DataLoader for the test set (used for evaluating the model)
        config: Dictionary containing analysis configuration parameters
    """

    print("\n" + "=" * 60)
    print("Step 2: Latent Space Analysis")
    print("=" * 60)

    # Load concept labels for ADE20K dataset (to be implemented)

    sampler = LatentSpaceSampler(model=model, device=config['device'])
    val_images, val_mus, val_logvars, val_labels = sampler.collect_latent_samples(val_loader, max_samples=config['max_samples'])
    val_mus = val_mus.to(config['device'])
    val_images = val_images.to(config['device'])
    val_logvars = val_logvars.to(config['device'])

    train_images, train_mus, train_logvars, train_labels = sampler.collect_latent_samples(train_loader, max_samples=config['max_samples'])
    train_mus = train_mus.to(config['device'])
    train_images = train_images.to(config['device'])
    train_logvars = train_logvars.to(config['device'])

    test_images, test_mus, test_logvars, test_labels = sampler.collect_latent_samples(test_loader, max_samples=config['max_samples'])
    test_mus = test_mus.to(config['device'])
    test_images = test_images.to(config['device'])
    test_logvars = test_logvars.to(config['device'])

    dataset = val_loader.dataset
    if hasattr(dataset, 'unique_classes'):
        val_labels = [dataset.unique_classes[lbl.item()] for lbl in val_labels]
        train_labels = [dataset.unique_classes[lbl.item()] for lbl in train_labels]
        test_labels = [dataset.unique_classes[lbl.item()] for lbl in test_labels]
    print(f"Collected {val_images.shape} samples for latent space analysis.")
    print(f"Latent space mean shape: {val_mus.shape}, logvar shape: {val_logvars.shape}")
    print(f"Shape of labels: {len(val_labels)}")

    viz_dir = Path(config['save_dir']) / config['experiment_name'] / 'visualizations'
    visualizer = LatentSpaceVisualizer(sampler=sampler, save_dir=viz_dir)
    concept_sampler = ConceptSampler(sampler=sampler, save_dir=viz_dir)

    print("\nVisualizing reconstructions for analysis...")

    #test_images = next(iter(val_loader)).to(config['device'])
    #print(f"Test images shape: {test_images.shape}")

    # Filter labels if specific concepts are provided in config
    if config['concepts'] is not None:
        filtered_val_labels = [val_labels[i] if val_labels[i] in config['concepts'] else '' for i in range(len(val_labels))]
        filtered_test_labels = [test_labels[i] if test_labels[i] in config['concepts'] else '' for i in range(len(test_labels))]
    else:
        filtered_val_labels = val_labels
        filtered_test_labels = test_labels

    visualizer.visualize_reconstructions(
        images=val_images,
        num_samples=16,
        errors=True,
        labels=filtered_val_labels,
        filename="reconstructions.png"
    )

    visualizer.visualize_latent_interpolation(
        z1=val_mus[0],
        z2=val_mus[1],
        num_steps=12,
        filename="latent_interpolation.png"
    )


    visualizer.visualize_latent_traversal(
        base_z=val_mus[0],
        num_steps=11,
        n_sigma=3,
        num_top_dims=7,
        dataset_mus=val_mus,
        filename="latent_traversal.png"
    )

    visualizer.visualize_directional_traversal(
        base_z=val_mus[0],
        direction=val_mus[1] - val_mus[0],
        num_steps=12,
        n_sigma=3,
        dataset_mus=val_mus,
        filename="directional_traversal.png"
    )

    #print(test_labels)

    visualizer.visualize_latent_distribution(
        dataset_mus=val_mus,
        filename="latent_distribution.png",
        labels=filtered_val_labels,
        num_top_concepts=30
    )

    visualizer.visualize_latent_distribution(
        dataset_mus=test_mus,
        filename="latent_distribution_all_labels.png",
        labels=filtered_test_labels,
        num_top_concepts=30
    )


    visualizer.visualize_images(
        sampler.decode_latent_vectors(
            sampler.sample_N(24)
        ),
        filename="random_samples.png")
    
    if config['concepts'] is not None:
        concepts = config['concepts']

        #should use train set to find vectors and visualize on val set?.
        

        concept_directions = concept_sampler.find_concept_directions(
            concepts=concepts,
            dataset_mus=train_mus,
            labels=train_labels
        )

        concept_dir = viz_dir / "concepts"

        Path(concept_dir).mkdir(parents=True, exist_ok=True)

        candidate_thresholds = np.linspace(0.1, 0.9, 17) 
        optimal_threshold = concept_sampler.tune_threshold_on_val(
            concept_sampler, 
            val_mus, 
            val_labels, 
            concept_directions, 
            concepts, 
            candidate_thresholds
        )

        predictions = concept_sampler.predict_concept_labels(
            latent_vectors=test_mus,
            concept_directions=concept_directions,
            threshold=optimal_threshold
        )

        metrics_report = {}
        for concept in concepts:

            if concept not in concept_directions:
                print(f"Concept '{concept}' not found in training data. Skipping visualization for this concept.")
                continue

            # Visualize traversal along concept direction
            direction = concept_directions[concept]
            visualizer.visualize_directional_traversal(
                base_z=test_mus[0],
                direction=direction,
                num_steps=12,
                n_sigma=3,
                dataset_mus=train_mus,
                filename=f"concepts/traversal_{concept}.png"
            )

            # Visualize concept vector
            visualizer.visualize_images(
                images=sampler.decode_latent_vectors(
                    latent_vectors=direction
                ),
                filename=f"concepts/sample_{concept}.png",
                in_row=1
            )

            tp, fp, tn, fn = concept_sampler.evaluate_concept_predictions(
                true_labels=test_labels,
                predicted_labels=predictions,
                label=concept
            )

            visualizer.visualize_confusion_matrix(
                true_positives=tp,
                false_positives=fp,
                true_negatives=tn,
                false_negatives=fn,
                filename=f"concepts/confusion_matrix_{concept}.png"
            )

            # correct / all
            accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0
            # correct positives out of predicted positives,
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            # correct positives out of actual positives
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            # harmonic mean of precision and recall
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics_report[concept] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'tp': tp,
                'fp': fp,
                'tn': tn,
                'fn': fn
            }
        
        with open(concept_dir / "concept_metrics_test.json", 'w') as f:
            json.dump(metrics_report, f, indent=4)


        print(f"Predicted {len(predictions)} concepts.")

        num_vis = 18

        # visualizer.visualize_images(
        #     images=sampler.decode_latent_vectors(
        #         latent_vectors=test_mus[:num_vis]
        #     ),
        #     filename="concept_predictions.png",
        #     in_row=6,
        #     image_labels=predictions[:num_vis]
        # )
        


        image_labels = [f"{pred}\n{label}" for pred, label in zip(predictions[:num_vis], filtered_test_labels[:num_vis])]

        visualizer.visualize_images(
            images=test_images[:num_vis],
            filename="concept_predictions.png",
            in_row=6,
            image_labels=image_labels
        )
    


    concepts = config['concepts'] if config['concepts'] is not None else sorted(set(val_labels))
    print(f"Unique concepts in dataset: {concepts}")

    concept_distances = sampler.concept_distances(
        mus=val_mus,
        all_labels=val_labels,
        concepts=concepts
    )

    visualizer.visualize_concept_distance(concept_distances=concept_distances, filename="concept_distances.png")

def save_latent_representations(model: VAE, dataloader: DataLoader, save_dir: str):
    """
    Extracts and saves the latent representation (mu) for each image in the dataset.
    Note: The dataloader must have shuffle=False to correctly map variables to filenames.
    """
    print(f"\nExtracting and saving latent representations to {save_dir}...")

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    device = next(model.parameters()).device
    model.eval()

    dataloader.dataset.return_paths = True

    dataset = dataloader.dataset
    if not hasattr(dataset, 'image_files'):
        raise AttributeError("Dataset must have 'image_files' attribute containing list of image file paths.")
    
    with torch.no_grad():
        global_index = 0
        for batch in tqdm(dataloader, desc="Processing batches", dynamic_ncols=True):
            if len(batch) == 3:
                images, labels, picture_ids = batch
            else:
                images, labels = batch
                picture_ids = [dataset.image_files[global_index + i].stem for i in range(images.size(0))]
            
            images = images.to(device)

            # Encode
            mus, logvars = model.encode(images)

            for i in range(mus.size(0)):
                image_file = dataset.image_files[global_index]
                picture_id = picture_ids[i]

                latent_tensor = mus[i].detach().cpu()

                file_desc = save_path / f"{picture_id}.pt"
                torch.save(latent_tensor, file_desc)

                global_index += 1

    dataloader.dataset.return_paths = False
    print("Latent representations saved successfully.")

def main():
    parser = argparse.ArgumentParser(description='Run interpretability study on Beta-VAE trained on ADE20K')
    project_root = Path(__file__).resolve().parent
    parser.add_argument('--data_dir', type=str, default=str(project_root / 'ade20k_data/ADEData2016'), help='Path to ADE20K dataset')
    parser.add_argument('--save_dir', type=str, default=str(project_root / 'experiments'), help='Directory to save model checkpoints and results')
    parser.add_argument('--experiment_name', type=str, default='beta_vae_experiment', help='Name of the experiment (used for organizing results)')
    parser.add_argument('--skip_training', action='store_true', help='Whether to skip training and only run interpretability analysis')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to a model checkpoint to load (optional)')

    #Model and training hyperparameters
    parser.add_argument('--latent_dim', type=int, default=128, help='Dimensionality of the latent space')
    parser.add_argument('--max_channels', type=int, default=256, help='Maximum number of channels in the encoder/decoder')
    parser.add_argument('--min_channels', type=int, default=8, help='Minimum number of channels in the encoder/decoder')
    parser.add_argument('--bottleneck_spatial', type=int, default=4, help='Spatial dimensions of the bottleneck feature map (e.g. 4 for 256x256 input)')

    parser.add_argument('--beta', type=float, default=1.0, help='Beta hyperparameter for Beta-VAE')
    parser.add_argument('--beta_start', type=float, default=None, help='Starting value of beta for warmup')
    parser.add_argument('--beta_warmup_epochs', type=int, default=10, help='Number of epochs to warm up beta from beta_start to beta')
    parser.add_argument('--recon_weight', type=float, default=1.0, help='Weight for the reconstruction loss')
    parser.add_argument('--ssim_weight', type=float, default=0.5, help='Weight for the SSIM loss')
    parser.add_argument('--label_distance_loss_weight', type=float, default=0.1, help='Weight for the label distance loss (requires labels in dataset and implementation in model)')

    # Training hyperparameters
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training and validation')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train the VAE')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate for training the VAE')
    parser.add_argument('--img_size', type=int, default=256, help='Input image size (images will be resized to this size)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of worker processes for data loading')
    parser.add_argument('--train_augmentation', default=True, action='store_true', help='Whether to apply data augmentation during training')
    parser.add_argument('--use_amp', default=True, action='store_true', help='Whether to use automatic mixed precision for training')
    parser.add_argument('--use_channels_last', default=False, action='store_true', help='Whether to use channels_last memory format for training (can improve performance on some GPUs)')
    parser.add_argument('--visualize_every', type=int, default=5, help='Frequency (in epochs) to visualize reconstructions during training')

    #Analysis hyperparameters
    parser.add_argument('--max_samples', type=int, default=2000, help='Maximum samples for analysis')
    parser.add_argument('--concepts', type=str, nargs='+', default=None, help='Concepts to discover with NP (e.g., "bedroom bathroom")')
    parser.add_argument('--latent_dir', type=str, default=None, help='Directory for latent representations')
    parser.add_argument('--n_common_labels', type=int, default=None, help='Number of most common labels to use for analysis and concept discovery')
    parser.add_argument('--exclude_concepts', type=str, nargs='+', default=None, help='Concepts to exclude from analysis (e.g., "wall floor")')
    args = parser.parse_args()

    config = {
        'data_dir': args.data_dir,
        'save_dir': args.save_dir,
        'experiment_name': args.experiment_name,
        'skip_training': args.skip_training,
        'checkpoint_path': args.checkpoint_path,
        'latent_dim': args.latent_dim,
        'max_channels': args.max_channels,
        'min_channels': args.min_channels,
        'bottleneck_spatial': args.bottleneck_spatial,
        'learning_rate': args.learning_rate,
        'beta': args.beta,
        'beta_start': args.beta_start if args.beta_start is not None else args.beta,
        'beta_warmup_epochs': args.beta_warmup_epochs,
        'recon_weight': args.recon_weight,
        'ssim_weight': args.ssim_weight,
        'label_distance_loss_weight': args.label_distance_loss_weight,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'img_size': args.img_size,
        'num_workers': args.num_workers,
        'train_augmentation': args.train_augmentation,
        'use_amp': args.use_amp,
        'use_channels_last': args.use_channels_last,
        'visualize_every': args.visualize_every,
        'max_samples': args.max_samples,
        'concepts': args.concepts,
        'latent_dir': args.latent_dir,
        'n_common_labels': args.n_common_labels,
        'exclude_concepts': args.exclude_concepts,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Step 1: Train Beta-VAE
    model, train_loader, val_loader, test_loader, trainer = train_beta_vae(config)

    # Step 2: Latent Space Analysis
    analyze_latent_space(model, val_loader, train_loader, test_loader, config)

    # Step 3: Save latent representations for cnn
    extract_train_loader = DataLoader(
        train_loader.dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
    )

    experiment_dir = Path(config['save_dir']) / config['experiment_name']

    save_latent_representations(model, extract_train_loader, experiment_dir / "latents" / "train")
    save_latent_representations(model, val_loader, experiment_dir / "latents" / "validation")
    save_latent_representations(model, test_loader, experiment_dir / "latents" / "test")
    

    


if __name__ == "__main__":    main()



