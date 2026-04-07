import torch
import argparse
import json
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

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

    train_loader, val_loader = get_dataloaders(
        root_dir=config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size'],
        num_workers=config['num_workers'],
        train_augmentation=config['train_augmentation'],
        pin_memory=config['device'] == 'cuda',
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
        beta_warmup_epochs=config['beta_warmup_epochs'])


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

    return model, train_loader, val_loader, trainer


def analyze_latent_space(model: VAE, val_loader: DataLoader, train_loader: DataLoader, config: dict):
    """
    Analyze the latent space of the trained VAE using Nevanlinna-Pick interpolation to discover concepts.

    Args:
        model: Trained VAE model
        val_loader: DataLoader for the validation set (used for sampling images for analysis)
        train_loader: DataLoader for the training set (used for finding concept vectors)
        config: Dictionary containing analysis configuration parameters
    """

    print("\n" + "=" * 60)
    print("Step 2: Latent Space Analysis")
    print("=" * 60)

    # Load concept labels for ADE20K dataset (to be implemented)

    sampler = LatentSpaceSampler(model=model, device=config['device'])
    dataset_images, mus, logvars, labels = sampler.collect_latent_samples(val_loader, max_samples=config['max_samples'])

    train_images, train_mus, train_logvars, train_labels = sampler.collect_latent_samples(train_loader, max_samples=config['max_samples'])
    train_mus = train_mus.to(config['device'])
    train_images = train_images.to(config['device'])
    train_logvars = train_logvars.to(config['device'])
    
    print(f"Collected {dataset_images.shape} samples for latent space analysis.")
    print(f"Latent space mean shape: {mus.shape}, logvar shape: {logvars.shape}")
    print(f"Shape of labels: {len(labels)}")

    viz_dir = Path(config['save_dir']) / config['experiment_name'] / 'visualizations'
    visualizer = LatentSpaceVisualizer(sampler=sampler, save_dir=viz_dir)
    concept_sampler = ConceptSampler(sampler=sampler, save_dir=viz_dir)

    print("\nVisualizing reconstructions for analysis...")

    #test_images = next(iter(val_loader)).to(config['device'])
    #print(f"Test images shape: {test_images.shape}")

    test_images = dataset_images.to(config['device'])
    test_mus = mus.to(config['device'])
    test_logvars = logvars.to(config['device'])
    if config['concepts'] is not None:
        test_labels = [labels[i] if labels[i] in config['concepts'] else '' for i in range(len(labels))]
    else:
        test_labels = labels

    visualizer.visualize_reconstructions(
        images=test_images,
        num_samples=16,
        errors=True,
        labels=labels,
        filename="reconstructions.png"
    )

    visualizer.visualize_latent_interpolation(
        z1=test_mus[0],
        z2=test_mus[1],
        num_steps=12,
        filename="latent_interpolation.png"
    )


    visualizer.visualize_latent_traversal(
        base_z=test_mus[0],
        num_steps=11,
        n_sigma=5,
        num_top_dims=7,
        dataset_mus=test_mus,
        filename="latent_traversal.png"
    )

    visualizer.visualize_directional_traversal(
        base_z=test_mus[0],
        direction=test_mus[1] - test_mus[0],
        num_steps=12,
        n_sigma=5,
        dataset_mus=test_mus,
        filename="directional_traversal.png"
    )

    #print(test_labels)

    visualizer.visualize_latent_distribution(
        dataset_mus=test_mus,
        filename="latent_distribution.png",
        labels=test_labels,
        num_top_concepts=30
    )

    visualizer.visualize_latent_distribution(
        dataset_mus=test_mus,
        filename="latent_distribution_all_labels.png",
        labels=labels,
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
                n_sigma=5,
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

        
        predictions = concept_sampler.predict_concept_labels(
            latent_vectors=test_mus,
            concept_directions=concept_directions,
            threshold=0.5
        )

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



        image_labels = [f"{pred}\n{label}" for pred, label in zip(predictions[:num_vis], labels[:num_vis])]

        visualizer.visualize_images(
            images=test_images[:num_vis],
            filename="concept_predictions.png",
            in_row=6,
            image_labels=image_labels
        )



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
    parser.add_argument('--max_channels', type=int, default=512, help='Maximum number of channels in the encoder/decoder')
    parser.add_argument('--min_channels', type=int, default=64, help='Minimum number of channels in the encoder/decoder')
    parser.add_argument('--bottleneck_spatial', type=int, default=4, help='Spatial dimensions of the bottleneck feature map (e.g. 4 for 256x256 input)')

    parser.add_argument('--beta', type=float, default=1.0, help='Beta hyperparameter for Beta-VAE')
    parser.add_argument('--beta_start', type=float, default=None, help='Starting value of beta for warmup')
    parser.add_argument('--beta_warmup_epochs', type=int, default=10, help='Number of epochs to warm up beta from beta_start to beta')
    parser.add_argument('--recon_weight', type=float, default=1.0, help='Weight for the reconstruction loss')
    parser.add_argument('--ssim_weight', type=float, default=0.0, help='Weight for the SSIM loss')

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
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    }

    print("=" * 60)
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Step 1: Train Beta-VAE
    model, train_loader, val_loader, trainer = train_beta_vae(config)

    # Step 2: Latent Space Analysis (to be implemented)
    # ...
    analyze_latent_space(model, val_loader, train_loader, config)
    

    


if __name__ == "__main__":    main()



