import argparse
import json
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

from cnn_model import CNN, CNN_1D, MLP
from train_cnn import CNNTrainer
from ade20k_dataset import get_dataloaders

def evaluate_model(trainer: CNNTrainer, test_loader, config: dict):
    print("\n" + "=" * 60)
    print(f"Evaluating Model: {config['model_path']} on Test Set")
    print("=" * 60)
    
    trainer.model.eval()
    
    all_preds = []
    all_labels = []
    
    loop = tqdm(test_loader, desc="Test Evaluation", dynamic_ncols=True)
    with torch.no_grad():
        for images, labels in loop:
            images = images.to(trainer.device)
            labels = labels.cpu()
            
            # Input shape adjustments based on model type
            if len(images.shape) == 2 and trainer.model.__class__.__name__ == "CNN":
                spatial_size = int(images.shape[1] ** 0.5)
                images = images.view(-1, 1, spatial_size, spatial_size)
            if len(images.shape) == 2 and trainer.model.__class__.__name__ == "CNN_1D":
                images = images.unsqueeze(1)
                
            outputs = trainer.model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Class mapping using dataset names
    dataset = test_loader.dataset
    if hasattr(dataset, 'unique_classes'):
        class_names = dataset.unique_classes
    else:
        class_names = [f"Class_{i}" for i in range(config['num_classes'])]

    # Calculate global metrics and per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(all_labels, all_preds, labels=range(len(class_names)), zero_division=0)
    cm = confusion_matrix(all_labels, all_preds, labels=range(len(class_names)))
    
    global_accuracy = accuracy_score(all_labels, all_preds)
    print(f"\nGlobal Test Accuracy: {global_accuracy:.4f}")

    metrics_report = {
        "global_accuracy": global_accuracy,
        "classes": {}
    }

    # Extract concept metric per class structurally identical to VAE metrics
    for idx, class_name in enumerate(class_names):
        # Extract True Positives, False Positives, False Negatives, True Negatives
        tp = cm[idx, idx]
        fp = cm[:, idx].sum() - tp
        fn = cm[idx, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)

        class_acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0

        metrics_report["classes"][class_name] = {
            "accuracy": float(class_acc),
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1_score": float(f1[idx]),
            "support": int(support[idx]),
            "tp": int(tp),
            "fp": int(fp),
            "tn": int(tn),
            "fn": int(fn)
        }
        
    # Sort classes descending by F1 score
    metrics_report["classes"] = dict(sorted(
        metrics_report["classes"].items(), 
        key=lambda item: item[1]["f1_score"], 
        reverse=True
    ))
        
    model_name = Path(config['model_path']).parent.name if config['model_path'] else "cnn_evaluation"
    save_dir = Path(config['save_dir']) / model_name
    save_dir.mkdir(parents=True, exist_ok=True)
    
    out_file = save_dir / "cnn_concept_metrics_test.json"
    with open(out_file, "w") as f:
        json.dump(metrics_report, f, indent=4)
        
    print(f"\nDetailed concept metrics saved to {out_file}")

def main():
    parser = argparse.ArgumentParser(description='Evaluate a CNN model on ADE20K concept classes')
    project_root = Path(__file__).resolve().parent
    
    parser.add_argument('--data_dir', type=str, default=str(project_root / 'ade20k_data/ADEData2016'), help='Path to dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the CNN model .pth checkpoint to evaluate')
    parser.add_argument('--save_dir', type=str, default=str(project_root / 'cnn_evaluations'), help='Where to save the evaluation results')
    parser.add_argument('--latent_dir', type=str, default=None, help='Directory containing extracted latent representations for 1D CNNs')
    
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    args = parser.parse_args()
    config = vars(args)
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Read model architecture from config.json in the same folder as the model
    model_dir = Path(config['model_path']).parent
    config_file = model_dir / "config.json"
    
    if config_file.exists():
        print(f"Loading architecture config from {config_file}")
        with open(config_file, "r") as f:
            model_config = json.load(f)
            
        config['model_type'] = model_config.get('model_type', 'CNN')
        config['in_channels'] = model_config.get('in_channels', 3)
        config['input_size'] = model_config.get('input_size', 256)
        config['num_classes'] = model_config.get('num_classes', 150)
        config['pooling'] = model_config.get('pooling', True)
        config['n_common_labels'] = model_config.get('n_common_labels', config['num_classes'])
        config['exclude_concepts'] = model_config.get('exclude_concepts', None)
        
        # If user didn't explicitly pass --latent_dir, try loading it from config
        if config.get('latent_dir') is None:
            config['latent_dir'] = model_config.get('latent_dir', None)
            
    else:
        print(f"Warning: Configuration file {config_file} not found. Using default architecture values.")
        config['model_type'] = 'CNN'
        config['in_channels'] = 3
        config['input_size'] = 256
        config['num_classes'] = 150
        config['pooling'] = True
        config['n_common_labels'] = 150
        config['exclude_concepts'] = None

    print("=" * 60)
    print("Configuration:")
    for key, val in config.items():
        print(f"  {key}: {val}")

    # 1. Load Data
    train_loader, val_loader, test_loader = get_dataloaders(
        root_dir=config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        img_size=config['input_size'],  # Derived from config.json
        pin_memory=config['device'] == 'cuda',
        n_common_labels=config['n_common_labels'],
        exclude_concepts=config['exclude_concepts'],
        latent_dir=config['latent_dir']
    )

    # 2. we only evaluate here
    if config['model_type'] == "CNN":
        model = CNN(
            in_channels=config['in_channels'],
            input_size=config['input_size'],
            num_classes=config['num_classes'],
            pooling=config['pooling']
        ).to(config['device'])
    elif config['model_type'] == "CNN_1D":
        model = CNN_1D(
            in_channels=config['in_channels'],
            input_size=config['input_size'],
            num_classes=config['num_classes']
        ).to(config['device'])
    elif config['model_type'] == "MLP":
        model = MLP(
            input_size=config['input_size'],
            num_classes=config['num_classes']
        ).to(config['device'])
    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")
    
    trainer = CNNTrainer(
        model=model,
        train_loader=None,
        val_loader=None, 
        device=config['device'],
        save_dir=Path("."),
        model_save_name="temp"
    )

    # 3. Load Model checkpoint
    try:
        trainer.load_model(config['model_path'])
    except Exception as e:
        print(f"Failed to load model from {config['model_path']}: {e}")
        return

    # 4. Strict evaluation on the test set alone
    evaluate_model(trainer, test_loader, config)

if __name__ == "__main__":
    main()
