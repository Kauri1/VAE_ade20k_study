

## Installation

Recommended to use virtual enviorment

```bash
pip install -r requirements.txt
```

## Data structure

```
ade20k_data/
    ADEData2016/
        images/
            training/
                ADE_train_00000001.jpg
                ...
            validation/
                ADE_val_00000001.jpg
                ...
        sceneCategories.txt
```


# How to use interpretability study

## see params

```
python run_interpretability_study.py --help
```

## Train new checkpoing

```
python run_interpretability_study.py --experiment_name good --latent_dim 256 --beta 1.5 --beta_start 0.5 --beta_warmup_epochs 10 --batch_size 64 --num_epochs 30 --learning_rate 1e-3 --concepts bathroom bedroom abbey alley airport_terminal
```

## Train old checkpoint more

```
python run_interpretability_study.py --checkpoint_path experiments/good/best_model_epoch_99.pth --num_epochs 5 --concepts bathroom bedroom abbey alley airport_terminal
```

## Just generate visualisations from checkpoint

```
python run_interpretability_study.py --num_epochs 0 --checkpoint_path experiments/good/best_model_epoch_99.pth --concepts bathroom bedroom abbey alley airport_terminal
```

