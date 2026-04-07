

## Dependency Installation

Recommended to use virtual enviorment

```bash
pip install -r requirements.txt
```

## Data structure

This project was done using the dataset from [kaggle ade20k-dataset](https://www.kaggle.com/datasets/awsaf49/ade20k-dataset)

```
{project dir}/
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

```bash
python run_interpretability_study.py --help
```

## Train new checkpoing

```bash
python run_interpretability_study.py --experiment_name VAE_study --latent_dim 256 --beta 1.5 --beta_start 0.5 --beta_warmup_epochs 10 --batch_size 64 --num_epochs 30 --concepts bathroom bedroom abbey alley airport_terminal --min_channels 8 --max_channels 128 --num_workers 8
```

## Train checkpoint more (load weights)

```bash
python run_interpretability_study.py --checkpoint_path experiments/VAE_study/best_model_epoch_32.pth --num_epochs 5 --concepts bathroom bedroom abbey alley airport_terminal
```

## Just generate visualisations from checkpoint (load weights)

```bash
python run_interpretability_study.py --num_epochs 0 --checkpoint_path experiments/VAE_study/best_model_epoch_32.pth --concepts bathroom bedroom abbey alley airport_terminal
```

## Inspect model training logs using TensorBoard

```bash
tensorboard --logdir experiments
```
