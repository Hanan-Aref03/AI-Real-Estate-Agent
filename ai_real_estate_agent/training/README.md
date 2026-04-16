# Training and Development

This folder contains scripts for model training, data preprocessing, and evaluation.

## Files

- **colab_model_training.py** - Exported Colab notebook for model training
- **data_preprocessing.py** - Data loading and feature engineering
- **model_evaluation.py** - Model performance evaluation and testing

## Usage

### Running Training Script

```bash
python training/colab_model_training.py
```

### Output

After training, the script should save:
- Model: `models/best_ames_model_random_forest.pkl`
- Features: `models/feature_names.pkl`
- Stats: `models/train_stats.json`

## Notes

- These scripts are for development only
- Use `models/` folder for production model artifacts
- Training data should be placed in a `data/` folder (not tracked in git)
