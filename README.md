# MLETT - ML Time Series Forecasting on ETT Dataset

A machine learning pipeline for time series forecasting using the ETT (Electricity Transformer Temperature) dataset.

## Project Structure

```
MLETT/
├── data/
│   ├── raw/                    # Raw data files
│   └── processed/              # Processed data files
├── src/                        # Source code
│   ├── config/                 # Configuration files
│   ├── data/                   # Data processing modules
│   ├── features/               # Feature engineering modules
│   ├── models/                 # Machine learning models
│   ├── training/               # Training and evaluation modules
│   └── utils/                  # Utility functions
├── scripts/                    # Executable scripts
│   ├── train.py               # Training script
│   ├── evaluate.py            # Evaluation script
│   └── predict.py             # Prediction script
├── models/                     # Saved models
├── logs/                       # Log files
├── results/                    # Experiment results
└── notebook/                   # Jupyter notebooks for analysis

```

## Installation

### Prerequisites
- Python 3.8 or higher
- Conda environment

### Setup

1. Create conda environment:
```bash
conda create -n data_science python=3.9
conda activate data_science
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Usage

### Training

Train a model using the default configuration:
```bash
python scripts/train.py
```

Train with custom model name:
```bash
python scripts/train.py --model-name my_model
```

Use custom configuration:
```bash
python scripts/train.py --config path/to/config.yaml
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py --model models/model_x.pkl --data data/processed/test_data.csv
```

### Prediction

Make predictions with a trained model:
```bash
python scripts/predict.py --model models/model_x.pkl --data data/processed/input.csv --output predictions.csv
```

## Configuration

The main configuration file is `src/config/config.yaml`. You can customize:
- Data paths and splitting ratios
- Feature engineering options
- Model parameters
- Training settings
- Evaluation metrics

## Data

The project uses the ETT (Electricity Transformer Temperature) dataset:
- **Features**: HUFL, HULL, MUFL, MULL, LUFL, LULL (various load types)
- **Target**: OT (Oil Temperature)
- **Granularity**: Hourly data
- **Dataset Size**: 17,421 samples

## Model

The project currently implements:
- **XGBoost**: Gradient boosting decision trees for regression

## Features

- Time series data splitting
- Feature engineering (time-based features, rolling windows, lag features)
- Model training and evaluation
- Comprehensive metrics (MSE, RMSE, MAE, R², MAPE)
- Logging and experiment tracking
- Model persistence

## Development

### Project Structure

The project follows a modular structure:
- **Data Module**: Data loading, cleaning, and splitting
- **Features Module**: Feature engineering and transformation
- **Models Module**: Machine learning model implementations
- **Training Module**: Model training and evaluation logic
- **Utils Module**: Helper functions (logging, I/O, metrics)

### Adding New Models

1. Create a new model class in `src/models/`
2. Inherit from `BaseModel`
3. Implement required abstract methods
4. Update the trainer to support the new model type

## Results

Training results are saved in:
- `models/`: Trained model files
- `results/`: Evaluation metrics and configurations
- `logs/`: Training and execution logs

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
