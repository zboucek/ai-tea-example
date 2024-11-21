# Iris Classification Machine Learning Examples

## Overview
This repository contains Iris flower classification examples implemented in multiple machine learning frameworks:
- PyTorch
- TensorFlow
- MATLAB

## Prerequisites
- Python 3.8+
- Conda
- Basic machine learning knowledge

## Environment Setup

### Create Conda Environment
```bash
conda create -n ai-tea-demo python=3.11
conda activate ai-tea-demo
```

### Install Dependencies

#### PyTorch
```bash
pip install torch scikit-learn numpy matplotlib
```

#### TensorFlow
```bash
pip install tensorflow scikit-learn numpy matplotlib
```

## Running Examples

### PyTorch
```bash
python pytorch_iris_classification.py
```

### TensorFlow
```bash
python tensorflow_iris_classification.py
```

### MATLAB
Open `matlab_iris_classification.m` in MATLAB and run

## Dataset
- Iris flower classification dataset
- 3 classes: Setosa, Versicolor, Virginica
- 4 features: Sepal length, Sepal width, Petal length, Petal width

## Performance
- Typical accuracy: 90-95%
- Varies by framework and model configuration

## Frameworks Comparison
| Framework | Pros | Cons |
|-----------|------|------|
| PyTorch   | Dynamic graphs, Research-friendly | Less production-ready |
| TensorFlow| Production-ready, Flexible | More complex |
| MATLAB    | Rapid prototyping, Signal processing | Expensive, Closed-source |