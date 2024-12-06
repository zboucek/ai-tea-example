# Iris Classification and Regression Machine Learning Examples

## Overview
This repository contains examples of machine learning models implemented using multiple frameworks:
- Classification: Iris flower classification.
- Regression: Real-world regression task with surface roughness prediction.

### Frameworks Covered
- **PyTorch**
- **TensorFlow**
- **MATLAB**

---

## Prerequisites
- Python 3.8+
- Conda
- MATLAB R2022a+ (for MATLAB examples)
- Basic machine learning knowledge

---

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

#### Workshop 2 Dependencies
```bash
pip install xgboost pandas
```

---

## Running Examples

### Classification (Iris Dataset)

#### PyTorch
```bash
python ai-pytorch.py
```

#### TensorFlow
```bash
python ai-tf.py
```

#### MATLAB
Open `ai_matlab.m` in MATLAB and run.

### Regression (Workshop 2)

#### Python
```bash
python ai-workshop2-example.py
```

#### MATLAB
Open `ai_maltab_workshop2.m` in MATLAB and run.

---

## Datasets

### Workshop 1: Iris Classification
- **Dataset:** Iris flower classification dataset
- **Classes:** 3 (Setosa, Versicolor, Virginica)
- **Features:** 4 (Sepal length, Sepal width, Petal length, Petal width)

### Workshop 2: Surface Roughness Regression
- **Dataset:** `Workshop02/data/DATA_1.csv`
- **Task:** Predict surface roughness (`Ra`) using:
  - Laser power (`lpow`)
  - Laser speed (`lspeed`)
  - Distance (`dist`)
  - Volume (`vol`)

---

## Performance

### Classification (Iris Dataset)
- **Typical Accuracy:** 90-95%
- **Variation:** Depends on the framework and model configuration.

### Regression (Workshop 2)
- **Metrics Reported:**
  - **MAE:** Mean Absolute Error
  - **MSE:** Mean Squared Error
  - **R²:** Coefficient of Determination

---

## Frameworks Comparison

| Framework    | Pros                                   | Cons                     |
|--------------|---------------------------------------|--------------------------|
| PyTorch      | Dynamic graphs, Research-friendly     | Less production-ready    |
| TensorFlow   | Production-ready, Flexible            | More complex             |
| MATLAB       | Rapid prototyping, Signal processing  | Expensive, Closed-source |

---

## Repository Structure

```plaintext
.
├── ai_maltab_workshop2.m            # MATLAB script for Workshop 2 regression
├── ai_matlab.m                      # MATLAB script for Iris classification
├── ai-pytorch.py                    # PyTorch implementation for Iris classification
├── ai-tf.py                         # TensorFlow implementation for Iris classification
├── ai-workshop2-example.py          # Python script for Workshop 2 regression
├── README.md                        # Repository documentation
├── Workshop01
│   ├── data
│   │   └── creditcard.zip           # Credit card fraud dataset (Workshop 1 extra)
│   └── snippets                     # Miscellaneous scripts for Workshop 1
│       ├── iris.py
│       ├── load_credit_card_fraud.py
│       └── load_titanic.py
└── Workshop02
    └── data
        └── DATA_1.csv               # Surface roughness regression dataset
```
