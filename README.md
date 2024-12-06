# Iris Classification and Regression Machine Learning Examples

## Overview
This repository contains examples of machine learning models implemented using multiple frameworks:
- **Classification**: Iris flower classification.
- **Regression**: Real-world regression task with surface roughness prediction.

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
- **Dataset Source:** [AI-RICE/Workshop GitHub Repository](https://github.com/AI-RICE/Workshop)
- **Dataset Location:** `Workshop02/data/DATA_1.csv`
- **Task:** Predict surface roughness (`Ra`) using:
  - Laser power (`lpow`)
  - Laser speed (`lspeed`)
  - Distance (`dist`)
  - Volume (`vol`)

---

## Performance and Results

### Classification (Iris Dataset)
- **Typical Accuracy:** 90-95%
- **Variation:** Depends on the framework and model configuration.

### Regression (Workshop 2)
#### Python Results:
| Model                         | MAE   | MSE   | R²     |
|-------------------------------|-------|-------|--------|
| Linear Regression             | 2.88  | 14.20 | 0.33   |
| Random Forest                 | 1.63  | 6.88  | 0.67   |
| XGBoost                       | 1.73  | 6.50  | 0.69   |
| Polynomial Regression (Ridge) | 2.55  | 7.56  | 0.47   |

#### MATLAB Results:
| Model                         | MAE   | MSE   | R²     |
|-------------------------------|-------|-------|--------|
| Linear Regression             | 3.80  | 23.56 | 0.40   |
| Random Forest                 | 2.01  | 7.41  | 0.79   |
| Polynomial Regression (Ridge) | 1.66  | 4.70  | 0.87   |

#### Comments on Results:
1. **Linear Regression**:
   - Performs poorly in both Python and MATLAB, with low `R²` and high error metrics. This suggests that the relationships in the data are non-linear and cannot be captured effectively by a simple linear model.

2. **Random Forest**:
   - Shows good performance in both frameworks, with a better `R²` in MATLAB. Differences may be due to hyperparameter defaults and randomness in tree construction.

3. **Polynomial Regression (Ridge)**:
   - **Python:** Performs moderately well but struggles to capture complex relationships.
   - **MATLAB:** Performs exceptionally well, likely due to better handling of polynomial features and regularization. Differences in feature generation or regularization tuning between frameworks may explain the discrepancy.

4. **XGBoost**:
   - Only implemented in Python, it performs slightly better than Random Forest. Adding an XGBoost implementation in MATLAB could provide further insights into model performance across frameworks.

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