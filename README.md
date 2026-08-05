# 🌸 Iris Flower Classification using Artificial Intelligence
CodTech IT Solutions — Artificial intelligence Internship    
Task name : Iris Flower Classification  
Intern : V.VINOTH  
Intern ID : CITS7154  
Domain :   Artificial Intelligence  
Duration : 4 Weeks  
Internship Period : 16 July 2026 - 13 August 2026  

# 🌸 Iris Flower Classification

**CodTech IT Solutions — Online Internship Project**

A Machine Learning project that classifies Iris flowers into three species —
**Setosa**, **Versicolor**, and **Virginica** — using a Support Vector
Machine (SVM) trained on the classic Iris dataset.

---

## 📌 Project Overview

| Detail | Description |
|---|---|
| **Domain** | Artificial Intelligence / Machine Learning |
| **Task** | Multi-class Classification |
| **Dataset** | Iris Dataset (150 samples, 3 classes, 4 features) |
| **Algorithm** | Support Vector Machine (SVM – Linear Kernel) |
| **Language** | Python 3 |
| **Libraries** | scikit-learn, pandas, numpy |

---

## 🎯 Objective

To build an AI model that can predict the **species of an Iris flower**
based on four measured features:

- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

## 📂 Repository Structure

```
iris-flower-classification/
│
├── iris_classification.py     # Main Python program
├── requirements.txt           # Project dependencies
├── output.txt                 # Sample program output (text)
├── iris_terminal_screenshot.png   # Terminal screenshot of execution
└── README.md                  # Project documentation
```

---

## ⚙️ How It Works

1. **Load Dataset** – The built-in Iris dataset is loaded via `sklearn.datasets`.
2. **Explore Data** – Basic structure and first few rows are displayed.
3. **Train-Test Split** – Data is split 80% (training) / 20% (testing).
4. **Feature Scaling** – `StandardScaler` normalizes feature values.
5. **Model Training** – An `SVC` (Support Vector Classifier) with a linear
   kernel is trained on the scaled training data.
6. **Evaluation** – Accuracy, confusion matrix, and a full classification
   report (precision, recall, f1-score) are generated.
7. **Prediction** – The trained model predicts the species of a new,
   unseen flower sample.

---

## 🚀 How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/iris-flower-classification.git
cd iris-flower-classification
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the program
```bash
python iris_classification.py
```

---


## 📊 Results

- **Model used:** Support Vector Machine (Linear Kernel)
- **Test set accuracy:** **100%** on the 30 held-out samples
- All three species were classified with perfect precision, recall, and
  f1-score on this split — expected for the well-separated Iris dataset.

---

## 🧠 Key Learnings

- Applying supervised ML classification algorithms to a real dataset.
- Importance of feature scaling before training distance/margin-based
  models like SVM.
- Evaluating a classifier using accuracy, confusion matrix, and
  classification report.
- Using a trained model to predict unseen/new data.

---

## 🛠️ Tech Stack

- **Python 3**
- **scikit-learn** – ML model, dataset, evaluation metrics
- **pandas** – data handling and display
- **numpy** – numerical operations

---
