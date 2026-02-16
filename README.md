<div align="center">

# 🚗 Car Price Prediction with Machine Learning
### CodeAlpha Data Science Internship — Task 3

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.7+-11557c?style=for-the-badge)](https://matplotlib.org)
[![Status](https://img.shields.io/badge/Status-✅%20Completed-2ecc71?style=for-the-badge)]()
[![Internship](https://img.shields.io/badge/CodeAlpha-Internship-FF6B6B?style=for-the-badge)]()

<br>

> **A complete ML regression pipeline** that predicts used car selling prices using 4 models — achieving up to **97% R² accuracy** with full EDA, feature engineering, model comparison, and price depreciation analysis.

<br>

[📓 View Notebook](#-how-to-run) • [📊 Results](#-model-results) • [📈 Visualizations](#-visualizations) • [📁 Structure](#-project-structure)

</div>

---

## 📌 Project Overview

This project is **Task 3** of the CodeAlpha Data Science Internship. The goal is to build a machine learning regression model that predicts the **selling price of used cars** based on features like brand, age, mileage, fuel type, and transmission.

The project covers the complete ML pipeline — from raw data cleaning through feature engineering, model training, comparison, and actionable price prediction insights.

---

## 🎯 Objectives

- ✅ Perform full EDA to understand price distribution and key influencing factors
- ✅ Engineer new features (Car Age, Price Depreciation %, KMs per Year)
- ✅ Encode categorical variables and scale features for optimal model performance
- ✅ Train and compare 4 regression models side-by-side
- ✅ Evaluate using R², MAE, RMSE and cross-validation
- ✅ Visualize Actual vs Predicted prices and residual analysis
- ✅ Simulate price depreciation over time using the best model

---

## 📂 Dataset

| Property | Detail |
|----------|--------|
| **File** | `car_data.csv` |
| **Rows** | 301 cars |
| **Original Features** | 9 |
| **Engineered Features** | 4 new features added |
| **Missing Values** | None |

**Original Features:**

| Feature | Type | Description |
|---------|------|-------------|
| `Car_Name` | Categorical | Car brand/model name |
| `Year` | Numerical | Manufacturing year |
| `Selling_Price` | Numerical | **Target** — price in Lakhs |
| `Present_Price` | Numerical | Current ex-showroom price |
| `Driven_kms` | Numerical | Total kilometers driven |
| `Fuel_Type` | Categorical | Petrol / Diesel / CNG |
| `Selling_type` | Categorical | Dealer / Individual |
| `Transmission` | Categorical | Manual / Automatic |
| `Owner` | Numerical | Number of previous owners |

**Engineered Features:**

| Feature | Formula | Insight |
|---------|---------|---------|
| `Car_Age` | `2024 - Year` | More intuitive than year |
| `Price_Drop` | `Present - Selling` | Absolute depreciation |
| `Price_Drop_Pct` | `Drop / Present × 100` | Depreciation percentage |
| `KMs_per_Year` | `Driven_kms / Car_Age` | Usage intensity |

---

## 🤖 Models Used

| # | Model | Key Parameters |
|---|-------|---------------|
| 1 | Linear Regression | Baseline model |
| 2 | Decision Tree Regressor | `max_depth=6` |
| 3 | Random Forest Regressor | `n_estimators=100, max_depth=8` |
| 4 | **Gradient Boosting** | `n_estimators=100, max_depth=4` |

All models trained with:
- **80/20 Train-Test Split** (240 train / 61 test)
- **StandardScaler** normalization
- **5-Fold Cross-Validation**

---

## 📊 Model Results

| Rank | Model | R² Score | MAE | RMSE |
|------|-------|:--------:|:---:|:----:|
| 🥇 | **Gradient Boosting** | **0.9699** | **0.519L** | **0.833L** |
| 🥈 | Random Forest | 0.9599 | 0.626L | 0.962L |
| 🥉 | Decision Tree | 0.9358 | 0.764L | 1.216L |
| 4 | Linear Regression | 0.8470 | 1.222L | 1.878L |

> 🏆 **Gradient Boosting wins** — predicts car prices within ₹0.52 Lakhs on average!

---

## 📈 Visualizations (9 Plots)

| # | Plot | File | Description |
|---|------|------|-------------|
| 1 | 💰 Price Distribution | `price_distribution.png` | Price histogram, by fuel type, by transmission |
| 2 | 🔗 Correlation Heatmap | `correlation_heatmap.png` | Feature correlation matrix |
| 3 | 🔍 EDA Insights | `eda_insights.png` | Scatter plots, boxplots, depreciation analysis |
| 4 | ⚙️ Feature Analysis | `feature_analysis.png` | Top cars by price, year trend, owner distribution |
| 5 | 🏆 Model Comparison | `model_comparison.png` | R², MAE, CV scores side-by-side |
| 6 | 🎯 Actual vs Predicted | `actual_vs_predicted.png` | All 4 models scatter plots |
| 7 | 🌳 Feature Importance | `feature_importance.png` | Random Forest feature importances |
| 8 | 📉 Residuals Analysis | `residuals_analysis.png` | Residual scatter, distribution, RMSE comparison |
| 9 | 🔮 Price Prediction | `price_prediction.png` | Depreciation curve + multi-model comparison |

---

## 💡 Key Insights

1. **Present Price is the strongest predictor** — Higher ex-showroom price directly correlates with higher resale value
2. **Car Age matters most for depreciation** — Price drops sharply in the first 3–5 years
3. **Diesel cars command higher resale prices** than Petrol or CNG
4. **Automatic transmission** fetches significantly higher prices than manual
5. **Gradient Boosting outperforms all** — captures non-linear relationships better than tree-based single models
6. **Average depreciation is ~50%** — cars lose roughly half their value over their lifetime

---

## 📁 Project Structure

```
CodeAlpha_CarPricePrediction/
│
├── 📓 car_price_prediction.ipynb    ← Main Jupyter Notebook
├── 📄 README.md                     ← This file
├── 📋 requirements.txt              ← Dependencies
├── 📂 car_data.csv                  ← Dataset
│
└── 📊 Plots/
    ├── price_distribution.png
    ├── correlation_heatmap.png
    ├── eda_insights.png
    ├── feature_analysis.png
    ├── model_comparison.png
    ├── actual_vs_predicted.png
    ├── feature_importance.png
    ├── residuals_analysis.png
    └── price_prediction.png
```

---

## 🚀 How to Run

### Option 1 — Google Colab

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MOHAMMED-ABUZAR317/CodeAlpha_CarPricePrediction/blob/main/car_price_prediction.ipynb)

### Option 2 — Run Locally

```bash
git clone https://github.com/MOHAMMED-ABUZAR317/CodeAlpha_CarPricePrediction.git
cd CodeAlpha_CarPricePrediction
pip install -r requirements.txt
jupyter notebook car_price_prediction.ipynb
```

---

## 📦 Requirements

```txt
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
jupyter>=1.0.0
```

---

## 📚 What I Learned

- How to engineer meaningful features from raw data (Car Age, Depreciation %)
- Label encoding for categorical variables and StandardScaler for numerical
- Comparing ensemble methods (Random Forest, Gradient Boosting) vs simple models
- Interpreting feature importance to understand what drives car prices
- Using residual analysis to validate model assumptions

---

## 🔗 Connect

<div align="center">

| Platform | Link |
|----------|------|
| 💼 LinkedIn | [Mohammed Abuzar](https://linkedin.com/in/mohammed-abuzar) |
| 🐙 GitHub | [MOHAMMED-ABUZAR317](https://github.com/MOHAMMED-ABUZAR317) |
| 🏢 Internship | [CodeAlpha](https://www.codealpha.tech) |

</div>

---

<div align="center">

**🚗 Made with ❤️ during the CodeAlpha Data Science Internship**

*If you found this helpful, give it a ⭐ on GitHub!*

</div>
