# 🚢 Titanic Dataset - Data Preprocessing

## 🎯 Objective
To apply data cleaning and preprocessing techniques on the Titanic dataset in preparation for machine learning models.

## 📁 Dataset
Original Titanic dataset from Kaggle:  
[https://www.kaggle.com/c/titanic/data](https://www.kaggle.com/c/titanic/data)

## 🛠 Technologies Used
- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn

## 📊 Steps Performed
1. **Loaded** the dataset (891 records, 12 columns)
2. **Handled missing values**:
   - Filled `Age` with **median**
   - Filled `Embarked` with **mode**
   - Dropped `Cabin` (too many nulls)
3. **Encoded categorical features**:
   - `Sex` → Label Encoding
   - `Embarked` → One-Hot Encoding
4. **Normalized** numerical features:
   - `Age`, `Fare` using `StandardScaler`
5. **Removed outliers**:
   - Detected and removed high outliers in `Fare` using IQR method
6. **Final dataset shape**: **775 rows × 12 columns**

## 📄 Files Included
📄 `main.py` – Script for data preprocessing  
📁 `titanic_preprocessed_dataset.csv` – Cleaned and preprocessed Titanic dataset

## ✅ Outcome
The dataset is clean, preprocessed, and ready for training machine learning models such as logistic regression or decision trees for survival prediction.

---
📂 _All code files and processed datasets are available in this repository._
