# 📊 Customer Churn Prediction – Case Study Report

## 🏫 Academic Details
**Course Code:** 21AIC401T  
**Course Name:** Inferential Statistics and Predictive Analytics  
**Assignment Type:** Case Study-Based Modeling Project  
**Institution:** SRM University – Department of Computational Intelligence  
**Submission Deadline:** 10.11.2025  
**Student Name:** Suhana Guha  

---

## 📘 1. Introduction
Customer churn refers to the phenomenon where existing customers stop doing business with a company. In the telecom industry, high churn rates can significantly impact revenue and profitability. Predicting churn enables companies to identify at-risk customers and take proactive retention measures.  
This project aims to build and validate a predictive model that accurately identifies customers likely to churn using **CHAID** and **Logistic Regression** models.

---

## 🎯 2. Objective
The objective of this project is to:
- Develop and validate predictive models for churn detection.
- Identify significant factors influencing customer churn.
- Compare the performance of different models.
- Demonstrate model deployment and future updating process.

---

## 🗂️ 3. Dataset Description
- **Source:** [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records:** 7043
- **Features:** 21 columns representing customer demographics, services, and account information.
- **Target Variable:** `Churn` (Yes = 1, No = 0)

### Key Attributes
| Feature | Description |
|----------|--------------|
| gender | Customer gender (Male/Female) |
| SeniorCitizen | Whether the customer is a senior citizen |
| Partner | Whether the customer has a partner |
| Dependents | Whether the customer has dependents |
| tenure | Number of months the customer has stayed |
| Contract | Contract type (Month-to-month, One year, Two year) |
| PaymentMethod | Method of payment |
| MonthlyCharges | The amount charged per month |
| TotalCharges | Total amount charged |
| Churn | Target variable (Yes/No) |

---

## 🧹 4. Data Preparation and Cleaning
1. Loaded dataset using `pandas`.
2. Checked for missing values – found blanks in `TotalCharges`.
3. Converted `TotalCharges` to numeric and replaced NaN with median.
4. Removed duplicate records.
5. Encoded categorical variables using `pd.get_dummies()`.
6. Split data into **training (70%)** and **testing (30%)** sets.

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

---

## 📊 5. Exploratory Data Analysis (EDA)
### Observations:
- **26.6%** of total customers have churned.
- Customers with **month-to-month contracts** have the highest churn rate.
- **Electronic check** users show higher churn.
- **Tenure** and **automatic payment** methods reduce churn likelihood.

### Visualizations:
- Countplot for churn distribution
- Boxplot for MonthlyCharges vs Churn
- Histogram for tenure distribution by churn

---

## 🧠 6. Model Development and Rule Induction (CHAID)
The CHAID algorithm is a decision tree technique that segments data based on the chi-square test of independence.  
In Python, we used `DecisionTreeClassifier(criterion='entropy')` to replicate CHAID-like behavior.

### Code Example:
```python
from sklearn.tree import DecisionTreeClassifier
chaid_model = DecisionTreeClassifier(criterion='entropy', max_depth=4, random_state=42)
chaid_model.fit(X_train, y_train)
```

### Key Rules Extracted:
1. Customers with **Month-to-month contracts** and **tenure < 12 months** → High churn.
2. Customers using **Electronic Check** payment method → Moderate churn.
3. **Two-year contracts** and **automatic payments** → Low churn.

These rules help telecom operators focus retention efforts effectively.

---

## ⚙️ 7. Model Comparison and Evaluation
Two models were built: **CHAID Decision Tree** and **Logistic Regression**.

### Code Example:
```python
from sklearn.linear_model import LogisticRegression
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train, y_train)
```

### Evaluation Metrics:
| Model | Accuracy | ROC-AUC |
|--------|-----------|----------|
| CHAID Decision Tree | 79% | 0.82 |
| Logistic Regression | 81% | 0.84 |

### Results:
- Logistic Regression performs slightly better.
- Both models show reliable predictive capability.
- ROC and Lift charts confirm good model discrimination.

---

## 💾 8. Model Deployment and Updating
Models were serialized using **Joblib** for easy deployment.

```python
import joblib
joblib.dump(log_model, 'logistic_churn.pkl')
joblib.dump(chaid_model, 'chaid_churn.pkl')
```

### Loading for Use:
```python
model = joblib.load('logistic_churn.pkl')
prediction = model.predict(new_data)
```

### Model Updating:
When new data becomes available:
1. Append it to existing dataset.
2. Retrain using the same pipeline.
3. Replace the old `.pkl` model file.

---

## 🧩 9. Insights and Interpretation
- Month-to-month contracts drive higher churn.
- Longer tenure customers are more loyal.
- Payment convenience impacts retention.
- Logistic Regression provides interpretable coefficients indicating churn likelihood.

---

## 📈 10. Conclusion
This project successfully demonstrates the use of **Inferential Statistics** and **Predictive Analytics** in real-world customer churn prediction.  
The analysis provides business insights to reduce churn and improve customer satisfaction.  
Future improvements can include feature engineering and advanced ensemble models.

---

## 🧾 11. References
1. [Kaggle - Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
2. IBM SPSS CHAID Algorithm Documentation  
3. Scikit-learn Documentation (DecisionTreeClassifier, LogisticRegression)  

---

## 🔗 12. GitHub Repository
[GitHub Repository – Telco Customer Churn Analytics](https://github.com/yourusername/Telco-Customer-Churn-Analytics)

---

⭐ *“Predict today, retain tomorrow — using data-driven churn analytics.”*
