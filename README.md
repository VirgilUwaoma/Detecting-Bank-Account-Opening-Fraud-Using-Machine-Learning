# Bank Account Opening Fraud Detection Using Machine Learning

This project focuses on building machine learning models to detect **fraudulent bank account opening applications**. By leveraging a large-scale dataset and evaluating multiple algorithms, this project demonstrates the application of data science in solving real-world fraud detection problems, with a focus on both **model accuracy** and **fairness**.

## Project Overview

Bank account fraud is an ongoing challenge for financial institutions, leading to significant financial losses and eroding customer trust. Traditional rule-based systems fail to adapt to the evolving nature of fraud. In this project, I explore **machine learning techniques** to detect fraudulent account applications using a dataset of over **1 million applications** across 30+ features. I address challenges such as class imbalance and algorithmic fairness, ensuring the solution is both practical and ethical.

## Key Features
- **Data Source**: Large-scale synthetic dataset representing real-world bank account fraud.
- **Models Evaluated**: 
  - Logistic Regression
  - Decision Trees
  - Random Forest
  - LightGBM (Gradient Boosting)
- **Performance**: Achieved **62% recall** on fraud detection using LightGBM, balancing accuracy and operational feasibility.
- **Fairness**: Evaluated model bias across protected demographic groups (e.g., age, income), ensuring that models perform equitably.

## Technologies and Tools Used
- **Languages**: Python (Pandas, NumPy, Scikit-learn, LightGBM)
- **Data Visualization**: Matplotlib, Seaborn
- **Machine Learning Techniques**:
  - Classification (Logistic Regression, Decision Trees, Random Forest, LightGBM)
  - Feature Engineering and Selection
  - Handling Imbalanced Data with Adaptive Synthetic Oversampling
  - Model Evaluation (Precision, Recall, F1-Score, ROC-AUC)
  - Fairness Metrics (Predictive Equality, Disparate Impact)
- **Jupyter Notebooks**: For data exploration, model training, and evaluation

## Project Objectives
1. **Detect Fraudulent Applications**: Build models capable of identifying fraudulent bank account applications from large datasets with a focus on **high recall**.
2. **Handle Imbalanced Data**: Employ techniques like **Adaptive Synthetic Oversampling** to mitigate the extreme rarity of fraud cases.
3. **Ensure Fairness**: Prevent algorithmic bias, ensuring fairness in fraud detection models across demographic groups.
4. **Operational Feasibility**: Evaluate the models based on practical constraints such as maintaining a **5% false positive rate**.

## Data
The dataset is a **synthetic, anonymized dataset** generated from real-world bank fraud data and includes:
- 1 million rows of data with over 30 features
- Temporal features (e.g., account application date)
- Financial and personal attributes (e.g., income, credit score, employment status)
- Fraud indicator (target variable)

### Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Understanding the distribution of features (e.g., income levels, applicant age).
- **Bivariate Analysis**: Studying relationships between key features (e.g., income vs. proposed credit limit, age vs. fraud likelihood).
- **Handling Missing Data**: Imputed missing values using domain-specific strategies.
- **Class Imbalance**: Addressed through synthetic resampling techniques.

### Modeling
- Multiple machine learning models were trained and tested, with a focus on:
  - **Logistic Regression**: Provides a strong baseline.
  - **Decision Tree**: Easy to interpret but prone to overfitting.
  - **Random Forest**: An ensemble method to reduce overfitting and improve predictive power.
  - **LightGBM**: Gradient boosting algorithm that yielded the highest recall and balanced accuracy.

### Results and Evaluation
- **Best Model**: LightGBM achieved the highest recall (62%) while adhering to operational constraints of maintaining a low false-positive rate (5%).
- **Fairness**: Models were tested for bias across demographic groups, and corrective measures were taken to ensure equal treatment.

## Key Learnings and Insights
- **Feature Engineering**: The importance of creating new features and selecting relevant ones for fraud detection.
- **Handling Imbalanced Datasets**: Implementing adaptive oversampling techniques can significantly improve model performance on rare classes like fraud cases.
- **Fairness in Machine Learning**: Ensuring that the model treats all demographic groups equitably was crucial in this project.

## Future Work
- **Model Ensembling**: Further exploration of model ensembling to enhance performance.
- **Hyperparameter Tuning**: Optimizing hyperparameters to further improve recall without sacrificing precision.
- **Real-time Fraud Detection**: Implementing real-time data pipelines to flag fraudulent applications as they occur.

## How to Run the Project
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/bank-fraud-detection.git
