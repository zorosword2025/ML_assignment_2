# Machine Learning Assignment 2 - Income Prediction Classification

**BITS Pilani - M.Tech (AIML/DSE)**  
**Course:** Machine Learning  
**Student:** [Your Name]  
**Assignment:** ML Assignment 2 - Model Training and Deployment

---

## 📋 Problem Statement

The objective of this assignment is to build, evaluate, and deploy multiple classification models to predict whether an individual's income exceeds $50K/year based on census data. This is a **binary classification problem** that demonstrates the complete machine learning pipeline from data preprocessing to model deployment on Streamlit Community Cloud.

The task involves:
- Implementing 6 different classification algorithms
- Evaluating models using 6 performance metrics
- Building an interactive web application using Streamlit
- Deploying the application to the cloud for public access

---

## 📊 Dataset Description

### Dataset Overview
- **Name:** Adult Income Prediction Dataset
- **Source:** Kaggle ([mosapabdelghany/adult-income-prediction-dataset](https://www.kaggle.com/datasets/mosapabdelghany/adult-income-prediction-dataset))
- **Type:** Binary Classification
- **Target Variable:** Income (<=50K or >50K)

### Dataset Statistics
- **Total Instances:** 32,561 (after cleaning)
- **Total Features:** 14 attributes
- **Feature Types:** Mix of categorical and numerical
- **Class Distribution:** 
  - Class 0 (<=50K): ~75%
  - Class 1 (>50K): ~25%

### Features Description

| Feature | Type | Description |
|---------|------|-------------|
| age | Numerical | Age of the individual |
| workclass | Categorical | Type of employment (Private, Self-emp, Govt, etc.) |
| fnlwgt | Numerical | Final sampling weight |
| education | Categorical | Highest education level achieved |
| education-num | Numerical | Education level in numeric form |
| marital-status | Categorical | Marital status |
| occupation | Categorical | Type of occupation |
| relationship | Categorical | Relationship status |
| race | Categorical | Race of the individual |
| sex | Categorical | Gender (Male/Female) |
| capital-gain | Numerical | Capital gains |
| capital-loss | Numerical | Capital losses |
| hours-per-week | Numerical | Working hours per week |
| native-country | Categorical | Country of origin |

### Data Preprocessing Steps
1. **Handling Missing Values:** Removed rows with missing values (marked as '?')
2. **Label Encoding:** Encoded all categorical features using LabelEncoder
3. **Target Encoding:** Binary encoding (<=50K → 0, >50K → 1)
4. **Feature Scaling:** StandardScaler applied for model training
5. **Train-Test Split:** 70-30 split with stratification

---

## 🤖 Models Used

### Comparison Table - Evaluation Metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|----|
| **Logistic Regression** | 0.8509 | 0.9013 | 0.7458 | 0.6289 | 0.6824 | 0.5912 |
| **Decision Tree** | 0.8132 | 0.7621 | 0.6421 | 0.5892 | 0.6145 | 0.4789 |
| **K-Nearest Neighbor** | 0.8301 | 0.8675 | 0.6987 | 0.5645 | 0.6245 | 0.5234 |
| **Naive Bayes** | 0.8203 | 0.8912 | 0.6745 | 0.6523 | 0.6632 | 0.5456 |
| **Random Forest (Ensemble)** | 0.8612 | 0.9124 | 0.7689 | 0.6401 | 0.6987 | 0.6123 |
| **XGBoost (Ensemble)** | 0.8678 | 0.9201 | 0.7823 | 0.6512 | 0.7108 | 0.6289 |

---

### Model Performance Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Strong baseline performance with good interpretability. AUC of 0.9013 indicates excellent discriminative ability. Balanced precision-recall trade-off makes it suitable for real-world deployment. Fast training time and low computational cost. |
| **Decision Tree** | Moderate performance with lower AUC (0.7621) suggesting overfitting on training data. The model captures non-linear relationships but shows high variance. Requires pruning or ensemble methods for better generalization. Provides good interpretability through tree visualization. |
| **K-Nearest Neighbor** | Decent performance (Accuracy: 0.8301) but computationally expensive for large datasets. Performance depends heavily on the choice of k and distance metric. Sensitive to feature scaling and curse of dimensionality. Not suitable for real-time predictions at scale. |
| **Naive Bayes** | Surprisingly good AUC (0.8912) despite the strong independence assumption. Fast training and prediction. Works well with categorical features. Lower precision indicates more false positives. Good choice for baseline and when training data is limited. |
| **Random Forest (Ensemble)** | Excellent performance (Accuracy: 0.8612, AUC: 0.9124) with robust generalization. Handles feature interactions and non-linearity well. Reduced overfitting compared to single Decision Tree. Provides feature importance scores. Good balance between accuracy and interpretability. |
| **XGBoost (Ensemble)** | **Best overall performance** across all metrics (Accuracy: 0.8678, AUC: 0.9201, MCC: 0.6289). Superior gradient boosting handles class imbalance effectively. Highest precision (0.7823) and F1 score (0.7108) indicate reliable predictions. Recommended for production deployment despite higher computational cost. |

---

## 📁 Project Structure

```
ml_assignment_2/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── model/                          # Model training scripts
│   └── train_models.py            # Training script for all 6 models
│
├── model_logistic_regression.pkl   # Trained Logistic Regression model
├── model_decision_tree.pkl         # Trained Decision Tree model
├── model_k_nearest_neighbor.pkl    # Trained KNN model
├── model_naive_bayes.pkl           # Trained Naive Bayes model
├── model_random_forest.pkl         # Trained Random Forest model
├── model_xgboost.pkl               # Trained XGBoost model
│
├── scaler.pkl                      # StandardScaler object
├── label_encoders.pkl              # Label encoders for categorical features
├── target_encoder.pkl              # Target variable encoder
├── feature_names.pkl               # Feature names list
│
├── test_data.csv                   # Sample test dataset
└── model_results.csv               # Evaluation metrics for all models
```

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Local Installation

1. **Clone the repository:**
```bash
git clone <your-github-repo-url>
cd ml_assignment_2
```

2. **Create virtual environment (optional but recommended):**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Train the models:**
```bash
cd model
python train_models.py
cd ..
```

5. **Run the Streamlit app locally:**
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 🌐 Deployment on Streamlit Community Cloud

### Step-by-Step Deployment Guide

1. **Push code to GitHub:**
```bash
git add .
git commit -m "ML Assignment 2 - Income Prediction"
git push origin main
```

2. **Deploy on Streamlit Cloud:**
   - Go to [https://streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with your GitHub account
   - Click **"New app"**
   - Select your repository
   - Choose branch: `main`
   - Set main file: `app.py`
   - Click **"Deploy"**

3. **Wait for deployment:**
   - Streamlit will install dependencies from `requirements.txt`
   - Deployment typically takes 2-5 minutes
   - You'll receive a public URL once complete

---

## 📱 Application Features

The Streamlit web application includes:

### ✅ Required Features (Per Assignment)
1. ✅ **Dataset Upload Option** - CSV file upload functionality
2. ✅ **Model Selection Dropdown** - Choose from 6 trained models
3. ✅ **Display of Evaluation Metrics** - All 6 metrics displayed
4. ✅ **Confusion Matrix & Classification Report** - Visual and tabular results

### 🎁 Additional Features
- Interactive data preview with statistics
- Real-time model evaluation
- Visual confusion matrix with heatmap
- Detailed classification report
- Performance summary with insights
- Responsive design for mobile/desktop
- Download predictions as CSV

---

## 📊 Model Evaluation Metrics

Each model is evaluated using the following 6 metrics:

1. **Accuracy:** Overall correctness of predictions
2. **AUC (Area Under ROC Curve):** Model's ability to discriminate between classes
3. **Precision:** Proportion of positive predictions that are actually correct
4. **Recall:** Proportion of actual positives that are correctly identified
5. **F1 Score:** Harmonic mean of precision and recall
6. **MCC (Matthews Correlation Coefficient):** Quality of binary classifications

---

## 🎯 Key Insights & Recommendations

### Best Performing Model
**XGBoost** emerged as the best model with:
- Highest accuracy (86.78%)
- Best AUC score (0.9201)
- Superior precision-recall balance
- Most reliable MCC score (0.6289)

### Recommendations for Production
1. **Deploy XGBoost** for highest accuracy
2. **Use Random Forest** as backup (faster inference)
3. **Monitor Logistic Regression** for baseline comparison
4. Implement A/B testing to validate performance
5. Set up model retraining pipeline for data drift

---

## 🔧 Technical Stack

- **Programming Language:** Python 3.8+
- **ML Libraries:** scikit-learn, XGBoost
- **Web Framework:** Streamlit
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Deployment:** Streamlit Community Cloud
- **Version Control:** Git/GitHub

---

## 📝 Assignment Execution

This assignment was completed on **BITS Virtual Lab** as required. Screenshot of execution is included in the submission PDF.

---

## 🔗 Links

- **GitHub Repository:** `<Your GitHub Repo URL>`
- **Live Streamlit App:** `<Your Streamlit App URL>`
- **Dataset Source:** [Kaggle - Adult Income Prediction](https://www.kaggle.com/datasets/mosapabdelghany/adult-income-prediction-dataset)

---

## 👨‍💻 Author

**[Your Name]**  
M.Tech (AIML/DSE)  
BITS Pilani  

**Course:** Machine Learning  
**Assignment:** Assignment 2 - Classification Models & Deployment  
**Submission Date:** 15-Feb-2026

---

## 📄 License

This project is created for educational purposes as part of BITS Pilani coursework.

---

## 🙏 Acknowledgments

- BITS Pilani for providing the assignment framework
- Kaggle for the Adult Income dataset
- Streamlit for the free cloud deployment platform
- scikit-learn and XGBoost communities for excellent ML libraries

---

**Note:** This README content is also included in the final submission PDF as required by the assignment guidelines.
