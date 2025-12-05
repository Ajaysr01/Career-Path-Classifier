# Job Role Predictor - Machine Learning Pipeline

A comprehensive machine learning application that predicts job roles based on user skills using multiple classification models.

## Features

**Three Classification Models:**
- Random Forest Classifier
- Support Vector Machine (SVM)
- Logistic Regression

**Advanced Metrics & Evaluation:**
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC Curves (One-vs-Rest)
- Confusion Matrices
- K-Fold Cross-Validation (5 folds)
- Feature Importance Analysis (Random Forest)

**Interactive Web Interface:**
- Beautiful gradient-based UI
- Real-time skill input
- Model selection dropdown
- Probability bars for each role
- Mobile-responsive design

**Comprehensive Visualizations:**
- Cross-validation performance charts
- Confusion matrices heatmaps
- ROC curves for multiclass classification
- Feature importance bar charts

## Dataset

**File:** `dataset9000.csv`
- **Samples:** 9,000 training examples
- **Features:** 17 skill levels (Database Fundamentals, Computer Architecture, Distributed Computing, etc.)
- **Target:** Job Role (multiple classifications)
- **Skill Levels:** Professional, Not Interested, Poor, Beginner, Average, Intermediate, Excellent

## Project Structure

.
├── enhanced_random_forest_model.py    # Enhanced RF with metrics and visualizations\n
├── svm_model.py                       # SVM classifier with ROC-AUC analysis\n
├── logistic_regression_model.py       # Logistic Regression baseline model\n
├── app.py                             # Flask web application\n
├── templates/\n
│   └── index.html                     # Interactive frontend\n
├── dataset9000.csv                    # Training dataset\n
├── requirements.txt                   # Python dependencies\n
└── README.md                          # This file\n

## Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### Step 1: Install Dependencies
#bash: 
pip install -r requirements.txt


### Step 2: Prepare Dataset
Ensure `dataset9000.csv` is in the project root directory.

### Step 3: Run Individual Models (Optional)
To view comprehensive metrics and visualizations for each model:

# Random Forest
python enhanced_random_forest_model.py

# SVM
python svm_model.py

# Logistic Regression
python logistic_regression_model.py

These scripts will generate:
- Performance metrics (Accuracy, Precision, Recall, F1-Score)
- Cross-validation analysis plots
- Confusion matrices
- ROC curves
- Feature importance charts

### Step 4: Run Flask Web Application
python app.py

The application will start at `http://localhost:5000`

## Usage

1. **Open Browser:** Navigate to `http://localhost:5000`

2. **Select Model:** Choose from Random Forest, SVM, or Logistic Regression

3. **Rate Skills:** Use dropdowns to rate your proficiency in 17 different skills:
   - Professional, Not Interested, Poor, Beginner, Average, Intermediate, Excellent

4. **Get Prediction:** Click "Predict Role" button

5. **View Results:**
   - Predicted job role with confidence percentage
   - Probability distribution across all roles
   - Model used for prediction

## Model Comparison

| Model | Type | Features | Strengths |
|-------|------|----------|-----------|
| **Random Forest** | Ensemble | Tree-based | Fast, Good generalization, Feature importance |
| **SVM** | Kernel-based | RBF Kernel | Handles non-linear patterns, High accuracy |
| **Logistic Regression** | Linear | Multinomial | Fast baseline, Interpretable, Good for baseline |

## Key Metrics

All models are evaluated using:
- **Train/Test Split:** 80/20 stratified split
- **Cross-Validation:** 5-fold stratified K-fold CV
- **Metrics:** Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Overfitting Check:** Train-test accuracy difference

## Performance Visualization

The application generates comprehensive plots:
- Cross-validation scores across folds
- Confusion matrices for each model
- ROC curves for multiclass classification (One-vs-Rest)
- Feature importance rankings (Random Forest)

## Frontend Features

- Responsive design (mobile, tablet, desktop)
- Real-time input validation
- Smooth animations
- Gradient-based purple theme
- Probability visualization with progress bars
- Model comparison capability

## Requirements

- Flask 2.3.2
- Pandas 2.0.3
- NumPy 1.24.3
- Scikit-learn 1.3.0
- Matplotlib 3.7.2
- Seaborn 0.12.2

## Troubleshooting

**Issue:** Models take time to load on first run
- **Solution:** First execution trains all three models. Subsequent runs use cached models.

**Issue:** Port 5000 already in use
- **Solution:** Modify `app.py` line: `app.run(debug=True, port=5001)`

**Issue:** Dataset not found
- **Solution:** Ensure `dataset9000.csv` is in the project root directory

## Model Details

### Random Forest
- n_estimators: 100
- max_depth: 15
- random_state: 42
- No feature scaling needed

### Support Vector Machine (SVM)
- kernel: RBF (Radial Basis Function)
- C: 1.5
- gamma: 0.002
- probability: True
- Feature scaling: StandardScaler required

### Logistic Regression
- max_iter: 1000
- multi_class: multinomial
- Feature scaling: StandardScaler required

## Future Enhancements

- Add more classification models (XGBoost, Neural Networks)
- Implement model ensemble predictions
- Add feature importance visualization in web UI
- Export predictions to CSV
- Add model performance comparison dashboard
- Implement user feedback loop for continuous learning

