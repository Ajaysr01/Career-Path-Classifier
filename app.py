from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Global variables to store models and encoders
models = {}
le_dict = None
feature_columns = None
scalers = {}

def load_models():
    """Load and train all three models"""
    global models, le_dict, feature_columns, scalers
    
    print("Loading and training models...")
    
    # Load data
    df = pd.read_csv('dataset9000.csv')
    df_modeling = df.copy()
    
    # Encode categorical variables
    le_dict = {}
    for col in df_modeling.columns:
        if df_modeling[col].dtype == 'object':
            le = LabelEncoder()
            df_modeling[col] = le.fit_transform(df_modeling[col])
            le_dict[col] = le
    
    # Prepare features and target
    X = df_modeling.drop('Role', axis=1)
    y = df_modeling['Role']
    feature_columns = X.columns.tolist()
    
    print("  - Training Random Forest...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=15, n_jobs=-1)
    rf_model.fit(X, y)
    models['random_forest'] = rf_model
    scalers['random_forest'] = None  # Random Forest doesn't need scaling
    
    print("  - Training SVM...")
    svm_scaler = StandardScaler()
    X_scaled_svm = svm_scaler.fit_transform(X)
    svm_model = SVC(kernel='rbf', C=100, gamma='scale', probability=True, random_state=42, max_iter=1000)
    svm_model.fit(X_scaled_svm, y)
    models['svm'] = svm_model
    scalers['svm'] = svm_scaler
    
    print("  - Training Logistic Regression...")
    lr_scaler = StandardScaler()
    X_scaled_lr = lr_scaler.fit_transform(X)
    lr_model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial', n_jobs=-1)
    lr_model.fit(X_scaled_lr, y)
    models['logistic_regression'] = lr_model
    scalers['logistic_regression'] = lr_scaler
    
    print("All models loaded and trained successfully!")

def predict_role(input_dict, model_name='random_forest'):
    """
    Make prediction for given input using specified model
    input_dict: dictionary with skill names as keys and skill levels as values
    model_name: name of the model to use
    """
    try:
        # Create feature vector with proper encoding
        feature_vector = []
        for col in feature_columns:
            if col in input_dict:
                skill_level = input_dict[col]
                try:
                    encoded_value = le_dict[col].transform([skill_level])[0]
                except ValueError:
                    # If encoding fails, find the closest match
                    print(f"[v0] Warning: Could not encode {skill_level} for {col}, using default")
                    encoded_value = 0
                feature_vector.append(encoded_value)
            else:
                print(f"[v0] Warning: Missing feature {col}")
                feature_vector.append(0)
        
        feature_vector = np.array(feature_vector).reshape(1, -1)
        
        # Scale if needed
        if scalers[model_name] is not None:
            feature_vector = scalers[model_name].transform(feature_vector)
        
        # Get prediction and probabilities
        model = models[model_name]
        prediction = model.predict(feature_vector)[0]
        probabilities = model.predict_proba(feature_vector)[0]
        
        # Decode prediction
        predicted_role = le_dict['Role'].inverse_transform([prediction])[0]
        
        # Get all roles with their probabilities
        roles = le_dict['Role'].classes_
        role_probabilities = {role: float(prob) for role, prob in zip(roles, probabilities)}
        
        return {
            'predicted_role': predicted_role,
            'confidence': float(max(probabilities)),
            'probabilities': role_probabilities,
            'model_used': model_name
        }
    except Exception as e:
        print(f"[v0] Prediction error: {str(e)}")
        return {'error': f"Prediction failed: {str(e)}"}

@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    return jsonify({
        'models': [
            {'id': 'random_forest', 'name': 'Random Forest', 'description': 'Ensemble method with high accuracy'},
            {'id': 'svm', 'name': 'Support Vector Machine', 'description': 'Powerful classifier for complex patterns'},
            {'id': 'logistic_regression', 'name': 'Logistic Regression', 'description': 'Fast baseline classifier'}
        ]
    })

@app.route('/api/skills', methods=['GET'])
def get_skills():
    """Get all available skills"""
    if not models:
        load_models()
    
    skills = [col for col in feature_columns if col != 'Role']
    skill_levels = le_dict[skills[0]].classes_.tolist() if skills else []
    
    print(f"[v0] Available skills: {skills}")
    print(f"[v0] Skill levels: {skill_levels}")
    
    return jsonify({
        'skills': skills,
        'levels': skill_levels
    })

@app.route('/api/predict', methods=['POST'])
def predict():
    """Make prediction based on input data"""
    if not models:
        load_models()
    
    try:
        data = request.json
        model_name = data.get('model', 'random_forest')
        
        print(f"[v0] Prediction request received for model: {model_name}")
        print(f"[v0] Input data: {data}")
        
        # Remove model field from data before prediction
        prediction_data = {k: v for k, v in data.items() if k != 'model'}
        
        result = predict_role(prediction_data, model_name)
        
        print(f"[v0] Prediction result: {result}")
        
        return jsonify(result)
    except Exception as e:
        print(f"[v0] API error: {str(e)}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    load_models()
    app.run(debug=True, port=5000)
