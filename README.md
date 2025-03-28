# Disease Prediction System

A machine learning system that predicts diseases based on patient symptoms using a Random Forest Classifier.

## Overview

This system analyzes patient symptoms and provides probabilistic predictions of potential diseases. It uses a carefully tuned Random Forest model trained on a dataset of disease-symptom relationships.

## Features

- Predicts diseases based on a list of symptoms
- Provides confidence scores for predictions
- Identifies the most relevant symptoms for disease prediction
- Achieves 98% accuracy in disease prediction

## Dataset

The model is trained on a structured dataset containing:
- 41 distinct diseases
- 131 unique symptoms
- 304 unique disease-symptom combinations after deduplication

## Technical Details

- **Algorithm**: Random Forest Classifier with optimized hyperparameters
- **Feature Selection**: Identifies the 53 most predictive symptoms
- **Performance**: 98.36% accuracy on test data, 95.05% average on cross-validation
- **Implementation**: Python with scikit-learn

## Usage

```python
# Load the model
import pickle
with open('disease_prediction_model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    
model = model_data['model']
selected_symptoms = model_data['selected_symptoms']
disease_mapping = model_data['disease_mapping']

# Predict based on symptoms
def predict_disease(symptoms_list):
    # Convert input symptoms to feature vector
    input_features = np.zeros(len(selected_symptoms))
    for symptom in symptoms_list:
        if symptom in selected_symptoms:
            idx = selected_symptoms.index(symptom)
            input_features[idx] = 1

    # Predict using the model
    prediction = model.predict([input_features])[0]
    disease = disease_mapping[prediction]
    
    # Get probabilities for top 3 predictions
    probabilities = model.predict_proba([input_features])[0]
    top_3_indices = probabilities.argsort()[-3:][::-1]
    top_3_diseases = [(disease_mapping[idx], probabilities[idx]) for idx in top_3_indices]
    
    return disease, top_3_diseases

# Example
symptoms = ['itching', 'skin_rash', 'nodal_skin_eruptions']
disease, top_3 = predict_disease(symptoms)
print(f"Predicted disease: {disease}")
print("Top 3 predictions:")
for d, prob in top_3:
    print(f"{d}: {prob:.4f}")
