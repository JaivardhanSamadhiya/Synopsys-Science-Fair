import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle
from flask import Flask, request, jsonify

# 1. Model Training

# Load the CSV file
df = pd.read_csv('cleaned_host_phage_data.csv')

# Preprocess the data
numerical_cols = ['Phage_GC_Content', 'Host_GC_Content'] + [f'Phage_Kmer_{i}' for i in range(3976)] + [f'Host_Kmer_{i}' for i in range(4096)]
df[numerical_cols] = (df[numerical_cols] - df[numerical_cols].mean()) / df[numerical_cols].std()
X = df.drop(['Phage_ID', 'Host_ID'], axis=1)
y = df.sample(len(df), random_state=42)['Phage_ID'].apply(lambda x: 0 if hash(x) % 2 == 0 else 1)

# Train the XGBoost classifier
model = XGBClassifier()
model.fit(X, y)
print(f'Model Accuracy: {accuracy_score(y, model.predict(X)):.2f}')
print(classification_report(y, model.predict(X)))

# Save the trained model
pickle.dump(model, open('xgboost_model.json', 'wb'))

# 2. Flask API

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded file
    file = request.files['file']

    # Load and preprocess the data
    df = pd.read_csv(file)
    num_cols = ['Phage_GC_Content', 'Host_GC_Content'] + [f'Phage_Kmer_{i}' for i in range(3976)] + [f'Host_Kmer_{i}' for i in range(4096)]
    df[num_cols] = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
    X = df.drop(['Phage_ID', 'Host_ID'], axis=1)

    # Load the trained model
    model = pickle.load(open('xgboost_model.json', 'rb'))

    # Make predictions
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Return the results
    return jsonify({
        'prediction': [int(pred) for pred in y_pred],
        'confidence': [float(prob) for prob in y_prob]
    })

if __name__ == '__main__':
    app.run(debug=True)
