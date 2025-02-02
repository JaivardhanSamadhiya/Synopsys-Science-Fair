from flask import Flask, request, jsonify
import pandas as pd
from io import StringIO
from flask_cors import CORS  # Allows frontend requests

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend communication

# Increase file upload size limit to 10MB
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        print("⚠️ No file received!")
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    
    if file.filename == '':
        print("⚠️ Empty file received!")
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read file size
        file_size = len(file.read())
        file.seek(0)  # Reset file pointer after reading

        print(f"✅ Received file: {file.filename}, Size: {file_size} bytes")

        if file_size > 100 * 1024 * 1024:  # If file exceeds 100M0B
            return jsonify({"error": "File too large"}), 400

        # Read CSV into DataFrame
        df = pd.read_csv(file, low_memory=False)

        print(f"📊 CSV Loaded: {df.shape}")  # Debugging info

        # Call your model or process the data
        results = process_phage_data(df)  # Replace with actual logic

        return jsonify({"prediction": results['predictions'], "confidence": results['confidence']})
    
    except Exception as e:
        print(f"❌ Error processing file: {e}")
        return jsonify({"error": str(e)}), 500

def process_phage_data(df):
    """ Dummy function to simulate predictions. Replace this with actual model logic. """
    predictions = ["Lytic" for _ in range(len(df))]
    confidence = [round(0.85, 2) for _ in range(len(df))]
    return {'predictions': predictions, 'confidence': confidence}

if __name__ == '__main__':
    app.run(debug=True)
