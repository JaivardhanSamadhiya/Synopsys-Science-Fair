from flask import Flask, request, jsonify
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    try:
        # Read the file into a DataFrame
        df = pd.read_csv(file)

        # Call your model or process the data here
        # For example, assuming `process_phage_data` is a function that analyzes the data
        results = process_phage_data(df)  # This should return predictions and confidence values
        
        # Return the results to the frontend
        return jsonify({"prediction": results['predictions'], "confidence": results['confidence']})
    
    except Exception as e:
        print(f"Error processing file: {e}")
        return jsonify({"error": str(e)}), 500

def process_phage_data(df):
    # Example function to process the data and make predictions (replace with your actual logic)
    # This would involve feeding the dataframe through your model or whatever analysis you have
    # For now, it will return dummy values
    predictions = ["Lytic" for _ in range(len(df))]
    confidence = [0.85 for _ in range(len(df))]
    return {'predictions': predictions, 'confidence': confidence}

if __name__ == '__main__':
    app.run(debug=True)
