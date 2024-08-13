from flask import Flask, request, jsonify
from transformers import pipeline

# Initialize Flask app
import tensorflow as tf
app = Flask(__name__)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Load the sentiment analysis pipeline
sentiment_pipeline = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Get the input data from the request
        data = request.get_json()
        input_text = data.get('text')

        if not input_text:
            return jsonify({'error': 'Input text is required'}), 400

        # Prepare the input data
        results = sentiment_pipeline([input_text])

        # Extract the result
        result = results[0]

        # Prepare the response
        response = {
            'text': input_text,
            'sentiment': result['label'],
            'score': result['score']
        }
        
        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
