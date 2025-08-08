# app.py
from flask import Flask, request, jsonify
from transformers import pipeline
import logging

# Configure basic logging to see application output
logging.basicConfig(level=logging.INFO)

# Initialize the Flask web application
app = Flask(__name__)

# Load the sentiment analysis model from Hugging Face.
# This happens only once when the application starts up.
try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )
    logging.info("Sentiment analysis model loaded successfully.")
except Exception as e:
    # If the model fails to load, log the error and set it to None
    logging.error(f"Fatal error: Could not load sentiment analysis model. Exception: {e}")
    sentiment_analyzer = None

@app.route('/predict', methods=['POST'])
def predict_sentiment():
    """
    API endpoint that accepts a JSON payload with a 'text' field 
    and returns the model's sentiment prediction.
    """
    # Check if the model was loaded successfully on startup
    if not sentiment_analyzer:
        return jsonify({"error": "Model is not available due to a startup error"}), 503

    # --- Secure MLOps Practice: Input Validation ---
    # 1. Ensure the request is in JSON format
    if not request.is_json:
        logging.warning("Received non-JSON request")
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json()
    text_input = data.get('text')

    # 2. Validate the presence, type, and length of the input
    if not text_input or not isinstance(text_input, str):
        logging.warning("Received request with missing or invalid 'text' field")
        return jsonify({"error": "Missing or invalid 'text' field in JSON payload"}), 400

    if len(text_input) > 512:
        logging.warning("Received request exceeding maximum length")
        return jsonify({"error": "Input text exceeds maximum length of 512 characters"}), 400

    # --- Model Prediction ---
    try:
        logging.info(f"Analyzing text: '{text_input}'")
        result = sentiment_analyzer(text_input)
        prediction = result[0] # The model returns a list, we need the first element
        logging.info(f"Prediction result: {prediction}")

        # Format and return the successful response
        return jsonify({
            "sentiment": prediction['label'],
            "score": round(prediction['score'], 4)
        })
    except Exception as e:
        logging.error(f"An error occurred during model prediction: {e}")
        return jsonify({"error": "An internal error occurred during prediction"}), 500

if __name__ == '__main__':
    # Run the app on host 0.0.0.0 to be accessible inside a Docker container
    app.run(host='0.0.0.0', port=8080) # nosec B104