from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Path to your cloned DialoGPT model directory
MODEL_DIR = './'  # Assuming model files are in the root directory

# Load the pre-trained DialoGPT model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

@app.route('/predict', methods=['POST'])
def predict():
    # Get the user input text from the frontend
    user_input = request.json['text']

    # Tokenize the input text and add the end-of-sequence token
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")

    # Generate a response using the model
    response_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return jsonify({"response": response_text})

# Set the port dynamically based on the environment (Render)
port = int(os.environ.get("PORT", 10000))  # Use the Render-provided PORT or default to 5000

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)  # Bind to the dynamic port for Render
