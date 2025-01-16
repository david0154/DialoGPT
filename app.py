from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os

# Initialize Flask app
app = Flask(__name__)

# Load the DialoGPT model and tokenizer
MODEL_DIR = './'  # Assuming model files are in the root directory
model = AutoModelForCausalLM.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json['text']
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(inputs, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    return jsonify({"response": response_text})

# Set the port dynamically based on Render's environment variable
port = int(os.environ.get("PORT", 4000))  # Default to 10000 if PORT is not set

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=port)
