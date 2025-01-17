from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load pre-trained model and tokenizer
model = AutoModelForCausalLM.from_pretrained("DialoGPT-medium")
tokenizer = AutoTokenizer.from_pretrained("DialoGPT-medium")

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json['text']  # Receive user input
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    
    # Create the attention mask
    attention_mask = torch.ones(inputs.shape, dtype=torch.long)

    # Handle padding if necessary
    # Ensure pad_token_id is set and not equal to eos_token_id
    if tokenizer.pad_token_id is not None and tokenizer.pad_token_id != tokenizer.eos_token_id:
        attention_mask = (inputs != tokenizer.pad_token_id).long()

    # Generate a response
    response_ids = model.generate(inputs, attention_mask=attention_mask, max_length=1000, pad_token_id=tokenizer.eos_token_id)
    
    # Decode the generated response
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    return jsonify({"response": response_text})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
