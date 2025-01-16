from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Initialize Flask app
app = Flask(__name__)

# Load pre-trained DialoGPT model and tokenizer
model_name = "microsoft/DialoGPT-medium"  # You can change this to other models if needed
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize conversation history
chat_history_ids = None

# David AI Self Information (static)
DAVID_AI_INFO = {
    "name": "David AI",
    "creator": "David",
    "version": "1.0",
    "company": "Nexuzy Tech Pvt Ltd",
    "description": "David AI is a conversational AI powered by DialoGPT, designed to assist users with natural conversations.",
    "language": "English",
    "creator_info": {
        "name": "David",
        "role": "Creator & Developer",
        "company": "Nexuzy Tech Pvt Ltd",
        "location": "India",
        "website": "https://imdavid.in",  # Creator's website
        "email": "davidk76011@gmail.com"  # Creator's email
    }
}

@app.route('/')
def home():
    return "Welcome to David AI! Send a POST request to /predict to start chatting."

@app.route('/predict', methods=['POST'])
def predict():
    global chat_history_ids

    # Get user input from the request
    user_input = request.json['text']

    # Encode the new user input, add the EOS token, and return a tensor in PyTorch
    new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

    # Append the new user input tokens to the chat history
    if chat_history_ids is None:
        chat_history_ids = new_user_input_ids
    else:
        chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)

    # Generate a response from the model
    bot_output = model.generate(chat_history_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

    # Decode the model's response
    bot_response = tokenizer.decode(bot_output[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True)

    # Update the chat history and return the response
    chat_history_ids = bot_output

    return jsonify({"response": bot_response})

@app.route('/info', methods=['GET'])
def get_ai_info():
    """Endpoint to return information about David AI."""
    return jsonify(DAVID_AI_INFO)

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
