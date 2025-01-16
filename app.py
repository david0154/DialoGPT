from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Flask app
app = Flask(__name__)

# Firebase setup (path to your Firebase key file)
cred = credentials.Certificate('firebase-key.json')  # Path to your Firebase key file in the root directory
firebase_admin.initialize_app(cred)
db = firestore.client()  # Firestore database instance

# Load the pre-trained model and tokenizer (from the root directory)
model = AutoModelForCausalLM.from_pretrained('./')  # Loads model from the current directory
tokenizer = AutoTokenizer.from_pretrained('./')   # Loads tokenizer from the current directory

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.json['text']

    # Tokenize and generate a response using the model
    inputs = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors="pt")
    response_ids = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id)
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

    # Store user input and response in Firebase Firestore
    db.collection('chat_history').add({
        'user_input': user_input,
        'response': response_text
    })

    return jsonify({"response": response_text})

# Run the Flask app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))  # Default to 5000 for local testing
    app.run(debug=True, host='0.0.0.0', port=port)
