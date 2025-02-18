from flask import Flask, request, jsonify
from flask_cors import CORS
import pyttsx3
import base64
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model from Hugging Face
model_name = "Lonewolf12/Kim_backend"
 # Your Hugging Face model repo
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

@app.get("/")
def home():
    return {"message": "Model is running from Hugging Face!"}

@app.route('/generate', methods=['POST'])
def generate_response():
    data = request.json
    context = data.get('context', '')
    gender = data.get('gender', 'female').lower()  # Defaults to female

    if not context:
        return jsonify({'error': 'Context is required.'}), 400

    try:
        # Generate text response
        inputs = tokenizer(context, return_tensors='pt', max_length=256, padding='max_length', truncation=True)
        with torch.no_grad():
            output_sequences = model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=256)
        response_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

        # Generate audio response
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        if gender == 'male':
            engine.setProperty('voice', voices[0].id)  # Male voice
        else:
            engine.setProperty('voice', voices[1].id)  # Female voice
        
        engine.save_to_file(response_text, 'response.mp3')
        engine.runAndWait()

        with open('response.mp3', 'rb') as f:
            audio_base64 = base64.b64encode(f.read()).decode('utf-8')

        return jsonify({'response': response_text, 'audio': audio_base64})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
