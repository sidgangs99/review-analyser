import os

# Import necessary libraries
from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer

# Initialize env

# Initialize Flask app
app = Flask(__name__)

# Initialize model
model_name = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Define API endpoint for receiving data and returning summary
@app.route('/get_summary', methods=['POST'])
def get_summary():
    try:
        # Get data from the request
        data = request.json

        data = request.get_json()
        text = data.get('text')

        # Assuming your ML model or function is named generate_summary
        input_ids = tokenizer.encode(text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(input_ids, max_length=1000, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

        # Return the summary as JSON response
        return jsonify({"summary": summary})

    except Exception as e:
        # Handle errors gracefully
        return jsonify({"error": str(e)})

# Run the server
if __name__ == '__main__':
    app.run(debug=True, port=5000)
