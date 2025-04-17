from flask import Flask, request, jsonify
from model.fallacy_model import predict_fallacies

app = Flask(__name__)

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    results = predict_fallacies(text)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
