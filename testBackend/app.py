from flask import Flask, request, jsonify, render_template
# from model.fallacy_model import predict_fallacies
import os
import re

# Define paths for templates and static files
template_dir = os.path.abspath('./../frontend/templates')
static_dir = os.path.abspath('./../frontend/static')

# Initialize Flask with custom template and static folders
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

lastUserInput = ""
lastUserModel = ""

@app.route("/", methods=["GET"])
def home():
    print("hello")
    return render_template('index.html', initial_text="")

@app.route("/analyze", methods=["POST"])
def analyze():
    global lastUserInput
    user_input = request.form.get("userArgument")  # or request.form["text_input"]
    lastUserInput = user_input
    model = request.form.get("model")  # or request.form["text_input"]
    print(user_input)
    return render_template('analysisPage.html', userArgument=user_input, model=model)

@app.route("/initial", methods=["GET"])
def initial():
    return render_template('initial.html', initial_text=lastUserInput)

@app.route("/test", methods=["GET"])
def test():
    return render_template('test.html')

HIGHLIGHTS = {
    "HTMX": "HTMX is a lightweight JavaScript library...",
    "tooltip": "A tooltip is a small pop-up box...",
    "highlight": "To emphasize or make something stand out visually"
}

@app.route('/load-content')
def load_content():
    original_text = """
    HTMX is a great library for adding interactivity to your web pages.
    With this implementation, we can highlight specific words and show tooltips
    when users hover over them. The tooltip will appear in a modern, stylish box.
    """
    
    # Process the text to add highlights
    processed_text = original_text
    for word, tooltip in HIGHLIGHTS.items():
        pattern = re.compile(rf'\b{word}\b', re.IGNORECASE)
        replacement = f'<span class="highlight">{word}<span class="tooltip">{tooltip}</span></span>'
        processed_text = pattern.sub(replacement, processed_text)
    
    return processed_text

if __name__ == "__main__":
    app.run(debug=True)