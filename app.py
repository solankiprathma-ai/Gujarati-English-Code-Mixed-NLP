from flask import Flask, render_template, request
from nlp_utils import analyze_text, ensure_models

app = Flask(__name__)
ensure_models()

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    input_text = ""
    if request.method == "POST":
        input_text = request.form.get("text", "").strip()
        if input_text:
            result = analyze_text(input_text)
    return render_template("index.html", result=result, input_text=input_text)

@app.route("/health")
def health():
    return {"status": "ok", "message": "Gujarati-English Code-Mixed NLP app is running"}

if __name__ == "__main__":
    app.run(debug=True)
