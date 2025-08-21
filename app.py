from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load(r"C:\Users\khush\OneDrive\Desktop\projects\Spam Detector\spam_model.pkl")
vectorizer = joblib.load(r"C:\Users\khush\OneDrive\Desktop\projects\Spam Detector\vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = ''
    important_words = []

    if request.method == "POST":
        msg = request.form["message"]

        # Transform the message
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]

        result = "🚨 Spam Detected!" if pred == 1 else "✅ Not Spam"

        # ---- Explainability ----
        feature_names = np.array(vectorizer.get_feature_names_out())
        
        # ✅ Fix: model.coef_ is already a numpy array
        coefs = model.coef_.flatten()  

        # Only consider words present in the message
        msg_features = vec.toarray().flatten()
        word_importance = [
            (feature_names[i], coefs[i] * msg_features[i]) 
            for i in range(len(feature_names)) if msg_features[i] != 0
        ]

        # Sort by absolute importance
        word_importance = sorted(word_importance, key=lambda x: abs(x[1]), reverse=True)
        important_words = [w for w, score in word_importance[:5]]

    return render_template("index.html", result=result, words=important_words)

if __name__ == "__main__":
    app.run(debug=True)
