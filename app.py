from flask import Flask, render_template, request
import joblib

app = Flask(__name__)
model = joblib.load(r"C:\Users\khush\OneDrive\Desktop\PyTraining\API\spam_model.pkl")
vectorizer = joblib.load(r"C:\Users\khush\OneDrive\Desktop\PyTraining\API\vectorizer.pkl")

@app.route('/', methods= ['GET', 'POST'])
def home():
    result = ''
    if request.method == 'POST':
        msg = request.form['message']
        # Transform the message using the loaded vectorizer
        vec = vectorizer.transform([msg])
        pred = model.predict(vec)[0]
        result = "Spam" if pred == 1 else "Not Spam"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)