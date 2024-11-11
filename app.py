import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence

from flask import Flask, render_template, request
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

app = Flask(__name__)

model = None
tokenizer = None
loaded = False  # Flag to ensure loading happens only once

def load_keras_model():
    global model
    try:
        model = load_model('models/uci_sentimentanalysis.h5')
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")

def load_tokenizer():
    global tokenizer
    try:
        with open('models/tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("Tokenizer loaded successfully.")
    except Exception as e:
        print(f"Error loading tokenizer: {e}")

# Befor_first_request was not working with the keras version i have isntalled
@app.before_request
def before_request():
    global loaded
    if not loaded:
        load_keras_model()
        load_tokenizer()
        loaded = True  # Set the flag to True to prevent reloading

def sentiment_analysis(input_text):
    if tokenizer is None or model is None:
        return "Model or tokenizer not loaded."
    
    user_sequences = tokenizer.texts_to_sequences([input_text])
    user_sequences_matrix = sequence.pad_sequences(user_sequences, maxlen=1225)
    prediction = model.predict(user_sequences_matrix)
    
    return round(float(prediction[0][0]), 2)

@app.route("/", methods=["GET", "POST"])
def index():
    sentiment = {}
    if request.method == "POST":
        text = request.form.get("user_text")  # Get user input
        if text:
            analyzer = SentimentIntensityAnalyzer()
            sentiment = analyzer.polarity_scores(text)  # VADER analysis
            sentiment["custom model positive"] = sentiment_analysis(text)  # Custom model analysis

    return render_template('form.html', sentiment=sentiment)

if __name__ == "__main__":
    app.run(debug=True)

