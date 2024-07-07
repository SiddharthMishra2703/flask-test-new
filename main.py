from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the model
model = joblib.load('xgb_classifier.joblib')


@app.route('/', methods=['GET'])
def home():
    return jsonify('hello')


@app.route('/predict', methods=['GET'])
def predict():
    # data = request.get_json(force=True)
    features = [885.157845, 853.763730, 9.063146, -
                0.000179, 2.143342, 2661.894136, 72.203287]
    prediction = model.predict([features])
    # prediction = model.predict([np.array(data['features'])])
    return jsonify(prediction.tolist())


PORT = os.environ.get('PORT', 3000)
if __name__ == '__main__':
    app.run(port=PORT, debug=True)
