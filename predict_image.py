import numpy as np
import pandas as pd
import os
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from neural_network import NeuralNetwork

app = Flask(__name__, static_folder='src/static', template_folder='src/templates')
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Assurez-vous que le dossier uploads existe
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def load_model():
    nn = NeuralNetwork()
    weights = np.load('model_weights.npz')
    nn.weights_input_hidden = weights['w1']
    nn.bias_input_hidden = weights['b1']
    nn.weights_hidden_middle = weights['w2']
    nn.bias_hidden_middle = weights['b2']
    nn.weights_middle_output = weights['w3']
    nn.bias_middle_output = weights['b3']
    return nn

def preprocess_image(features):
    if len(features) != 42:
        raise ValueError("L'image doit avoir 42 coordonnées")
    return np.array([features])

def predict(features):
    nn = load_model()
    X = preprocess_image(features)
    probabilities = nn.forward(X)
    prediction = nn.predict(X)[0]
    letters = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}
    return letters[prediction]

def extract_features_from_image(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"L'image {image_path} n'existe pas")

    data = pd.read_csv('src/datas/data_formatted.csv', header=None)
    filename = os.path.basename(image_path)
    try:
        image_num = int(''.join(filter(str.isdigit, filename))) - 1
    except ValueError:
        raise ValueError(f"Impossible d'extraire le numéro de l'image depuis {filename}")

    if image_num < 0 or image_num >= len(data):
        raise ValueError(f"Numéro d'image {image_num} hors plage (0 à {len(data) - 1})")

    features = data.iloc[image_num, :42].values
    true_label_onehot = data.iloc[image_num, 42:].values
    true_label = np.argmax(true_label_onehot) + 1
    return features, true_label

@app.route('/')
def upload_form():
    return render_template("upload.html")

@app.route('/predict', methods=['POST'])
def predict_image():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        try:
            features, true_label = extract_features_from_image(file_path)
            predicted_letter = predict(features)
            letters = {1: 'A', 2: 'B', 3: 'C', 4: 'D', 5: 'E'}

            # Supprimez le fichier après traitement
            os.remove(file_path)

            return render_template('upload.html',
                                 prediction=predicted_letter,
                                 true_label=letters[true_label])
        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5001)
