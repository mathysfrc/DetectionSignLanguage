# Sign Language Letter Recognition Project

## Overview

This project implements a **neural network** to classify hand gestures representing letters (A, B, C, D, E) in sign language. The dataset consists of preprocessed images, where each image is represented by 42 normalized coordinates (21 points with x, y coordinates) extracted from hand gestures. The goal is to train a neural network to perform 5-class classification (A, B, C, D, E) and deploy it as a web application using Flask.

The project includes:
- A neural network implemented from scratch (no external ML libraries like TensorFlow or PyTorch are used).
- A Flask web application to upload images and predict the corresponding letter.
- Training and validation on a dataset of 300 images, split into a learning dataset (250 images) and a validation dataset (50 images).


## Project Structure

```bash
SignLanguageRecognition/
├── src/
│   ├── datas/
│   │   ├── images/                # Folder containing 300 sign language images (for reference)
│   │   ├── learningDataset/       # Learning dataset (250 images, 50 per sign)
│   │   └── validationDataset/     # Validation dataset (50 images, 10 per sign)
│   ├── static/                    # Static files (e.g., CSS for the web interface)
│   ├── templates/                 # HTML templates for the web interface
│   ├── uploads/                   # Temporary folder for uploaded images
│   ├── model_weights.npz          # Saved model weights after training
│   ├── neural_network.py          # Neural network implementation
│   ├── predict_IMAGE.py           # Flask application for prediction
│   └── train_model.py             # Script for training the neural network
└── README.md                      # Project documentation
```


## Implementation Details

### Dataset

The dataset is provided in the `src/datas/` directory:
- **learningDataset/**: Contains 250 images (50 per sign: A, B, C, D, E), preprocessed into 42 normalized coordinates per image.
- **validationDataset/**: Contains 50 images (10 per sign: A, B, C, D, E), preprocessed similarly.
- **images/**: Contains the 300 original sign language images (for reference only; not used directly in training).

Each image is represented by 42 features (normalized x, y coordinates of 21 points) and a label (1 for A, 2 for B, ..., 5 for E). The dataset is split as per the requirements:
- **Learning Dataset**: 250 images (50 per sign).
- **Validation Dataset**: 50 images (10 per sign).

### Neural Network

The neural network is implemented in `neural_network.py` with the following architecture:
- **Input Layer**: 42 features (normalized coordinates).
- **Hidden Layer 1**: 64 neurons with ReLU activation.
- **Hidden Layer 2**: 32 neurons with ReLU activation.
- **Output Layer**: 5 neurons with softmax activation (for 5-class classification: A, B, C, D, E).
- **Loss Function**: Cross-entropy loss.
- **Optimizer**: Gradient descent with backpropagation.

The network is trained for 1000 epochs with a learning rate of 0.5. Weights are saved in `model_weights.npz` after training.

### Web Application

The web application is implemented in `predict_IMAGE.py` using Flask. It allows users to:
- Upload an image of a sign language gesture.
- Extract features from the image (using the preprocessed data in `src/datas/`).
- Predict the corresponding letter (A, B, C, D, or E) using the trained neural network.
- Display the predicted letter and the true label.

The web interface uses `templates/` for HTML templates and `static/` for CSS styling.

### Requirements

- Python 3.x
- Libraries used:
  - `numpy`: For numerical computations.
  - `pandas`: For reading and handling CSV data.
  - `flask`: For the web application.
  - `werkzeug`: For secure file uploads.


## How to Run

### 1. Clone the Repository

```bash
git clone <repository-url>
cd SignLanguageRecognition
```

## 2. Install Dependencies

Ensure you have Python installed, then install the required libraries:
```bash
pip install numpy pandas flask werkzeug
```

## 3. Train the Model

Run the training script to train the neural network and save the weights:
```bash
python src/train_model.py
```
This will:

- Load and split the dataset into training (250 images) and validation (50 images) sets from src/datas/learningDataset/ and src/datas/validationDataset/.
- Train the neural network for 1000 epochs.
- Save the trained weights in src/model_weights.npz.
- Print the validation accuracy.


## 4. Run the Web Application

Start the Flask server to launch the web application:
```bash
python src/predict_image.py
```
- Open your browser and go to http://localhost:5000.
- Upload an image (e.g., from the src/datas/images/ folder) to get a prediction.


## Usage 

## 1. Training:
  The train_model.py script loads the dataset from src/datas/, splits it into training and validation sets, and trains the neural network.
  After training, the model weights are saved for use in prediction.
  
## 2. Prediction:
  The web application (predict_image.py) allows users to upload an image.
  Features are extracted from the image using the corresponding preprocessed data in src/datas/.
  The neural network predicts the letter (A, B, C, D, or E) and displays the result along with the true label.


## Results

- The neural network achieves a validation accuracy of approximately 90-95% (depending on the random split and training dynamics).
- The web interface provides an intuitive way to test the model by uploading images and viewing predictions.


## Collaborators

This project was developed by the following team members:

- [Mathys Franco](https://linkedin.com/in/mathys-franco)  

- [Cameron Noupoue](https://linkedin.com/in/cnoupoue)  

- [Jobelin Kom](https://www.linkedin.com/in/jobelin-kom)  



