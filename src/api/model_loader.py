import os
import json
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json

def load_artifacts():
    model_path = os.path.join("models", "lstm_text_classifier.h5")
    tokenizer_path = os.path.join("models", "tokenizer.json")
    labels_path = os.path.join("models", "label_classes.npy")  

    model = load_model(model_path)

    with open(tokenizer_path, 'r') as f:
        tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(json.dumps(tokenizer_json))

    label_classes = np.load(labels_path, allow_pickle=True).tolist()

    return model, tokenizer, label_classes
