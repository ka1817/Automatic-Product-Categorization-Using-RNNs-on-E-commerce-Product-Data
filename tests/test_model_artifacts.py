import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json

def test_model_files_exist():
    assert os.path.exists("models/lstm_text_classifier.h5")
    assert os.path.exists("models/tokenizer.json")
    assert os.path.exists("models/label_classes.npy")

def test_model_loads():
    model = load_model("models/lstm_text_classifier.h5")
    assert model is not None

def test_tokenizer_loads():
    with open("models/tokenizer.json") as f:
       json_str = f.read()
    tokenizer = tokenizer_from_json(json_str)
    assert tokenizer is not None

def test_label_classes_load():
    classes = np.load("models/label_classes.npy", allow_pickle=True)
    assert len(classes) > 0
