import numpy as np
import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.preprocessing.sequence import pad_sequences


def predict_category(text, model, tokenizer,label_classes):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=100)  

    prediction = model.predict(padded)[0]
    predicted_index = np.argmax(prediction)
    confidence = float(prediction[predicted_index])
    predicted_label = label_classes[predicted_index]

    return {
        "label": predicted_label,
        "confidence": confidence
    }
