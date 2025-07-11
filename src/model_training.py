import os
import mlflow
import mlflow.tensorflow
import numpy as np
import tensorflow as tf
import warnings
warnings.filterwarnings('ignore')
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from src.data_preprocessing import data_preprocessing
from src.logger import get_logger

logger = get_logger(__name__, "model_training.log")

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("ecommerce-text-categorization")

def train_model():
    try:
        df = data_preprocessing()
        texts = df['Description'].astype(str).tolist()
        labels = df['class'].astype(str).tolist()

        logger.info("Preprocessing text and labels")

        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(labels)
        num_classes = len(label_encoder.classes_)

        tokenizer = Tokenizer(num_words=10000, oov_token="<OOV>")
        tokenizer.fit_on_texts(texts)
        sequences = tokenizer.texts_to_sequences(texts)
        padded_sequences = pad_sequences(sequences, maxlen=300, padding='post', truncating='post')

        X_train, X_test, y_train, y_test = train_test_split(
            padded_sequences, encoded_labels, test_size=0.2, random_state=42
        )

        logger.info("Building LSTM model.")
        vocab_size = min(len(tokenizer.word_index) + 1, 10000)

        model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=128, input_length=300),
            LSTM(128, return_sequences=False),
            Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.3),
            Dense(num_classes, activation='softmax')
        ])

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True,
            verbose=1
        )

        with mlflow.start_run():
            mlflow.log_param("model_type", "LSTM")
            mlflow.log_param("vocab_size", vocab_size)
            mlflow.log_param("sequence_length", 300)

            logger.info("Training model with early stopping.")
            history = model.fit(
                X_train, y_train,
                epochs=10,
                batch_size=32,
                validation_split=0.1,
                callbacks=[early_stopping]
            )

            loss, accuracy = model.evaluate(X_test, y_test)
            mlflow.log_metric("test_loss", loss)
            mlflow.log_metric("test_accuracy", accuracy)

            models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
            os.makedirs(models_dir, exist_ok=True)

            model_path = os.path.join(models_dir, "lstm_text_classifier.h5")
            tokenizer_path = os.path.join(models_dir, "tokenizer.json")
            label_encoder_path = os.path.join(models_dir, "label_classes.npy")

            model.save(model_path)
            with open(tokenizer_path, 'w') as f:
                f.write(tokenizer.to_json())
            np.save(label_encoder_path, label_encoder.classes_)

            mlflow.log_artifact(model_path)
            mlflow.log_artifact(tokenizer_path)
            mlflow.log_artifact(label_encoder_path)

            logger.info(f"Model training completed with accuracy: {accuracy}")

    except Exception as e:
        logger.error(f"Error during model training: {e}", exc_info=True)
        raise

if __name__ == '__main__':
    train_model()
