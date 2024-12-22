import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class ChatbotModel:
    def __init__(self, vocab_size, embedding_dim=128, lstm_units=64, num_classes=10, max_sequence_length=100):
        """
        Initializes the ChatbotModel with customizable parameters.
        :param vocab_size: Size of the vocabulary (number of unique tokens in the dataset).
        :param embedding_dim: Dimension of the embedding layer.
        :param lstm_units: Number of units in the LSTM layer.
        :param num_classes: Number of output classes (possible intents).
        :param max_sequence_length: Maximum length of input sequences (pad sequences to this length).
        """
        self.model = keras.Sequential([
            # Embedding layer for converting word indices into dense vectors
            layers.Embedding(vocab_size, embedding_dim, input_length=max_sequence_length),

            # LSTM layer for learning dependencies in the sequence
            layers.LSTM(lstm_units, return_sequences=False),

            # Dropout layer for regularization to prevent overfitting
            layers.Dropout(0.5),

            # Dense output layer with softmax activation for classification
            layers.Dense(num_classes, activation='softmax')
        ])

        # Compile the model with Adam optimizer and sparse categorical crossentropy loss function
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    def train(self, X_train, y_train, epochs=100, batch_size=32, validation_data=None):
        """
        Trains the chatbot model on the provided dataset.
        :param X_train: Training input data (e.g., tokenized and padded sequences).
        :param y_train: Training labels (e.g., intents).
        :param epochs: Number of epochs for training.
        :param batch_size: Batch size for training.
        :param validation_data: Tuple (X_val, y_val) for validation during training.
        :return: History object containing training and validation metrics.
        """
        return self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    def save(self, filepath):
        """
        Saves the model to the specified file path.
        :param filepath: Path where the model will be saved.
        """
        self.model.save(filepath)

    def load(self, filepath):
        """
        Loads a pre-trained model from the specified file path.
        :param filepath: Path from where the model will be loaded.
        """
        self.model = tf.keras.models.load_model(filepath)

    def predict(self, X_input):
        """
        Predicts the intent for a given input sequence.
        :param X_input: Input sequence (tokenized and padded).
        :return: Predicted intent (the class index with highest probability).
        """
        predictions = self.model.predict(X_input)
        return predictions.argmax(axis=-1)  # Return the index of the class with the highest probability
