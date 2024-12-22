from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

class Trainer:
    def __init__(self, preprocessor, model, dataset, model_path):
        """
        Initializes the Trainer class.

        :param preprocessor: Preprocessor object for tokenizing and encoding input data
        :param model: Chatbot model object to be trained
        :param dataset: Dataset containing tokens and encoded labels
        :param model_path: Directory path to save the trained model
        """
        self.preprocessor = preprocessor
        self.model = model
        self.dataset = dataset
        self.model_path = model_path

    def prepare_data(self, max_vocab_size=5000, max_length=20):
        """
        Prepares the training and validation data by tokenizing and padding sequences.

        :param max_vocab_size: Maximum size of the tokenizer vocabulary
        :param max_length: Maximum length for padded sequences
        :return: Tuple of training and validation data, and the tokenizer object
        """
        tokenizer = Tokenizer(num_words=max_vocab_size, oov_token="<OOV>")
        tokenizer.fit_on_texts(self.dataset['tokens'])

        # Tokenize and pad sequences
        X = tokenizer.texts_to_sequences(self.dataset['tokens'])
        X = pad_sequences(X, maxlen=max_length, padding='post')

        # Labels
        y = self.dataset['encoded_labels']

        # Split data into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        return X_train, X_val, y_train, y_val, tokenizer

    def train_model(self, epochs=100, batch_size=32):
        """
        Trains the chatbot model and saves it to the specified path.

        :param epochs: Number of training epochs
        :param batch_size: Size of training batches
        :return: Training history dictionary containing accuracy and loss metrics
        """
        # Prepare data
        X_train, X_val, y_train, y_val, tokenizer = self.prepare_data()

        # Train the model
        history = self.model.train(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val)
        )

        # Ensure the model directory exists
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        # Save the model in the Keras format (chatbot_model.keras)
        self.model.save(os.path.join(self.model_path, 'chatbot_model.keras'))

        # Save the tokenizer (optional)
        with open(os.path.join(self.model_path, 'tokenizer.json'), 'w') as f:
            f.write(tokenizer.to_json())

        # Return the training history dictionary (accuracies and losses)
        return history.history
