import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt', quiet=True)


class Preprocessor:
    def __init__(self, dataset_path, vocab_size=5000, max_len=50):
        """
        Initializes the Preprocessor class.

        :param dataset_path: Path to the dataset CSV file
        :param vocab_size: Maximum vocabulary size for the tokenizer
        :param max_len: Maximum sequence length for padding
        """
        self.dataset_path = dataset_path
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.label_encoder = LabelEncoder()
        self.tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")  # Initialize Tokenizer

    def load_and_preprocess(self):
        """
        Loads the dataset, preprocesses user input, and encodes intent labels.

        :return: Preprocessed dataset and the fitted label encoder
        """
        # Load dataset
        try:
            data = pd.read_csv(self.dataset_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"Dataset file not found at path: {self.dataset_path}")

        # Ensure required columns exist
        if 'intent' not in data.columns or 'user_input' not in data.columns:
            raise ValueError("Dataset must contain 'intent' and 'user_input' columns in that order.")

        # Drop missing values
        data.dropna(subset=['intent', 'user_input'], inplace=True)

        # Tokenize user input
        data['tokens'] = data['user_input'].apply(lambda text: word_tokenize(str(text).lower().strip()))

        # Fit the tokenizer on user input text
        self.tokenizer.fit_on_texts(data['user_input'])

        # Encode intent labels
        data['encoded_labels'] = self.label_encoder.fit_transform(data['intent'])

        return data, self.label_encoder

    def encode_input(self, input_text):
        """
        Encodes a single input text into a padded sequence.

        :param input_text: User input string to be encoded
        :return: Padded sequence for model prediction
        """
        if not isinstance(input_text, str) or not input_text.strip():
            raise ValueError("Input text must be a non-empty string.")

        # Tokenize the input text
        tokens = word_tokenize(input_text.lower().strip())

        # Convert tokens to integer indices using the tokenizer
        encoded_input = self.tokenizer.texts_to_sequences([tokens])

        # Check for empty sequence after tokenization
        if not encoded_input[0]:
            raise ValueError("Input text contains no recognizable tokens.")

        # Pad sequences to ensure consistent input length
        padded_input = pad_sequences(encoded_input, maxlen=self.max_len, padding='post', truncating='post')

        return padded_input
