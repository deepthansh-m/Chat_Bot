from rest_framework.views import APIView
from rest_framework.response import Response
from django.conf import settings
from .models import ChatMessage, TrainedModel
from .serializers import ChatMessageSerializer, TrainedModelSerializer
from .utils.preprocessor import Preprocessor
from .utils.model import ChatbotModel
from .utils.trainer import Trainer  # Ensure Trainer is properly imported
import os
import logging
import numpy as np
import random

logger = logging.getLogger(__name__)  # Setting up logging


class ChatbotView(APIView):
    """
    Handles user input, processes it using the trained model, and saves chat messages to the database.
    """

    def post(self, request):
        user_input = request.data.get('user_input')
        if not user_input:
            logger.warning("No user_input provided")
            return Response({'error': 'No input provided'}, status=400)

        try:
            # Initialize preprocessor and model
            preprocessor = Preprocessor(settings.DATASET_PATH)
            _, label_encoder = preprocessor.load_and_preprocess()

            model = ChatbotModel(vocab_size=5000)
            model.load(os.path.join(settings.MODEL_PATH, 'chatbot_model.keras'))

            # Preprocess and encode the input text
            encoded_input = preprocessor.encode_input(user_input)

            # Log the shape of the input for debugging
            logger.debug(f"Encoded input shape: {np.shape(encoded_input)}")

            # Ensure input is a numpy array and reshape if required
            encoded_input = np.array(encoded_input).reshape(1, -1)

            # Predict the intent
            prediction = model.model.predict(encoded_input)
            predicted_label = label_encoder.inverse_transform([prediction.argmax()])[0]

            bot_response = self.generate_response(predicted_label)  # Generate response based on intent

            # Save the chat message to the database
            chat_message = ChatMessage.objects.create(
                user_input=user_input,
                bot_response=bot_response,
                intent=predicted_label,
                confidence_score=float(prediction.max()),
                conversation_id="12345"  # Replace with a dynamic conversation ID if required
            )
            serializer = ChatMessageSerializer(chat_message)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error in ChatbotView: {e}", exc_info=True)
            return Response({'error': 'An error occurred while processing the request.'}, status=500)

    def generate_response(self, intent):
        return self.get_random_response(intent)

    # Define the response map with multiple responses for each intent
    response_map = {
        "greeting": [
            "Hello! How can I assist you today?",
            "Hi there! What can I do for you?",
            "Greetings! How may I help you?"
        ],
        "farewell": [
            "Goodbye! Have a great day!",
            "See you later! Take care!",
            "Bye! Feel free to return if you have more questions!"
        ],
        "payment_issue": [
            "Please contact our customer support for assistance.",
            "Our support team is here to help with payment issues. Reach out to them!",
            "Facing payment troubles? Customer support is just a message away."
        ],
        "technical_support": [
            "We will get back to you shortly.",
            "Technical support will respond to you soon. Please wait.",
            "Thank you for reaching out. Our technical team will assist shortly."
        ],
        "product_inquiry": [
            "Please contact our customer support for assistance.",
            "Have questions about our products? Support is ready to help.",
            "Our support team can provide detailed information about the product."
        ]
    }

    # Function to randomly select a response for the given intent
    def get_random_response(self, intent):
        if intent in self.response_map:
            return random.choice(self.response_map[intent])
        else:
            return "I'm sorry, I couldn't understand your request. Could you please rephrase?"

    # Example usage
    def chatbot_logic(self, user_input, intent):
        # Get a random response for the detected intent
        bot_response = self.get_random_response(intent)
        return bot_response


class TrainModelView(APIView):
    """
    Handles the training of the chatbot model and saves the trained model information in the database.
    """

    def post(self, request):
        try:
            # Load and preprocess dataset
            preprocessor = Preprocessor(settings.DATASET_PATH)
            dataset, label_encoder = preprocessor.load_and_preprocess()

            # Initialize the model
            model = ChatbotModel(
                vocab_size=5000,
                num_classes=len(label_encoder.classes_)
            )

            # Initialize and run the trainer
            trainer = Trainer(preprocessor, model, dataset, settings.MODEL_PATH)
            training_results = trainer.train_model()

            # Assuming training_results is a dictionary, access accuracy and loss
            final_accuracy = training_results.get('accuracy', [None])[-1]
            final_loss = training_results.get('loss', [None])[-1]

            if final_accuracy is None or final_loss is None:
                raise ValueError("Training results missing accuracy or loss")

            # Save the trained model details in the database
            trained_model = TrainedModel.objects.create(
                name="Chatbot Model",
                version="1.0",
                accuracy=final_accuracy,
                loss=final_loss,
                model_path=os.path.join(settings.MODEL_PATH, 'chatbot_model.keras')  # Save as .keras format
            )
            serializer = TrainedModelSerializer(trained_model)
            return Response(serializer.data)

        except Exception as e:
            logger.error(f"Error in TrainModelView: {e}", exc_info=True)
            return Response({'error': 'An error occurred during model training.'}, status=500)
