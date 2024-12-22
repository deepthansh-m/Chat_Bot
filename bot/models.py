from django.db import models

class ChatMessage(models.Model):
    """
    Model to store user and bot conversation messages.
    """
    user_input = models.TextField()
    bot_response = models.TextField()
    intent = models.CharField(max_length=100)
    confidence_score = models.FloatField()
    conversation_id = models.CharField(max_length=100)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Conversation {self.conversation_id}: {self.user_input[:50]}"

class TrainedModel(models.Model):
    """
    Model to store information about trained chatbot models.
    """
    name = models.CharField(max_length=100)
    version = models.CharField(max_length=20)
    accuracy = models.FloatField()
    loss = models.FloatField()
    model_file = models.FileField(upload_to='models/')
    model_path = models.CharField(max_length=255)
    trained_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.name} (v{self.version})"
