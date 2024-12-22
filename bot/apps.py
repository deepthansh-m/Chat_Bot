from django.apps import AppConfig
import nltk

class BotConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'bot'

    def ready(self):
        # Download punkt tokenizer
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab')
