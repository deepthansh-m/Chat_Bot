from django.contrib import admin
from .models import ChatMessage, TrainedModel

admin.site.register(ChatMessage)
admin.site.register(TrainedModel)
