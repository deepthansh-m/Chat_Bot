from django.urls import path
from .views import ChatbotView, TrainModelView

urlpatterns = [
    path('chat/', ChatbotView.as_view(), name='chatbot'),
    path('train/', TrainModelView.as_view(), name='train_model'),
]
