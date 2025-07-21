from django.db import models
from django.contrib.auth.models import User


class ChatSession(models.Model):
    user = models.ForeignKey(
        User, on_delete=models.CASCADE, null=True, blank=True,
        related_name='chat_sessions'
    )
    session_key = models.CharField(max_length=255)
    pdf_file = models.CharField(max_length=255)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Session {self.session_key} - {self.user.username if self.user else 'Anonymous'}"


class ChatMessage(models.Model):
    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name='messages')
    sender = models.CharField(max_length=10)  # 'user' or 'bot'
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{'User' if self.is_user else 'Bot'}: {self.message[:30]}"
    

