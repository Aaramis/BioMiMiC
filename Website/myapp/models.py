from django.db import models

# Create your models here.

class TodoItem(models.Model):
    title = models.CharField(max_length=200)
    completed = models.BooleanField(default=False)


class Biomimic(models.Model):
    Biomimic_image = models.ImageField(null=True, blank=True, upload_to="images/")
