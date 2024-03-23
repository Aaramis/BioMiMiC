from django.db import models

# Create your models here.

class TodoItem(models.Model):
    title = models.CharField(max_length=200)
    completed = models.BooleanField(default=False)


class Biomimic(models.Model):
    molecule_choice = [
        ("Antioxydant", 'Antioxydant'),
        ('Antibiotique', 'Antibiotique'),
        ('Antifongique', 'Antifongique'),
        ('Analgésique', 'Analgésique')
    ]
    Biomimic_image = models.ImageField(null=True, blank=True, upload_to="images/")
    molecule_fct = models.CharField(choices=molecule_choice, max_length=50, blank=True, null=True)