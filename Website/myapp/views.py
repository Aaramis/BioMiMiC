from django.shortcuts import render, HttpResponse
from .models import TodoItem, Biomimic


# def todos_2(request):
#     items = TodoItem.objects.all()
#     return render(request, "todo.html", {"todos": items})

def landing(request):
    return render(request, "landing.html")