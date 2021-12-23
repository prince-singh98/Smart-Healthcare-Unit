from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path("", views.home),
    path("train_models/", views.train_models),
    path("diabetes_predict/", views.diabetes_predict),
    path("heart_predict/", views.heart_predict),
    path("diabetes_predict/diabetes_result", views.diabetes_result),
    path("heart_predict/heart_result", views.heart_result),
]
