# clasificacion/urls.py
from django.urls import path
from . import views

urlpatterns = [
      path('predictimage/', views.predict_image, name='predict_image'), 
      path('', views.predict_image, name='predict_image'),
]
