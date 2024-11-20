from django.contrib import admin
from django.urls import path

from .views import index, configure_parameters, dashboard

urlpatterns = [
    path('', index, name="index"),
    path('configure-parameters/', configure_parameters, name="configure_parameters"),
    path('dashboard/', dashboard, name="dashboard"),
]
