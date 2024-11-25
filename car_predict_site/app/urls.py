from django.contrib import admin
from django.urls import path

from .views import index, configure_parameters, dashboard, car_prediction

urlpatterns = [
    path('', index, name="index"),
    path('configure-parameters/', configure_parameters, name="configure_parameters"),
    path('dashboard/', dashboard, name="dashboard"),
    path('prediction/', car_prediction, name="car_prediction"),
]
