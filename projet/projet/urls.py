from django.contrib import admin
from django.urls import path
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', views.View.home_view),
    path('appliquer_options/', views.View.appliquer_options),
    path('statistiques/', views.View.statistiques),
    path('generate/', views.View.generate),
]
