"""
URL configuration for my_mediapipe project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
# from django.contrib import admin
from django.urls import path
from main import views

urlpatterns = [
    #    path('admin/', admin.site.urls),
    path("check_frame/", views.check_frame, name="check_frame"),
    path("upload_passport_and_verify/", views.upload_passport_and_verify, name="upload_passport_and_verify"),
    path("", views.liveness_test, name="check_1frame"),
]
