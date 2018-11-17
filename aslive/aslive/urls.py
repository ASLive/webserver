from django.conf.urls import include, url

urlpatterns = [
    url(r'^app/', include('app.urls')),
]
