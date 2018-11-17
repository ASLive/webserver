from django.conf.urls import url

urlpatterns = [
    url(r'^(?P<room_name>[^/]+)/$',lambda:None),
]
