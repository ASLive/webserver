from django.conf.urls import url

from . import consumers

websocket_urlpatterns = [
    url(r'^ws/app/(?P<room_name>[^/]+)/$', consumers.AslConsumer),
]