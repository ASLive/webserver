from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
import json
from . import translate

class AslConsumer(WebsocketConsumer):

    def connect(self):
        self.room_name = "" #self.scope['url_route']['kwargs']['room_name']
        self.room_group_name = 'client_%s' % self.room_name
        self.translator = translate.Translator(self)

        # Join room group
        async_to_sync(self.channel_layer.group_add)(
            self.room_group_name,
            self.channel_name
        )

        self.accept()

    def disconnect(self, close_code):
        async_to_sync(self.channel_layer.group_discard)(
            self.room_group_name,
            self.channel_name
        )

    def receive(self, text_data):
        self.translator.process(json.loads(text_data)['input'])
