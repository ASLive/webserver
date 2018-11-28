from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
import json
import uuid
from . import translate

class AslConsumer(WebsocketConsumer):

    def connect(self):
        self.client_name = 'client_%s' % uuid.uuid4()
        self.translator = translate.Translator(self)
        async_to_sync(self.channel_layer.group_add)(self.client_name,self.channel_name)
        self.accept()

    def disconnect(self, close_code):
        async_to_sync(self.channel_layer.group_discard)(self.client_name,self.channel_name)

    def receive(self, text_data):
        self.translator.process(json.loads(text_data)['input'])
