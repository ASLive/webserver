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

        # write video data to mp4 file
        f = open('test.mp4','w')
        f.write(text_data)

        # print data for debugging
        print("\n\n\n"+text_data+"\n\n\n");
        #self.translator.process(json.loads(text_data)['input'])
