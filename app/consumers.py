from asgiref.sync import async_to_sync
from channels.generic.websocket import WebsocketConsumer
import json
import sys
import uuid
from . import translate
import base64
from datetime import datetime
import cv2
from contextlib import closing

class AslConsumer(WebsocketConsumer):

    def connect(self):
        self.client_name = 'client_%s' % uuid.uuid4()
        self.translator = translate.Translator(self)
        async_to_sync(self.channel_layer.group_add)(self.client_name,self.channel_name)
        self.accept()

    def disconnect(self, close_code):
        async_to_sync(self.channel_layer.group_discard)(self.client_name,self.channel_name)

    def receive(self, text_data):
        print('receiving video @ ' + str(datetime.now().time()))

        # decode video data from base64
        # remove leading padding header (data:video/mp4;base64,)
        # TODO: find a safer way to strip header
        decoded_string = base64.b64decode(text_data[22:])

        # write video data to mp4 file
        f = open('test.mp4','wb')
        f.write(decoded_string)
        f.close()

        vidcap = cv2.VideoCapture('test.mp4')
        success, image = vidcap.read()
        count = 0
        success = True
        while success:
          cv2.imwrite("frames/frame_%d.jpg" % count, image)     # save frame as JPEG file
          success, image = vidcap.read()
          count += 1

        print('stored file @ ' + str(datetime.now().time()))

        # print data for debugging
        # print("\n\n\n"+text_data+"\n\n\n");
        #self.translator.process(json.loads(text_data)['input'])
