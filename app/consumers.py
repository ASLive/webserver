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
        # TODO: use logging
        print('receiving video @ ' + str(datetime.now().time()))

        # decode video data from base64
        # remove leading padding header (data:video/mp4;base64,)
        # TODO: find a safer way to strip header
        decoded_string = base64.b64decode(text_data[22:])

        # write video data to mp4 file
        video_file_path = 'test.mp4'
        f = open(video_file_path,'wb')
        f.write(decoded_string)
        f.close()

        # get video frames
        video_frames = get_frames(video_file_path)

        # process frame with ML
        # TODO: process every frame, or every i-th frame
        video_frames = [video_frames[0]]
        self.translator.process(video_frames)

        print('processed request @ ' + str(datetime.now().time()))

    # TODO: get frames without saving video file for speed
    def get_frames(video_file):
        frames = []
        vidcap = cv2.VideoCapture(video_file)
        success = True
        while success:
          # cv2.imwrite("frames/frame_%d.jpg" % count, image)     # save frame as JPEG file
          success, image = vidcap.read()
          if success:
              frames.append(image)

        return frames
