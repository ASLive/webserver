import json

class Translator():

    raw_data_cache = []
    asl_word_cache = []
    english_cache = []

    def __init__(self,consumer):
        self.consumer = consumer

    def process(self,data):
        self.raw_data_cache.append(data)
        self.detect_asl_word()

    def detect_asl_word(self):
        # detect asl words/gestures
        detected = "word"
        self.consumer.send(text_data=json.dumps({'translation': detected}))
