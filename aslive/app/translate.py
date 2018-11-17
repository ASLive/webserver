import json

class Translator():

    raw_data_cache = []
    asl_word_cache = []
    english_cache = []

    def __init__(self,consumer):
        self.client = consumer

    def process(self,data):
        self.raw_data_cache.append(data)
        self.detect_asl_word()

    def detect_asl_word(self):
        last_word = self.raw_data_cache[-1]
        if "asl" in last_word:
            self.client.send(text_data=json.dumps(
                {'translation': last_word}
            ))
