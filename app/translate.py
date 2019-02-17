import json
from app.ml import read_model, read_classes

class Translator():

    raw_data_cache = []
    asl_word_cache = []
    english_cache = []
    model = read_model()
    classes = read_classes()

    def __init__(self,consumer):
        self.client = consumer

    def process(self,video_frames):
        self.detect_asl_letter(video_frames)

    def detect_asl_letter(self,video_frames):
        predictions = model.predict(video_frames)
        for eval in predictions:
            predicted_label = np.argmax(eval)
            predicted_letter = classes[predicted_label]
            # TODO: have threshold of confidence
            # to not spam client every frame
            self.client.send(text_data=json.dumps(
                {'translation': predicted_letter}
            ))
