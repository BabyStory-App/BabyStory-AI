import tensorflow as tf
import numpy as np
import librosa
import io
import csv
import json
import warnings

warnings.filterwarnings(action='ignore')


class YamNet:
    def __init__(self, model_path, csv_path):
        self.model_path = model_path
        self.csv_path = csv_path

        self.interpreter = tf.lite.Interpreter(model_path)
        self.waveform_input_index = None
        self.scores_output_index = None
        self.embeddings_output_index = None
        self.spectrogram_output_index = None
        self.class_names = self.get_class_names()
        self.set_interpreter()

    def summary(self):
        return tf.lite.experimental.Analyzer.analyze(self.model_path)

    def __call__(self, waveform) -> bool:
        prediction = self.get_prediction(waveform, top_n=5)
        return self.is_baby_cry(prediction)

    def set_interpreter(self):
        input_details = self.interpreter.get_input_details()
        output_details = self.interpreter.get_output_details()

        print(input_details)
        print(output_details)

        self.waveform_input_index = input_details[0]['index']
        self.scores_output_index = output_details[0]['index']
        self.embeddings_output_index = output_details[1]['index']
        self.spectrogram_output_index = output_details[2]['index']

    def get_class_names(self):
        csv_text = open(self.csv_path).read()
        class_map_csv = io.StringIO(csv_text)
        class_names = [display_name for (
            class_index, mid, display_name) in csv.reader(class_map_csv)]
        class_names = class_names[1:]  # Skip CSV header
        return class_names

    def get_prediction(self, waveform, top_n=None, with_score=False):
        if top_n != None and top_n >= len(self.class_names):
            raise ValueError('top_n is bigger than classes length')

        self.interpreter.resize_tensor_input(
            self.waveform_input_index, [len(waveform)], strict=True)
        self.interpreter.allocate_tensors()
        self.interpreter.set_tensor(self.waveform_input_index, waveform)
        self.interpreter.invoke()
        scores = (
            self.interpreter.get_tensor(self.scores_output_index),
            self.interpreter.get_tensor(self.embeddings_output_index),
            self.interpreter.get_tensor(self.spectrogram_output_index))[0]
        # print(scores.shape, embeddings.shape, spectrogram.shape)

        index_sort_list = scores.mean(axis=0).argsort()[::-1]

        if top_n != None:
            index_sort_list = index_sort_list[:top_n]

        if with_score == False:
            return [self.class_names[index] for index in index_sort_list]
        else:
            return {self.class_names[index]: scores.mean(axis=0)[index] for index in index_sort_list}

    def predict(self, waveform, top_n=None):
        return self.get_prediction(waveform, top_n=top_n, with_score=True)

    def is_baby_cry(self, prediction) -> bool:
        return any([target in prediction for target in ['Crying, sobbing', 'Baby cry, infant cry']])


if __name__ == '__main__':
    import os

    cur_path = os.getcwd()
    csv_path = os.path.join(cur_path, 'yamnet_class_map.csv')
    model_path = os.path.join(cur_path, 'yamnet.tflite')

    # 모델을 인스턴스 생성
    yamNet = YamNet(model_path, csv_path)

    # 음성을 불러온다.
    waveform = librosa.load(
        '/Users/jaewone/Downloads/model_test/data/baby_cry.wav', sr=16000)[0]

    # 예측을 수행한 뒤 출력한다.
    prediction = yamNet.predict(waveform, top_n=5)
    print(json.dumps({k: round(float(v), 2)
          for k, v in prediction.items()}, indent=4))
    print(f'Is baby cry: {yamNet.is_baby_cry(list(prediction.keys()))}')
