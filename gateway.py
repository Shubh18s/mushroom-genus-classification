import os
import grpc
from PIL import Image
import numpy as np

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from keras_image_helper.base import download_image

from flask import Flask
from flask import request
from flask import jsonify

from proto import np_to_protobuf

# host = 'localhost:8500'
host = os.getenv('TF_SERVING_HOST', '0.0.0.0:8500')
channel = grpc.insecure_channel(host)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)


def resize_image(img, target_size=(299,299)):
    return img.resize(target_size, Image.Resampling.NEAREST)

def image_to_array(img):
    # one image loaded
    x = np.array(img)
    # capital case (batch of images)
    X = np.array([x], dtype=np.float32)
    return X

def load_image(path):
    with Image.open(path) as img:
        img = img.resize((299,299), Image.Resampling.NEAREST)
        X = image_to_array(img)
        return X

def load_image_from_url(url):
    img = download_image(url)
    img = resize_image(img, (299, 299))
    X = image_to_array(img)
    return X


def prepare_request(X):
    pb_request = predict_pb2.PredictRequest()
    pb_request.model_spec.name = 'mushroom-model'
    pb_request.model_spec.signature_name = 'serving_default'
    pb_request.inputs['input_2'].CopyFrom(np_to_protobuf(X))

    return pb_request


classes = ['Agaricus',
 'Amanita',
 'Boletus',
 'Cortinarius',
 'Entoloma',
 'Hygrocybe',
 'Lactarius',
 'Russula',
 'Suillus']


def prepare_response(pb_response):
    preds = pb_response.outputs['dense_3'].float_val
    return dict(zip(classes, preds))


def predict(url):
    X = load_image_from_url(url)
    pb_request = prepare_request(X)
    pb_response = stub.Predict(pb_request, timeout=20.0)
    response = prepare_response(pb_response)

    return response


app = Flask('mushroom classification')

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    data = request.get_json()
    url = data['url']
    result = predict(url)
    return jsonify(result)


if __name__ == '__main__':
    # url = "https://storage.googleapis.com/kagglesdsdata/datasets/130737/312053/Mushrooms/Agaricus/000_ePQknW8cTp8.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20231227%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231227T045019Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=052703779ba0667a05b223fbd60d05639587e61f710f1a9778334d0c865d5a4d01d453e327f698dbb1b50f9df39b74de3c31aae869a281ababfa094c2f0a7226304aa4bcd450c50f8c59d5adcde7168c9bcda63d11b4517abc765abbc4ec2d3b4ba1d219065f4921b57b32cd95ce733fd0d27670793ef95e2591de8308df712a6d385da8c995c2e7aa353c7c1a78f4eb5a995865ae2e1fb5efa46f10dcd56abd5295dd22c344c58ee052cb775a7fa7d7319b9b2797eede84f201a9e68b12208e74a0c0cc94f3af737f0900e5301859e205c2b1cc4f4f855a5e4c6744a70428555a64384ed77c69b90422901c1911cdafc90937dccf97941cfdfd6ec74b45f987"
    url = "https://storage.googleapis.com/kagglesdsdata/datasets/130737/312053/Mushrooms/Agaricus/001_2jP9N_ipAo8.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20240109%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20240109T123315Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=c313106f5c2796100994472df76254aba79e47db3ae27247cc0daac3323229a61a73ec8c6f6c3de29873f65ccf3fe643189c5bbbedbf40e40ecb234088acb80fd1618b5a4ce409280140cffe7d99b75869edf1e91a64e223f6e4ee8a83d293b591ff01fbcb8638885200384caacab9ea2de093b691fa12eb4c13bd22182ba81a60d3ca6bbfe3417e774bc8e8dbd063c9b18967536aa594d7c5af68b4aa9c768092ca28da6b35eff936198df19c07f5c356b9b238e10a0fa0d47232451dfc6a4ef3a5a90d39bc497e25357f580ba2765e9edef7d72d8d870b883a8d97a1bc0e25bb1ad5c7d134fb2d2978cecefeea6e1dbe97eddd9e349f279f6259a14d50c1db"
    response = predict(url)
    print(response)
    # app.run(debug=True, host='0.0.0.0', port=9696)

