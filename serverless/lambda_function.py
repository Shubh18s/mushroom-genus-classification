
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
from keras_image_helper.base import download_image 


# path = 'mushroom-data-small/Lactarius/401_79lQJ0MGorw.jpg'
# url = "https://storage.googleapis.com/kagglesdsdata/datasets/130737/312053/Mushrooms/Agaricus/000_ePQknW8cTp8.jpg?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20231227%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20231227T045019Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=052703779ba0667a05b223fbd60d05639587e61f710f1a9778334d0c865d5a4d01d453e327f698dbb1b50f9df39b74de3c31aae869a281ababfa094c2f0a7226304aa4bcd450c50f8c59d5adcde7168c9bcda63d11b4517abc765abbc4ec2d3b4ba1d219065f4921b57b32cd95ce733fd0d27670793ef95e2591de8308df712a6d385da8c995c2e7aa353c7c1a78f4eb5a995865ae2e1fb5efa46f10dcd56abd5295dd22c344c58ee052cb775a7fa7d7319b9b2797eede84f201a9e68b12208e74a0c0cc94f3af737f0900e5301859e205c2b1cc4f4f855a5e4c6744a70428555a64384ed77c69b90422901c1911cdafc90937dccf97941cfdfd6ec74b45f987"

classes = ['Agaricus',
 'Amanita',
 'Boletus',
 'Cortinarius',
 'Entoloma',
 'Hygrocybe',
 'Lactarius',
 'Russula',
 'Suillus']

interpreter = tflite.Interpreter(model_path = "mushroom-model.tflite")
# loading weights into memory as well for tflite
interpreter.allocate_tensors()
# finding index for input to model
input_index = interpreter.get_input_details()[0]['index']
# finding index for output to model
output_index = interpreter.get_output_details()[0]['index']


def resize_image(img, target_size=(299,299)):
    return img.resize(target_size, Image.Resampling.NEAREST)

def load_image(path):
    with Image.open(path) as img:
        img = img.resize((299,299), Image.Resampling.NEAREST)
        return img

def load_image_from_url(url):
    img = download_image(url)
    img = resize_image(img, (299, 299))
    return img

def image_to_array(img):
    # one image loaded
    x = np.array(img)
    #capital case (batch of images)
    X = np.array([x], dtype=np.float32)
    return X


def predict(url):
    img = load_image_from_url(url)
    X = image_to_array(img)
    interpreter.set_tensor(input_index, X)

    # Now we initialized the input of the interpreter with X 
    # Now we need to invoke all the CONVOLUTIONS IN THE NEURAL Network
    interpreter.invoke()
    # fetching all results from
    preds = interpreter.get_tensor(output_index)

    float_predictions = preds[0].tolist()
    return dict(zip(classes, float_predictions))


def lambda_handler(event, context):
    url = event['url']
    result = predict(url)
    return result
