FROM tensorflow/serving:2.7.0

COPY mushroom-model /models/mushroom-model/1

ENV MODEL_NAME="mushroom-model"