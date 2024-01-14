FROM public.ecr.aws/lambda/python:3.9

RUN pip install keras_image_helper
RUN pip install tflite_runtime

COPY mushroom-model.tflite .
COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]