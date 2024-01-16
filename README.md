# mushroom-genus-classification
Machine learning project to classify mushroom images into 9 distinct genus classes.

# Task
With over 14000 mushroom species on earth from varying biological families/Genuses, classifying mushrooms is not an easy feat, especially when it depends on so many factors including shape, size, texture and whether they produce mushroom fruit bodies or not. You can read more about classifying mushrooms here - https://en.wikipedia.org/wiki/Mushroom#Classification
For our particular task, I set out to classify the common mushroom genus' using the image dataset available on Kaggle - https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images

# Data
The dataset contains 9 common mushroom genus' images with 300-1500 images for each genus. While the final model is trained on the specific images from the dataset curated from Northern Europe, it can be translated into classifying Australian mushroom images as some of the common genus' are available here as well.

# Modeling
As for the modeling part, I utilized #transferlearning leveraging state-of-the-art architectures available from Keras (https://keras.io/).
- The convolutional layers and their corresponding weights are similar to the ImageNet (https://image-net.org/), the most common and benchmark dataset for common objects.
- The dense layers have been designed specifically to classify mushroom images into 9 distinct classes - 'Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula' and 'Suillus'.
- Initial model training and experimentation was done using smaller images (150x150), followed by the larger model training using 299x299 images.
- The architectures I used for the task include Xception (66% accuracy), ResNet50v2 (64% accuracy), InceptionV3 (64% accuracy) and finally EfficientNetV2 (87% accuracy).
- The final model (which I also published on Kaggle for those of you curious) performed quite well on test dataset with an accuracy of 85%.
