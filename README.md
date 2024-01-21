# Mushroom Genus Classification
Machine learning project to classify mushroom images into 9 distinct genus classes.

![alt hello](https://github.com/Shubh18s/mushroom-genus-classification/blob/main/images/mushroom_classification_image_bing_generated.jpg)


# Task
With over 14000 mushroom species on earth from varying biological families/Genuses, classifying mushrooms is not an easy feat, especially when it depends on so many factors including shape, size, texture and whether they produce mushroom fruit bodies or not. You can read more about classifying mushrooms here - https://en.wikipedia.org/wiki/Mushroom#Classification

For our particular task, I set out to classify the common mushroom genus' using the image dataset available on Kaggle - https://www.kaggle.com/datasets/maysee/mushrooms-classification-common-genuss-images

# Data
The dataset contains 9 common mushroom genus' images with 300-1500 images for each genus. While the final model is trained on the specific images from the dataset curated from Northern Europe, it can be translated into classifying Australian mushroom images as some of the common genus' are available here as well.

![alt text](https://github.com/Shubh18s/mushroom-genus-classification/blob/main/images/mushroom_genus_distribution.png)

# Project Structure

The files in the project and roughly defined in below 3 segments

### Model creation and tf-lite serving
- [mushroom-classification.ipynb](mushroom-classification.ipynb) - Mushroom classification training notebook
- [mushroom-classification-model.h5](mushroom-classification-model.h5) - Final model with 88% accuracy
- [EfficientNetV2B0_v3_26_0.885.h5](EfficientNetV2B0_v3_26_0.885.h5) - Final model with 88% accuracy
- [mushroom-classification-tf-lite.ipynb](mushroom-classification-tf-lite.ipynb) - TF-Lite notebook with lightweight packages for prediction
- [mushroom-model.tflite](mushroom-model.tflite) - TF-Lite model

### Model and gateway deployment
- [gateway.py](gateway.py) - Flask app for gateway service
- [image-gateway.dockerfile](image-gateway.dockerfile) - Dockerfile for gateway container
- [image-model.dockerfile](image-model.dockerfile)- Dockerfile for model container
- [mushroom-model](mushroom-model) - saved model directory for mushroom classification
- [proto.py](proto.py) - Helper file for tensor to proto conversion
- [cloudbuild.yaml](cloudbuild.yaml) - Cloud build configuration file
- [kube-config-local](kube-config-local) - Deployment manifests for local kubernetes deployment
- [kube-config-gke](kube-config-gke) - Deployment manifests for kubernetes deployment to Google Kubernetes Engine
- [docker-compose.yaml](docker-compose.yaml) - Docker compose file for local run of gateway and model service

### Other files/directories
- [serverless](serverless) - directory for serverless deployment not currently used.
- [images](images) - image directory
- [Pipfile](Pipfile) - Python dependency file
- [test.py](test.py) - File to test local/cloud deployment

# Modeling
As for the modeling part, I utilized #transferlearning leveraging state-of-the-art architectures available from Keras (https://keras.io/).
- The convolutional layers and their corresponding weights are similar to the ImageNet (https://image-net.org/), the most common and benchmark dataset for common objects.
- The dense layers have been designed specifically to classify mushroom images into 9 distinct classes - 'Agaricus', 'Amanita', 'Boletus', 'Cortinarius', 'Entoloma', 'Hygrocybe', 'Lactarius', 'Russula' and 'Suillus'.
- Initial model training and experimentation was done using smaller images (150x150), followed by the larger model training using 299x299 images.
- The architectures I used for the task include Xception (66% accuracy), ResNet50v2 (64% accuracy), InceptionV3 (64% accuracy) and finally EfficientNetV2 (88% accuracy).
- The final model was the EfficientNetV2 with the highest accuracy (88%) on test dataset and was used for deployment.


# Deployment - Tensorflow Serving with Kubernetes

## Architecture

![alt text](https://github.com/Shubh18s/mushroom-genus-classification/blob/main/images/mushroom_classification_architecture.jpg)

Kubernetes cluster with 2 pods - one for gateway deployment and the other for TF-serving model deployment was used for our usecase. The 

gRPC and Protobuf and TF-serving

## Local

1. Install Kubectl and Kind
2. Create a new default cluster -
    `kind create cluster`
3. Load the model and gateway images to cluster -
    `kind load docker-image mushroom-classification-model:efficientnet-v3-001`
    `kind load docker-image mushroom-classification-gateway:001`
4. Create deployment and Service -
    `kubectl apply -f kube-config-local`

## Cloud

Cloud deployment makes use of Google Cloud services including - [Google Kubernetes Engine (GKE)](https://cloud.google.com/kubernetes-engine/?utm_source=bing&utm_medium=cpc&utm_campaign=japac-AU-all-en-dr-BKWS-all-super-trial-PHR-dr-1605216&utm_content=text-ad-none-none-DEV_c-CRE_-ADGP_Hybrid+%7C+BKWS+-+PHR+%7C+Txt+~+Containers_Kubernetes+Engine_google+kubernetes_main-KWID_43700079238685177-kwd-71606489890140:loc-9&userloc_122876-network_o&utm_term=KW_google+kubernetes+engine&gclsrc=3p.ds&&gclid=d2a059b180b9139664243c1a5309bd4f&gclsrc=3p.ds&&hl=en), [Google Cloud Build](https://cloud.google.com/build?hl=en), [Artifact Registry](https://cloud.google.com/artifact-registry/) and [Cloud Storage](https://cloud.google.com/storage/?hl=en).

1. Create a service account with below roles
    - Artifact Registry Admin
    - Cloud Build Editor
    - Create Service Accounts
    - Kubernetes Engine Admin
    - Service Account User
    - Service Usage Admin
    - Storage Admin

2. Install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install#deb) and [GKE GCloud Auth Plugin](https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke)


### Build and Push Model and Gateway image to Artifact Registry

1. Get you project id using - `gcloud config get-value project`

2. Create a new repository in Artifact Registry. Replace location and project parameters  - 
`gcloud artifacts repositories create mushroom-classification-repo --project={PROJECT_ID} --repository-format=docker --location={REGION} --description="Docker repository"`

3. Build and push images to Artifact Registry -
    Replace Line 9 and 10 in [cloudbuild.yaml](cloudbuild.yaml) with your `YOUR_REPOSITORY_REGION`, `YOUR_PROJECT_ID` and run `gcloud builds submit --config=cloudbuild.yaml .`

![alt text](https://github.com/Shubh18s/mushroom-genus-classification/blob/main/images/artifact_registry_sc.png)

### Deploying to Google Kubernetes Engine
1. Create Cluster 
`gcloud container clusters create-auto mushroom-classification-gke --location {REGION}`

2. Replace `YOUR_REPOSITORY_REGION` and `YOUR_PROJECT_ID` project parameters in [gateway-deployment.yaml](kube-config-gke/gateway-deployment.yaml) and [model-deployment.yaml](kube-config-gke/model-deployment.yaml)

3. Run - `kubectl apply -f kube-config-gke`

4. To check current deployments use - 
`kubectl get deployments`

Currently the service is deployed at [http://34.173.137.46:80/predict](http://34.173.137.46:80/predict). Make sure to send a POST request with JSON object containing `url` as below. Refer to postman screenshot below -

```
{"url":"<REPLACE_WITH_IMAGE_LINK_FROM_KAGGLE_REPO_OR_SEND_ANOTHER_MUSHROOM_IMAGE>"}
```

![alt text](https://github.com/Shubh18s/mushroom-genus-classification/blob/main/images/gke_deployments.png)

![alt text](https://github.com/Shubh18s/mushroom-genus-classification/blob/main/images/postman_sc.png)

<!-- ![alt text](https://github.com/Shubh18s/mushroom-genus-classification/blob/main/images/gke_deployment_test.png) -->

# Developer

### Shubhdeep Singh (singh18shubhdeep@gmail.com)