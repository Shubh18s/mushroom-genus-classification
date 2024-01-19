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


# Deployment


## Local

## Cloud

1. Install [Google Cloud CLI](https://cloud.google.com/sdk/docs/install#deb) and [GKE GCloud Auth Plugin](https://cloud.google.com/blog/products/containers-kubernetes/kubectl-auth-changes-in-gke)

2. Make sure to use the service account with below roles
    - Artifact Registry Admin
    - Cloud Build Editor
    - Create Service Accounts
    - Kubernetes Engine Admin
    - Service Account User
    - Service Usage Admin
    - Storage Admin

### Cloud build Setup

    1. `gcloud config get-value project`

	2. Create a new repository in Artifact Registry - 
    `gcloud artifacts repositories create mushroom-classification-repo --project={PROJECT_ID} --repository-format=docker --location={REGION} --description="Docker repository"`

    3. Build and push images to Artifact Registry - 
    `gcloud builds submit --config=cloudbuild.yaml .`

### Deploying to Google Kubernetes Engine
    1. Create Cluster 
    `gcloud container clusters create-auto mushroom-classification-gke --location {REGION}`

    2. `kubectl apply -f kube-config-gke/model-deployment.yaml`
	3. `kubectl apply -f kube-config-gke/model-service.yaml`
	4. `kubectl apply -f kube-config-gke/gateway-deployment.yaml`
    5. `kubectl apply -f kube-config-gke/gateway-service.yaml`

    6. To check current deployments use - 
    `kubectl get deployments`

    Note - If the Error: ImagePullBackOff error occurs, use below command to create iam-policy-binding to artifact registry repo - 
    gcloud artifacts repositories add-iam-policy-binding mushroom-classification-repo --location={REGION} --member=serviceAccount:self-managed-svc-account@{PROJECT_ID}.iam.gserviceaccount.com --role="roles/artifactregistry.reader"
