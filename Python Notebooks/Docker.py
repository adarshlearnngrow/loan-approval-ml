import mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Creating the docker
from mlflow.models import build_docker
build_docker(model_uri="models:/Credit Risk Model/2", name="credit-risk-model")