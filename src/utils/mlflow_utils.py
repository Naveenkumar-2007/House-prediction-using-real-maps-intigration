import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from datetime import datetime

class MLflowTracker:
    def __init__(self, experiment_name="indian_housing_prediction"):
        """Initialize MLflow tracker"""
        # Set tracking URI (local or remote)
        mlflow.set_tracking_uri("file:./mlruns")
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        self.client = MlflowClient()
        self.run_id = None
    
    def start_run(self, run_name=None):
        """Start MLflow run"""
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run = mlflow.start_run(run_name=run_name)
        self.run_id = self.run.info.run_id
        return self.run
    
    def log_params(self, params):
        """Log parameters"""
        mlflow.log_params(params)
    
    def log_metrics(self, metrics):
        """Log metrics"""
        mlflow.log_metrics(metrics)
    
    def log_model(self, model, artifact_path="model"):
        """Log model"""
        mlflow.sklearn.log_model(model, artifact_path)
    
    def log_artifact(self, local_path, artifact_path=None):
        """Log artifact"""
        mlflow.log_artifact(local_path, artifact_path)
    
    def log_data_info(self, train_shape, test_shape):
        """Log dataset information"""
        mlflow.log_params({
            "train_samples": train_shape[0],
            "train_features": train_shape[1],
            "test_samples": test_shape[0],
            "test_features": test_shape[1]
        })
    
    def end_run(self):
        """End MLflow run"""
        mlflow.end_run()
    
    def set_tags(self, tags):
        """Set tags for the run"""
        mlflow.set_tags(tags)