import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dataclasses import dataclass
import joblib,mlflow, json
from src.utils.mlflow_utils import MLflowTracker
from datetime import datetime

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifects', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        self.mlflow_tracker=MLflowTracker(experiment_name="indian_housing_prediction")


    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        """
        Evaluate multiple models and return performance metrics
        """
        try:
            report = {}

            for model_name, model in models.items():
                print(f"Training {model_name}...")

                #creating mlflow with nested loop
                with mlflow.start_run(nested=True,run_name=f"{model_name}_evalutions"):
                
                    # Train model
                    model.fit(X_train, y_train)

                    # Predictions
                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    # Metrics
                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)
                    test_mae = mean_absolute_error(y_test, y_test_pred)
                    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
                    
                    # MAPE (Mean Absolute Percentage Error)
                    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100

                    report[model_name] = {
                        'train_r2': train_r2,
                        'test_r2': test_r2,
                        'test_mae': test_mae,
                        'test_rmse': test_rmse,
                        'test_mape': test_mape
                    }
                    
                    #hyperparamters tracking
                    if hasattr(model,"get_params"):
                        mlflow.log_params({
                            f"{model_name}_param_{k}":v
                            for k,v in model.get_params().items()
                        })

                    #log all metrics of mlflow
                    mlflow.log_metrics({
                        f"{model_name}_train_r2":train_r2,
                        f"{model_name}_test_r2":test_r2,
                        f"{model_name}_test_mae": test_mae,
                        f"{model_name}_test_rmse": test_rmse,
                        f"{model_name}_test_mape": test_mape
                    })

                    # Log the trained model artifact for this nested run
                    # Use mlflow.sklearn.log_model (works for scikit-learn models)
                    try:
                        mlflow.sklearn.log_model(model, artifact_path=f"models/{model_name}")
                    except Exception:
                        # Fallback to generic artifact logging if model cannot be serialized by mlflow.sklearn
                        temp_path = os.path.join("artifects", f"{model_name.replace(' ', '_')}_model.pkl")
                        joblib.dump(model, temp_path)
                        mlflow.log_artifact(temp_path, artifact_path=f"models/{model_name}")

                    print(f"   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
                    print(f"   MAE: ₹{test_mae:.2f}L | RMSE: ₹{test_rmse:.2f}L | MAPE: {test_mape:.2f}%")

            return report

        except Exception as e:
            print(f" Error in model evaluation: {str(e)}")
            raise e

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train and save the best model for Indian housing data
        """
        try:
            # Start main MLflow run with timestamped name
            run_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.mlflow_tracker.start_run(run_name=run_name)
            
            # Set tags for this run - helps filter/search runs in MLflow UI
            self.mlflow_tracker.set_tags({
                "project": "Indian Housing Price Prediction",
                "model_type": "regression",
                "dataset": "indian_housing",
                "version": "1.0",
                "author": "Data Science Team",
                "stage": "training"
            })
            
            print("Splitting training and test input data...")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            print(f"   X_train shape: {X_train.shape}")
            print(f"   X_test shape: {X_test.shape}")

            # Log dataset information to MLflow
            # This helps track which data was used for training
            self.mlflow_tracker.log_params({
                "train_samples": X_train.shape[0],
                "train_features": X_train.shape[1],
                "test_samples": X_test.shape[0],
                "test_features": X_test.shape[1],
                "total_samples": X_train.shape[0] + X_test.shape[0]
            })

            print("Training multiple regression models...\n")
            
            models = {
                "Random Forest": RandomForestRegressor(
                    n_estimators=200,
                    max_depth=20,
                    min_samples_split=5,
                    min_samples_leaf=2,
                    random_state=42,
                    n_jobs=-1
                ),
                "Gradient Boosting": GradientBoostingRegressor(
                    n_estimators=200,
                    learning_rate=0.1,
                    max_depth=7,
                    random_state=42
                ),
                "Extra Trees": ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=20,
                    random_state=42,
                    n_jobs=-1
                ),
                "Linear Regression": LinearRegression(),
                "Ridge Regression": Ridge(alpha=1.0),
                "Lasso Regression": Lasso(alpha=1.0),
                "Decision Tree": DecisionTreeRegressor(
                    max_depth=20,
                    min_samples_split=5,
                    random_state=42
                )
            }

            model_report = self.evaluate_models(X_train, y_train, X_test, y_test, models)

            # Get best model based on test R2 score
            best_model_name = max(model_report, key=lambda x: model_report[x]['test_r2'])
            best_model_score = model_report[best_model_name]['test_r2']
            best_model_mae = model_report[best_model_name]['test_mae']
            best_model_mape = model_report[best_model_name]['test_mape']
            best_model_rmse = model_report[best_model_name]['test_rmse']

            print(f"\n{'='*60}")
            print(f"BEST MODEL: {best_model_name}")
            print(f"{'='*60}")
            print(f"   R² Score: {best_model_score:.4f}")
            print(f"   MAE: ₹{best_model_mae:.2f} Lakhs")
            print(f"   MAPE: {best_model_mape:.2f}%")
            print(f"{'='*60}\n")

            if best_model_score < 0.6:
                print("Warning: Best model R² score is below 0.6")

            best_model = models[best_model_name]

            # Log best model information to main MLflow run
            self.mlflow_tracker.log_params({
                "best_model_name": best_model_name,
                "best_model_type": type(best_model).__name__
            })
            
            # Log best model's hyperparameters
            if hasattr(best_model, 'get_params'):
                best_params = {f"best_{k}": v for k, v in best_model.get_params().items()}
                self.mlflow_tracker.log_params(best_params)
            
            # Log best model metrics to main run
            self.mlflow_tracker.log_metrics({
                "best_r2_score": best_model_score,
                "best_mae": best_model_mae,
                "best_rmse": best_model_rmse,
                "best_mape": best_model_mape,
                "train_test_r2_diff": model_report[best_model_name]['train_r2'] - best_model_score
            })

            # Save the best model locally
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)

            print(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")

            # Log the best model to MLflow (main run)
            try:
                mlflow.sklearn.log_model(best_model, artifact_path="best_model")
            except Exception:
                # Fallback: log pickled model file as an artifact
                mlflow.log_artifact(self.model_trainer_config.trained_model_file_path, artifact_path="best_model")

            # Log metrics JSON as artifact
            metrics = {
                "best_model_name": best_model_name,
                "r2_score": float(best_model_score),
                "mae": float(best_model_mae),
                "mape": float(best_model_mape)
            }
            metrics_path = os.path.join("artifects", "metrics.json")
            os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)

            mlflow.log_artifact(metrics_path, artifact_path="metrics")

            print("Model training completed successfully!\n")

            return best_model_score

        except Exception as e:
            print(f"Error in model training: {str(e)}")
            raise e

if __name__ == "__main__":
    from src.components.data_ingestion import DATA_INGESTION
    from src.components.data_transformation import data_transformation
    
    # Ingest data
    print("STEP 1: DATA INGESTION")
    print("-"*60)
    obj =DATA_INGESTION()
    train_data,test_data=obj.data_intiation()

    # Transform data
    print("\nSTEP 2: DATA TRANSFORMATION")
    print("-"*60)
    data_transformation = data_transformation()
    train_arr, test_arr, _ = data_transformation.data_intiate_trans(train_data, test_data)

    # Train model
    print("\nSTEP 3: MODEL TRAINING")
    print("-"*60)
    model_trainer = ModelTrainer()
    r2_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
    
    print(f"Pipeline completed! Final R² Score: {r2_score:.4f}")