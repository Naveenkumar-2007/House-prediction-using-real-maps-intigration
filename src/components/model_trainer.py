import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from dataclasses import dataclass
import joblib

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join('artifects', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def evaluate_models(self, X_train, y_train, X_test, y_test, models):
        """
        Evaluate multiple models and return performance metrics
        """
        try:
            report = {}

            for model_name, model in models.items():
                print(f"Training {model_name}...")
                
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
            print("Splitting training and test input data...")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            
            print(f"   X_train shape: {X_train.shape}")
            print(f"   X_test shape: {X_test.shape}")

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

            # Save the best model
            os.makedirs(os.path.dirname(self.model_trainer_config.trained_model_file_path), exist_ok=True)
            joblib.dump(best_model, self.model_trainer_config.trained_model_file_path)

            print(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")
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