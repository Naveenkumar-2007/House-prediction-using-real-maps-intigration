import pandas as pd
import numpy as np
from src.utils.geo_utils import GeoUtils

class KNNImputer:
    def __init__(self, k=5):
        self.k = k
        self.geo_utils = GeoUtils()

    def impute_features(self, target_lat, target_lon, training_data, features_to_impute):
        """
        Impute missing features using KNN based on geographic proximity
        Specifically designed for Indian housing dataset
        
        Args:
            target_lat: Latitude of target location
            target_lon: Longitude of target location
            training_data: DataFrame with training data
            features_to_impute: List of features to impute
        
        Returns:
            Dictionary with imputed feature values
        """
        try:
            # Calculate distances to all training samples
            distances = []
            for idx, row in training_data.iterrows():
                if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                    dist = self.geo_utils.calculate_distance(
                        target_lat, target_lon,
                        row['Latitude'], row['Longitude']
                    )
                    distances.append((idx, dist))

            if not distances:
                print(" No valid geographic data found for KNN imputation")
                return {feature: None for feature in features_to_impute}

            # Sort by distance and get k nearest neighbors
            distances.sort(key=lambda x: x[1])
            k_actual = min(self.k, len(distances))
            nearest_indices = [idx for idx, dist in distances[:k_actual]]

            # Get nearest neighbors
            nearest_neighbors = training_data.loc[nearest_indices]

            # Impute features using median/mode of nearest neighbors
            imputed_values = {}
            
            for feature in features_to_impute:
                if feature in nearest_neighbors.columns:
                    # Use mode for categorical features, median for numerical
                    if nearest_neighbors[feature].dtype == 'object':
                        imputed_values[feature] = nearest_neighbors[feature].mode()[0] if len(nearest_neighbors[feature].mode()) > 0 else None
                    else:
                        imputed_values[feature] = nearest_neighbors[feature].median()
                else:
                    imputed_values[feature] = None

            # Add distance info
            imputed_values['nearest_distance_km'] = distances[0][1]
            imputed_values['neighbors_found'] = k_actual

            return imputed_values

        except Exception as e:
            print(f"Error in KNN imputation: {str(e)}")
            return {feature: None for feature in features_to_impute}

    def get_neighborhood_stats(self, target_lat, target_lon, training_data):
        """
        Get statistics about the neighborhood based on KNN for Indian properties
        """
        try:
            # Calculate distances
            distances = []
            for idx, row in training_data.iterrows():
                if pd.notna(row.get('Latitude')) and pd.notna(row.get('Longitude')):
                    dist = self.geo_utils.calculate_distance(
                        target_lat, target_lon,
                        row['Latitude'], row['Longitude']
                    )
                    distances.append((idx, dist))

            if not distances:
                return None

            # Get nearest neighbors
            distances.sort(key=lambda x: x[1])
            k_actual = min(self.k, len(distances))
            nearest_indices = [idx for idx, dist in distances[:k_actual]]
            nearest_neighbors = training_data.loc[nearest_indices]

            # Calculate statistics
            stats = {
                'avg_price_lakhs': nearest_neighbors['Price_in_Lakhs'].mean(),
                'median_price_lakhs': nearest_neighbors['Price_in_Lakhs'].median(),
                'min_price_lakhs': nearest_neighbors['Price_in_Lakhs'].min(),
                'max_price_lakhs': nearest_neighbors['Price_in_Lakhs'].max(),
                'avg_size_sqft': nearest_neighbors['Size_in_SqFt'].mean(),
                'avg_bhk': nearest_neighbors['BHK'].mean(),
                'avg_price_per_sqft': nearest_neighbors['Price_per_SqFt'].mean(),
                'common_property_type': nearest_neighbors['Property_Type'].mode()[0] if len(nearest_neighbors['Property_Type'].mode()) > 0 else 'Unknown',
                'avg_age': nearest_neighbors['Age_of_Property'].mean() if 'Age_of_Property' in nearest_neighbors.columns else None,
                'nearest_distance_km': distances[0][1],
                'neighbors_count': k_actual,
                'avg_nearby_schools': nearest_neighbors['Nearby_Schools'].mean() if 'Nearby_Schools' in nearest_neighbors.columns else None,
                'avg_nearby_hospitals': nearest_neighbors['Nearby_Hospitals'].mean() if 'Nearby_Hospitals' in nearest_neighbors.columns else None
            }

            return stats

        except Exception as e:
            print(f"Error getting neighborhood stats: {str(e)}")
            return None