import sys
import pandas as pd
import joblib
import os
import numpy as np
from src.utils.location_predictor import LocationBasedPredictor

class PredictPipeline:
    def __init__(self):
        self.model_path = 'artifects/model.pkl'
        self.preprocessor_path = 'artifects/preprocessing.pkl'
        self.label_encoders_path = 'artifects/labelencoding.pkl'
        self.location_predictor = LocationBasedPredictor()
        
    def predict(self, features_df):
        """
        Make predictions using the trained model for Indian housing data with location enhancement
        """
        try:
            # Load model, preprocessor and label encoders
            print(" Loading model and preprocessors...")
            model = joblib.load(self.model_path)
            preprocessor = joblib.load(self.preprocessor_path)
            label_encoders = joblib.load(self.label_encoders_path)
            
            # Get location information for enhancement
            city = features_df.iloc[0].get('City', 'Mumbai')
            locality = features_df.iloc[0].get('Locality', 'Unknown')
            state = features_df.iloc[0].get('State', 'Maharashtra')
            
            # Get real coordinates if not provided
            if 'Latitude' not in features_df.columns or pd.isna(features_df.iloc[0].get('Latitude')):
                lat, lon = self.location_predictor.get_coordinates(city, locality, state)
                features_df = features_df.copy()
                features_df['Latitude'] = lat
                features_df['Longitude'] = lon
            else:
                lat = features_df.iloc[0]['Latitude']
                lon = features_df.iloc[0]['Longitude']
            
            # Calculate location-based features
            location_features = self.location_predictor.calculate_location_features(
                city, locality, state, lat, lon
            )
            
            # Prepare features for model
            features_prepared = self.prepare_features(features_df, label_encoders)
            
            # Transform features
            data_scaled = preprocessor.transform(features_prepared)
            
            # Get base prediction
            base_predictions = model.predict(data_scaled)
            
            # Enhance prediction with location intelligence
            enhanced_predictions = []
            for i, base_pred in enumerate(base_predictions):
                property_features = {
                    'Size_in_SqFt': features_df.iloc[i].get('Size_in_SqFt', 1000),
                    'BHK': features_df.iloc[i].get('BHK', 2),
                    'Property_Type': features_df.iloc[i].get('Property_Type', 'Apartment')
                }
                
                enhanced_pred = self.location_predictor.enhance_prediction_with_location(
                    base_pred, location_features, property_features
                )
                enhanced_predictions.append(enhanced_pred)
            
            print(f"🎯 Base prediction: ₹{base_predictions[0]:.2f} Lakhs")
            print(f"🌟 Enhanced prediction: ₹{enhanced_predictions[0]:.2f} Lakhs")
            print(f"📍 Location: {locality}, {city}, {state}")
            print(f"🗺️ Coordinates: {lat:.4f}, {lon:.4f}")
            
            return np.array(enhanced_predictions)
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise e
    
    def prepare_features(self, df, label_encoders):
        """
        Prepare features for prediction (encoding, amenities processing)
        """
        try:
            df_copy = df.copy()
            
            # Process Amenities if present
            if 'Amenities' in df_copy.columns:
                df_copy['Amenities_Count'] = df_copy['Amenities'].apply(
                    lambda x: len(str(x).split(',')) if pd.notna(x) else 0
                )
                df_copy['Has_Gym'] = df_copy['Amenities'].str.contains('Gym', case=False, na=False).astype(int)
                df_copy['Has_Pool'] = df_copy['Amenities'].str.contains('Pool', case=False, na=False).astype(int)
                df_copy['Has_Playground'] = df_copy['Amenities'].str.contains('Playground', case=False, na=False).astype(int)
                df_copy['Has_Garden'] = df_copy['Amenities'].str.contains('Garden', case=False, na=False).astype(int)
                df_copy['Has_Clubhouse'] = df_copy['Amenities'].str.contains('Clubhouse', case=False, na=False).astype(int)
                df_copy = df_copy.drop('Amenities', axis=1)
            
            # Encode categorical features
            categorical_columns = [
                'State', 'City', 'Locality', 'Property_Type',
                'Furnished_Status', 'Public_Transport_Accessibility',
                'Parking_Space', 'Security', 'Facing', 'Owner_Type',
                'Availability_Status'
            ]
            
            for col in categorical_columns:
                if col in df_copy.columns and col in label_encoders:
                    le = label_encoders[col]
                    # Handle unknown categories by using the most common class
                    def encode_with_fallback(x):
                        try:
                            if x in le.classes_:
                                return le.transform([x])[0]
                            else:
                                print(f"⚠️ Warning: '{x}' not found in {col} classes. Using most common value.")
                                # Use the first class (index 0) as fallback
                                return 0
                        except Exception as e:
                            print(f"Error encoding {col}: {e}")
                            return 0
                    
                    df_copy[col] = df_copy[col].astype(str).apply(encode_with_fallback)
            
            return df_copy
            
        except Exception as e:
            print(f"Error preparing features: {str(e)}")
            raise e


class CustomData:
    """
    Custom data class for Indian housing features
    """
    def __init__(
        self,
        State: str,
        City: str,
        Locality: str,
        Property_Type: str,
        BHK: int,
        Size_in_SqFt: float,
        Price_per_SqFt: float,
        Year_Built: int,
        Latitude: float,
        Longitude: float,
        Age_of_Property: int,
        Nearby_Schools: int,
        Nearby_Hospitals: int,
        Furnishing_Status: str,
        Public_Transport_Accessibility: str,
        Parking_Space: str,
        Security: str,
        Amenities: str,
        Facing: str,
        Owner_Type: str,
        Availability_Status: str,
        Floor_No: int = 2,  # Default middle floor
        Total_Floors: int = 5  # Default total floors
    ):
        self.State = State
        self.City = City
        self.Locality = Locality
        self.Property_Type = Property_Type
        self.BHK = BHK
        self.Size_in_SqFt = Size_in_SqFt
        self.Price_per_SqFt = Price_per_SqFt
        self.Year_Built = Year_Built
        self.Latitude = Latitude
        self.Longitude = Longitude
        self.Age_of_Property = Age_of_Property
        self.Nearby_Schools = Nearby_Schools
        self.Nearby_Hospitals = Nearby_Hospitals
        self.Furnished_Status = Furnishing_Status  # Note: Changed to Furnished_Status
        self.Public_Transport_Accessibility = Public_Transport_Accessibility
        self.Parking_Space = Parking_Space
        self.Security = Security
        self.Amenities = Amenities
        self.Facing = Facing
        self.Owner_Type = Owner_Type
        self.Availability_Status = Availability_Status
        self.Floor_No = Floor_No
        self.Total_Floors = Total_Floors

    def get_data_as_dataframe(self):
        """
        Convert custom data to pandas DataFrame
        """
        try:
            custom_data_input_dict = {
                "State": [self.State],
                "City": [self.City],
                "Locality": [self.Locality],
                "Property_Type": [self.Property_Type],
                "BHK": [self.BHK],
                "Size_in_SqFt": [self.Size_in_SqFt],
                "Price_per_SqFt": [self.Price_per_SqFt],
                "Year_Built": [self.Year_Built],
                "Latitude": [self.Latitude],
                "Longitude": [self.Longitude],
                "Age_of_Property": [self.Age_of_Property],
                "Nearby_Schools": [self.Nearby_Schools],
                "Nearby_Hospitals": [self.Nearby_Hospitals],
                "Furnished_Status": [self.Furnished_Status],  # Changed column name
                "Public_Transport_Accessibility": [self.Public_Transport_Accessibility],
                "Parking_Space": [self.Parking_Space],
                "Security": [self.Security],
                "Amenities": [self.Amenities],
                "Facing": [self.Facing],
                "Owner_Type": [self.Owner_Type],
                "Availability_Status": [self.Availability_Status],
                "Floor_No": [self.Floor_No],  # Added missing column
                "Total_Floors": [self.Total_Floors]  # Added missing column
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            print(f"Error creating dataframe: {str(e)}")
            raise e