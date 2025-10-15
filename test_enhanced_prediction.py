import pandas as pd
import sys
import os

# Test the enhanced location-based prediction
try:
    from src.pipeline.predict_pipeline import PredictPipeline
    
    print("🏠 Testing Enhanced Location-Based Prediction System")
    print("=" * 60)
    
    pipeline = PredictPipeline()
    
    # Test different locations to see varying predictions
    test_locations = [
        {
            'State': ['Maharashtra'],
            'City': ['Mumbai'], 
            'Locality': ['Bandra'],
            'Property_Type': ['Apartment'],
            'BHK': [2],
            'Size_in_SqFt': [1000.0],
            'Price_per_SqFt': [15000.0],
            'Year_Built': [2018],
            'Age_of_Property': [5],
            'Nearby_Schools': [3],
            'Nearby_Hospitals': [2],
            'Furnished_Status': ['Semi-furnished'],
            'Public_Transport_Accessibility': ['High'],
            'Parking_Space': ['Yes'],
            'Security': ['Yes'],
            'Amenities': ['Swimming Pool, Gym'],
            'Facing': ['North'],
            'Owner_Type': ['Owner'],
            'Availability_Status': ['Ready_to_Move'],
            'Floor_No': [5],
            'Total_Floors': [10]
        },
        {
            'State': ['Karnataka'],
            'City': ['Bangalore'], 
            'Locality': ['Koramangala'],
            'Property_Type': ['Apartment'],
            'BHK': [3],
            'Size_in_SqFt': [1200.0],
            'Price_per_SqFt': [12000.0],
            'Year_Built': [2020],
            'Age_of_Property': [3],
            'Nearby_Schools': [4],
            'Nearby_Hospitals': [3],
            'Furnished_Status': ['Furnished'],
            'Public_Transport_Accessibility': ['Medium'],
            'Parking_Space': ['Yes'],
            'Security': ['Yes'],
            'Amenities': ['Clubhouse, Garden'],
            'Facing': ['East'],
            'Owner_Type': ['Builder'],
            'Availability_Status': ['Ready_to_Move'],
            'Floor_No': [3],
            'Total_Floors': [8]
        },
        {
            'State': ['Uttar Pradesh'],
            'City': ['Lucknow'], 
            'Locality': ['Gomti Nagar'],
            'Property_Type': ['Villa'],
            'BHK': [4],
            'Size_in_SqFt': [2000.0],
            'Price_per_SqFt': [8000.0],
            'Year_Built': [2019],
            'Age_of_Property': [4],
            'Nearby_Schools': [2],
            'Nearby_Hospitals': [1],
            'Furnished_Status': ['Unfurnished'],
            'Public_Transport_Accessibility': ['Low'],
            'Parking_Space': ['Yes'],
            'Security': ['No'],
            'Amenities': ['Garden'],
            'Facing': ['South'],
            'Owner_Type': ['Owner'],
            'Availability_Status': ['Ready_to_Move'],
            'Floor_No': [1],
            'Total_Floors': [2]
        }
    ]
    
    for i, test_data in enumerate(test_locations, 1):
        print(f"\n🏘️ Test {i}: {test_data['Locality'][0]}, {test_data['City'][0]}")
        print("-" * 40)
        
        df = pd.DataFrame(test_data)
        predictions = pipeline.predict(df)
        
        print(f"💰 Final Prediction: ₹{predictions[0]:,.2f} Lakhs")
        print()

except Exception as e:
    print(f"❌ Error occurred: {str(e)}")
    import traceback
    traceback.print_exc()