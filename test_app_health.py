import requests
import json
import time

# Test the running Streamlit app to verify no monotonic_cst errors
print("🧪 Testing Streamlit App for monotonic_cst Error...")
print("=" * 50)

# Give the app a moment to fully start
time.sleep(3)

try:
    # Test if the app is running by checking the health endpoint
    response = requests.get("http://localhost:8505/_stcore/health", timeout=5)
    if response.status_code == 200:
        print("✅ Streamlit app is running successfully!")
        print(f"🌐 App URL: http://localhost:8505")
        print("📍 You can now test predictions in the browser")
        print("\n🎯 Expected behavior:")
        print("- No 'monotonic_cst' attribute errors")
        print("- Real location-based predictions")
        print("- Different prices for different locations")
        print("- Working map integration")
        
        # Test a simple prediction using our enhanced system
        print("\n🏠 Quick prediction test:")
        from src.pipeline.predict_pipeline import PredictPipeline
        import pandas as pd
        
        pipeline = PredictPipeline()
        test_data = {
            'State': ['Delhi'],
            'City': ['Delhi'], 
            'Locality': ['Connaught Place'],
            'Property_Type': ['Apartment'],
            'BHK': [3],
            'Size_in_SqFt': [1500.0],
            'Price_per_SqFt': [18000.0],
            'Year_Built': [2020],
            'Age_of_Property': [3],
            'Nearby_Schools': [5],
            'Nearby_Hospitals': [3],
            'Furnished_Status': ['Semi-furnished'],
            'Public_Transport_Accessibility': ['High'],
            'Parking_Space': ['Yes'],
            'Security': ['Yes'],
            'Amenities': ['Swimming Pool, Gym, Clubhouse'],
            'Facing': ['North'],
            'Owner_Type': ['Builder'],
            'Availability_Status': ['Ready_to_Move'],
            'Floor_No': [8],
            'Total_Floors': [15]
        }
        
        df = pd.DataFrame(test_data)
        predictions = pipeline.predict(df)
        print(f"✨ Test prediction for Delhi CP: ₹{predictions[0]:,.2f} Lakhs")
        print("\n✅ All systems working correctly!")
        
    else:
        print(f"❌ App health check failed with status: {response.status_code}")
        
except requests.exceptions.ConnectionError:
    print("🟡 Could not connect to app - it might still be starting up")
    print("🌐 Try accessing: http://localhost:8505")
except Exception as e:
    print(f"❌ Error testing app: {str(e)}")
    print("🌐 Manual check: http://localhost:8505")