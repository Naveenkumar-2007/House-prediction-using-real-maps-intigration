import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import st_folium
import joblib
import os
import sys
from datetime import datetime
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline.predict_pipeline import PredictPipeline, CustomData
from src.utils.geo_utils import GeoUtils
from src.utils.knn_imputation import KNNImputer

# Page config
st.set_page_config(
    page_title="🏠 Indian Housing Price Predictor",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better design
st.markdown("""
<style>
    /* Main styles */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Header */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }
    
    .main-header h1 {
        color: white;
        font-size: 3rem;
        margin: 0;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: #fff;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.95;
    }
    
    /* Cards */
    .info-box {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        border-left: 5px solid #667eea;
    }
    
    .search-section {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 3rem;
        border-radius: 20px;
        text-align: center;
        color: white;
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        margin: 2rem 0;
    }
    
    .prediction-result h1 {
        font-size: 4rem;
        margin: 0;
        font-weight: 900;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
    }
    
    .prediction-result p {
        font-size: 1.5rem;
        margin: 0.5rem 0;
        opacity: 0.95;
    }
    
    /* Buttons */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 12px;
        padding: 0.75rem 1.5rem;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.5);
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        padding: 0.5rem;
        transition: all 0.3s;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 0.2rem rgba(102, 126, 234, 0.25);
    }
    
    /* Map container */
    .map-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        margin-bottom: 2rem;
    }
    
    /* Metrics */
    div[data-testid="metric-container"] {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    section[data-testid="stSidebar"] * {
        color: white !important;
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo, .stWarning {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize utilities
geo_utils = GeoUtils()

# Load training data
@st.cache_data
def load_training_data():
    try:
        df = pd.read_csv("artifects/train.csv")
        return df
    except:
        try:
            df = pd.read_csv("C:\\Users\\navee\\Cisco Packet Tracer 8.2.2\\saves\\india_house\\notebook\\indian_housing_data.csv")
            return df
        except:
            return None

training_data = load_training_data()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    input_mode = st.radio(
        "Choose Input Method:",
        ["🗺️ Map + Address", "✍️ Manual Entry"],
        help="Select how you want to predict prices"
    )
    
    st.markdown("---")
    
    if training_data is not None:
        st.markdown("### 📊 Dataset Info")
        st.write(f"**Total Properties:** {len(training_data):,}")
        st.write(f"**States:** {training_data['State'].nunique()}")
        st.write(f"**Cities:** {training_data['City'].nunique()}")
        
        st.markdown("### 💰 Price Stats")
        st.write(f"**Min:** ₹{training_data['Price_in_Lakhs'].min():.2f}L")
        st.write(f"**Max:** ₹{training_data['Price_in_Lakhs'].max():.2f}L")
        st.write(f"**Avg:** ₹{training_data['Price_in_Lakhs'].mean():.2f}L")
    
    st.markdown("---")
    st.markdown("### 🌟 Features")
    st.write("✅ All India Coverage")
    st.write("✅ Smart Address Input")
    st.write("✅ AI Auto-Fill")
    st.write("✅ Real Predictions")

def predict_price(data_dict):
    """Make prediction"""
    try:
        custom_data = CustomData(**data_dict)
        pred_df = custom_data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(pred_df)
        return prediction[0]
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
        return None

def geocode_address(state, city, district="", pincode=""):
    """Geocode address from components"""
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="housing_predictor_v3")
        
        # Build address string
        address_parts = []
        if district:
            address_parts.append(district)
        if city:
            address_parts.append(city)
        if state:
            address_parts.append(state)
        if pincode:
            address_parts.append(pincode)
        address_parts.append("India")
        
        address_string = ", ".join(address_parts)
        
        with st.spinner(f"🔍 Finding location for: {address_string}..."):
            time.sleep(1)
            location = geolocator.geocode(address_string, timeout=10)
        
        if location:
            return {
                'latitude': location.latitude,
                'longitude': location.longitude,
                'display_name': location.address,
                'found': True
            }
        else:
            return {'found': False}
    except Exception as e:
        st.error(f"Geocoding error: {str(e)}")
        return {'found': False}

# Header
st.markdown("""
<div class="main-header">
    <h1>🏠 Indian Housing Price Predictor</h1>
    <p>🇮🇳 AI-Powered Real Estate Price Predictions Across India</p>
</div>
""", unsafe_allow_html=True)

# ==================== MAP + ADDRESS MODE ====================
if "Map + Address" in input_mode:
    
    # Address Input Section
    st.markdown('<div class="search-section">', unsafe_allow_html=True)
    st.markdown("### 📍 Enter Property Address")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        state = st.selectbox(
            "🏛️ State *",
            ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Gujarat", 
             "Rajasthan", "Punjab", "Haryana", "West Bengal", "Telangana", 
             "Uttar Pradesh", "Odisha", "Kerala", "Madhya Pradesh", "Andhra Pradesh",
             "Bihar", "Chhattisgarh", "Jharkhand", "Assam", "Uttarakhand"],
            key="addr_state"
        )
    
    with col2:
        city = st.text_input(
            "🏙️ City *",
            placeholder="e.g., Mumbai, Bangalore",
            key="addr_city"
        )
    
    with col3:
        district = st.text_input(
            "📍 District/Area",
            placeholder="e.g., Andheri, Whitefield (Optional)",
            key="addr_district"
        )
    
    with col4:
        pincode = st.text_input(
            "📮 Pincode",
            placeholder="e.g., 560001 (Optional)",
            key="addr_pincode",
            max_chars=6
        )
    
    col_btn1, col_btn2 = st.columns([1, 3])
    with col_btn1:
        find_location_btn = st.button("🔍 Find on Map", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize session state
    if 'location_found' not in st.session_state:
        st.session_state.location_found = False
        st.session_state.map_center = [20.5937, 78.9629]
        st.session_state.map_zoom = 5
    
    # Handle location finding
    if find_location_btn:
        if not city:
            st.error("❌ Please enter at least State and City")
        else:
            result = geocode_address(state, city, district, pincode)
            
            if result['found']:
                st.success(f"✅ Location found: {result['display_name']}")
                st.session_state.map_center = [result['latitude'], result['longitude']]
                st.session_state.map_zoom = 13
                st.session_state.location_found = True
                st.session_state.found_location = result
            else:
                st.error("❌ Location not found. Try with just State and City.")
    
    # Map Section
    st.markdown('<div class="map-container">', unsafe_allow_html=True)
    st.markdown("### 🗺️ Interactive Map")
    st.write("**Instructions:** Click on the exact property location on the map")
    
    # Create map
    m = folium.Map(
        location=st.session_state.map_center,
        zoom_start=st.session_state.map_zoom,
        tiles="OpenStreetMap"
    )
    
    # Add marker if location was found
    if st.session_state.location_found and hasattr(st.session_state, 'found_location'):
        folium.Marker(
            [st.session_state.found_location['latitude'], 
             st.session_state.found_location['longitude']],
            popup="📍 Found Location",
            tooltip="Click nearby for exact property location",
            icon=folium.Icon(color='green', icon='map-marker', prefix='fa')
        ).add_to(m)
    
    # Add click popup
    m.add_child(folium.LatLngPopup())
    
    # Display map
    map_data = st_folium(
        m,
        width=None,
        height=500,
        returned_objects=["last_clicked"]
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Handle map click
    if map_data and map_data.get('last_clicked'):
        clicked_lat = map_data['last_clicked']['lat']
        clicked_lon = map_data['last_clicked']['lng']
        
        st.success(f"✅ Property location selected: {clicked_lat:.6f}°N, {clicked_lon:.6f}°E")
        
        # Get detailed location
        with st.spinner("📍 Getting location details..."):
            location_info = geo_utils.reverse_geocode(clicked_lat, clicked_lon)
        
        # Display location info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"**🏛️ State**\n\n{location_info['state']}")
        with col2:
            st.info(f"**🏙️ City**\n\n{location_info['city']}")
        with col3:
            st.info(f"**🏘️ Locality**\n\n{location_info['locality']}")
        with col4:
            st.info(f"**📮 Pincode**\n\n{location_info['postcode']}")
        
        st.info(f"📍 **Full Address:** {location_info['full_address']}")
        
        st.markdown("---")
        
        # Property Details
        st.markdown('<div class="info-box">', unsafe_allow_html=True)
        st.markdown("### 🏗️ Property Details")
        
        # AI Auto-fill option
        use_knn = st.checkbox(
            "🤖 Auto-fill property details using AI (from nearby properties)",
            value=True
        )
        
        # Default values
        defaults = {
            'BHK': 3, 'Size_in_SqFt': 1500, 'Year_Built': 2010,
            'Nearby_Schools': 5, 'Nearby_Hospitals': 3,
            'Property_Type': 'Apartment', 'Furnishing_Status': 'Semi-Furnished',
            'Public_Transport_Accessibility': 'Medium',
            'Parking_Space': 'Yes', 'Security': 'Yes',
            'Facing': 'East', 'Owner_Type': 'Owner',
            'Availability_Status': 'Ready_to_Move',
            'Floor_No': 1, 'Total_Floors': 5
        }
        
        # KNN imputation
        if use_knn and training_data is not None:
            with st.spinner("🔄 Analyzing nearby properties..."):
                knn_imputer = KNNImputer(k=5)
                features = ['BHK', 'Size_in_SqFt', 'Year_Built', 'Nearby_Schools', 
                           'Nearby_Hospitals', 'Property_Type', 'Furnishing_Status',
                           'Public_Transport_Accessibility', 'Parking_Space', 
                           'Security', 'Facing', 'Owner_Type', 'Availability_Status']
                
                imputed = knn_imputer.impute_features(clicked_lat, clicked_lon, training_data, features)
                
                for key, value in imputed.items():
                    if key in defaults and value is not None:
                        defaults[key] = value
                
                if imputed.get('neighbors_found', 0) > 0:
                    st.success(f"✅ Found {imputed['neighbors_found']} nearby properties ({imputed.get('nearest_distance_km', 0):.2f} km away)")
        
        # Input form
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            bhk = st.number_input("🛏️ BHK", 1, 10, int(defaults['BHK']))
            size_sqft = st.number_input("📏 Size (Sq.Ft)", 100, 50000, int(defaults['Size_in_SqFt']))
            year_built = st.number_input("📅 Year Built", 1900, 2025, int(defaults['Year_Built']))
        
        with col2:
            property_type = st.selectbox("🏘️ Property Type", 
                                        ["Apartment", "Independent House", "Villa", "Penthouse"])
            furnishing = st.selectbox("🛋️ Furnishing", 
                                     ["Furnished", "Semi-Furnished", "Unfurnished"])
            facing = st.selectbox("🧭 Facing", 
                                 ["North", "South", "East", "West", "NE", "NW", "SE", "SW"])
        
        with col3:
            nearby_schools = st.number_input("🏫 Nearby Schools", 0, 20, int(defaults['Nearby_Schools']))
            nearby_hospitals = st.number_input("🏥 Nearby Hospitals", 0, 20, int(defaults['Nearby_Hospitals']))
            transport = st.selectbox("🚇 Public Transport", ["High", "Medium", "Low"])
        
        with col4:
            parking = st.selectbox("🅿️ Parking", ["Yes", "No"])
            security = st.selectbox("🔒 Security", ["Yes", "No"])
            owner_type = st.selectbox("👤 Owner", ["Owner", "Builder", "Broker"])
        
        col1, col2, col3 = st.columns(3)
        with col1:
            availability = st.selectbox("✅ Status", ["Ready_to_Move", "Under_Construction"])
        with col2:
            floor_no = st.number_input("🔢 Floor No", 0, 50, 1)
        with col3:
            total_floors = st.number_input("🏢 Total Floors", 1, 50, 5)
        
        # Amenities
        st.write("**🎯 Amenities:**")
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            gym = st.checkbox("💪 Gym", True)
        with col2:
            pool = st.checkbox("🏊 Pool", True)
        with col3:
            playground = st.checkbox("🎮 Playground", True)
        with col4:
            garden = st.checkbox("🌳 Garden", True)
        with col5:
            clubhouse = st.checkbox("🏛️ Clubhouse", True)
        
        amenities_list = []
        if gym: amenities_list.append("Gym")
        if pool: amenities_list.append("Pool")
        if playground: amenities_list.append("Playground")
        if garden: amenities_list.append("Garden")
        if clubhouse: amenities_list.append("Clubhouse")
        amenities = ", ".join(amenities_list) if amenities_list else "None"
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Predict button
        if st.button("🔮 Predict House Price", type="primary", use_container_width=True):
            with st.spinner("🤖 AI is analyzing the property..."):
                current_year = datetime.now().year
                age = current_year - year_built
                
                data_dict = {
                    'State': location_info['state'],
                    'City': location_info['city'],
                    'Locality': location_info['locality'],
                    'Property_Type': property_type,
                    'BHK': int(bhk),
                    'Size_in_SqFt': float(size_sqft),
                    'Price_per_SqFt': 0.08,
                    'Year_Built': int(year_built),
                    'Latitude': float(clicked_lat),
                    'Longitude': float(clicked_lon),
                    'Age_of_Property': int(age),
                    'Nearby_Schools': int(nearby_schools),
                    'Nearby_Hospitals': int(nearby_hospitals),
                    'Furnishing_Status': furnishing,
                    'Public_Transport_Accessibility': transport,
                    'Parking_Space': parking,
                    'Security': security,
                    'Amenities': amenities,
                    'Facing': facing,
                    'Owner_Type': owner_type,
                    'Availability_Status': availability,
                    'Floor_No': int(floor_no),
                    'Total_Floors': int(total_floors)
                }
                
                predicted_price = predict_price(data_dict)
                
                if predicted_price:
                    st.balloons()
                    
                    # Display prediction
                    st.markdown(f"""
                    <div class="prediction-result">
                        <h1>₹ {predicted_price:.2f} Lakhs</h1>
                        <p>≈ ₹ {predicted_price * 100000:,.0f}</p>
                        <p style="font-size: 1rem; margin-top: 1rem;">AI Predicted Price</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("💵 Price/Sq.Ft", f"₹{(predicted_price * 100000 / size_sqft):,.0f}")
                    with col2:
                        st.metric("🏠 Type", property_type)
                    with col3:
                        st.metric("📅 Age", f"{age} years")
                    with col4:
                        st.metric("🛏️ Config", f"{bhk} BHK")
                    
                    # Neighborhood stats
                    if training_data is not None:
                        st.markdown("### 📊 Neighborhood Analysis")
                        knn_imputer = KNNImputer(k=10)
                        stats = knn_imputer.get_neighborhood_stats(clicked_lat, clicked_lon, training_data)
                        
                        if stats:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Avg Price (Nearby)", f"₹{stats['avg_price_lakhs']:.2f}L")
                                st.metric("Median Price", f"₹{stats['median_price_lakhs']:.2f}L")
                            with col2:
                                st.metric("Avg Size", f"{stats['avg_size_sqft']:.0f} sq.ft")
                                st.metric("Avg BHK", f"{stats['avg_bhk']:.1f}")
                            with col3:
                                st.metric("Nearest Property", f"{stats['nearest_distance_km']:.2f} km")
                                st.metric("Common Type", stats['common_property_type'])
                            
                            price_diff = predicted_price - stats['median_price_lakhs']
                            price_diff_pct = (price_diff / stats['median_price_lakhs']) * 100
                            
                            if abs(price_diff_pct) < 5:
                                st.success(f"💡 Fairly priced (±{abs(price_diff_pct):.1f}%)")
                            elif price_diff > 0:
                                st.info(f"💡 {abs(price_diff_pct):.1f}% above market")
                            else:
                                st.success(f"💡 Great deal! {abs(price_diff_pct):.1f}% below market")

# ==================== MANUAL ENTRY MODE ====================
else:
    st.markdown('<div class="info-box">', unsafe_allow_html=True)
    st.markdown("### ✍️ Enter Property Details Manually")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        state = st.selectbox("🏛️ State", 
                            ["Maharashtra", "Karnataka", "Tamil Nadu", "Delhi", "Gujarat", 
                             "Rajasthan", "Punjab", "Haryana", "West Bengal", "Telangana", 
                             "Uttar Pradesh", "Odisha", "Kerala"])
        city = st.text_input("🏙️ City", "Mumbai")
        locality = st.text_input("🏘️ Locality", "Andheri")
        property_type = st.selectbox("🏘️ Type", ["Apartment", "Independent House", "Villa", "Penthouse"])
    
    with col2:
        bhk = st.number_input("🛏️ BHK", 1, 10, 3)
        size_sqft = st.number_input("📏 Size", 100, 50000, 1500)
        year_built = st.number_input("📅 Year", 1900, 2025, 2010)
        furnishing = st.selectbox("🛋️ Furnishing", ["Furnished", "Semi-Furnished", "Unfurnished"])
    
    with col3:
        latitude = st.number_input("🌍 Latitude", 8.0, 35.0, 19.0760, format="%.6f")
        longitude = st.number_input("🌍 Longitude", 68.0, 97.0, 72.8777, format="%.6f")
        nearby_schools = st.number_input("🏫 Schools", 0, 20, 5)
        nearby_hospitals = st.number_input("🏥 Hospitals", 0, 20, 3)
    
    with col4:
        transport = st.selectbox("🚇 Transport", ["High", "Medium", "Low"])
        parking = st.selectbox("🅿️ Parking", ["Yes", "No"])
        security = st.selectbox("🔒 Security", ["Yes", "No"])
        facing = st.selectbox("🧭 Facing", ["North", "South", "East", "West"])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        owner_type = st.selectbox("👤 Owner", ["Owner", "Builder", "Broker"])
    with col2:
        availability = st.selectbox("✅ Status", ["Ready_to_Move", "Under_Construction"])
    with col3:
        floor_no = st.number_input("🔢 Floor", 0, 50, 1)
    with col4:
        total_floors = st.number_input("🏢 Total", 1, 50, 5)
    
    st.write("**🎯 Amenities:**")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        gym = st.checkbox("💪 Gym", True)
    with col2:
        pool = st.checkbox("🏊 Pool", True)
    with col3:
        playground = st.checkbox("🎮 Playground", True)
    with col4:
        garden = st.checkbox("🌳 Garden", True)
    with col5:
        clubhouse = st.checkbox("🏛️ Clubhouse", True)
    
    amenities_list = []
    if gym: amenities_list.append("Gym")
    if pool: amenities_list.append("Pool")
    if playground: amenities_list.append("Playground")
    if garden: amenities_list.append("Garden")
    if clubhouse: amenities_list.append("Clubhouse")
    amenities = ", ".join(amenities_list) if amenities_list else "None"
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🔮 Predict Price", type="primary", use_container_width=True):
        with st.spinner("🤖 Analyzing..."):
            current_year = datetime.now().year
            age = current_year - year_built
            
            data_dict = {
                'State': state, 'City': city, 'Locality': locality,
                'Property_Type': property_type, 'BHK': int(bhk),
                'Size_in_SqFt': float(size_sqft), 'Price_per_SqFt': 0.08,
                'Year_Built': int(year_built), 'Latitude': float(latitude),
                'Longitude': float(longitude), 'Age_of_Property': int(age),
                'Nearby_Schools': int(nearby_schools),
                'Nearby_Hospitals': int(nearby_hospitals),
                'Furnishing_Status': furnishing,
                'Public_Transport_Accessibility': transport,
                'Parking_Space': parking, 'Security': security,
                'Amenities': amenities, 'Facing': facing,
                'Owner_Type': owner_type, 'Availability_Status': availability,
                'Floor_No': int(floor_no), 'Total_Floors': int(total_floors)
            }
            
            predicted_price = predict_price(data_dict)
            
            if predicted_price:
                st.balloons()
                st.markdown(f"""
                <div class="prediction-result">
                    <h1>₹ {predicted_price:.2f} Lakhs</h1>
                    <p>≈ ₹ {predicted_price * 100000:,.0f}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("💵 Price/Sq.Ft", f"₹{(predicted_price * 100000 / size_sqft):,.0f}")
                with col2:
                    st.metric("🏠 Type", property_type)
                with col3:
                    st.metric("📅 Age", f"{age} years")
                with col4:
                    st.metric("🛏️ Config", f"{bhk} BHK")

