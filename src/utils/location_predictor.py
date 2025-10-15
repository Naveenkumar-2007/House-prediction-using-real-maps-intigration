import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
from src.utils.geo_utils import GeoUtils

class LocationBasedPredictor:
    def __init__(self):
        self.geo_utils = GeoUtils()
        self.geolocator = Nominatim(user_agent="housing_predictor_2025")
        
        # Major Indian cities with their typical price ranges (in lakhs per sq ft)
        self.city_price_multipliers = {
            'Mumbai': 2.5, 'Delhi': 2.2, 'Bangalore': 1.8, 'Gurgaon': 2.0,
            'Pune': 1.6, 'Hyderabad': 1.5, 'Chennai': 1.7, 'Kolkata': 1.3,
            'Noida': 1.8, 'Ahmedabad': 1.2, 'Jaipur': 1.0, 'Lucknow': 0.8,
            'Kochi': 1.4, 'Chandigarh': 1.5, 'Indore': 0.9, 'Bhopal': 0.8,
            'Coimbatore': 1.1, 'Vadodara': 1.0, 'Nagpur': 0.9, 'Surat': 1.1
        }
        
        # Metro proximity bonuses
        self.metro_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune']
        
        # Locality premium mapping for major areas
        self.premium_localities = {
            'Mumbai': ['Bandra', 'Juhu', 'Andheri West', 'Powai', 'Lower Parel', 'Worli'],
            'Delhi': ['Connaught Place', 'Defence Colony', 'Greater Kailash', 'Hauz Khas', 'Khan Market'],
            'Bangalore': ['Koramangala', 'Indiranagar', 'Whitefield', 'Electronic City', 'HSR Layout'],
            'Pune': ['Koregaon Park', 'Kalyani Nagar', 'Aundh', 'Baner', 'Viman Nagar'],
            'Hyderabad': ['Banjara Hills', 'Jubilee Hills', 'Hitech City', 'Gachibowli', 'Kondapur']
        }

    def get_coordinates(self, city, locality=None, state=None):
        """Get real coordinates from location name"""
        try:
            # Build search query
            if locality and locality != 'Unknown':
                search_query = f"{locality}, {city}"
            else:
                search_query = city
                
            if state and state != 'Unknown':
                search_query += f", {state}, India"
            else:
                search_query += ", India"
            
            print(f"🌍 Geocoding: {search_query}")
            
            # Add delay to respect rate limits
            time.sleep(1)
            
            location = self.geolocator.geocode(search_query, country_codes=['in'], timeout=10)
            
            if location:
                lat, lon = location.latitude, location.longitude
                print(f"✅ Found coordinates: {lat:.4f}, {lon:.4f}")
                return lat, lon
            else:
                # Fallback to city coordinates
                city_coords = self.geo_utils.get_indian_city_coordinates()
                if city in city_coords:
                    lat, lon = city_coords[city]
                    print(f"📍 Using default city coordinates: {lat:.4f}, {lon:.4f}")
                    return lat, lon
                else:
                    # Default to Mumbai coordinates
                    print("⚠️ Using default Mumbai coordinates")
                    return 19.0760, 72.8777
                    
        except Exception as e:
            print(f"❌ Geocoding error: {str(e)}")
            # Return Mumbai coordinates as fallback
            return 19.0760, 72.8777

    def calculate_location_features(self, city, locality, state, latitude, longitude):
        """Calculate location-based features for better prediction"""
        try:
            features = {}
            
            # City-based price multiplier
            city_clean = str(city).strip()
            features['city_price_multiplier'] = self.city_price_multipliers.get(city_clean, 1.0)
            
            # Metro city bonus
            features['is_metro_city'] = 1 if city_clean in self.metro_cities else 0
            
            # Premium locality bonus
            locality_clean = str(locality).strip()
            is_premium = False
            if city_clean in self.premium_localities:
                premium_areas = self.premium_localities[city_clean]
                is_premium = any(area.lower() in locality_clean.lower() for area in premium_areas)
            features['is_premium_locality'] = 1 if is_premium else 0
            
            # Distance to major business hubs
            business_hubs = {
                'Mumbai_BKC': (19.0647, 72.8678),
                'Delhi_CP': (28.6315, 77.2167),
                'Bangalore_Electronic_City': (12.8456, 77.6647),
                'Hyderabad_Hitech': (17.4435, 78.3772),
                'Pune_Hinjewadi': (18.5912, 73.7389)
            }
            
            min_hub_distance = float('inf')
            for hub_name, (hub_lat, hub_lon) in business_hubs.items():
                distance = geodesic((latitude, longitude), (hub_lat, hub_lon)).kilometers
                min_hub_distance = min(min_hub_distance, distance)
            
            features['distance_to_business_hub'] = min_hub_distance
            
            # Tier classification
            tier_1_cities = ['Mumbai', 'Delhi', 'Bangalore', 'Hyderabad', 'Chennai', 'Kolkata', 'Pune', 'Ahmedabad']
            tier_2_cities = ['Jaipur', 'Lucknow', 'Kochi', 'Chandigarh', 'Indore', 'Bhopal', 'Coimbatore', 'Vadodara']
            
            if city_clean in tier_1_cities:
                features['city_tier'] = 1
            elif city_clean in tier_2_cities:
                features['city_tier'] = 2
            else:
                features['city_tier'] = 3
                
            return features
            
        except Exception as e:
            print(f"Error calculating location features: {str(e)}")
            return {
                'city_price_multiplier': 1.0,
                'is_metro_city': 0,
                'is_premium_locality': 0,
                'distance_to_business_hub': 50.0,
                'city_tier': 3
            }

    def enhance_prediction_with_location(self, base_prediction, location_features, property_features):
        """Enhance base prediction with location-based adjustments"""
        try:
            enhanced_price = float(base_prediction)
            
            # Apply city multiplier
            enhanced_price *= location_features['city_price_multiplier']
            
            # Metro city bonus (10-20% increase)
            if location_features['is_metro_city']:
                enhanced_price *= 1.15
            
            # Premium locality bonus (15-30% increase)
            if location_features['is_premium_locality']:
                enhanced_price *= 1.25
            
            # Distance penalty/bonus
            distance = location_features['distance_to_business_hub']
            if distance < 5:  # Very close to business hub
                enhanced_price *= 1.3
            elif distance < 15:  # Moderately close
                enhanced_price *= 1.1
            elif distance > 50:  # Far from business hubs
                enhanced_price *= 0.9
            
            # Size-based adjustments
            size_sqft = property_features.get('Size_in_SqFt', 1000)
            if size_sqft > 2000:  # Large properties
                enhanced_price *= 1.05
            elif size_sqft < 500:  # Small properties
                enhanced_price *= 0.95
            
            # BHK-based adjustments
            bhk = property_features.get('BHK', 2)
            if bhk >= 4:
                enhanced_price *= 1.1
            elif bhk == 1:
                enhanced_price *= 0.9
            
            # Add some realistic variation (±5%)
            import random
            variation = random.uniform(0.95, 1.05)
            enhanced_price *= variation
            
            return max(enhanced_price, 10.0)  # Minimum 10 lakhs
            
        except Exception as e:
            print(f"Error enhancing prediction: {str(e)}")
            return float(base_prediction)