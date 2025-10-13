import requests
from geopy.geocoders import Nominatim
from geopy.distance import geodesic
import time
import pandas as pd

class GeoUtils:
    def __init__(self):
        self.geolocator = Nominatim(user_agent="indian_housing_price_predictor_2025")

    def reverse_geocode(self, latitude, longitude):
        """
        Get address information from latitude and longitude for Indian locations
        """
        try:
            time.sleep(1)  # Respect rate limits
            location = self.geolocator.reverse(f"{latitude}, {longitude}", language='en')
            
            if location and location.raw.get('address'):
                address = location.raw['address']
                
                # Extract Indian-specific location details
                state = address.get('state', 'Unknown')
                city = (address.get('city') or 
                       address.get('town') or 
                       address.get('village') or 
                       address.get('municipality') or 
                       'Unknown')
                
                locality = (address.get('suburb') or 
                           address.get('neighbourhood') or 
                           address.get('residential') or 
                           address.get('locality') or
                           address.get('city_district') or
                           'Unknown')
                
                postcode = address.get('postcode', 'Unknown')
                country = address.get('country', 'Unknown')
                
                return {
                    'state': state,
                    'city': city,
                    'locality': locality,
                    'postcode': postcode,
                    'country': country,
                    'full_address': location.address,
                    'latitude': latitude,
                    'longitude': longitude
                }
            else:
                return {
                    'state': 'Unknown',
                    'city': 'Unknown',
                    'locality': 'Unknown',
                    'postcode': 'Unknown',
                    'country': 'India',
                    'full_address': f'Location at {latitude:.4f}, {longitude:.4f}',
                    'latitude': latitude,
                    'longitude': longitude
                }

        except Exception as e:
            print(f" Error in reverse geocoding: {str(e)}")
            return {
                'state': 'Unknown',
                'city': 'Unknown',
                'locality': 'Unknown',
                'postcode': 'Unknown',
                'country': 'India',
                'full_address': f'Location at {latitude:.4f}, {longitude:.4f}',
                'latitude': latitude,
                'longitude': longitude
            }

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        """
        Calculate distance between two coordinates in kilometers
        """
        try:
            coords_1 = (lat1, lon1)
            coords_2 = (lat2, lon2)
            return geodesic(coords_1, coords_2).kilometers
        except Exception as e:
            print(f"Error calculating distance: {str(e)}")
            return float('inf')
    
    def get_indian_city_coordinates(self):
        """
        Return major Indian cities with their coordinates for map initialization
        """
        return {
            'Mumbai': (19.0760, 72.8777),
            'Delhi': (28.7041, 77.1025),
            'Bangalore': (12.9716, 77.5946),
            'Hyderabad': (17.3850, 78.4867),
            'Chennai': (13.0827, 80.2707),
            'Kolkata': (22.5726, 88.3639),
            'Pune': (18.5204, 73.8567),
            'Ahmedabad': (23.0225, 72.5714),
            'Jaipur': (26.9124, 75.7873),
            'Lucknow': (26.8467, 80.9462)
        }