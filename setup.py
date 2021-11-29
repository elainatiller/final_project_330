!pip install geopandas
!pip install geopy
!pip install pgeocode
import matplotlib.pyplot as plt
import pandas as pd
from geopy.geocoders import Photon
import pgeocode
import geopandas
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from time import sleep
import re
import json
import seaborn as sns
import os

#Inputs: string as county to query geolocator with (St. Louis city, MO)
#Output: tuple wit zipcode (int) and str of lat/long
def zip_and_lat_long_for_crime(county):
    if type(county) != type('string'):
        print('Inputs were not correct. Ensure text is a string.')
    else:
        location = geolocator.geocode(county)
        if(location != None):
            return location[1]
        
#Inputs: string address to query geolocator with ex:("1975 ALPHA STE 100 ROCKWALL TX 75087")
#Output: str of lat/long
def lat_long_for_hospital(address):
    if type(address) != type('string'):
        print('Inputs were not correct. Ensure text is a string.')
    else:
        location = geolocator.geocode(address)
        if(location != None):
            return location[1]

def file_pre_processing():
    global geolocator = Photon(user_agent="Final Project1", timeout=10)
    
    required_columns = ['Provider ID', 'Hospital Name', 'Address', 'City', 'State', 'ZIP Code', 'Hospital Type', \
               'Hospital Ownership', 'Emergency Services', 'Hospital overall rating', \
               'Patient experience national comparison', 'Mortality national comparison', \
               'Effectiveness of care national comparison']

    crime_columns = ['county_name', 'crime_rate_per_100000', 'MURDER', 'RAPE', 'ROBBERY', \
                 'AGASSLT', 'BURGLRY', 'LARCENY', 'population']
    
    crime_data = pd.read_csv('crime_data_w_population_and_crime_rate.csv', usecols=crime_columns)
    hospitals = pd.read_csv('Hospital.csv', encoding='Windows-1252', usecols=required_columns)

    hospitals['ZIP Code'] = hospitals['ZIP Code'].astype(str)
    hospitals['full_address'] = hospitals['Address'] + ' ' + hospitals['City'] + ' ' + hospitals['State'] + ' ' + hospitals['ZIP Code']
    #use functions above w API to get lat long
    crime_data['lat_long'] = crime_data['county_name'].map(zip_and_lat_long_for_crime)
    hospitals['lat_long'] = hospitals['full_address'].map(lat_long_for_hospital)
    
    
    crime_data.to_csv('crime_data_post_processing.csv')
    hospitals.to_csv('hospitals_data_post_processing.csv')

    return None

