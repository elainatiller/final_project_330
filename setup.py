import matplotlib.pyplot as plt
import pandas as pd
import re
import os

def file_pre_processing():

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

    ####
    #
    # Your work here Elaina to get lat, long for both datasets
    # e.g. hospitals['lat_long'] = ...
    #
    ####

    crime_data.to_csv('crime_data_post_processing.csv')
    hospitals.to_csv('hospitals_data_post_processing.csv')

    return None
