import ast
from haversine import haversine, Unit
from matplotlib.pyplot import figure
import matplotlib.patches as mpatches
from setup import *
from charts_and_processing import *
import matplotlib.pyplot as plt



def find_hospital_closest_county(hospital_lat_long, crime_data):
    minimum = float('inf')
    index_of_minimum = 0
    for i, county_lat_long in enumerate(crime_data['lat_long'].tolist()):

        distance = haversine(hospital_lat_long, county_lat_long, unit=Unit.MILES)
        if distance < minimum:
            minimum = distance
            index_of_minimum = i
    
    return (crime_data.iloc[index_of_minimum]['county_name'], minimum)


def clean_drop_data_hospitals(hospitals):
    hospitals = hospitals[hospitals['Hospital overall rating']!='Not Available']
    hospitals = hospitals[hospitals['lat_long'].notna()]
    hospitals['Hospital overall rating']=hospitals['Hospital overall rating'].astype(int)
    hospitals['lat_long'] = hospitals['lat_long'].apply(lambda x : ast.literal_eval(x))
    return hospitals

def clean_drop_data_crime(crime_data):
    crime_data = crime_data[crime_data['lat_long'] != "(Na,Na)"]
    crime_data['lat_long'] = crime_data['lat_long'].apply(lambda x : ast.literal_eval(x))
    return crime_data

def crime_vs_care_chart(hospitals, crime_data):
    hosp_grouped = hospitals.groupby('county')
    hospital_mean_rating_by_county = hosp_grouped['Hospital overall rating'].mean()
    #get counties with avg. rating of 5 and avg. rating of 1
    top_rated_counties = hospital_mean_rating_by_county[hospital_mean_rating_by_county==5.0].index
    lowest_rated_counties = hospital_mean_rating_by_county[hospital_mean_rating_by_county==1.0].index
    crime_for_highest_rated=[]
    counties_highest_rated=[]

    for county in top_rated_counties:
        counties_highest_rated.append(county)
        crime_for_highest_rated.append(crime_data[crime_data['county_name']==county]['crime_rate_per_100000'].values[0])
        
    crime_for_lowest_rated=[]
    counties_lowest_rated=[]
    for county in lowest_rated_counties:
        counties_lowest_rated.append(county)
        crime_for_lowest_rated.append(crime_data[crime_data['county_name']==county]['crime_rate_per_100000'].values[0])
    
    avg_crime_for_lowest_rated = sum(crime_for_lowest_rated)/len(crime_for_lowest_rated)
    avg_crime_for_highest_rated = sum(crime_for_highest_rated)/len(crime_for_highest_rated)
    


    width = 0.35
    font = {'size'   : '13','family' : 'DejaVu Sans','weight' : 'bold'}
    plt.rc('font', **font)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.bar(counties_highest_rated, crime_for_highest_rated, width)
    ax1.set_ylabel('Crime Rate Per 100000')
    ax1.set_xlabel('Counties')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_title('                                           Crime rate for Highest rated (left) and Lowest rated (right) hospital care in counties')
    ax1.set_ylim([0, 650])
    ax1.axhline(avg_crime_for_highest_rated, color='red', ls='dotted')




    ax2.bar(counties_lowest_rated, crime_for_lowest_rated, width)
    # ax2.set_ylabel('Crime Rate Per 100000')
    ax2.set_xlabel('Counties')
    ax2.tick_params(axis='x', rotation=60)
    # ax2.set_title('Crime rate for Lowest rated hospital care in counties')
    ax2.set_ylim([0, 650])
    avg = ax2.axhline(avg_crime_for_lowest_rated, color='red', ls='dotted')
    plt.rcParams['figure.figsize'] = [10, 7]
    # plt.legend('avg')
    font = {'size'   : '8','family' : 'DejaVu Sans'}
    plt.rc('font', **font)
    red_patch = mpatches.Patch(color='red', ls='dotted', label='Average Crime')
    plt.legend(handles=[red_patch], loc='upper right')

    plt.show()
    
def mortality_murder_chart(crime_data,hospitals):
    top_5_murder_counties = list(crime_data.sort_values(['MURDER'], ascending=False)[:5]['county_name'])
    dict_mortality_options = {
    'Not Available':-1,
    "Below the national average":0,
    "Same as the national average":1,
    "Above the national average":2
    }
    def Mortality_num_category(mortality_comparision):
        return dict_mortality_options[mortality_comparision]

    hospitals['num_mortality_comparision'] = hospitals['Mortality national comparison'].map(dict_mortality_options)
    hospitals_filtered = hospitals[hospitals['num_mortality_comparision']!=-1]
    hosp_grouped = hospitals.groupby('county')['num_mortality_comparision'].mean()

    def get_mortality_avg(county):
        return hosp_grouped[county]
    values = crime_data.sort_values(['MURDER'], ascending=False)[:5]['county_name'].apply(get_mortality_avg)
    width = 0.4
    font = {'size'   : '13','family' : 'DejaVu Sans','weight' : 'bold'}
    plt.rc('font', **font)
    fig,ax1 = plt.subplots(1)
    ax1.bar(top_5_murder_counties, values, width)
    ax1.set_ylabel('Avg. Mortality Rate in hospitals Compared to National Average')
    ax1.set_xlabel('Counties')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_title('Top 5 Highest Murder rate Counties Mortality Rate')
    ax1.set_ylim([0, 3])
    font = {'size'   : '8','family' : 'DejaVu Sans'}
    plt.rc('font', **font)
    avg = ax1.axhline(1, color='red', ls='dotted')
    red_patch = mpatches.Patch(color='red', ls='dotted', label='Average mortality Rate')
    plt.legend(handles=[red_patch], loc='upper right')


    plt.show()
