from setup import *
from charts_and_processing import *
import numpy as np
from sklearn.linear_model import LinearRegression
import scipy.stats as st
import sys
import scipy
import sklearn
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import linear_model


def merge_dataframes_and_analysis(hospital_df, crime_df):
    
    hospitals = hospital_df.rename({'county':'county_name'}, axis=1)

    hospitals_merged = hospitals.merge(crime_df, on='county_name', how='left').drop(['full_address', 'lat_long_x', 'lat_long_y'], axis=1)
    distances = hospitals_merged['distance_in_miles']

    hospitals_merged['pop_bin'] = pd.qcut(hospitals_merged['population'], 10)

    no_unrated_hospitals = hospitals_merged[hospitals_merged['Hospital overall rating'] != 'Not Available']
    no_unrated_hospitals['Hospital overall rating'] = pd.to_numeric(no_unrated_hospitals['Hospital overall rating'])
    crime_rate_by_pop = no_unrated_hospitals.groupby('pop_bin')['crime_rate_per_100000'].mean()

    return distances, no_unrated_hospitals, crime_rate_by_pop

def hospital_rating_vs_crime(no_unrated_hospitals):
    
    rating_vs_crime = no_unrated_hospitals.groupby('Hospital overall rating')['crime_rate_per_100000'].mean()
    x = rating_vs_crime.index
    y = rating_vs_crime.values
    plt.plot(x, y, 'o', color='black')
    m, b = np.polyfit(x, y, 1)
    plt.plot(x, m*x+b, color='red')
    plt.xticks([1, 2, 3, 4, 5])
    plt.xlabel("Hospital Rating")
    plt.ylabel("Average County Crime Rate per 100000")
    plt.title("Average County Crime Rate vs. Hospital Rating")

    x_as_np = np.array(x).reshape((-1, 1))
    y_as_np = np.array(y)
    model = LinearRegression().fit(x_as_np, y_as_np)
    r_sq = model.score(x_as_np, y_as_np)

    return 'Coefficient of Determination: ' + str(r_sq)

def crime_vs_hospital_rating(no_unrated_hospitals):

    global grouped_df
    grouped_df = no_unrated_hospitals.copy()
    no_unrated_hospitals['crime_rate_bin'] = pd.qcut(no_unrated_hospitals['crime_rate_per_100000'], 10)
    crime_vs_rating = no_unrated_hospitals.groupby('crime_rate_bin')['Hospital overall rating'].mean()

    x = crime_vs_rating.index
    y = crime_vs_rating.values
    x = x.astype(str)
    idex = np.asarray([i+1 for i in range(len(x))])
    plt.plot(idex, y, 'o', color='black')
    m, b = np.polyfit(idex, y, 1)
    plt.plot(idex, m*idex+b, color='red')
    plt.xticks(idex)
    plt.xticks(np.arange(10), x,
        rotation=45)  # Set text labels and properties.
    plt.yticks([2.5, 3, 3.5])
    plt.xlabel("Crime Rate Bin")
    plt.ylabel("Hospital Rating")
    plt.title("Hospital Rating vs. Crime Rate (bins)")

    x_as_np = np.array(idex).reshape((-1, 1))
    y_as_np = np.array(y)
    model = LinearRegression().fit(x_as_np, y_as_np)
    r_sq = model.score(x_as_np, y_as_np)

    return (crime_vs_rating, 'Coefficient of Determination: ' + str(r_sq))

def control_for_population_size(no_unrated_hospitals):

    no_unrated_hospitals['pop_bin'] = pd.qcut(no_unrated_hospitals['population'], 20)

    grouped_by_pop_and_hospital_rating = no_unrated_hospitals.groupby(['pop_bin', 'Hospital overall rating'])['crime_rate_per_100000'].agg('mean').unstack(level=0)
    df = grouped_by_pop_and_hospital_rating.copy()
    df = df.apply(lambda col: (col-col.mean())/col.std(), axis=0) # standardizing

    count_where_higher_quality_hospital_and_crime_reduced = 0
    count_where_higher_quality_hospital_and_crime_increased = 0
    higher_quality_hospital_and_crime_reduced_std = 0
    higher_quality_hospital_and_crime_increased_std = 0
    for column in df.columns:
        this_column_values = df[column].values
        if pd.isnull(this_column_values[1]) or pd.isnull(this_column_values[3]):
            pass
        else: 
            if this_column_values[3] < this_column_values[1]:
                higher_quality_hospital_and_crime_reduced_std += this_column_values[3] - this_column_values[1]
                count_where_higher_quality_hospital_and_crime_reduced += 1
            else:
                higher_quality_hospital_and_crime_increased_std += this_column_values[1] - this_column_values[3]
                count_where_higher_quality_hospital_and_crime_increased += 1
            
    decreased = f'# of instances where crime DECREASED as hospital quality increased: {count_where_higher_quality_hospital_and_crime_reduced}'
    increased = f'# of instances where crime INCREASED as hospital quality increased: {count_where_higher_quality_hospital_and_crime_increased}'
    std_reduction = f'\nAvg # of standard deviations of REDUCTION in crime as hospital quality increased: {-higher_quality_hospital_and_crime_reduced_std/count_where_higher_quality_hospital_and_crime_reduced}'
    std_increase = f'Avg # of standard deviations of INCREASE in crime as hospital quality increased: {-higher_quality_hospital_and_crime_increased_std/count_where_higher_quality_hospital_and_crime_increased}'

    return grouped_by_pop_and_hospital_rating, df, decreased, increased, std_reduction, std_increase

def graph_crime_rates_between_hospital_ratings(grouped_by_pop_and_hospital_rating):

    #grouped_by_pop_and_hospital_rating.columns.tolist()
    X = []
    for item in grouped_by_pop_and_hospital_rating.columns.tolist():
        X.append(str(item))
    rating_4 = grouped_by_pop_and_hospital_rating.loc[4].tolist()
    rating_2 = grouped_by_pop_and_hospital_rating.loc[2].tolist()
    X_axis = np.arange(len(X))
  
    plt.bar(X_axis - 0.2, rating_2, 0.4, label = 'Hospital Rating: 2', color='orange')
    plt.bar(X_axis + 0.2, rating_4, 0.4, label = 'Hospital Rating: 4', color='lightgreen')
    plt.xticks(X_axis, X, rotation=90)
    plt.xlabel("Population Bin")
    plt.ylabel("Average Crime Rate per 100000")
    plt.title("Crime Rates Between Hospital Ratings")
    plt.legend()
    plt.show()

    return None

def create_model(no_unrated_hospitals):
    
    cols_to_keep = ['Hospital Type', 'Hospital Ownership', 'Emergency Services',
       'Hospital overall rating', 'Mortality national comparison',
       'Patient experience national comparison',
       'Effectiveness of care national comparison', 'crime_rate_per_100000',
       'MURDER', 'RAPE', 'ROBBERY', 'AGASSLT', 'BURGLRY', 'LARCENY',
       'population']

    df = no_unrated_hospitals[cols_to_keep]
    
    df['Hospital Type'] = df['Hospital Type'].astype('category')
    df['Hospital Type'] = df['Hospital Type'].cat.codes
    # 0: Acute Care Hospitals
    # 1: Critical Access Hospitals

    df['Hospital Ownership'] = df['Hospital Ownership'].astype('category')
    df['Hospital Ownership'] = df['Hospital Ownership'].cat.codes
    # 0: Government - Federal
    # 1: Government - Hospital District or Authority
    # 2: Government - Local
    # 3: Government - State
    # 4: Physician
    # 5: Proprietary
    # 6: Tribal
    # 7: Voluntary non-profit - Church
    # 8: Voluntary non-profit - Other
    # 9: Voluntary non-profit - Private

    df['Emergency Services'] = df['Emergency Services'].astype('category')
    df['Emergency Services'] = df['Emergency Services'].cat.codes
    # 0: No
    # 1: Yes

    df['Mortality national comparison'] = df['Mortality national comparison'].astype('category')
    df['Mortality national comparison'] = df['Mortality national comparison'].cat.codes
    # 0: Above the national average
    # 1: Below the national average
    # 2: Not Available
    # 3: Same as the national average

    df['Patient experience national comparison'] = df['Patient experience national comparison'].astype('category')
    df['Patient experience national comparison'] = df['Patient experience national comparison'].cat.codes
    # 0: Above the national average
    # 1: Below the national average
    # 2: Not Available
    # 3: Same as the national average

    df['Effectiveness of care national comparison'] = df['Effectiveness of care national comparison'].astype('category')
    df['Effectiveness of care national comparison'] = df['Effectiveness of care national comparison'].cat.codes
    # 0: Above the national average
    # 1: Below the national average
    # 2: Not Available
    # 3: Same as the national average

    # df.hist()
    # pyplot.show()
    # scatter_matrix(df)
    # pyplot.show()

    array = df.values
    X = np.concatenate([array[:, 0:3], array[:, 4:15]],axis = 1) # .shape = (3000, 14)
    y = array[:, 3]                                              # .shape = (3000,)

    X_train, X_validation, Y_train, Y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
    #models = []
    #models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
    #models.append(('LDA', LinearDiscriminantAnalysis()))
    #models.append(('KNN', KNeighborsClassifier()))
    #models.append(('CART', DecisionTreeClassifier()))
    #models.append(('NB', GaussianNB()))
    #models.append(('SVM', SVC(gamma='auto')))

    #results = []
    #names = []
    #for name, model in models:
        
        #steps = [('pca', PCA(n_components=10)), ('m', model)]
        #model = Pipeline(steps=steps)
        
        #kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
        #cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        #results.append(cv_results)
        #names.append(name)
        #print('%s: %f (%f)' % (name, cv_results.mean(), cv_results.std()))

        #regr = linear_model.LinearRegression()
        #regr.fit(X_train, Y_train)

    model = LinearDiscriminantAnalysis()
    model.fit(X_train, Y_train)

    #predictions = model.predict(X_validation)
    #print(accuracy_score(Y_validation, predictions))
    #print(confusion_matrix(Y_validation, predictions))
    #print(classification_report(Y_validation, predictions))

    return df, model

def model_make_prediction(crime_rate, population, murder_input=None, robbery_input=None, assault_input=None):

    df, model = create_model(grouped_df)

    if isinstance(murder_input, type(None)):
        murder = df['MURDER'].mean()
    else:
        murder = murder_input

    if isinstance(robbery_input, type(None)):
        robbery = df['ROBBERY'].mean()
    else:
        robbery = robbery_input

    if isinstance(assault_input, type(None)):
        assault = df['AGASSLT'].mean()
    else:
        assault = assault_input

    hosp_type = df['Hospital Type'].median()
    ownership = df['Hospital Ownership'].median()
    emergency_services = df['Emergency Services'].median()
    mortality = df['Mortality national comparison'].median()
    experience = df['Patient experience national comparison'].median()
    effectiveness = df['Effectiveness of care national comparison'].median()
    rape = df['RAPE'].mean()
    burglary = df['BURGLRY'].mean()
    larceny = df['LARCENY'].mean()
    array = np.array([hosp_type, ownership, emergency_services, mortality,
       experience, effectiveness, crime_rate, murder,
       rape, robbery, assault, burglary,
       larceny, population])

    prediction = int(model.predict(array.reshape(1, -1).astype(np.float64))[0])
    print(f'Predicted Hospital Overall Rating: {prediction}')
    
    return None


def predict():

    crime = grouped_df['crime_rate_per_100000'].mean()
    population = grouped_df['population'].mean()
    murder = grouped_df['MURDER'].mean()
    robbery = grouped_df['ROBBERY'].mean()
    assault = grouped_df['AGASSLT'].mean()

    print(f'Set your crime rate (Avg: {crime})')
    crime_rate_input = input()
    #print(type(crime_rate_input))
    #if type(crime_rate_input) != int:
    #    print('Incorrect input. Please try again!')
    #    return None
    print(f'Set your population (Avg: {population})')
    pop = input()
    #if type(pop) != int:
    #    print('Incorrect input. Please try again!')
    #    return None
    print('Would you like to set murder, robbery, and assault rates? Respond Y/N')
    additional = input()
    #if not (additional == 'Y' or additional == 'N'):
    #    print("Incorrect input. Please type 'Y' or 'N'!")
    #    return None
    if additional == 'Y':
        print(f'Set # of murders (Avg: {murder})')
        murder_input = input()
    #    if type(murder_input) != int:
    #        print('Incorrect input. Please try again!')
    #        return None
        print(f'Set # of robberies: (Avg: {robbery})')
        robbery_input = input()
    #    if type(robbery_input) != int:
    #        print('Incorrect input. Please try again!')
    #        return None
        print(f'Set # of assaults: (Avg: {assault})')
        assault_input = input()
    #    if type(assault_input) != int:
    #        print('Incorrect input. Please try again!')
    #        return None

        model_make_prediction(crime_rate_input, pop, murder_input, robbery_input, assault_input)
    
    else:

        model_make_prediction(crime_rate_input, pop)
        
    return None