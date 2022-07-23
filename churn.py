import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import sklearn.model_selection
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
# import telecom dataset into a pandas data frame
df_telco = pd.read_csv("Customer-Churn.csv")

# visualize column names
#OK print(df_telco.columns)

# check unique values of each column
#OK for column in df_telco.columns:
#OK   print('Column: {0} - Unique Values: {1}'.format(column, df_telco[column].unique()))

# summary of the data frame

# df_telco.info()
# ..there is a error in datafrsme that is the column MonthlyCharges is detected as an object type so we
# need to transform this to a numeric value


# transform the column TotalCharges into a numeric data type
df_telco['TotalCharges'] = pd.to_numeric(df_telco['TotalCharges'], errors='coerce')
#df_telco.info() #now its showing MonthlyCharges as an float type but still we have 11 missing data that is notNull

# print(df_telco[df_telco['TotalCharges'].isnull()])


# drop observations with null values
df_telco.dropna(inplace=True)

# drop the customerID column from the dataset
df_telco.drop(columns='customerID', inplace=True)

# unique elements of the PaymentMethod column..there is a big word automatic so we need to delete this
#ok  print(df_telco.PaymentMethod.unique())

# remove (automatic) from payment method names
df_telco['PaymentMethod'] = df_telco['PaymentMethod'].str.replace(' (automatic)', '', regex=False)
#ok 'automatic' removed print(df_telco['PaymentMethod'])

# unique elements of the PaymentMethod column after the modification
# print(df_telco.PaymentMethod.unique())


# Data Visualization
# create a figure
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)

# proportion of observation of each class
prop_response = df_telco['Churn'].value_counts(normalize=True)

# create a bar plot showing the percentage of churn
prop_response.plot(kind='bar',
                   ax=ax,
                   color=['springgreen','salmon'])

# set title and labels
ax.set_title('Proportion of observations of the response variable',
             fontsize=18, loc='left')
ax.set_xlabel('churn',
              fontsize=14)
ax.set_ylabel('proportion of observations',
              fontsize=14)
ax.tick_params(rotation='auto')

# eliminate the frame from the plot
spine_names = ('top', 'right', 'bottom', 'left')
for spine_name in spine_names:
    ax.spines[spine_name].set_visible(False)
#ok plt.show()

# Demographic Information

def percentage_stacked_plot(columns_to_plot, super_title):
    '''
    Prints a 100% stacked plot of the response variable for independent variable of the list columns_to_plot.
            Parameters:
                    columns_to_plot (list of string): Names of the variables to plot
                    super_title (string): Super title of the visualization
            Returns:
                    None
    '''

    number_of_columns = 2
    number_of_rows = math.ceil(len(columns_to_plot) / 2)

    # create a figure
    fig = plt.figure(figsize=(12, 5 * number_of_rows))
    fig.suptitle(super_title, fontsize=22, y=.95)

    # loop to each column name to create a subplot
    for index, column in enumerate(columns_to_plot, 1):

        # create the subplot
        ax = fig.add_subplot(number_of_rows, number_of_columns, index)

        # calculate the percentage of observations of the response variable for each group of the independent variable
        # 100% stacked bar plot
        prop_by_independent = pd.crosstab(df_telco[column], df_telco['Churn']).apply(lambda x: x / x.sum() * 100,
                                                                                     axis=1)

        prop_by_independent.plot(kind='bar', ax=ax, stacked=True,
                                 rot=0, color=['springgreen', 'salmon'])

        # set the legend in the upper right corner
        ax.legend(loc="upper right", bbox_to_anchor=(0.62, 0.5, 0.5, 0.5),
                  title='Churn', fancybox=True)

        # set title and labels
        ax.set_title('Proportion of observations by ' + column,
                     fontsize=16, loc='left')

        ax.tick_params(rotation='auto')

        # eliminate the frame from the plot
        spine_names = ('top', 'right', 'bottom', 'left')
        for spine_name in spine_names:
            ax.spines[spine_name].set_visible(False)


# demographic column names
demographic_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents']

# stacked plot of demographic columns
percentage_stacked_plot(demographic_columns, 'Demographic Information')
# ok # plt.show()

# customer account column names
account_columns = ['Contract', 'PaperlessBilling', 'PaymentMethod']

# stacked plot of customer account columns
percentage_stacked_plot(account_columns, 'Customer Account Information')

#not working! denographics for other data columns:
"""
def histogram_plots(columns_to_plot, super_title):
    '''
    Prints a histogram for each independent variable of the list columns_to_plot.
           Parameters:
                   columns_to_plot (list of string): Names of the variables to plot
                   super_title (string): Super title of the visualization
           Returns:
                   None
   '''
    # set number of rows and number of columns


number_of_columns = 2
number_of_rows = math.ceil(len(columns_to_plot) / 2)

# create a figure
fig = plt.figure(figsize=(12, 5 * number_of_rows))
fig.suptitle(super_title, fontsize=22, y=.95)

# loop to each demographic column name to create a subplot
for index, column in enumerate(columns_to_plot, 1):

    # create the subplot
    ax = fig.add_subplot(number_of_rows, number_of_columns, index)

    # histograms for each class (normalized histogram)
    df_telco[df_telco['Churn'] == 'No'][column].plot(kind='hist', ax=ax, density=True,
                                                     alpha=0.5, color='springgreen', label='No')
    df_telco[df_telco['Churn'] == 'Yes'][column].plot(kind='hist', ax=ax, density=True,
                                                      alpha=0.5, color='salmon', label='Yes')

    # set the legend in the upper right corner
    ax.legend(loc="upper right", bbox_to_anchor=(0.5, 0.5, 0.5, 0.5),
              title='Churn', fancybox=True)

    # set title and labels
    ax.set_title('Distribution of ' + column + ' by churn',
                 fontsize=16, loc='left')

    ax.tick_params(rotation='auto')

    # eliminate the frame from the plot
    spine_names = ('top', 'right', 'bottom', 'left')
    for spine_name in spine_names:
        ax.spines[spine_name].set_visible(False)

# customer account column names
account_columns_numeric = ['tenure', 'MonthlyCharges', 'TotalCharges']
# histogram of costumer account columns
histogram_plots(account_columns_numeric, 'Customer Account Information')

plt.show()
"""

# services column names
services_columns = ['PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                   'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

# stacked plot of services columns
percentage_stacked_plot(services_columns, 'Services Information')
#ok plt.show()



#mutual information
from sklearn.metrics import mutual_info_score
# function that computes the mutual infomation score between a categorical serie and the column Churn
# function that computes the mutual infomation score between a categorical serie and the column Churn
def compute_mutual_information(categorical_serie):
    return mutual_info_score(categorical_serie, df_telco.Churn)

# select categorial variables excluding the response variable
categorical_variables = df_telco.select_dtypes(include=object).drop('Churn', axis=1)

# compute the mutual information score between each categorical variable and the target
feature_importance = categorical_variables.apply(compute_mutual_information).sort_values(ascending=False)

# visualize feature importance
#ok print(feature_importance)


# label encoding
df_telco_transformed = df_telco.copy()

# label encoding (binary variables)
label_encoding_columns = ['gender', 'Partner', 'Dependents', 'PaperlessBilling', 'PhoneService', 'Churn']

# encode categorical binary features using label encoding
for column in label_encoding_columns:
    if column == 'gender':
        df_telco_transformed[column] = df_telco_transformed[column].map({'Female': 1, 'Male': 0})
    else:
        df_telco_transformed[column] = df_telco_transformed[column].map({'Yes': 1, 'No': 0})

#ok print(df_telco_transformed.columns)

# one-hot encoding (categorical variables with more than two levels)
one_hot_encoding_columns = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                            'TechSupport', 'StreamingTV',  'StreamingMovies', 'Contract', 'PaymentMethod']

# encode categorical variables with more than two levels using one-hot encoding
df_telco_transformed = pd.get_dummies(df_telco_transformed, columns = one_hot_encoding_columns)




""""" data normalization:why need? in dataset there are some columns In machine learning, some feature values differ from 
others multiple times. The features with higher values will dominate the learning process; however, it does not mean 
those variables are more important to predict the target. Data normalization transforms multiscaled data to the same scale"""""

#data normalization

# min-max normalization (numeric variables)
min_max_columns = ['tenure', 'MonthlyCharges', 'TotalCharges']

# scale numerical variables using min max scaler
for column in min_max_columns:
        # minimum value of the column
        min_column = df_telco_transformed[column].min()
        # maximum value of the column
        max_column = df_telco_transformed[column].max()
        # min max scaler
        df_telco_transformed[column] = (df_telco_transformed[column] - min_column) / (max_column - min_column)


# algorithm part

# training and test
# select independent variables or target
X = df_telco_transformed.drop(columns='Churn')

# select dependent variables
y = df_telco_transformed.loc[:, 'Churn']

# prove that the variables were selected correctly
#ok print(X.columns)

# prove that the variables were selected correctly
#ok print(y.name)

# use train and test function to spit dataset in two groups

# split the data in training and testing sets

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.55,random_state=40, shuffle=True)
#ok print(X_train, X_test, y_train, y_test)



# models we used here:6



def create_models(seed=2):
    '''
    Create a list of machine learning models.
            Parameters:
                    seed (integer): random seed of the models
            Returns:
                    models (list): list containing the models
    '''

    models = []
    models.append(('dummy_classifier', DummyClassifier(random_state=seed, strategy='most_frequent')))
    models.append(('k_nearest_neighbors', KNeighborsClassifier()))
    models.append(('logistic_regression', LogisticRegression(solver='lbfgs', max_iter=1000,random_state=seed)))
    models.append(('support_vector_machines', SVC(random_state=seed)))
    models.append(('random_forest', RandomForestClassifier(random_state=seed)))
    models.append(('gradient_boosting', GradientBoostingClassifier(random_state=seed)))


    return models



# create a list with all the algorithms we are going to assess
models = create_models()
#ok print(models)

# result for each model

# test the accuracy of each model using default hyperparameters
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    # fit the model with the training data
    model.fit(X_train, y_train).predict(X_test)
    # make predictions with the testing data
    predictions = model.predict(X_test)
    # calculate accuracy
    accuracy = accuracy_score(y_test, predictions)
    # append the model name and the accuracy to the lists
    results.append(accuracy)
    names.append(name)
    # print classifier accuracy
    print('Classifier: {}, Accuracy: {}'.format(name, accuracy))