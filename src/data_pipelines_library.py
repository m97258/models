# data_pipelines_library.py
# Author: Mauricio
# Date: 30/11/2024
# Purpose: library used by the depression prediction model

import pandas as pd
import pickle as pk
import constants_library as constants

# PARAMETRIZATION OF NAMES OF USER CREATED COLUMNS FOR PASSING MEANS AND STDEVS
variables_column                    = constants.NAME_VARIABLES_COLUMN
values_column                       = constants.NAME_VALUES_COLUMN

# PARAMETRIZATION OF PATH AND FILE NAMES
name_file_categorical               = constants.NAME_FILE_CATEGORICAL_VARIABLES
name_file_non_categorical           = constants.NAME_FILE_NON_CATEGORICAL_VARIABLES
name_file_dummy_variables           = constants.NAME_FILE_DUMMY_VARIABLES
name_file_all_model_columns         = constants.NAME_FILE_ALL_MODEL_COLUMNS
name_file_mean                      = constants.NAME_FILE_MEAN
name_file_std                       = constants.NAME_FILE_STD
name_file_xgb                       = constants.NAME_FILE_XGB


def import_model(name_path):
    """
    This function imports all the files that make the depression predictive model:
    Model input files:  - std_variables.csv   - categorical_variables.csv      - dummy_variables.csv    - xgb_model.pkl
                        - mean_variables.csv  - non_categorical_variables.csv  - all_model_columns.csv
    
    Inputs:
    - name_path: the name of the path where the model file is
    Model input files:  - std_variables.csv   - categorical_variables.csv      - dummy_variables.csv    - xgb_model.pkl
                        - mean_variables.csv  - non_categorical_variables.csv  - all_model_columns.csv
    
    Outputs:
    - cols_binary: list of names of columns of binary variables
    - cols_categorical: list of names of columns of categorical variables
    - cols_non_categorical: list of names of columns of non-categorical variables
    - cols_dummy: list of names of columns of dummy variables
    - cols_all_variables: list of names of all the columns of the model, after standardization
    - dfmean: a dataframe with the mean used for standardization of categorical variables
    - dfstd: a dataframe with the standard deviation used for standardization of categorical variables
    - model_xgb: a model object    
    """
    
    cols_categorical     = list(pd.read_csv(name_path + name_file_categorical,       sep = ',', low_memory=False)[values_column])
    cols_non_categorical = list(pd.read_csv(name_path + name_file_non_categorical,   sep = ',', low_memory=False)[values_column])
    cols_dummy           = list(pd.read_csv(name_path + name_file_dummy_variables,   sep = ',', low_memory=False)[values_column])
    cols_all_variables   = list(pd.read_csv(name_path + name_file_all_model_columns, sep = ',', low_memory=False)[values_column])
    dfmean = pd.read_csv(name_path + name_file_mean, sep = ',', low_memory=False)
    dfstd  = pd.read_csv(name_path + name_file_std, sep = ',', low_memory=False)
    model_xgb = pk.load(open(name_path + name_file_xgb, "rb"))
    return(cols_categorical, cols_non_categorical, cols_dummy, cols_all_variables, dfmean, dfstd, model_xgb)


def loading_dataset(name_path, name_file):
    """
    This function loads a file into a dataframe. 
    
    Inputs:
    - name_path: the name of the path where the dataset is
    - name_file: the name of the csv file
    
    Outputs:
    - a dataframe
    """
    
    if (name_file[-4:] == '.csv'):
        df = pd.read_csv(name_path + name_file, sep = ',', low_memory=False)
    if (name_file[-8:] == '.parquet'):
        df = pd.read_parquet(name_path + name_file, engine='pyarrow')
    return(df)


def storing_dataset(df, name_path, name_file):
    """
    This function stores a dataframe into a file. 
    
    Inputs:
    - df: a dataframe
    - name_path: the name of the path where the dataset is
    - name_file: the name of the csv file
    
    Outputs:
    - a file storing the data of the dataframe df
    """
    
    if (name_file[-4:] == '.csv'):
        df.to_csv(name_path + name_file, sep = ',')
    if (name_file[-8:] == '.parquet'):
        df.to_parquet(name_path + name_file, engine='pyarrow')


def remove_discarded_columns(df, columns_to_be_removed):
    """
    This function remove discarded columns from a dataframe. 
    
    Inputs:
    - df: a dataframe
    - columns_to_be_removed: a list of columns to be removed
    
    Outputs:
    - a dataframe with the columns specified removed.
    """
    
    columns_df = df.columns
    columns_pruned = list(columns_to_be_removed)
    for i in range(0,len(columns_to_be_removed)):
        not_found = True
        for j in range(0,len(columns_df)):
            if (columns_df[j] == columns_to_be_removed[i]):
                not_found = False
        if (not_found):
            columns_pruned.remove(columns_to_be_removed[i])
    df = df.drop(columns_pruned, axis=1)
    return(df)


def standardizing_dataset_production(df, cols_categorical, cols_non_categorical, cols_dummy, cols_all_variables, \
                                     means_non_categorical, std_non_categorical):
    """
    This function creates a dataframe with indicator variables for categorical variables and normalized variables for 
    non-categorical variables. For that standardization it uses the means and standard deviations computed during modeling. 
    
    Inputs:
    - df: a dataframe
    - cols_categorical: a list of columns deemed to be categorical
    - cols_non_categorical: a list of columns deemed to be non-categorical
    - cols_dummy: a list of column names of dummy variables
    - cols_all_variables: a list of all the columns of the model, after standardization, in the right order
    - means_non_categorical: a dataframe with the means used for standardizing the non-categorical variables of the model
    - std_non_categorical: a dataframe with the stdev used for standardizing the non-categorical variables of the model
    
    Outputs:
    - a stardardized dataframe
    """
    
    # Creation of normalized variables and indicator variables
    dfnorm = pd.DataFrame()
    # Feature transformations: creation of normalized variables for non-categorical variables
    for col in cols_non_categorical:
        mean_value = float(means_non_categorical[means_non_categorical[variables_column] == col][values_column])
        std_value  = float(std_non_categorical[std_non_categorical[variables_column] == col][values_column])
        dfnorm[col] = (df[col] - mean_value)/std_value
    # Feature transformations: creation of indicator variables for categorical variables
    for col in cols_categorical:
        dfnorm[col] = df[col]
        dfnorm = pd.get_dummies(dfnorm, columns=[col])
    new_dummy_variables = [item for item in dfnorm.columns if item not in cols_categorical]
    new_dummy_variables = [item for item in new_dummy_variables if item not in cols_non_categorical]
    # Here the new_dummy_variables are corrected, in order to get exactly the dummy variables built during the modeling stage
    dummy_variables_that_were_previously_discarded = []
    for col in new_dummy_variables:
        if (col not in cols_dummy):
            dummy_variables_that_were_previously_discarded = dummy_variables_that_were_previously_discarded + [col] 
    dfnorm = remove_discarded_columns(dfnorm, dummy_variables_that_were_previously_discarded)
    for col in cols_dummy:
        if (col not in dfnorm.columns):
            dfnorm[col] = 0     
    return(dfnorm)
    