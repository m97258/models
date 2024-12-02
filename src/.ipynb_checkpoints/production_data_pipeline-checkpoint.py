# production_data_pipeline.py
# Author: Mauricio
# Date: 30/11/2024
# Purpose: data pipeline that runs in production a predictive model for depression
# Instructions for running it:
#      This program can be run either with no parameter or with the name of the path and the input files, as shown below:
#      python production_data_pipeline.py
#      python production_data_pipeline.py ./ depression_data.csv
#
#  --------------- Production data pipeline ---------------
#
#    - Importing the Model
#    - Data Collection from Data Sources: Extracting the data from the data sources                 ---- This is used in Modeling data pipeline too
#    - Data Transformation/Processing:                                                              ---- This is used in Modeling data pipeline too
#            Data Standardization                                                                   ---- This is used in Modeling data pipeline too
#    - Data Ingestion: The data on which the model would be used is properly stored                 ---- This is used in Modeling data pipeline too
#    - Running Model in Production: Executing the model in production on new data
#    - Data Delivered at Destination Point for Consumption

import sys
import constants_library as constants
import data_pipelines_library as data_pipelines_library 

# PARAMETRIZATION OF NAMES OF USER CREATED COLUMNS FOR EXPORTING THE PREDICTIONS RUN IN PRODUCTION
probability_target_column = constants.NAME_PROBABILITY_TARGET_COLUMN

# PARAMETRIZATION OF FILE NAMES
default_name_file_current_data      = constants.NAME_FILE_DATA
default_name_file_Xtrain_production = constants.NAME_FILE_INGESTED_XTRAIN_PRODUCTION


default_name_path = 'C:/Users/T4925/PythonCode/AXA/data/'


def run_production_data_pipeline(name_path, name_file_current_data):
    """
    This function runs the production data pipeline that gets the probabilities of depression of a series of people.
    
    Inputs:
    - name_path:              the path where the input files are
    - name_file_current_data: the name of the input file with the details of current data
    
    Outputs:
    - a dataframe with the details of the file 'name_file_current_data' updated with probabilities of the depression. That dataframe is saved
      updating the original 'name_file_current_data' file.
    """
    
    # Importing the model
    cols_categorical, cols_non_categorical, cols_dummy, cols_all_variables, dfmean, dfstd, model_xgb = data_pipelines_library.import_model(name_path)

    # Data Collection from Data Sources: Extracting the data from the data sources
    df_current_data = data_pipelines_library.loading_dataset(name_path, name_file_current_data)

    # Data Transformation/Processing: Data Standardization
    Xcurrent_data_norm = data_pipelines_library.standardizing_dataset_production(df_current_data, cols_categorical, cols_non_categorical, \
                                                                                 cols_dummy, cols_all_variables, dfmean, dfstd)

    # Data Ingestion: The data on which the model would be used is properly stored
    data_pipelines_library.storing_dataset(Xcurrent_data_norm, name_path, default_name_file_Xtrain_production)

    # Running Model in Production: Executing the model in production on new data
    df_current_data[probability_target_column] = model_xgb.predict_proba(Xcurrent_data_norm)[:, 1]

    # Data Delivered at Destination Point for Consumption: The data is delivered at destination
    data_pipelines_library.storing_dataset(df_current_data, name_path, name_file_current_data)

    return (df_current_data)


def main(args):
    for i, arg in enumerate(args):
        if (i == 2):
            name_path = arg
        else:
            name_path = default_name_path
        if (i == 3):
            name_file_current_data = arg
        else:
            name_file_current_data = default_name_file_current_data

    predictions = run_production_data_pipeline(name_path, name_file_current_data)


if __name__ == "__main__":
    main(sys.argv)
