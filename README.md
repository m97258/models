# models
#  -------------------------------------------------------------------------------------------------------------------------------------
#  ----- README - Depression Model -----------------------------------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------------------------------------------------------
# Author: Mauricio
# Date: 30/11/2024
# Purpose: application that gets predictions of depression
#
# The model provides:
#    - a production data pipeline, already exported to run the model in other environment, either in Unix or Windows.
#    - a modeling data pipeline, to reproduce the model development process, including nice graphics in a Jupyter Notebook.
#    - different libraries for different purposes (modeling library, visualization library, performance library, constants library and
#      data pipelines library).
#    - a Gradient Boosting Machine model, that runs in production.
#    - a data engineering, already embedded in the production data pipeline.
#
#  -------------------------------------------------------------------------------------------------------------------------------------
#  ----- Running the model in production using the command line -------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------------------------------------------------------
#
# You can run the model in production, either in Unix or in Windows, and getting predictions on new data points. This program can be run
# either with no parameter or with the name of the path and the input file, as shown below, or even with the name of the path, the input
# file, the output path for exported test results and the name of the file for exporting test results:
#
#      python production_data_pipeline.py
#      python production_data_pipeline.py ./data depression_data.csv
#      python production_data_pipeline.py ./data depression_data.csv ./test Xtest_exported.csv
#
#
#  -------------------------------------------------------------------------------------------------------------------------------------
#  ----- Running the model in production by means of including it in a Python application -----------------------------------------------
#  -------------------------------------------------------------------------------------------------------------------------------------
#
# To run the model in production including its code in Python, include the following code in the system that will use the
# data in that environment:
#
# import sys
# import constants_library as constants
# import production_data_pipeline as production_data_pipeline
#
# PARAMETRIZATION OF NAMES OF USER CREATED COLUMNS FOR EXPORTING THE PREDICTIONS RUN IN PRODUCTION
# probability_target_column = constants.NAME_PROBABILITY_TARGET_COLUMN
#
# PARAMETRIZATION OF FILE NAMES
# default_name_file_current_data      = constants.NAME_FILE_DATA
# default_name_file_Xtrain_production = constants.NAME_FILE_INGESTED_XTRAIN_PRODUCTION
# default_name_exported_predictions   = constants.NAME_FILE_EXPORTED_XTEST_RESULTS
#
# # ---- name_path ./.....                -- Your own current path
# # ---- name_file_current_data ./.....   -- Your own file name for storing the input data
# # ---- name_path_test ./.....           -- Your own file name for storing the model results
# df_test_results = production_data_pipeline.run_production_data_pipeline(name_path, name_file_current_data, name_path_test)
#
#  -------------------------------------------------------------------------------------------------------------------------------------
#  ----- Model files -------------------------------------------------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------------------------------------------------------
#
# The model uses a series of files, which are detailed below:
#
# Data pipelines code files: - modeling_data_pipeline.ipynb - production_data_pipeline.py
#
# Libraries code files:      - constants_library.py         - modeling_library.py            - data_pipelines_library.py
#                            - visualization_library.py     - performance_library.py
#
# Model output files:        - xgb_model.pkl                - categorical_variables.csv      - dummy_variables.csv
#                            - std_variables.csv            - non_categorical_variables.csv  - all_model_columns.csv
#                            - mean_variables.csv           - dummy_variables.csv
#
#  -------------------------------------------------------------------------------------------------------------------------------------
#  ----- Data input files required by the model in production --------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------------------------------------------------------
#
# The structure of the input files on which the model will feed should be like the files shown below:
#
#    - depression_data.csv
#
#  -------------------------------------------------------------------------------------------------------------------------------------
#  ----- Production data pipeline ------------------------------------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------------------------------------------------------
#
# The production data pipeline is given in the file 'production_data_pipeline.py' and also in the file 'production_data_pipeline.ipynb'.
# The production data pipeline stages are shown below:
#
#    - Importing the Model
#    - Data Collection from Data Sources: Extracting the data from the data sources
#    - Data Transformation/Processing:
#            Data Standardization
#    - Data Ingestion: The data on which the model would be used is properly stored
#    - Running Model in Production: Executing the model in production on new data
#    - Data Delivered at Destination Point for Consumption
#
#  -------------------------------------------------------------------------------------------------------------------------------------
#  ----- Modeling data pipeline --------------------------------------------------------------------------------------------------------
#  -------------------------------------------------------------------------------------------------------------------------------------
#
# The modeling data pipeline is given in the file Jupyter notebook 'modeling_data_pipeline.ipynb'. # The production data pipeline stages
# are shown below:
#
#    - Data Collection from Data Sources:
#    - Exploratory Analysis
#    - Data Transformation/Processing:
#            Splitting the dataset in Trainvalidation and Test datasets
#            Data Standardization
#            Removing features
#    - Data Ingestion: The data on which the model would be used is properly stored
#    - Modeling
#    - Model Evaluation
#    - Model Deployment: Exporting model at destination point (consumption)