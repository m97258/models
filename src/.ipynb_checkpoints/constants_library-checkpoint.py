# constants_library.py
# Author: Mauricio
# Date: 30/11/2024
# Purpose: library used by the depression prediction model.

# PARAMETRIZATION OF COLUMN NAMES OF INPUT AND OUTPUT FILES
NAME_NAME_COLUMN                                  = 'Name'
NAME_AGE_COLUMN                                   = 'Age'
NAME_NUMBER_OF_CHILDREN_COLUMN                    = 'Number of Children'
NAME_INCOME_COLUMN                                = 'Income'
NAME_MARITAL_STATUS_COLUMN                        = 'Marital Status'
NAME_EDUCATION_LEVEL_COLUMN                       = 'Education Level'
NAME_SMOKING_STATUS_COLUMN                        = 'Smoking Status'
NAME_PHYSICAL_ACTIVITY_LEVEL_COLUMN               = 'Physical Activity Level'
NAME_EMPLOYMENT_STATUS_COLUMN                     = 'Employment Status'
NAME_ALCOHOL_CONSUMPTION_COLUMN                   = 'Alcohol Consumption'
NAME_DIETARY_HABITS_COLUMN                        = 'Dietary Habits'
NAME_SLEEP_PATTERNS_COLUMN                        = 'Sleep Patterns'
NAME_HISTORY_OF_SUBSTANCE_ABUSE_COLUMN            = 'History of Substance Abuse'
NAME_FAMILY_HISTORY_OF_DEPRESSION_COLUMN          = 'Family History of Depression'
NAME_CHRONIC_MEDICAL_CONDITIONS_COLUMN            = 'Chronic Medical Conditions'
NAME_TARGET_COLUMN                                = 'History of Mental Illness'

# PARAMETRIZATION OF NAMES OF USER CREATED COLUMNS DERIVED FROM NAMES
NAME_FORENAME_COLUMN                              = 'Forename'
NAME_LAST_NAME_COLUMN                             = 'Last Name'
NAME_PREDICTED_GENDER_COLUMN                      = 'Predicted Gender'
NAME_LAST_NAME_ORIGIN_COLUMN                      = 'Last Name Origin'

# PARAMETRIZATION OF NAMES OF USER CREATED COLUMNS FOR BUILDING CATEGORIES OF INCOME VARIABLE
NAME_INCOME_CATEGORY_COLUMN                       = 'Income Category'

# PARAMETRIZATION OF NAMES OF USER CREATED COLUMNS FOR PASSING MEANS AND STDEVS
NAME_VARIABLES_COLUMN                             = 'variables'   # A name of a column used to export mean and stdev
NAME_VALUES_COLUMN                                = 'values'      # A name of a column used to export mean and stdev

# PARAMETRIZATION OF NAMES OF USER CREATED COLUMNS FOR EXPORTING THE PREDICTIONS RUN IN PRODUCTION
NAME_PROBABILITY_TARGET_COLUMN                    = 'mental_illness_probability'

# PARAMETRIZATION OF COLUMN NAMES FOR FINDING PREFIXES AND SUFFIXES
NAME_WORDS_IN_NAME_COLUMN                         = 'Words in name'
NAME_NUMBER_OF_WORDS_IN_NAME_COLUMN               = 'Number of words in name'
NAME_PREFIXES_AND_SUFFIXES_COLUMN                 = 'Prefix/Suffix'
NAME_PREFIXES_AND_SUFFIXES_NUMBER_OF_CASES_COLUMN = 'Number of cases'

# PARAMETRIZATION OF COLUMN NAMES OF ERRORS TABLE
NAME_PROBABILITY_COLUMN                           = 'probability'
NAME_ACCUMULATED_PROBABILITY_COLUMN               = 'accumulated_probability'
NAME_ERROR_ACCUMULATED_I_COLUMN                   = 'error_accumulated_I'
NAME_ERROR_ACCUMULATED_II_COLUMN                  = 'error_accumulated_II'

# PARAMETRIZATION OF FILE NAMES
NAME_FILE_DATA                                    = 'depression_data.csv'
NAME_FILE_LAST_NAMES_ORIGIN                       = 'last_names_origin.csv'
NAME_FILE_INGESTED_XTRAIN_MODELING                = 'Xtrain_modeling.parquet'
NAME_FILE_INGESTED_YTRAIN_MODELING                = 'Ytrain_modeling.parquet'
NAME_FILE_INGESTED_XTRAIN_PRODUCTION              = 'Xtrain_production.parquet'
NAME_FILE_CURRENT_DATA                            = 'interactions_holdout_predictions.parquet'
NAME_FILE_PREDICTIONS                             = 'predictions_file.csv'
NAME_FILE_CATEGORICAL_VARIABLES                   = 'categorical_variables.csv'
NAME_FILE_NON_CATEGORICAL_VARIABLES               = 'non_categorical_variables.csv'
NAME_FILE_XGB                                     = 'xgb_model.pkl'
NAME_FILE_MEAN                                    = 'mean_variables.csv'
NAME_FILE_STD                                     = 'std_variables.csv'
NAME_FILE_DUMMY_VARIABLES                         = 'dummy_variables.csv'
NAME_FILE_ALL_MODEL_COLUMNS                       = 'all_model_columns.csv'

# PARAMETRIZATION OF VALUES OF CATEGORICAL VARIABLES
VALUE_EDUCATION_LEVEL_ASSOCIATE_DEGREE            = 'Associate Degree'
VALUE_EDUCATION_LEVEL_BACHELORS_DEGREE            = 'Bachelor\'s Degree'
VALUE_EDUCATION_LEVEL_HIGH_SCHOOL                 = 'High School'
VALUE_EDUCATION_LEVEL_MASTERS_DEGREE              = 'Master\'s Degree'
VALUE_EDUCATION_LEVEL_PHD                         = 'PhD'
VALUE_EMPLOYMENT_STATUS_EMPLOYED                  = 'Employed'
VALUE_EMPLOYMENT_STATUS_UNEMPLOYED                = 'Unemployed'
VALUE_DIETARY_HABITS_HEALTHY                      = 'Healthy'
VALUE_DIETARY_HABITS_MODERATE                     = 'Moderate'
VALUE_DIETARY_HABITS_UNHEALTHY                    = 'Unhealthy'
VALUE_SLEEP_PATTERNS_FAIR                         = 'Fair'
VALUE_SLEEP_PATTERNS_GOOD                         = 'Good'
VALUE_SLEEP_PATTERNS_POOR                         = 'Poor'

# PARAMETRIZATION OF COLORS FOR GRAPHICS
VALUE_COLOR_CREAM_CALYPSO                         = '#e8f4f0'
VALUE_COLOR_CREAM_YELLOW                          = '#f7ecb0'
VALUE_COLOR_CREAM_PINK                            = '#ffb3e6'
VALUE_COLOR_CREAM_GREEN                           = '#99ff99'
VALUE_COLOR_CREAM_BLUE                            = '#66b3ff'
VALUE_COLOR_CREAM_PURPLE                          = '#c7b3fb'
VALUE_COLOR_CREAM_RED                             = '#ff6666'
VALUE_COLOR_CREAM_ORANGE                          = '#f9c3b7'
VALUE_COLOR_CREAM_BLACK                           = '#525252'

# PARAMETRIZATION FOR BUILDING XGBOOST MODELS
PARAMS_COL_SAMPLE_BY_TREE                         = [0.6]
PARAMS_LEARNING_RATE                              = [0.1]
PARAMS_MAX_DEPTH                                  = [5, 6]
PARAMS_N_ESTIMATORS                               = [1000, 5000]
K_FOLDS                                           = 5

