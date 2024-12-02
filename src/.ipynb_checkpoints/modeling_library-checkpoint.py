# modeling_library.py
# Author: Mauricio
# Date: 30/11/2024
# Purpose: library used by the depression prediction model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from xgboost import XGBClassifier
import pickle as pk
import constants_library as constants
import performance_library as performance_library

# PARAMETRIZATION OF NAMES OF USER CREATED COLUMNS FOR PASSING MEANS AND STDEVS
variables_column                    = constants.NAME_VARIABLES_COLUMN
values_column                       = constants.NAME_VALUES_COLUMN

# PARAMETRIZATION OF COLUMN NAMES OF ERRORS TABLE
name_accumulated_probability_column = constants.NAME_ACCUMULATED_PROBABILITY_COLUMN
name_error_accumulated_I_column     = constants.NAME_ERROR_ACCUMULATED_I_COLUMN
name_error_accumulated_II_column    = constants.NAME_ERROR_ACCUMULATED_II_COLUMN

# PARAMETRIZATION OF FILE NAMES
name_file_categorical               = constants.NAME_FILE_CATEGORICAL_VARIABLES
name_file_non_categorical           = constants.NAME_FILE_NON_CATEGORICAL_VARIABLES
name_file_dummy_variables           = constants.NAME_FILE_DUMMY_VARIABLES
name_file_all_model_columns         = constants.NAME_FILE_ALL_MODEL_COLUMNS
name_file_mean                      = constants.NAME_FILE_MEAN
name_file_std                       = constants.NAME_FILE_STD
name_file_xgb                       = constants.NAME_FILE_XGB

# PARAMETRIZATION FOR BUILDING XGBOOST MODELS
params_col_sample_by_tree           = constants.PARAMS_COL_SAMPLE_BY_TREE
params_learning_rate                = constants.PARAMS_LEARNING_RATE
params_max_depth                    = constants.PARAMS_MAX_DEPTH                 
params_n_estimators                 = constants.PARAMS_N_ESTIMATORS
k_folds                             = constants.K_FOLDS


def trainvalidation_testing_split(df, column_target, percentage):
    """
    This function splits the data in a "training-validation" dataset and a "testing" dataset.
    
    Inputs:
    - df: the dataframe with the data to be split
    - column_target: the column name of the target variable in the dataset
    - percentage: percentage of the data to be allocated to the testing dataset
    
    Outputs:
    - four dataframes, two with the training-validation data and two with the testing data
    """
    
    X = df
    Y = X[column_target]
    Xtrainval, Xtest, Ytrainval, Ytest = train_test_split(X, Y, test_size=percentage)
    cols = [col for col in df.columns if col not in [column_target]]
    Xtrainval_result = Xtrainval[cols]
    Xtest_result     = Xtest[cols]
    return (Xtrainval_result, Xtest_result, Ytrainval, Ytest)


def standardizing_dataset_modeling(df, cols_categorical, cols_non_categorical):
    """
    This function creates a dataframe with indicator variables for categorical variables and normalized variables for  non-categorical 
    variables. It computes the means and standard deviations that will have to be used by the standardizations run during production. 
    
    Inputs:
    - df: a dataframe
    - cols_categorical: a list of columns deemed to be categorical
    - cols_non_categorical: a list of columns deemed to be non-categorical
    
    Outputs:
    - a stardardized dataframe
    - a list of column names of dummy variables
    - a list of column names of all variables of the standardized dataset
    - a dataframe for the means of non-categorical variables
    - a dataframe for the stdev of non-categorical variables
    """

    dfnorm = pd.DataFrame()
    for col in cols_non_categorical: # Creation of normalized variables for non-categorical variables
        dfnorm[col] = (df[col] - df[col].mean())/df[col].std()
    for col in cols_categorical:     # Creation of indicator variables for categorical variables
        dfnorm[col] = df[col]
        dfnorm = pd.get_dummies(dfnorm, columns=[col])
    cols_dummy = [item for item in dfnorm.columns if item not in cols_categorical]
    cols_dummy = [item for item in cols_dummy if item not in cols_non_categorical]
    cols_all_variables = dfnorm.columns
    means_non_categorical = pd.DataFrame(df[cols_non_categorical].mean())
    means_non_categorical = means_non_categorical.reset_index()
    means_non_categorical.columns = [variables_column, values_column]
    std_non_categorical = pd.DataFrame(df[cols_non_categorical].std())
    std_non_categorical = std_non_categorical.reset_index()
    std_non_categorical.columns = [variables_column, values_column]
    return(dfnorm, cols_dummy, cols_all_variables, means_non_categorical, std_non_categorical)


def gridsearch_XGBoost(Xtrain, Ytrain):
    """
    This function performs a grid search on different sets of parameters of a XGBoost model and returns the best paramaters together with
    the average accuracy score on the training and validation dataset during crossvalidation. 

    Inputs:
    - Xtrain: training design matrix
    - Ytrain: target values for training
    
     Outputs:
    - a dictionary with a summary of the results of the grid search, including the best parameters found and the avergare gini score on
      the training and validation datasets
    """
    result = {}
    kf = KFold(n_splits=k_folds, shuffle=False)
    previous_avg_gini_val = 0
    for col_sample_bytree_value in params_col_sample_by_tree:
        for learning_rate_value in params_learning_rate:
            for max_depth_value in params_max_depth:
                for n_estimators_value in params_n_estimators:
                    model = XGBClassifier(colsample_bytree = col_sample_bytree_value, learning_rate = learning_rate_value, \
                                          max_depth = max_depth_value, n_estimators = n_estimators_value, use_label_encoder = False, \
                                          disable_default_eval_metric = True, verbosity = 0)
                    avg_accuracy_val   = 0
                    avg_accuracy_train = 0
                    for indexes_Xtrain_crossval, indexes_Xval_crossval in kf.split(Xtrain):
                        Xtrain_crossval = Xtrain.iloc[indexes_Xtrain_crossval]
                        Ytrain_crossval = Ytrain.iloc[indexes_Xtrain_crossval]
                        Xval_crossval   = Xtrain.iloc[indexes_Xval_crossval]
                        Yval_crossval   = Ytrain.iloc[indexes_Xval_crossval]
                        model.fit(Xtrain_crossval, Ytrain_crossval)
                        Ypredprobval    = model.predict(Xval_crossval)
                        Ypredprobtrain  = model.predict(Xtrain_crossval)
                        avg_accuracy_train += performance_library.accuracy_score(Ytrain_crossval, Ypredprobtrain)/k_folds
                        avg_accuracy_val   += performance_library.accuracy_score(Yval_crossval, Ypredprobval)/k_folds
                    if (avg_accuracy_val > previous_avg_accuracy_val):
                        result['col_sample_bytree']  = col_sample_bytree_value
                        result['learning_rate']      = learning_rate_value
                        result['max_depth']          = max_depth_value
                        result['n_estimators']       = n_estimators_value
                        result['avg_accuracy_train'] = avg_accuracy_train
                        result['avg_accuracy_val']   = avg_accuracy_val  
                        previous_avg_accuracy_val    = avg_accuracy_val 
    return(result)
    

def export_model(name_path, cols_categorical, cols_non_categorical, cols_dummy, cols_all_variables, dfmean, dfstd, xgb_model):
    """
    This function exports all the output files that make the predictive model. 
    
    Inputs:
    - name_path: the name of the path where the dataset is
    - cols_categorical: list of names of columns of categorical variables
    - cols_non_categorical: list of names of columns of non-categorical variables
    - cols_dummy: list of names of columns of dummy variables
    - cols_all_variables: list of names of all the columns of the model, after standardization
    - dfmean: a dataframe with the mean used for standardization of non-categorical variables
    - dfstd: a dataframe with the standard deviation used for standardization of non-categorical variables
    - xgb_model: a model object
    
    Outputs:
    Model output files: - std_variables.csv   - categorical_variables.csv      - dummy_variables.csv    - xgb_model.pkl  
                        - mean_variables.csv  - non_categorical_variables.csv  - all_model_columns.csv 
    """
    
    pd.DataFrame(cols_categorical, columns = [values_column]).to_csv(name_path + name_file_categorical, index = False, )
    pd.DataFrame(cols_non_categorical, columns = [values_column]).to_csv(name_path + name_file_non_categorical, index = False)
    pd.DataFrame(cols_dummy, columns = [values_column]).to_csv(name_path + name_file_dummy_variables, index = False)
    pd.DataFrame(cols_all_variables, columns = [values_column]).to_csv(name_path + name_file_all_model_columns, index = False)
    dfmean.to_csv(name_path + name_file_mean, index = False)
    dfstd.to_csv(name_path + name_file_std, index = False)
    pk.dump(xgb_model, open(name_path + name_file_xgb, "wb"))

    
def build_errors_table(Ranking_probabilities_main_class, Ranking_probabilities_secondary_class, splits):
    """
    It buils a table of errors type I and II of the predictions made by a classification model. 

    Inputs:
    - Ranking_probabilities_main_class: this is a ranking of data points truly belonging to the 'main class', where the ranking score is
      the probability computed by the model for a data point to belong to that 'main class'.
    - Ranking_probabilities_secondary_class: this is a ranking of data points truly belonging to the 'secondary class', where the ranking
      score is the probability computed by the model for a data point to belong to the 'main class'.
    - splits: the number of splits of the horizontal variable for displaying purposes
    
     Outputs:
    - a dataframe with the errors table
    """
    
    accumulated_probability = 0
    error_accumulated_I     = 0
    error_accumulated_II    = 1
    number_datapoints_main_class   = Ranking_probabilities_main_class.shape[0]
    number_datapoints_second_class = Ranking_probabilities_secondary_class.shape[0]
    errors_table = pd.DataFrame()
    errors_table[name_accumulated_probability_column] = [0]*splits
    errors_table[name_error_accumulated_I_column]     = [0]*splits
    errors_table[name_error_accumulated_II_column]    = [0]*splits
    index_ranking_I  = 0
    index_ranking_II = 0
    for i in range(0, splits):
        accumulated_probability += 1/splits
        while ((index_ranking_I < number_datapoints_main_class) and \
               (Ranking_probabilities_main_class.iloc[index_ranking_I] < accumulated_probability)):
            error_accumulated_I += 1/number_datapoints_main_class
            index_ranking_I += 1
        while ((index_ranking_II < number_datapoints_second_class) and \
               (Ranking_probabilities_secondary_class.iloc[index_ranking_II] < accumulated_probability)): 
            error_accumulated_II -= 1/number_datapoints_second_class
            index_ranking_II += 1
        errors_table.iloc[i] = [accumulated_probability, error_accumulated_I, error_accumulated_II]
    errors_table[name_accumulated_probability_column] = errors_table[name_accumulated_probability_column]*100
    errors_table[name_error_accumulated_I_column]     = errors_table[name_error_accumulated_I_column]*100
    errors_table[name_error_accumulated_II_column]    = errors_table[name_error_accumulated_II_column]*100
    return(errors_table)