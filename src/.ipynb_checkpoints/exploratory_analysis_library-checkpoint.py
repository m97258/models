# exploratory_analysis_library.py
# Author: Mauricio
# Date: 30/11/2024
# Purpose: library used to run exploratory analysis on the depression dataset

import pandas as pd
import gender_guesser.detector as gender
from collections import Counter
import constants_library as constants

# PARAMETRIZATION OF COLUMN NAMES FOR FINDING PREFIXES AND SUFFIXES
words_in_name_column                              = constants.NAME_WORDS_IN_NAME_COLUMN
number_words_in_name_column                       = constants.NAME_NUMBER_OF_WORDS_IN_NAME_COLUMN
name_prefixes_and_suffixes_column                 = constants.NAME_PREFIXES_AND_SUFFIXES_COLUMN
name_prefixes_and_suffixes_number_of_cases_column = constants.NAME_PREFIXES_AND_SUFFIXES_NUMBER_OF_CASES_COLUMN

def discover_prefixes_and_suffixes_in_name(df, name_column):
    """
    This function iscovers prefixes and suffixes from the names in a column of a dataframe.
    
    Inputs:
    - df: a dataframe with a column with names
    - name_column: the name of the column that holds names of persons
    
    Outputs:
    - a dataframe with the prefixes and suffixes of names from the column that originally held those names.
    """  
    df[words_in_name_column]        = df[name_column].str.split(' ')
    df[number_words_in_name_column] = df[words_in_name_column].apply(lambda x: len(x))

    set_of_words = set()
    for x in df[words_in_name_column]:
        set_of_words.update(x)
        
    counter_of_words = Counter({}) # Creating a counter of words, initially set to zero for every word
    for x in set_of_words:
        counter_of_words[x] = 0
        
    # Counting the occurrences of words in combinations of three words (forename, last name and a third word that is either 
    # a prefix or a suffix
    for index, row in df[df[number_words_in_name_column] > 2].iterrows():
        for i in range(len(row[words_in_name_column])):
            if (i == 0 or i == (len(row[words_in_name_column]) - 1)):
                word_to_be_updated = row[words_in_name_column][i]
                counter_of_words[word_to_be_updated] = counter_of_words[word_to_be_updated] + 1
    
    # Discarding the words that were observed in combinations of two words in a name (forename and last name)
    for index, row in df[df[number_words_in_name_column] == 2].iterrows():
        for i in range(len(row[words_in_name_column])):
            if (i == 0 or i == (len(row[words_in_name_column]) - 1)):
                word_to_be_updated = row[words_in_name_column][i]
                counter_of_words[word_to_be_updated] = 0

    df = df.drop([words_in_name_column, number_words_in_name_column], axis=1)
    df_prefixes_and_suffixes_in_name = pd.DataFrame(counter_of_words.items(), columns=[name_prefixes_and_suffixes_column, \
                                                                                       name_prefixes_and_suffixes_number_of_cases_column])
    df_prefixes_and_suffixes_in_name = df_prefixes_and_suffixes_in_name[df_prefixes_and_suffixes_in_name[name_prefixes_and_suffixes_number_of_cases_column]>0]
    return(df_prefixes_and_suffixes_in_name)


def remove_prefixes_and_suffixes(df, name_column):
    """
    This function removes prefixes and suffixes from the names in a column of a dataframe.
    
    Inputs:
    - df: a dataframe with a column with names
    - name_column: the name of the column that holds names of persons
    
    Outputs:
    - a dataframe with the prefixes and suffixes of names removed from the column that originally held those names.
    """  
    
    df[name_column] = df[name_column].str.replace('Dr. ', '')
    df[name_column] = df[name_column].str.replace(' Jr.', '')
    df[name_column] = df[name_column].str.replace(' DSS', '')
    df[name_column] = df[name_column].str.replace('Mr. ', '')
    df[name_column] = df[name_column].str.replace(' MD', '')
    df[name_column] = df[name_column].str.replace(' DVM', '')
    df[name_column] = df[name_column].str.replace('Mrs. ', '')
    df[name_column] = df[name_column].str.replace('Miss ', '')
    df[name_column] = df[name_column].str.replace(' PhD', '')
    df[name_column] = df[name_column].str.replace('Ms. ', '')
    df[name_column] = [x[:-3] if x[-3:] == ' II'  else x for x in df['Name']]
    df[name_column] = [x[:-4] if x[-4:] == ' III' else x for x in df['Name']]
    df[name_column] = [x[:-3] if x[-3:] == ' IV'  else x for x in df['Name']]
    df[name_column] = [x[:-2] if x[-2:] == ' V'   else x for x in df['Name']]
    return(df)


def predict_gender(name):
    """
    This function predicts the gender of a person based on his or her name.
    
    Inputs:
    - name: the name of a person
    
    Outputs:
    - a string with the predicted gender of a person.
    """  
    
    d = gender.Detector()
    guessed_gender = d.get_gender(name)
    if   guessed_gender == 'mostly_male':
        return 'male'
    elif guessed_gender == 'mostly_female':
        return 'female'
    elif guessed_gender == 'andy':
        return 'unknown'    
    else:
        return guessed_gender

def get_genders(df, forename_column, predicted_gender_column):
    """
    This function predicts the genders of the persons in a dataframe given their forenames.
    
    Inputs:
    - df: a dataframe with a collection of data of people
    - forename_column: the name of the column that holds the forenames.
    - predicted_gender_column: the name of the column that will hold the predicted gender.
    
    Outputs:
    - a datafrane with a column for the predicted gender of every person in the dataset.
    """  

    df_forenames = pd.DataFrame()
    df_forenames[forename_column] = df[forename_column].unique()
    df_forenames[predicted_gender_column]=df_forenames[forename_column].apply(lambda x: predict_gender(x))
    df = pd.merge(df, df_forenames, how='inner', on=[forename_column])
    return(df)
