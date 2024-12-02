# visualization_library.py
# Author: Mauricio
# Date: 30/11/2024
# Purpose: library used by the depression prediction model

import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import constants_library as constants

import performance_library as performance_library
import modeling_library as modeling_library

# PARAMETRIZATION OF COLUMN NAMES OF INPUT AND OUTPUT FILES
name_column                         = constants.NAME_NAME_COLUMN
age_column                          = constants.NAME_AGE_COLUMN
number_of_children_colum            = constants.NAME_NUMBER_OF_CHILDREN_COLUMN
target_column                       = constants.NAME_TARGET_COLUMN

# PARAMETRIZATION OF NAMES OF USER CREATED COLUMNS DERIVED FROM NAMES
forename_column                     = constants.NAME_FORENAME_COLUMN
last_name_column                    = constants.NAME_LAST_NAME_COLUMN
predicted_gender_column             = constants.NAME_PREDICTED_GENDER_COLUMN
last_name_origin_column             = constants.NAME_LAST_NAME_ORIGIN_COLUMN

# PARAMETRIZATION OF NAMES OF USER CREATED COLUMNS FOR BUILDING CATEGORIES OF INCOME VARIABLE
income_category_column              = constants.NAME_INCOME_CATEGORY_COLUMN

# PARAMETRIZATION OF COLORS FOR GRAPHICS
value_color_cream_calypso           = constants.VALUE_COLOR_CREAM_CALYPSO
value_color_cream_yellow            = constants.VALUE_COLOR_CREAM_YELLOW
value_color_cream_pink              = constants.VALUE_COLOR_CREAM_PINK
value_color_cream_green             = constants.VALUE_COLOR_CREAM_GREEN
value_color_cream_blue              = constants.VALUE_COLOR_CREAM_BLUE
value_color_cream_purple            = constants.VALUE_COLOR_CREAM_PURPLE
value_color_cream_red               = constants.VALUE_COLOR_CREAM_RED
value_color_cream_orange            = constants.VALUE_COLOR_CREAM_ORANGE
value_color_cream_black             = constants.VALUE_COLOR_CREAM_BLACK

def plot_imbalance(df, target_column, title):
    """
    This function shows the imbalance in a binary target variable.
    
    Inputs:
    - df: a dataframe with a column with the values of the target variable
    - target_column: the name of the column of the dataframe with the values of the target variable
    - title: the title of the graph
    
    Outputs:
    - a bar chart and a pie chart displayed on the screen
    """

    font_color = value_color_cream_black
    colors = [value_color_cream_yellow, value_color_cream_pink]
    sizes  = df[target_column].value_counts(dropna = True)
    target_labels2 = df[target_column].value_counts(dropna = True).index
    explode    = (0.1, 0)  # explode 1st slice
    fig = plt.figure(figsize=(8, 4), facecolor=value_color_cream_calypso)
    fig.tight_layout(pad=3.0)
    figL, figR = fig.subfigures(1, 2)
    figL.set_facecolor(value_color_cream_calypso)  
    figL.suptitle('Split of the Sample', size = 16, color=font_color, weight="bold")
    axL = figL.subplots(sharex=True)
    width = 0.7
    barlist = axL.bar(sizes.index, pd.Series(sizes), width=0.6)
    barlist[0].set_color(colors[0])
    barlist[1].set_color(colors[1]) 
    barlist[0].set_edgecolor(value_color_cream_black)
    barlist[1].set_edgecolor(value_color_cream_black)
    axL.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
    y_title = 'Number of people'
    axL.set_ylabel(y_title, color=font_color, fontsize=16)
    x_ticks = list(pd.Series(sizes).index)
    axL.set_xticklabels(x_ticks,color=font_color, fontsize=16)
    figR.set_facecolor(value_color_cream_calypso)
    axR = figR.subplots(sharey=True)
    wedges, texts, autotexts = axR.pie(sizes, explode=explode, labels=target_labels2, startangle=30, colors=colors, autopct='%1.1f%%', shadow=True, \
                                   textprops={'color':font_color})
    plt.setp(texts, size=16)
    plt.setp(autotexts, size=16)
    axR.set_title(title, size = 18, color=font_color, weight="bold")


def plot_categorical_variable_versus_target(df_data, categorial_variable, target_column):
    """
    This function shows bar charts and a pie charts with the percentages of the different values of the target variable for the different values of
    the column 'categorical_variable' of the dataframe 'df_data'
    
    Inputs:
    - df_data: a dataframe with a column with categorical variables and another column with the target variable
    - target_column: the name of the column of the dataframe with the values of the target variable
    
    Outputs:
    - bar charts and a pie chart displayed on the screen
    """

    font_color = value_color_cream_black
    colors = [value_color_cream_yellow, value_color_cream_pink]
    df = df_data[[categorial_variable, target_column]]
    df_grouped = df.groupby([categorial_variable, target_column]).size().reset_index()
    table = pd.pivot_table(df_grouped, index=[categorial_variable], columns=[target_column])
    table = table.dropna(axis=1).astype(int)
    number_of_subgraphics = table.shape[0]
    if (number_of_subgraphics == 2):
        num_columns = 3
        fig, axes = plt.subplots(1, 4, figsize=(11, 4), facecolor=value_color_cream_calypso)
        fig.delaxes(ax= axes[3])
        legend_bbox_to_anchor = (1, 0.7)
    elif (number_of_subgraphics == 3):
        num_columns = 2
        fig, axes = plt.subplots(2, 3, figsize=(11, 6), facecolor=value_color_cream_calypso)
        fig.delaxes(ax= axes[1,2])
        legend_bbox_to_anchor = (1.6, 0.7)
    elif (number_of_subgraphics == 4): 
        num_columns = 3
        fig, axes = plt.subplots(2, 4, figsize=(11, 6), facecolor=value_color_cream_calypso)
        fig.delaxes(ax= axes[1,2])
        fig.delaxes(ax= axes[1,3])
        legend_bbox_to_anchor = (1.2, 0.7)
    elif (number_of_subgraphics == 5): 
        num_columns = 3
        fig, axes = plt.subplots(2, 4, figsize=(11, 6), facecolor=value_color_cream_calypso)
        fig.delaxes(ax= axes[1,3])
        legend_bbox_to_anchor = (1.2, 0.7)
        
    fig.tight_layout(pad=3.0)
    title = fig.suptitle(target_column + ' vs ' +  categorial_variable, y=.95, fontsize=15, color=font_color, fontweight='bold')
    sublots_adjust_top = 0.85
    subplots_adjust_bottom = 0.1   
    for i, (idx, row) in enumerate(table.iterrows()):
        if (number_of_subgraphics == 2):
            ax = axes[i + 1]
        else:
            ax = axes[i // num_columns, (i % num_columns) + 1]
        row = row[row.gt(row.sum() * .01)]
        patches, texts, autotexts = ax.pie(row, labels=['',''], startangle=30, wedgeprops=dict(width=.5), colors=colors, autopct='%1.1f%%', \
                                           shadow=True, textprops={'color':font_color})
        for autotext in autotexts: # Customizing percent labels
            autotext.set_horizontalalignment('center')
            autotext.set_fontstyle('italic')
            autotext.set_fontsize(12)
        ax.set_title(idx, fontsize=15, color=font_color, y=-0.1)
        legend = plt.legend([x[1] for x in row.index], bbox_to_anchor=legend_bbox_to_anchor, loc='upper left', ncol=1, fancybox=True, fontsize=12)
        legend.set_title(target_column,prop={'size':'large'})
        for text in legend.get_texts():
            plt.setp(text, color=font_color, fontsize=14) # Legend font color
        fig.subplots_adjust(wspace=.02) # Space between charts  
    plt.subplots_adjust(top=sublots_adjust_top, bottom=subplots_adjust_bottom) # To prevent the title from being cropped
    sizes  = df_data[target_column].value_counts(dropna = True)
    target_labels2 = df_data[target_column].value_counts(dropna = True).index
    if (number_of_subgraphics == 2):
        axU = axes[0]
    else:
        sizes_by_category  = df_data[categorial_variable].value_counts(dropna = True)
        target_labels2 = df_data[target_column].value_counts(dropna = True).index
        axL = axes[1,0]
        width = 0.7
        barlistL = axL.bar(sizes_by_category.index, pd.Series(sizes_by_category), width=0.6, edgecolor=value_color_cream_black, color=value_color_cream_blue)
        for i in range(len(barlistL)):
            barlistL[i].set_edgecolor(value_color_cream_black)
        axL.yaxis.set_ticks([0,50000,100000,150000,200000,250000,300000])
        axL.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
        y_title = 'Number of people'
        axL.set_ylabel(y_title, color=font_color, fontsize=14)
        x_ticks = list(pd.Series(sizes_by_category).index)
        axL.set_xticklabels(x_ticks,color=font_color, rotation=75, fontsize=14)
        axU = axes[0,0]        
    width = 0.7
    barlistU = axU.bar(sizes.index, pd.Series(sizes), width=0.6, edgecolor=value_color_cream_black)
    barlistU[0].set_color(colors[0])
    barlistU[1].set_color(colors[1])
    barlistU[0].set_edgecolor(value_color_cream_black)
    barlistU[1].set_edgecolor(value_color_cream_black)
    axU.yaxis.set_ticks([0,50000,100000,150000,200000,250000,300000])
    axU.get_yaxis().set_major_formatter(ticker.FuncFormatter(lambda y, p: format(int(y), ',')))
    y_title = 'Number of people'
    axU.set_ylabel(y_title, color=font_color, fontsize=14)
    x_ticks = list(pd.Series(sizes).index)
    axU.set_xticklabels(x_ticks,color=font_color, fontsize=14)


def plot_non_categorical_variable_versus_target(df_data, non_categorical_variable, target_column):
    """
    This function shows linear plots with the percentages of the binary target variable for the different values of the column 
    'non categorical_variable' of the dataframe 'df_data'
    
    Inputs:
    - df_data: a dataframe with a column with non categorical variables and another column with the target variable
    - target_column: the name of the column of the dataframe with the values of the target variable
    
    Outputs:
    - linear plots displayed on the screen
    """
    
    font_color = value_color_cream_black
    df = df_data[[non_categorical_variable, target_column]]
    df_grouped = df.groupby([non_categorical_variable, target_column]).size().reset_index()
    table = pd.pivot_table(df_grouped, index=[non_categorical_variable], columns=[target_column])
    table = table.dropna(axis=1).astype(int)
    list_of_non_categorical_values = table.index.to_list()
    percentage_mental_illness = []
    persons_with_mental_illness    = table[0]['Yes'].to_list()
    persons_without_mental_illness = table[0]['No'].to_list()
    number_of_different_non_categorical_variables = len(table[0])
    for i in range(0,number_of_different_non_categorical_variables):
        total_person_of_that_non_categorical_value = persons_with_mental_illness[i] + persons_without_mental_illness[i]
        percentage_mental_illness = percentage_mental_illness + [persons_with_mental_illness[i] / total_person_of_that_non_categorical_value * 100]
    average_percentage_of_mental_illness = [30.4]*number_of_different_non_categorical_variables
    fig = plt.figure(figsize=(11, 5), facecolor=value_color_cream_calypso)
    title = fig.suptitle(target_column + ' vs ' +  non_categorical_variable, y=.95, fontsize=15, color=font_color, fontweight='bold')
    plt.plot(list_of_non_categorical_values, percentage_mental_illness, color=value_color_cream_red,   label=['% Mental illness by age'])
    plt.plot(list_of_non_categorical_values, average_percentage_of_mental_illness, color=value_color_cream_blue, label=['Avg % mental illness'])
    plt.legend()
    plt.legend(fontsize=14)
    labels_titles = [non_categorical_variable, '% Mental Illness']
    plt.xlabel(labels_titles[0], fontsize=14)
    plt.ylabel(labels_titles[1], fontsize=14)
    if (non_categorical_variable == age_column):
        plt.xticks(np.arange(15,85, 5))
        plt.yticks(np.arange(27, 35, 1))
    elif (non_categorical_variable == number_of_children_colum):
        plt.xticks(np.arange(0, max(list_of_non_categorical_values)+1, 1))
        plt.yticks(np.arange(27, 35, 1))
    elif (non_categorical_variable == income_category_column):
        plt.xticks(['0K','10K','20K','30K','40K','50K','60K','70K','80K','90K','100K','110K','120K','130K','140K'])
        plt.yticks(np.arange(20, 45, 5))
    legend_bbox_to_anchor = (0.55, 0.32)
    sublots_adjust_top = 0.88
    subplots_adjust_bottom = 0.1
    legend = plt.legend(bbox_to_anchor=legend_bbox_to_anchor, loc='upper left', ncol=1, fancybox=True, fontsize=12)
    plt.subplots_adjust(top=sublots_adjust_top, bottom=subplots_adjust_bottom) # To prevent the title from being cropped
    plt.show()


def plot_correlations(df):
    """
    This function shows a correlations matrix between the variables that are candidate to become explanatory features in the model
    
    Inputs:
    - df: a dataframe with columns that hold variables that are candidate to be explanatory features
    
    Outputs:
    - a heat map in which the correlations between variables are displayed on the screen
    """
    
    f = plt.figure(figsize=(34, 34))
    plt.matshow(df.corr(method='pearson'), fignum=f.number)
    cols = df.columns
    plt.xticks(range(len(cols)), cols, fontsize=27, rotation=90)
    plt.yticks(range(len(cols)), cols, fontsize=27)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=27)


def highlight_highest_abs_correlations(df, threshold_percentage):
    """
    This function shows in a matrix the pairs of variables that have the highest correlation in absolute value
    
    Inputs:
    - df: a dataframe
    - threshold_percentage: only the correlations above that percentage in absolute value will be highlighted.
    
    Outputs:
    - a heat map in which the highest correlations are displayed on the screen
    """
    
    df_correlations = df.corr(method='pearson')
    df_correlations_abs = df_correlations.abs()
    df_high_correlations = df_correlations_abs.mask(df_correlations_abs < threshold_percentage, 0)
    f = plt.figure(figsize=(10, 10))
    plt.matshow(df_high_correlations.corr(method='pearson'), fignum=f.number)
    cols = df.columns
    plt.xticks(range(len(cols)), cols, fontsize=9, rotation=90)
    plt.yticks(range(len(cols)), cols, fontsize=9)

    
def plot_model_performance(Ytest, y_predtest_model, title, labels_titles):
    """
    This function shows the performance of the model across different metrics (accuracy, precision, recall, specificity)
    
    Inputs:
    - Ytest: a list with the true values of the target variable
    - y_pred_test_model: a list with the predicted values of the target variable
    - title: the title of the graph
    - labels_titles: a list with the titles of the axis of the graph
    
    Outputs:
    - a bar graph displayed on the screen
    """
    
    test_performance_model = {}
    test_performance_model['accuracy']    = performance_library.accuracy_score(Ytest, y_predtest_model)*100
    test_performance_model['precision']   = performance_library.precision_score(Ytest, y_predtest_model)*100
    test_performance_model['recall']      = performance_library.recall_score(Ytest, y_predtest_model)*100
    test_performance_model['specificity'] = performance_library.specificity_score(Ytest, y_predtest_model)*100
    plt.figure(figsize=(10, 5))
    fig = pd.Series(test_performance_model.values()).plot(kind='bar')
    fig.set_title(title, fontsize=18)
    fig.set_xlabel(labels_titles[0], fontsize=16)
    fig.set_ylabel(labels_titles[1], fontsize=16)
    fig.set_xticklabels(pd.Series(test_performance_model.keys()), fontsize=16)
    plt.xticks(rotation=0)
    labels = np.round(list(test_performance_model.values()),1)
    rects = fig.patches
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        fig.text(rect.get_x() + rect.get_width() / 2, height + 0.005, label, ha="center", va="bottom", fontsize=12)
    plt.show()    
    

def plot_performance_train_val_test(avg_accuracy_train, avg_accuracy_val, accuracy_test, title, labels_titles):
    """
   This function compares the performances of the model across its training, validation and testing datasets.
    
    Inputs:
    - avg_accuracy_train: average accuracy score on the training dataset on a crossvalidation framework (on a training/validation split)
    - avg_accuracy_val: average accuracy score on the validation dataset on a crossvalidation framework (on a training/validation split)
    - accuracy_test: accuracy score on the testing dataset
    - title: the title of the graph
    - labels_titles: a list with the titles of the axis of the graph
    
    Outputs:
    - a bar graph displayed on the screen
    """
     
    test_performance_model = {}
    test_performance_model['Avg Training']   = avg_accuracy_train*100
    test_performance_model['Avg Validation'] = avg_accuracy_val*100
    test_performance_model['Test']           = accuracy_test*100
    plt.figure(figsize=(10, 5))
    fig = pd.Series(test_performance_model.values()).plot(kind='bar')
    fig.set_title(title, fontsize=18)
    fig.set_xlabel(labels_titles[0], fontsize=16)
    fig.set_ylabel(labels_titles[1], fontsize=16)
    fig.set_xticklabels(pd.Series(test_performance_model.keys()), fontsize=16)
    plt.xticks(rotation=0)
    labels = np.round(list(test_performance_model.values()),1)
    rects = fig.patches
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        fig.text(rect.get_x() + rect.get_width() / 2, height + 0.005, label, ha="center", va="bottom", fontsize=12)
    plt.show() 
 
    