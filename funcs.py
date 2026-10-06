import pandas as pd
import numpy as np
import statistics
import funcs as fn


def nan_median_replace(df, column= "col_name"):
    '''
    This function will replace NaN/null values in a column to its average value.
    params:
            df = pandas dataframe
            column (str) = column name to fill the missing values
    '''
    #get boolean data of known values
    NotNullE = pd.notnull(df[column])

    #turn data back into dataframe
    NotNullEd = df[NotNullE]

    #get mean of known values
    NotNME = statistics.mean(NotNullEd[column])

    #replace NaN with mean
    FinishedE = df[column].fillna(NotNME)

    df[column] = FinishedE

    return df


def nan_mode_replace(df, column= "col_name"):
    '''
    This function will replace NaN/null values in a column to its mode value.
    params:
            df = pandas dataframe
            column (str) = column name to fill the missing values
    '''

    # replace UNKNOWN with NaN
    df[column] = df[column].replace('UNKNOWN', np.nan)

    #identify KNOWN patients
    dfknown = df[df[column].notnull()]

    #find mode of known
    mode_str = statistics.mode(dfknown[column])

    # replace NaN with mode
    df[column] = df[column].fillna(mode_str)

    return df


def clean_data(df):
    """
    This function will clean the data by removing columns that are entirely NaN, mean-imputing numeric columns that contain at least one real value, and mode-imputing string columns.
    params: 
            df = pandas dataframe
    """
    # Go through a fixed list of the original column names
    for col in list(df.columns):

        # Remove 'UNKNOWN' values and replace with NaN
        df[col] = df[col].replace('UNKNOWN', np.nan)

        # Remove columns that are entirely NaN
        if df[col].isna().all():
            df = df.drop(columns=[col])
            continue

        # Remove columns that are 70% NaN or more
        if df[col].isna().mean() >= 0.7:
            df = df.drop(columns=[col])
            continue
        
        # Only mean-impute numeric columns that contain at least one real value
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].notna().any():
            df = fn.nan_median_replace(df, col)

        # string columns use mode
        elif pd.api.types.is_numeric_dtype(df[col]) == False:
            df = fn.nan_mode_replace(df, col)

    return df


def structure_missing_values(df):

    unstructured_missing_vals = ['', '?', ' ', 'nan', 'N/A', None, 'na', 'None', 'none']

    for col in df.columns:
        df[col] = df[col].replace(unstructured_missing_vals, np.nan)

    for col in list(df.columns):
        # Remove columns that are entirely NaN
        if df[col].isna().all():
            df = df.drop(columns=[col])
            continue
        
        # Remove columns that are 70% NaN or more
        if df[col].isna().mean() >= 0.7:
            df = df.drop(columns=[col])
            continue

        if pd.api.types.is_numeric_dtype(df[col]) == False:
            df = df.drop(columns=col)

    return df       

