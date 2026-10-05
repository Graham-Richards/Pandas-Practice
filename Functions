import pandas as pd
import numpy as np
import statistics
import funcs as fn       # Download this file as funcs.py

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
