import pandas as pd

def clip_upper(df, column_name):
    """
    Clips the values of the feature based on the upper limit.
    """
    Q1 = df[column_name].quantile(0.25)
    Q3 = df[column_name].quantile(0.75)
    IQR = Q3 - Q1
    upper_limit = Q3 + 1.5 * IQR
    df[column_name] = df[column_name].clip(upper=upper_limit)
    
    return df[column_name]
