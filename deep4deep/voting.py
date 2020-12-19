import numpy as np
import pandas as pd

def take_measure(row, function):
    '''
    Uses the given function as 'voting strategy' to combine the values found in row
    If NLP prediction is NaN, then only takes the main model's prediction.
    '''
    if row['y_pred_NLP'] == np.nan:
        return row['y_pred_forest']
    if function == 'mean':
        return row[['y_pred_forest','y_pred_NLP']].mean()
    if function == 'min':
        return row[['y_pred_forest','y_pred_NLP']].min()
    if function == 'max':
        return row[['y_pred_forest','y_pred_NLP']].max()
    else:
        print("not a known function")
    return None
