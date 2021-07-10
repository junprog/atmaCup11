import pandas as pd

def one_hot_encode(df):
    one_hot_encoded_df = pd.get_dummies(df, columns=['name'])
    one_hot_encoded_df = one_hot_encoded_df.set_index('object_id')
    one_hot_encoded_df = one_hot_encoded_df.sum(level=0).reset_index()

    return one_hot_encoded_df