import pandas as pd

def one_hot_encode(df):
    one_hot_encoded_df = pd.get_dummies(df, columns=['name'])
    one_hot_encoded_df = one_hot_encoded_df.set_index('object_id')
    one_hot_encoded_df = one_hot_encoded_df.sum(level=0).reset_index()

    # 余分な列名削除
    new_name = {}
    for name in one_hot_encoded_df:
        new_name[name] = name.replace('name_', '')

    one_hot_encoded_df = one_hot_encoded_df.rename(columns=new_name)

    return one_hot_encoded_df