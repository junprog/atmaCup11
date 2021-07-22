import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

def one_hot_encode(df):
    one_hot_encoded_df = pd.get_dummies(df, columns=['name'])
    one_hot_encoded_df = one_hot_encoded_df.set_index('object_id')
    one_hot_encoded_df = one_hot_encoded_df.sum(level=0).reset_index()

    # csv に重複があるため、1にする
    for name in one_hot_encoded_df:
        if name is not 'object_id':
            one_hot_encoded_df[(one_hot_encoded_df[name] != 1) & (one_hot_encoded_df[name] != 0)] = 1

    # 余分な列名削除
    new_name = {}
    for name in one_hot_encoded_df:
        new_name[name] = name.replace('name_', '')

    one_hot_encoded_df = one_hot_encoded_df.rename(columns=new_name)

    return one_hot_encoded_df