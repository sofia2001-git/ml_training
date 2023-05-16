import pandas as pd
import argparse
import os
import re
from bs4 import BeautifulSoup
import joblib


def createParser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path_to_file')
    parser.add_argument('path_to_model')
    return parser


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()


def clear_data(data):
    texts = []
    for idx in range(data.content.shape[0]):
        try:
            text = BeautifulSoup(data.content[idx])
            texts.append(clean_str(text.get_text()))
        except:
            print(idx)
    return texts


def get_predict(X, model):
    '''
    :X: data for analysis
    :model: path to model
    :return: vector with classes
    '''
    loaded_rf = joblib.load(model)
    return loaded_rf.predict(X)


if __name__ == '__main__':
    # get args
    parser = createParser()
    namespace = parser.parse_args()
    file = namespace.path_to_file
    model = namespace.path_to_model

    # read file
    if '.xlsx' in file or '.xls' in file:
        df = pd.read_excel(file)
        df.to_csv(f'./{os.path.splitext(os.path.basename(file))[0]}.csv')
    data_for_predict = pd.read_csv(f'./{os.path.splitext(os.path.basename(file))[0]}.csv')

    # preproccessing data
    data = pd.DataFrame()
    data['content'] = data_for_predict['Содержание сообщения']
    data.dropna()
    X = clear_data(data)

    # get predict
    pred = get_predict(X, model)

    # writing the result to a file
    data_with_pred = pd.DataFrame()
    data_with_pred['X'] = X
    data_with_pred['pred'] = pred
    data_with_pred.to_csv(f'./{os.path.splitext(os.path.basename(file))[0]}_result.csv')
