import numpy as np
import torch
import torch.utils.data as data
import math
import os
from tqdm import tqdm
import pandas as pd
import re
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer

class Dataset:
    def __init__(self):
        self.basedir = 'C:/Users/zdaaaaa/Desktop/Repository/软件安全/选中的论文/MSR2019---NN'
        self.datapath = self.basedir + '/data/all_cves_cleaned.csv'

    def extract_features(self, config, start_word_ngram, end_word_ngram):
        if config == 1:
            return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=False, min_df=0.001,
                                norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+')
        elif config == 2:
            return TfidfVectorizer(stop_words=['aka'], ngram_range=(1, 1), use_idf=True, min_df=0.001,
                                norm='l2', token_pattern=r'\S*[A-Za-z]\S+')
        elif config < 6:
            return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=False,
                                min_df=0.001, norm=None, smooth_idf=False, token_pattern=r'\S*[A-Za-z]\S+')

        return TfidfVectorizer(stop_words=['aka'], ngram_range=(start_word_ngram, end_word_ngram), use_idf=True,
                            min_df=0.001, norm='l2', token_pattern=r'\S*[A-Za-z]\S+')

    def feature_model(self, X_train, X_test, y_test, config, start_word_ngram, end_word_ngram):
        # Create vectorizer
        vectorizer = self.extract_features(config=config, start_word_ngram=start_word_ngram, end_word_ngram=end_word_ngram)

        X_train = X_train.astype(str)
        X_test = X_test.astype(str)

        X_train_transformed = vectorizer.fit_transform(X_train)
        X_train_transformed = X_train_transformed.toarray()

        X_test_transformed = vectorizer.transform(X_test)
        X_test_transformed = X_test_transformed.toarray()

        # Remove rows with all zero values
        test_df = pd.DataFrame(X_test_transformed)
        results = test_df.apply(lambda x: x.value_counts().get(0.0, 0), axis=1)
        non_zero_indices = np.where(results < len(test_df.columns))[0]

        X_train_transformed = X_train_transformed.astype(np.float64)
        X_test_transformed = X_test_transformed.astype(np.float64)

        return X_train_transformed, X_test_transformed[non_zero_indices], y_test[non_zero_indices]

    def extractYearFromId(self, id):
        return re.match(r'CVE-(\d+)-\d+', id).group(1)


    def fetch_data(self, label, config, start_word_ngram, end_word_ngram, year):
        df = pd.read_csv(self.datapath, low_memory=False)

        df['Year'] = df.ID.map(self.extractYearFromId).astype(np.int64)
        notnull_indices = np.where(df.CVSS2_Avail.notnull())[0]
        df_notnull = df.iloc[notnull_indices]

        split_year = 2016
        all_train_indices = np.where(df_notnull.Year < split_year)[0]
        X = df_notnull.Cleaned_Desc.iloc[all_train_indices].values
        y = df_notnull[label].iloc[all_train_indices].values

        train_indices = np.where(df_notnull.iloc[all_train_indices].Year < year)[0]
        test_indices = np.where(df_notnull.iloc[all_train_indices].Year == year)[0]
        X_train, X_test = X[train_indices], X[test_indices]
        y_train, y_test = y[train_indices], y[test_indices]
        X_train_transformed, X_test_transformed, y_test = self.feature_model(X_train, X_test, y_test, config, start_word_ngram, end_word_ngram)

        # 把y中的label变为数字
        label_set = set()
        for item in y_train:
            label_set.add(item)
        for index, item in enumerate(y_train):
            item = list(label_set).index(item)
            y_train[index] = item
        for index, item in enumerate(y_test):
            item = list(label_set).index(item)
            y_test[index] = item

        return X_train_transformed, X_test_transformed, y_train, y_test

    def load_data(self, label, config, start_word_ngram, end_word_ngram, year):
        X_train_transformed, X_test_transformed, y_train, y_test = self.fetch_data(label, config, start_word_ngram, end_word_ngram, year)

        class TempClass:
            def __init__(self_2):
                self_2.x_train = X_train_transformed
                self_2.x_valid = X_test_transformed
                self_2.y_train = y_train
                self_2.y_valid = y_test

        return TempClass()


class TorchDataset(data.Dataset):
    def __init__(self, ds, mode='train'):
        super(TorchDataset, self).__init__()
        self.ds = ds
        self.mode = mode

    def __getitem__(self, index):
        if self.mode == 'train':
            x = torch.from_numpy(self.ds.x_train[index])
            y = torch.tensor(self.ds.y_train[index]) 
        else:
            x = torch.from_numpy(self.ds.x_valid[index])
            y = torch.tensor(self.ds.y_valid[index])

        return x.float(), y.float()

    def __len__(self):
        if self.mode == 'train':
            return self.ds.x_train.shape[0]
        else:
            return self.ds.x_valid.shape[0]

class DatasetFactory(object):
    def __init__(self, label, config, start_word_ngram, end_word_ngram, year):
        self.dataset = Dataset()
        self.ds = self.dataset.load_data(label, config, start_word_ngram, end_word_ngram, year)

    def get_train_dataset(self):
        return TorchDataset(self.ds, 'train')

    def get_valid_dataset(self):
        return TorchDataset(self.ds, 'valid')