'''Social Credibility Feature class deifnition'''
from .base_feature import BaseFeature
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from .utils import *


class SocialCredibilityFeature(BaseFeature):

    __datasets = {
                'train': ('1EZ1jrYSFagyZ6ltkiGAklm5enqijojjH', 'train_news_with_tweets.csv'),
                'test': ('1vt0xnaKwvUmRoeriAmlprdweumgxnz-6', 'test_news_with_tweets.csv'),
                'valid': ('19cpmTufNyqVTBaGdgrajrlB3wN5tvCih', 'valid_news_with_tweets.csv')
            }
    labels = ['original','true','mostly-true','half-true',
              'barely-true','false','pants-fire']

    def __init__(self, datasets=None):
        if datasets:
            self.__datasets = datasets
        super().__init__("social_credibility.pickle")
        self.__load_datasets()

    def __load_datasets(self):
        train = super()._load_dataset_from_gdrive(*self.__datasets['train'])
        test = super()._load_dataset_from_gdrive(*self.__datasets['test'])
        valid = super()._load_dataset_from_gdrive(*self.__datasets['valid'])
        train = self.__clense_data(train)
        test = self.__clense_data(test)
        valid = self.__clense_data(valid)
        self.train = self.__text_preprocess(train).dropna(how='any',axis=0)
        self.test = self.__text_preprocess(test).dropna(how='any',axis=0)
        self.valid = self.__text_preprocess(valid).dropna(how='any',axis=0)
        self.tfidf = TfidfVectorizer(min_df=3, max_features=None,
                                     strip_accents='unicode',
                                     analyzer='word', token_pattern=r'\w{1,}',
                                     ngram_range=(1,5), use_idf=1,
                                     smooth_idf=1, sublinear_tf=1)
        self.tfidf.fit_transform(self.train['speaker'])
        self.X_train, self.y_train = self.__vectorize(self.train)

    def __clense_data(self, df):
        df = df[np.isfinite(df['followers_count'])]
        indecies = df[(df['followers_count']==0)].index
        df.drop(indecies , inplace=True)
        return df.dropna()

    def __text_preprocess(self, df):
        #convert to lower case
        df['speaker'] = df['speaker'].str.lower()
        #remove stop words
        df['speaker'] = df['speaker'].apply(remove_stopwords)
        #Lemmetize
        df['speaker'] = df['speaker'].apply(lemmatize_stemming)
        #stemming
        df['speaker'] = df['speaker'].apply(stemming)
        #remove punctuation
        df['speaker'] = df['speaker'].apply(remove_punctuation)
        #remove less than 3 letter words
        df['speaker']    = df.speaker.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))
        df['label'] = df['label'].str.lower()
        label_ids = []
        for i in df.label:
            label_ids.append(self.labels.index(i))
        df.insert(1, "label_id", label_ids)
        df = self.__generate_ratio(df)
        return df[['speaker', 'label_id', 'ratio']]

    def __generate_ratio(self, df):
        df = df.copy()
        for index, row in df.iterrows():
            followers_count = row['followers_count']
            friends_count = row['friends_count']
            ratio = friends_count / followers_count
            df.at[index, 'ratio'] = ratio
        return df

    def __vectorize(self, df):
        x_text = self.tfidf.transform(df['speaker'])
        x_val = df.drop(['label_id','speaker'], axis=1).values
        x = sparse.hstack([x_val, x_text]).tocsr()
        y = self.train['label_id'].values
        return x, y

    def predict(self, speaker_name, followers_count, friends_count, return_int=False):
        model = self._train_multinomial_bayes()
        df = pd.DataFrame(data={"label": ["false"],
                          "speaker": [speaker_name],
                          "followers_count": [followers_count],
                          "friends_count": [friends_count]})
        df = self.__text_preprocess(df)
        X, y = self.__vectorize(df)
        prediction = model.predict(X)
        if return_int:
            return prediction[0]/float(len(self.labels)-1)
        return self.labels[prediction[0]]
