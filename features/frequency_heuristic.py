'''Frequency Heuristic Feature class deifnition'''
from .base_feature import BaseFeature
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from .utils import *


class FrequencyHeuristicFeature(BaseFeature):

    __datasets = {
                'train': ('1-0cMe3FdhCjPeynHb-0emu7ZVEiARcGo', 'train_tweets.csv'),
                'test': ('1--avSX9A4E0BGxwGpcVGdud9f1FhaCx5', 'test_tweets.csv'),
                'valid': ('1--EX1lqaoNjEVLoaBIaTc3Q_17TymK11', 'valid_tweets.csv')
            }
    labels = ['true', 'mostly-true', 'half-true', 'barely-true',
              'mostly-false', 'false', 'pants-fire']

    def __init__(self, datasets=None):
        if datasets:
            self.__datasets = datasets
        super().__init__("freq_heuristics.pickle")
        self.__load_datasets()

    def __load_datasets(self):
        train = super()._load_dataset_from_gdrive(*self.__datasets['train'])
        test = super()._load_dataset_from_gdrive(*self.__datasets['test'])
        valid = super()._load_dataset_from_gdrive(*self.__datasets['valid'])
        for df in [train, test, valid]:
            self._clense_data(df)
            label_ids = []
            for i in df.label:
                label_ids.append(self.labels.index(i))
            df.insert(1, "label_id", label_ids)
        self.train = self.__text_preprocess(train).dropna()
        self.test = self.__text_preprocess(test).dropna()
        self.valid = self.__text_preprocess(valid).dropna()
        self.tfidf = TfidfVectorizer(min_df=3, max_features=None,
                                    strip_accents='unicode',
                                    analyzer='word', token_pattern=r'\w{1,}',
                                    ngram_range=(1,5), use_idf=1,
                                    smooth_idf=1, sublinear_tf=1)
        self.tfidf.fit_transform(self.train['headline_text'])
        self.X_train, self.y_train = self.__vectorize(self.train)

    @staticmethod
    def __text_preprocess(df):
        #convert to lower case
        df['headline_text'] = df['headline_text'].str.lower()
        #remove stop words
        df['headline_text'] = df['headline_text'].apply(remove_stopwords)
        #Lemmetize
        df['headline_text'] = df['headline_text'].apply(lemmatize_stemming)
        #stemming
        df['headline_text'] = df['headline_text'].apply(stemming)
        #remove punctuation
        df['headline_text'] = df['headline_text'].apply(remove_punctuation)
        #remove less than 3 letter words
        df['headline_text'] = df.headline_text.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))
        return df[['headline_text', 'label_id', 'tweet_count', 'source_list_cnt']]

    def __vectorize(self, df):
        x_text = self.tfidf.transform(df['headline_text'])
        x_val = df.drop(['label_id','headline_text'], axis=1).values
        x = sparse.hstack([x_val, x_text]).tocsr()
        y = self.train['label_id'].values
        return x, y

    def predict(self, headline_text,
                tweet_count, source_list_count, return_int=False):
        model = self._train_multinomial_bayes()
        df = pd.DataFrame(data={"headline_text": [headline_text],
                                "label_id": [-1],
                                "tweet_count": [tweet_count],
                                "source_list_cnt": [source_list_count]})
        df = self.__text_preprocess(df)
        x, y = self.__vectorize(df)
        prediction = model.predict(x)
        if return_int:
            return prediction[0]/float(len(self.labels)-1)
        return self.labels[prediction[0]]
