'''Reliable Source Feature class deifnition'''
from .base_feature import BaseFeature
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from .utils import *


class ReliableSourceFeature(BaseFeature):

    __datasets = {
        'train': ('1JZ4xVfDofvxlu-y0O3vAATujIyAUTduv', 'train_news_sources.csv'),
        'test': ('15Ls7ZLpbpmus8sq6RIHK2G9TbHVRZsuS', 'test_news_sources.csv') }
    labels = ['original','true','mostly-true','half-true',
              'barely-true','false','pants-fire']

    def __init__(self, datasets=None):
        if datasets:
            self.__datasets = datasets
        super().__init__("reliable_source.pickle")
        self.__load_datasets()

    def __load_datasets(self):
        train = super()._load_dataset_from_gdrive(*self.__datasets['train'])
        test = super()._load_dataset_from_gdrive(*self.__datasets['test'])
        for df in [train, test]:
            label_ids = []
            for i in df.label:
                label_ids.append(self.labels.index(i))
            df.insert(1, "label_id", label_ids)
        self.train = self.__text_preprocess(train).dropna()
        self.test = self.__text_preprocess(test).dropna()
        self.tfidf = TfidfVectorizer(min_df=3, max_features=None,
                                                                 strip_accents='unicode',
                                                                 analyzer='word', token_pattern=r'\w{1,}',
                                                                 ngram_range=(1,5), use_idf=1,
                                                                 smooth_idf=1, sublinear_tf=1)
        self.tfidf.fit_transform(self.train['headline_text'])
        self.tfidf.fit_transform(self.train['context'])
        self.tfidf.fit_transform(self.train['source'])
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
        df['headline_text']    = df.headline_text.apply(lambda i: ' '.join(filter(lambda j: len(j) > 3, i.split())))

        #convert to lower case
        df['context'] = df['context'].str.lower()
        #remove stop words
        df['context'] = df['context'].apply(remove_stopwords)
        #Lemmetize
        df['context'] = df['context'].apply(lemmatize_stemming)
        #stemming
        df['context'] = df['context'].apply(stemming)
        #remove punctuation
        df['context'] = df['context'].apply(remove_punctuation)

        return df[['headline_text', 'label_id', 'context', 'source']]

    def __vectorize(self, df):
        x_text = self.tfidf.transform(df['headline_text'])
        x_context = self.tfidf.transform(df['context'])
        x_source = self.tfidf.transform(df['source'])
        x = sparse.hstack([x_text, x_context, x_source]).tocsr()
        y = self.train['label_id'].values
        return x, y

    def predict(self, headline_text, context, source):
        model = self._train_multinomial_bayes()
        df = pd.DataFrame(data={"headline_text": [headline_text],
                                                        "label_id": [-1],
                                                        "context": [context],
                                                        "source": [source]})
        df = self.__text_preprocess(df)
        x, y = self.__vectorize(df)
        prediction = model.predict(x)
        return self.labels[prediction[0]]
