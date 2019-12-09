'''Frequency Heuristic Feature class deifnition'''
from google.colab import auth
from oauth2client.client import GoogleCredentials
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from .utils import *


class FrequencyHeuristicFeature(object):
    __datasets = {
                'train': ('1-0cMe3FdhCjPeynHb-0emu7ZVEiARcGo', 'train_tweets.csv'),
                'test': ('1--avSX9A4E0BGxwGpcVGdud9f1FhaCx5', 'test_tweets.csv'),
                'valid': ('1--EX1lqaoNjEVLoaBIaTc3Q_17TymK11', 'valid_tweets.csv')
            }
    labels = ['original', 'pants-fire', 'false', 'barely-true',
              'half-true', 'mostly-true', 'true']

    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)
        self.__load_datasets()
        self.stop_words = None
        self.headline_words = None

    def __load_dataset_from_gdrive(self, file_id, file_name):
        downloaded = self.drive.CreateFile({'id':file_id})
        downloaded.GetContentFile(file_name)
        return pd.read_csv(file_name)

    def __load_datasets(self):
        train = self.__load_dataset_from_gdrive(*self.__datasets['train'])
        test = self.__load_dataset_from_gdrive(*self.__datasets['test'])
        valid = self.__load_dataset_from_gdrive(*self.__datasets['valid'])
        for df in [train, test, valid]:
            self.__clense_data(df)
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

    def __clense_data(self, df):
        indecies = df[(df['tweet_count']==0)&(df['source_list_cnt']==0)].index
        df.drop(indecies , inplace=True)

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
        return df[['headline_text', 'label_id', 'tweet_count', 'source_list_cnt']]

    def __vectorize(self, df):
        x_text = self.tfidf.transform(df['headline_text'])
        x_val = df.drop(['label_id','headline_text'], axis=1).values
        x = sparse.hstack([x_val, x_text]).tocsr()
        y = self.train['label_id'].values
        return x, y

    def __train_multinomial_bayes(self):
        nb = MultinomialNB()
        nb.fit(self.X_train, self.y_train)
        return nb

    def predict(self, headline_text, tweet_count, source_list_count, return_int=False):
        model = self.__train_multinomial_bayes()
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
