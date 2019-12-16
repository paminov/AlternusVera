'''Frequency Heuristic Feature class deifnition'''
from .base_feature import BaseFeature
import nltk
import nltk.sentiment
import pandas as pd


class SocialMediaActivitiesFeature(BaseFeature):

    __datasets = {
           'train': ('12HZeJfwd3B4ayHX380mkf_pWsGbvRYbb', 'pf_news_tweets.csv'),
           'test': ('1GFFIpbwqv7R-eRRhqcKpcMXryzfZMeEt', 'test_tweets_social_posts.csv.csv'),
           'valid': ('1--v0c1uOblbBcB6XVtwf7T5GZxEWjONd', 'valid_tweets_social_posts.csv.csv')
        }
    labels = ['pants-fire', 'mostly-false', 'false', 'barely-true',
            'half-true', 'mostly-true', 'true']
    def __init__(self, datasets=None):
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('vader_lexicon')
        if datasets:
            self.__datasets = datasets
        super().__init__("social_media_activities.pickle")
        self.__load_datasets()

    def __load_dataset_from_gdrive(self, file_id, file_name):
        downloaded = self.drive.CreateFile({'id':file_id})
        downloaded.GetContentFile(file_name)
        return pd.read_csv(file_name)

    def __load_datasets(self):
        train = super()._load_dataset_from_gdrive(*self.__datasets['train'])
        test = super()._load_dataset_from_gdrive(*self.__datasets['test'])
        valid = super()._load_dataset_from_gdrive(*self.__datasets['valid'])
        train['label']=train['label'].str.lower()
        self.train, self.test, self.valid = train, test, valid
        self.data_all = test.append(train, ignore_index=True, sort=True)\
                            .append(valid, ignore_index=True, sort=True)

    def __prepare_x_y(self, df):
        df = df.copy()
        senti = nltk.sentiment.vader.SentimentIntensityAnalyzer()
        compound=0
        for index, row in df.iterrows():
            favourite_count = row['favourite_count']
            retweet_count = row['retweet_count']
            #tweet_text = row['tweet_text']
            tweet_count=len(row['tweet_text'])
            for tweet in row['tweet_text']:
                snt = senti.polarity_scores(tweet)
                compound+=snt['compound']
            social_score = compound/tweet_count
            df.at[index, 'social_score'] = social_score
        X = df[['social_score','favourite_count', 'retweet_count']]
        y = df['label']
        return X, y

    def __train_model(self):
        self.X_train, self.y_train = self.__prepare_x_y(self.train)
        return self._train_gaussian_nb()

    def predict(self, favourite_count,
                retweet_count, tweet_text, return_int=False):
        model = self.__train_model()
        df = pd.DataFrame(data={"label": [None],
                          "favourite_count": [favourite_count],
                          "retweet_count": [retweet_count],
                          "tweet_text": [tweet_text]})
        X, y = self.__prepare_x_y(df)
        prediction = model.predict(X)
        if return_int:
            return self.labels.index(prediction[0])/float(len(self.labels)-1)
        return prediction[0]
