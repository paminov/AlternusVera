from google.colab import auth
from oauth2client.client import GoogleCredentials
import os
import pandas as pd
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier


class BaseFeature(object):

    labels = ['original','true','mostly-true','half-true',
              'barely-true','false','pants-fire']

    def __init__(self, pickle_file):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)
        self.model_pickle = pickle_file

    def _load_dataset_from_gdrive(self, file_id, file_name):
        downloaded = self.drive.CreateFile({'id':file_id})
        downloaded.GetContentFile(file_name)
        return pd.read_csv(file_name)

    def __clense_data(self, df):
        indecies = df[(df['tweet_count']==0)&(df['source_list_cnt']==0)].index
        df.drop(indecies , inplace=True)

    def _pickle_model(self, model):
        with open(self.model_pickle, 'wb') as file:
            pickle.dump(model, file)
        print("exported pickle")

    def _load_pickle(self):
        if not os.path.exists(self.model_pickle):
            return None
        with open(self.model_pickle, 'rb') as f:
            model = pickle.load(f)
        print("loaded pickle")
        return model

    def _train_multinomial_bayes(self):
        nb = self._load_pickle()
        if not nb:
            nb = MultinomialNB()
            nb.fit(self.X_train, self.y_train)
            self._pickle_model(nb)
        return nb

    def _train_decision_tree(self):
        clf = self._load_pickle()
        if not clf:
            clf = DecisionTreeClassifier(criterion='entropy', max_depth =2,
                                         min_samples_split=2, min_samples_leaf=6)
            clf.fit(self.X_train, self.y_train)
            self._pickle_model(clf)
        return clf
