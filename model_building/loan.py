# importer les librairies
import pandas as pd
import seaborn as sns

# importer les packages machine learning
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as score
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, cross_val_score
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC


# importer les datas
dataframe = pd.read_csv("dataframe_log.csv")

df = dataframe.copy()

target_mapper = {'loan accoded' : 1, 'loan refused':0}

# Préparer les données d'entrainement et test
X = df.iloc[:, : -1]
y = df.iloc[:, -1]

# Utilisation du modele Random Forest
clf = XGBClassifier()
clf.fit(X, y)

# Sauvegarde du model
import pickle
pickle.dump(clf, open('loan_clf.pkl', 'wb'))



