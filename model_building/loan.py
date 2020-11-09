# importer les librairies
import pandas as pd
import seaborn as sns

# importer les packages machine learning
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split, cross_val_score
from sklearn.metrics import plot_confusion_matrix, classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC


# importer les datas
dataframe = pd.read_csv("dataframe_log2.csv")

df = dataframe.copy()

# Préparer les données d'entrainement et test
X = df.iloc[:, : -1]
y = df.iloc[:, -1]

# standardisation des données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(X)

X_scaled = pd.DataFrame(data_scaled, index=X.index, columns=X.columns)

# Utilisation du modele Random Forest
clf = XGBClassifier()
clf.fit(X_scaled, y)

# Sauvegarde du model
import pickle
pickle.dump(clf, open('loan_clf.pkl', 'wb'))



