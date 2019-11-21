from self_learning_model import SelfLearningModel
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd

df = pd.read_csv('data_labeled.csv')
dfU = pd.read_csv('data_unlabeled.csv')

X = ['b3001', 'b3002', 'b3003', 'b3004', 'b3005', 'b3006', 'b3007', 'b3008', 'b3009', 'b3010', 'b3011', 'b3012', 'b3013']
y = ['x', 'y']

regressor = MultiOutputRegressor(GradientBoostingRegressor(random_state=0))

slm = SelfLearningModel(regressor, 10, 0.8)

slm.set_labeled_dataset(df)
slm.set_unlabeled_dataset(dfU)
slm.set_features(X)
slm.set_target_variables(y)
slm.train()

print('Score: ' + str(slm.get_score(df[X], df[y])))
print('F1 Score: ' + str(slm.get_score(df[X], df[y])))