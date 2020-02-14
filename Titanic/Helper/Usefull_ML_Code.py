# Disabling warnings
import warnings
warnings.simplefilter("ignore")
####################numpy###############################################################################################################################################
import numpy as np # linear algebra


#####################pandas##############################################################################################################################################
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#create a pd.DataFrame
train_df = pd.read_csv("/kaggle/input/titanic/train.csv")
test_df = pd.read_csv("/kaggle/input/titanic/test.csv")
train_df.head()
train_df.info()

#columns with "object" data type must convert to numbers.
objects_cols = train_df.select_dtypes("object").columns

# Show if any NAN data (do it for both train and test)
train_df.isnull().sum()

#.loc: Access a group of rows and columns by label(s) or a boolean array.
women = train_df.loc[(train_df.Sex == 'female') & (train_df.Age == 22.0) ]["Survived"]
women = train_df.loc[train_df.Sex == 'female']["Survived"]
print(train_df.loc[1]) #access row 1
print(women)

#default is axis = 0 which mean row wise but we want column wise
train_df = train_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)

#drop the null values
train_df = train_df.dropna()

#fillna: We can also propagate non-null values forward or backward.
#ffill: propagate[s] last valid observation forward to next valid
train_df.Embarked.fillna(method='ffill', inplace=True)

#correlation: if the correlation of 2 features are too high it means it is redundant
train_df.corr()
#correlation based on one parameter only
train_df.corr().Survived.sort_values()

#Age and Pclass are correlated so we estimate the missing Age based on Pclass
def predict_age(row_age_pclass):
    age = row_age_pclass[0]
    pclass = row_age_pclass[1]
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 29
        else:
            return 25
    else:
        return age 
#apply: get a column and apply a change row by row
train_df['Age'] = train_df[['Age', 'Pclass']].apply(predict_age, axis=1)

#get_dummies: converting categorical features
sex = pd.get_dummies(train_df['Sex'], drop_first=True)
embarked = pd.get_dummies(train_df['Embarked'], drop_first=True)
#concatenation
train_df = pd.concat([train_df, sex, embarked], axis=1)

#################seaborn and matplotib######################################################################################################################################
import seaborn as sns #data visualization is more grafical and user friendly
import matplotlib.pyplot as plt #data visualization

#plot the null entries
plt.figure(figsize=(8,8))
sns.heatmap(train_df.isnull(), yticklabels=False, cbar=False, cmap="viridis")

#see if the our target is balances
sns.countplot(x="Survived", data=train_df)
sns.countplot(x="Survived", data=train_df, hue="Sex")

#distribution plot
#dropna drops the nulls
sns.distplot(train_df.Age.dropna(), kde=False, bins=30)
sns.distplot(train_df['Age'], bins=24, color='b')

#distribution using pandas
train_df.Age.hist(alpha=0.7)

#distriuation of all columns based on each other
sns.pairplot(train_df)

#Correlation of all features
plt.figure(figsize=(12, 8))
plt.title('Titanic Correlation of Features', y=1.05, size=15)
sns.heatmap(train_df.corr(), linewidths=0.1, vmax=1.0, square=True, linecolor='white', annot=True)


##################scikit Learn##########################################################################################################################################
from sklearn #machine learning library

#Scaling: standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler
scale = StandardScaler()
scale.fit(X)
X = scale.transform(X)

#SimpleImputer: sklearn library for Imputation of missing values You Can find all of them here: 
#https://scikit-learn.org/stable/modules/impute.html#univariate-feature-imputation
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(np.nan, "mean")
train_df['Age'] = imputer.fit_transform(np.array(train_df['Age']).reshape(891, 1))

#columns with "object" data type must convert to numbers.
objects_cols = train_df.select_dtypes("object").columns
#Encode target labels with value between 0 and (n_classes - 1)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
train_df[objects_cols] = train_df[objects_cols].apply(le.fit_transform)

#split the data to training and test
from sklearn.model_selection import train_test_split
X = train_df.drop('Survived', axis=1)
y = train_df.Survived
#split the training data t0 70%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#evaluation
#1
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, logmodel_predict))
print(confusion_matrix(y_test, logmodel_predict))
#2
from sklearn.metrics import accuracy_score
acc = accuracy_score(y_pred, y_test)
print(f"{self.model_name()} Model Accuracy: ", acc)

#cross validarion
from sklearn.model_selection import cross_val_score
CVS = cross_val_score(self.model, self.X, self.y, scoring='accuracy', cv=cv)
print(CVS)
print("="*60, "\nMean accuracy of cross-validation: ", CVS.mean())

#Logistic Regression
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
#train
logmodel.fit(X_train, y_train)
#predict
logmodel_predict = logmodel.predict(X_test)
#evaluate
from sklearn.metrics import classification_report, confusion_matrix
print(classification_report(y_test, logmodel_predict))
print(confusion_matrix(y_test, logmodel_predict))

#Random Forest
from sklearn.ensemble import RandomForestClassifier
#train
randmodel = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=1)
randmodel.fit(X_train, y_train)
#predict
randmodel_predict = randmodel.predict(X_test)

#SVC
from sklearn.svm import SVC
svc_model = SVC()
#train
svc_model.fit(X_train, y_train)
#predict
svc_predict = svc_model.predict(X_test)

#GaussianNB
from sklearn.naive_bayes import GaussianNB

#############tensorflow#####################################################################################################################################
import tensorflow as tf

#Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout

X_train_nn = X_train.values
y_train_nn = y_train.values
X_test_nn = X_test.values
y_test_nn = y_test.values

nn_model = Sequential()
#add layer
nn_model.add(Dense(units=8, activation='relu')) #unit is the # nodes which is roughly # of features
nn_model.add(Dropout(0.5)) #prevent NN from overfitting by disabling 50% of the activation nodes
#add layer
nn_model.add(Dense(units=4, activation='relu'))
nn_model.add(Dropout(0.5)) 
#add layer (final layer)
nn_model.add(Dense(units=1, activation='sigmoid'))
nn_model.compile(loss='binary_crossentropy', optimizer='adam')
#if model not working stop before the iterations (epochs)
from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
#train
nn_model.fit(x=X_train_nn, y=y_train_nn, epochs=200, validation_data=(X_test_nn, y_test_nn), verbose=1, callbacks=[early_stop])


###########xgboost############################################################################################################################################
from xgboost import XGBClassifier

###########pipeline############################################################################################################################################

def predict_age(row_age_pclass):
    age = row_age_pclass[0]
    pclass = row_age_pclass[1]
    if pd.isnull(age):
        if pclass == 1:
            return 38
        elif pclass == 2:
            return 29
        else:
            return 25
    else:
        return age

def clean_titanic_data(data_path):
	train_df = pd.read_csv(data_path)
	train_df = train_df.drop(["PassengerId", "Name", "Ticket", "Cabin"], axis=1)  
	#apply get a column and apply a change row by row
	train_df['Age'] = train_df[['Age', 'Pclass']].apply(predict_age, axis=1) 
	train_df = train_df.dropna()
	sex = pd.get_dummies(train_df['Sex'], drop_first=True)
	embarked = pd.get_dummies(train_df['Embarked'], drop_first=True)
	train_df = pd.concat([train_df, sex, embarked], axis=1)
	train_df = train_df.drop(['Sex', 'Embarked'], axis=1)
	return train_df


from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
class Model:
    def __init__(self, model):
        self.model = model
        self.X, self.y = X, y
        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test
        
        self.train()
    
    def model_name(self):
        model_name = type(self.model).__name__
        return model_name
        
    def cross_validation(self, cv=5):
        print(f"Evaluate {self.model_name()} score by cross-validation...")
        CVS = cross_val_score(self.model, self.X, self.y, scoring='accuracy', cv=cv)
        print(CVS)
        print("="*60, "\nMean accuracy of cross-validation: ", CVS.mean())
    
    def train(self):
        print(f"Training {self.model_name()} Model...")
        self.model.fit(X_train, y_train)
        print("Model Trained.")
        
    def prediction(self, test_x=None, test=False):
        if test == False:
            y_pred = self.model.predict(self.X_test)
        else:
            y_pred = self.model.predict(test_x)
            
        return y_pred
    
    def accuracy(self):
        y_pred = self.prediction()
        y_test = self.y_test
        
        acc = accuracy_score(y_pred, y_test)
        print(f"{self.model_name()} Model Accuracy: ", acc)