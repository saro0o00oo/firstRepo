from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

class MLclass:
# Machine Learning Class
	def __init__(self, model, X, y, X_test_final, test_size=0.3):
		self.model = model
		#Training data
		self.X, self.y = X, y
		#Splitted training data
		self.X_train, self.X_test, self.y_train, self.y_test = self.split_data(test_size)
		#Testing data
		self.X_test_final = X_test_final

	def split_data(self, test_size=0.3, random_state=42):
		X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=random_state)
		return X_train, X_test, y_train, y_test

	def model_name(self):
		model_name = type(self.model).__name__
		return model_name

	def train_with_splitted_data(self):
		print("Training {} Model with splitted training data...".format(self.model_name()))
		self.model.fit(self.X_train, self.y_train)
		print("Model trained.")

	def train_with_whole_data(self):
		#train with aal of the data to predict the testing data
		print("Training {} Model with ALL of training data...".format(self.model_name()))
		self.model.fit(self.X, self.y)
		print("Model trained.")

	def train_and_evaluate_with_cross_validation(self, cv=5):
		print("Train and evaluate the accuracy of {} model by cross_val_scoress-validation".format(self.model_name()))
		CVS = cross_val_score(self.model, self.X, self.y, scoring='accuracy', cv=cv)
		print(CVS)
		print("\n Mean accuracy of cross-validation: {}".format(CVS.mean()))

	def predict_with_splitted_data(self):
		return self.model.predict(self.X_test)

	def predict_with_whole_data(self):
		return self.model.predict(self.X_test_final)

	def evaluate_accuracy(self):
		y_pred = self.predict_with_splitted_data()
		acc = accuracy_score(y_pred, self.y_test)
		print("Accuracy of {} model is:{}".format(self.model_name(), acc))





