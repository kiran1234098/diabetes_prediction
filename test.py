from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score
from typing import Tuple
import pandas as pd
import numpy as np

class DataLoader:
    def _init_(self, path: str):
        self.path = path
    
    def load_data(self) -> Tuple[np.ndarray, np.ndarray]:
        data = pd.read_excel(self.path)
        X = data.iloc[:, :8].values
        y = data.iloc[:, 8].values
        return X, y

class ModelTrainer:
    def _init_(self, model):
        self.model = model
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        self.model.fit(X_train, y_train)
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Tuple[float, float]:
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        return accuracy 


class ModelComparer:
    def _init_(self, model1, model2,model3,model4,model5,model6):
        self.model1 = model1
        self.model2 = model2
        self.model3 = model3
        self.model4 = model4
        self.model5 = model5
        self.model6 = model6
    
    def compare(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        trainer1 = ModelTrainer(self.model1)
        trainer2 = ModelTrainer(self.model2)
        trainer3 = ModelTrainer(self.model3)
        trainer4 = ModelTrainer(self.model4)
        trainer5 = ModelTrainer(self.model5)
        trainer6 = ModelTrainer(self.model6)
        trainer1.train(X_train, y_train)
        trainer2.train(X_train, y_train)
        trainer3.train(X_train, y_train)
        trainer4.train(X_train, y_train)
        trainer5.train(X_train, y_train)
        trainer6.train(X_train, y_train)
        accuracy1 = trainer1.evaluate(X_test, y_test)
        accuracy2 = trainer2.evaluate(X_test, y_test)
        accuracy3 = trainer3.evaluate(X_test, y_test)
        accuracy4 = trainer4.evaluate(X_test, y_test)
        accuracy5 = trainer5.evaluate(X_test, y_test)
        accuracy6 = trainer6.evaluate(X_test, y_test)
        print(f"Model 1: accuracy = {accuracy1}")

        print(f"Model 2: accuracy = {accuracy2}")
        print(f"Model 3: accuracy = {accuracy3}")
        print(f"Model 4: accuracy = {accuracy4}")
        print(f"Model 5: accuracy = {accuracy5}")
        print(f"Model 6: accuracy = {accuracy6}")

if _name_ == "_main_":
    data_loader = DataLoader("notebooks/diabetes.csv")
    X, y = data_loader.load_data()
    X_train, X_test = X[:600], X[600:]
    y_train, y_test = y[:600], y[600:]
    model1 = LogisticRegression()
    model2 = GaussianNB()
    model3 = DecisionTreeClassifier()
    model4 =RandomForestClassifier()
    model5 =xgb.XGBClassifier()
    model6 =SVC()

    comparer = ModelComparer(model1, model2,model3,model4,model5,model6)
    comparer.compare(X_train, y_train, X_test, y_test)