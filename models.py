from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

def entrainer_modele_classification(X_train, y_train):
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    return model

def entrainer_modele_regression(X_train, y_train):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    return model
