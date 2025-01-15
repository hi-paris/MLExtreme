from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR, SVC
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def Get_Model(Name, problem_type='regression',**kwargs):
    """
    Get a model based on the provided name and problem type.

    Parameters:
    Name (str): The name of the model to return.
    problem_type (str): 'regression' or 'classification'. Specifies the type of problem.

    Returns:
    model: The model instance.
    """
    if problem_type == 'regression':
        if Name == 'OLS':
            return LinearRegression(**kwargs)
        elif Name == 'Ridge':
            return Ridge(**kwargs)
        elif Name == 'Lasso':
            return Lasso(**kwargs)
        elif Name == 'SVR':
            return SVR(kernel='rbf',**kwargs)
        elif Name == 'RandomForest':
            return RandomForestRegressor(**kwargs)
        elif Name == 'DecisionTree':
            return DecisionTreeRegressor(**kwargs)
        elif Name == 'KNN':
            return KNeighborsRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown regression model: {Name}")
    
    elif problem_type == 'classification':
        if Name == 'SVC':
            return SVC(kernel='rbf',**kwargs)
        elif Name == 'RandomForest':
            return RandomForestClassifier(**kwargs)
        elif Name == 'DecisionTree':
            return DecisionTreeClassifier(**kwargs)
        elif Name == 'KNN':
            return KNeighborsClassifier(**kwargs)
        elif Name == 'NaiveBayes':
            return GaussianNB(**kwargs)
        else:
            raise ValueError(f"Unknown classification model: {Name}")

    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
