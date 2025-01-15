import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


## norm recalculé ou non


class Classifier:
    def __init__(self, model, norm_func, k=0):
        """
        Initialisation

        model :  the model that you want to use. Must have a .fit method

        norm_func : a norm

        k
        """
        self.model = model
        self.norm_func = norm_func
        self.k = k

        self.threshold=0
        

    def fit(self, X_train, X_test, y_train):
        if self.k == 0:
            self.k = 4 * int(np.sqrt(len(X_train) + len(X_test)))
        
        # Sélection des points extrêmes
        Norm_X_train = self.norm_func(X_train)
        threshold = np.percentile(Norm_X_train, 100 * (1 - self.k / len(Norm_X_train)))
        X_train_extrem = X_train[Norm_X_train >= threshold]
        
        # Normalisation sur la sphère unité
        X_train_unit = X_train_extrem / ((Norm_X_train[Norm_X_train >= threshold])[:, np.newaxis])
        y_train_extrem = y_train[Norm_X_train >= threshold]
        
        # Entraînement du modèle
        self.model.fit(X_train_unit, y_train_extrem)

        #Sauvegarde threshold & model
        self.threshold=threshold
        #self.model=model
        
        return threshold, X_train_unit

    def predict(self, X_test, threshold=0):
        """
        Prédiction des étiquettes sur les données de test en utilisant les points extrêmes.
        """
        #Refaire le threshold
        if threshold==0:
            if self.threshold==0:
                print('error')
                return 0
            else:
                threshold=self.threshold

        
        Norm_X_test = self.norm_func(X_test)
        mask_test = Norm_X_test >= threshold
        X_test_extrem = X_test[mask_test]
        
        # Normalisation sur la sphère unité
        X_test_unit = X_test_extrem / ((Norm_X_test[Norm_X_test >= threshold])[:, np.newaxis])
        y_pred = self.model.predict(X_test_unit)
        
        return y_pred, mask_test, X_test_unit

    def plot_classif(self, X, y_test, y_pred):
        """
        Affiche les points classifiés en fonction des prédictions et des valeurs réelles.
        """
        plt.scatter(X[:, 0][(y_pred == 0) & (y_test == 0)], X[:, 1][(y_pred == 0) & (y_test == 0)], color='red', marker='o')
        plt.scatter(X[:, 0][(y_pred == 0) & (y_test == 1)], X[:, 1][(y_pred == 0) & (y_test == 1)], color='blue', marker='x')

        plt.scatter(X[:, 0][(y_pred == 1) & (y_test == 0)], X[:, 1][(y_pred == 1) & (y_test == 0)], color='red', marker='x')
        plt.scatter(X[:, 0][(y_pred == 1) & (y_test == 1)], X[:, 1][(y_pred == 1) & (y_test == 1)], color='blue', marker='o')
        plt.show()
