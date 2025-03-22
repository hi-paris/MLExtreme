
# #######################################
# ## From data_generation: begin
# #######################################

def gen_classif_data_diriClasses(Mu0, wei0, lnu0=None, alpha=4,
                                 size0=1, size1=1):
    wei1 = wei0[::-1]
    if lnu0 is None:
        lnu0 = np.log(2 / np.min(Mu0, axis=1))
    data0 = gen_rv_dirimix(alpha, Mu0, wei0, lnu0, scale_weight_noise=0.5,
                           index_weight_noise=0.5, size=size0)
    data1 = gen_rv_dirimix(alpha, Mu0, wei1, lnu0, scale_weight_noise=0.5,
                           index_weight_noise=0.5, size=size1)
    y = np.vstack((np.zeros(size0).reshape(-1, 1),
                  np.ones(size1).reshape(-1, 1))).flatten()
    X =  np.vstack((data0, data1))
    permut = np.random.permutation(size0 + size1)
    X = X[permut, :]
    y=y[permut]
    return X,  y



# ## target generation for classification models
def gen_label(Matrix, angle=0.2):
    """
    Generate labels for a 2D classification toy example with and
    explicit decision boundary and no difference between tail and bulk
    behaviour

    Parameters:
    -----------
    Matrix : ndarray, shape (n, 2)
        2D array where each row represents a point in 2D space.
    angle : float, optional
        Angular range parameter. Points within this range are labeled as 1.
        Default is 0.2.

    Returns:
    --------
    ndarray, shape (n,)
        Array of labels (0 or 1) for each point in the input Matrix.
    """
    Vect_angle = np.arctan2(Matrix[:, 1], Matrix[:, 0])
    lower_bound = angle * np.pi / 2
    upper_bound = (1 - angle) * np.pi / 2
    label = [
        0 if (point_angle < lower_bound or point_angle > upper_bound) else 1
        for point_angle in Vect_angle
    ]

    return np.array(label)


def gen_label2(Matrix, angle=0.2):
    """
    Generate labels for a 2D toy example with an explicit decision boundary and
    explicit decision boundary and no difference between tail and bulk
    behaviour

    Parameters:
    -----------
    Matrix : ndarray, shape (n, 2)
        2D array where each row represents a point in 2D space.
    angle : float, optional
        Angular range parameter. Points within this range are labeled as 0.
    Default is 0.2.

    Returns:
    --------
    ndarray, shape (n,)
        Array of labels (0 or 1) for each point in the input Matrix.
    """
    Vect_angle = np.arctan2(Matrix[:, 1], Matrix[:, 0])
    label = []
    for point_angle in Vect_angle:
        first_condition = 0 <= point_angle < angle * np.pi / 4
        second_condition = (1 - angle) * np.pi / 4 <= point_angle < np.pi / 4
        third_condition = (
            np.pi / 4 + angle * np.pi / 4 <= point_angle <
            np.pi / 4 + (1 - angle) * np.pi / 4
        )

        if first_condition or second_condition or third_condition:
            label.append(0)
        else:
            label.append(1)
    return np.array(label)


# #######################################
# ## From data_generation: end
# #######################################


# #########################
# from supervised_classif.py : begin
# #########################

# ## 1. toy example with  hard decision boundary in 2D
# all data follow the tail structure (angular decision boundary)
# no bias variance compromise in the choice of k


# Parameters for data generation
n = 1000
Dim = 2
split = 0.2
alpha_dep = 0.9
angle = 0.25

# Data generation
data = mlx.gen_multilog(Dim, alpha_dep, size=n)**(1/4)
label = mlx.gen_label2(data, angle=angle)

# Visualization of the generated data
colors = np.where(label == 1, 'red', 'blue')
plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.7)
plt.show()

# Splitting the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data, label,
                                                    test_size=split,
                                                    random_state=42)

# Choice of an off-the-shelf classification algorithm
# see https://scikit-learn.org/stable/supervised_learning.html

# Pick  a classifier model in sklearn, previously imported
model = RandomForestClassifier()

# Normalization function
def norm_func(x):
    return np.linalg.norm(x, ord=2, axis=1)


# Classifier class initialization
classifier = mlx.Classifier(model, norm_func)

# Model training
threshold, ratio, X_train_extreme = classifier.fit(X_train,  y_train, k=200)

# predict above a larger threshold (extrapolation)
ratio_ext = ratio / 2
Norm_test = norm_func(X_test)
thresh_predict = np.quantile(Norm_test, 1- ratio_ext)

# Prediction on the test data
y_pred_extreme,  X_test_extreme, mask_test = classifier.predict(
                                            X_test, thresh_predict)

# Accuracy evaluation
y_test_extreme = y_test[mask_test]
accuracy = accuracy_score(y_test_extreme, y_pred_extreme)
print(f'Accuracy: {accuracy:.4f}')
hamming = hamming_loss(y_test_extreme, y_pred_extreme)
print(f'0-1 loss: {hamming:.4f}')

# Display classification results
# show X
classifier.plot_classif(X_test_extreme, y_test_extreme, y_pred_extreme)

# show Theta
X_test_extrem_unit = X_test_extreme / norm_func(X_test_extreme)[:, np.newaxis]
classifier.plot_classif(X_test_extrem_unit, y_test_extreme, y_pred_extreme)


# 2. generating dirichlet mixtures ##
Mu0 = np.array([[0.2, 0.8], [0.8, 0.2]])  # Means of the two components
lnu0 = np.log(10) * np.ones(2)  # log(10) for each component
wei0 = np.array([0.1, 0.9])  # Mixture weights

# # class 0: dirichlet parameters
# Mu0 = np.array([[0.2, 0.8], [0.8, 0.2]])  # Means of the two components
# wei0 = np.array([0.9, 0.1])  # Mixture weights
# lnu0 = np.log(10 / np.min(Mu0, axis=1)) # log concentration for each component
# Plot the mixture density
mlx.plot_pdf_dirimix_2D(Mu0, wei0, lnu0)
n0=5000
n1=n0

data, label = mlx.gen_classif_data_diriClasses(Mu0, wei0, lnu0,
                                               size0=n0, size1=n1)
# Visualization of the generated data
colors = np.where(label == 1, 'red', 'blue').flatten()
plt.scatter(data[:, 0], data[:, 1], c=colors, alpha=0.5)
plt.show()



# #########################
# from supervised_classif.py - end
# #########################
