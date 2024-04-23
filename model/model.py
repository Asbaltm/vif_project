from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
from utils.base import Base


class Model(Base):

    def __init__(self, x, y):
        self.classifiers = self.set_classifiers()
        self.x = x
        self.y = y

    @staticmethod
    def set_classifiers() -> dict:
        # Define the classification models
        classifiers = {
            'XGBClassifier': XGBClassifier(),
            'RandomForestClassifier': RandomForestClassifier()
        }
        return classifiers

    def best_classifier_cv(self, cv=5):
        """
        This function performs cross-validation using specified classifiers
        It chooses best model based on cross validation scores than fits the best model

        Parameters:
           cv (int): The number of folds in the cross-validation. The default is 5.

        Returns:
           best model for classifiers (ex: RandomForestClassifier): The best fitted model.
        """

        # Initialize the best model and the highest score to empty objects
        best_model = None
        best_model_name = None
        highest_score = 0
        # runs classifiers cross validation and fit steps
        self.logger.info(f"Start Cross Validation Step")
        # Iterate on classifiers
        for classifier_name, classifier in self.classifiers.items():
            self.logger.info(f"Cross Validation Step using {classifier_name}")
            # returns a list here
            scores = cross_val_score(classifier, self.x, self.y, cv=cv)
            # Check if this model's score is higher than the current highest score
            if scores.mean() > highest_score:
                highest_score = scores.mean()
                best_model = classifier
                best_model_name = classifier_name

        self.logger.info(f"The best model is {best_model_name} with an accuracy score of {highest_score}")
        # Fit the model on all data available
        best_model.fit(self.x, self.y)
        return best_model

    @staticmethod
    def split_train_test(x, y, nrows: int = 2000):
        """
        Splits the input data and target labels into training and test sets.

        Parameters:
            x (numpy.array or pandas.DataFrame): The input data matrix.
            y (numpy.array or pandas.Series): The target data.
            nrows (int, optional): The number of rows to use for the training set. Defaults to 2000.

        Returns:
            tuple: The split data as four arrays - training data, training labels, test data, test labels.
        """
        return x[0:nrows], y[0:nrows], x[nrows:], y[nrows:]

    def run(self):
        """
        Runs Model Selection and Fit/Predict Steps using best model

        Returns:
           mean_accuracy(float): final accuracy on the test set
        """
        # Splits data into train and test sets
        x_train, y_train, x_test, y_test = self.split_train_test(x=self.x, y=self.y)
        model = self.best_classifier_cv()
        # Predict step on test set
        mean_accuracy = model.score(x_test, y_test)
        return mean_accuracy






