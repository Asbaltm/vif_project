import unittest
from unittest.mock import patch, MagicMock
import numpy as np
from model.model import Model
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier


class TestModel(unittest.TestCase):

    def test_init(self):
        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        # Instantiate Model Class
        model = Model(x, y)
        self.assertIsInstance(model.classifiers["XGBClassifier"], XGBClassifier)
        self.assertIsInstance(model.classifiers["RandomForestClassifier"], RandomForestClassifier)
        np.testing.assert_array_equal(model.x, x)
        np.testing.assert_array_equal(model.y, y)

    @patch("sklearn.model_selection.cross_val_score")
    def test_best_classifier_cv(self, mock_cross_val_score):
        mock_cross_val_score.return_value = np.array([0.8, 0.9, 0.85])

        x = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([0, 1, 0])
        model = Model(x, y)
        result = model.best_classifier_cv(cv=2)
        self.assertEqual(result, model.classifiers["XGBClassifier"])

    def test_split_train_test(self):
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        x_train, y_train, x_test, y_test = Model.split_train_test(x, y, nrows=2)
        np.testing.assert_array_equal(x_train, np.array([[1, 2], [3, 4]]))
        np.testing.assert_array_equal(y_train, np.array([0, 1]))
        np.testing.assert_array_equal(x_test, np.array([[5, 6], [7, 8]]))
        np.testing.assert_array_equal(y_test, np.array([0, 1]))

    @patch.object(Model, "best_classifier_cv")
    @patch.object(Model, "split_train_test")
    def test_run(self, mock_split_train_test, mock_best_classifier_cv):
        x = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([0, 1, 0, 1])
        model = Model(x, y)
        mock_model = MagicMock()
        mock_model.score.return_value = 0.85
        mock_best_classifier_cv.return_value = mock_model
        mock_split_train_test.return_value = (x[:2], y[:2], x[2:], y[2:])
        result = model.run()
        self.assertEqual(result, 0.85)



