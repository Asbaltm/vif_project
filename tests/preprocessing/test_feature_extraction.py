import unittest
from unittest.mock import patch, MagicMock
from pandas import DataFrame
from numpy import array
from preprocessing.feature_extraction import FeatureSelector, f_classif


class TestFeatureSelector(unittest.TestCase):
    @patch('preprocessing.feature_extraction.SelectKBest')
    def test_run(self, mock_selectkBest):
        X = DataFrame({'col1': array([1, 2, 3]), 'col2': array([4, 5, 6]), 'col3': array([7, 8, 9])})
        target_df = DataFrame({'target': array([1, 0, 1])})
        target_column = 'target'
        mock_selector = MagicMock()
        mock_selectkBest.return_value = mock_selector
        mock_selector.get_support.return_value = array([True, False, True])
        fs = FeatureSelector(k=2)
        result = fs.run(X, target_df, target_column)
        mock_selectkBest.assert_called_once()
        mock_selector.fit.assert_called_once_with(X, target_df[target_column])
        self.assertEqual(result.columns.tolist(), ['col1', 'col3'])

