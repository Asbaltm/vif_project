import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from preprocessing.preprocessing import Inputs


class TestInputs(unittest.TestCase):

    @patch("pandas.read_csv")
    def test_load_file_to_dataframe(self, mock_read_csv):
        mock_read_csv.return_value = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        inputs = Inputs("file1", "file2", "file3")
        result = inputs.load_file_to_dataframe("file")
        # Assert
        mock_read_csv.assert_called_once_with("file", sep='\t', header=None)
        pd.testing.assert_frame_equal(result, pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]}))

    def test_encode_column(self):
        df = pd.DataFrame({'A': [100, 200, 300], 'B': [4, 5, 6]})
        inputs = Inputs("file1", "file2", "file3")
        result = inputs.encode_column(df, 0)
        expected_df = pd.DataFrame({'A': [100, 200, 300], 'B': [4, 5, 6], 'target': [1, 0, 0]})
        pd.testing.assert_frame_equal(result, expected_df)

    @patch("sklearn.preprocessing.StandardScaler.fit_transform")
    def test_standardize_df(self, mock_fit_transform):
        mock_fit_transform.return_value = np.array([[0.0, 1.0], [-1.0, 0.0], [1.0, -1.0]])
        df = pd.DataFrame({'A': [2, 1, 3], 'B': [5, 6, 4]})
        inputs = Inputs("file1", "file2", "file3")
        result = inputs.standardize_df(df)
        expected_df = pd.DataFrame({'A': [0.0, -1.0, 1.0], 'B': [1.0, 0.0, -1.0]})
        pd.testing.assert_frame_equal(result, expected_df)

    @patch("sklearn.decomposition.PCA.fit_transform")
    def test_apply_pca(self, mock_fit_transform):
        mock_fit_transform.return_value = np.array([[0.0, 1.0], [-1.0, 0.0], [1.0, -1.0]])
        df = pd.DataFrame({'A': [2, 1, 3], 'B': [5, 6, 4]})
        inputs = Inputs("file1", "file2", "file3")
        result = inputs.apply_pca(df, 2)
        expected_df = pd.DataFrame({'PC1': [0.0, -1.0, 1.0], 'PC2': [1.0, 0.0, -1.0]})
        pd.testing.assert_frame_equal(result, expected_df)

    @patch('os.path.join')
    @patch('pandas.DataFrame.to_csv')
    @patch('pandas.DataFrame.corr')
    @patch.object(Inputs, 'load_file_to_dataframe')
    def test_compute_and_save_corr(self, mock_load_file_to_dataframe, mock_corr, mock_to_csv, mock_path_join):
        mock_load_file_to_dataframe.return_value = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_corr.return_value = pd.DataFrame({'A': [1, 0.5], 'B': [0.5, 1]})
        mock_path_join.return_value = './outputs/X_all_correlation_matrix.csv'
        inputs = Inputs("file1", "file2", "file3")
        inputs.compute_and_save_corr(X_all=mock_load_file_to_dataframe.return_value, analysis=True)
        mock_corr.assert_called_once()
        mock_to_csv.assert_called_once_with('./outputs/X_all_correlation_matrix.csv')

    @patch('pandas.concat')
    @patch.object(Inputs, 'apply_pca')
    @patch.object(Inputs, 'standardize_df')
    @patch.object(Inputs, 'encode_column')
    @patch.object(Inputs, 'load_file_to_dataframe')
    def test_run(self, mock_load_file_to_dataframe, mock_encode_column, mock_standardize_df, mock_apply_pca,
                 mock_concat):
        mock_concat.return_value = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_load_file_to_dataframe.return_value = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        mock_encode_column.return_value = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6], 'target': [0, 1, 0]})
        mock_standardize_df.return_value = pd.DataFrame({'A': [-1, 0, 1], 'B': [1, 0, -1]})
        mock_apply_pca.return_value = pd.DataFrame({'PC1': [-1, 0, 1], 'PC2': [1, 0, -1]})
        inputs = Inputs("file1", "file2", "file3")
        result = inputs.run()
        expected_result = {"X": mock_apply_pca.return_value, "target_df": mock_encode_column.return_value}
        self.assertEqual(result, expected_result)

