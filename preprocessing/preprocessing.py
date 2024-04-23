import os
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from utils.base import Base


class Inputs(Base):

    def __init__(self, file_path1, file_path2, file_path3):
        """
        Initialize Inputs with three file paths and outputs dir.

        Parameters:
        file_path1, file_path2, file_path3 (str): The paths to the text files to load.
        """
        self.file_path1 = file_path1
        self.file_path2 = file_path2
        self.file_path3 = file_path3
        self.output_dir = "./outputs"

    @staticmethod
    def load_file_to_dataframe(file_path):
        """
        This function loads a text file and returns a pandas dataframe.

        Parameters:
        file_path (str): The path to the text file to load.

        Returns:
        df (pandas.DataFrame): The loaded data as a pandas dataframe.
        """
        df = pd.read_csv(file_path, sep='\t', header=None)
        return df

    @staticmethod
    def encode_column(df, column_position):
        """
        This function encodes column values to 1 when it is 100, else 0.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        column_position (int): The position of the column to encode.

        Returns:
        df (pandas.DataFrame): The DataFrame with the encoded column added.
        """
        column_name = df.columns[column_position]
        df["target"] = (df[column_name] == 100).astype(int)
        return df

    @staticmethod
    def standardize_df(df):
        """
        This function standardizes a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.

        Returns:
        standardized_df (pandas.DataFrame): The standardized DataFrame.
        """
        scaler = StandardScaler()
        standardized_data = scaler.fit_transform(df)

        # Create a DataFrame with the standardized data
        standardized_df = pd.DataFrame(standardized_data, columns=df.columns)

        return standardized_df

    @staticmethod
    def apply_pca(df, n_components):
        """
        This function applies PCA to a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        n_components (int): The number of principal components to return.

        Returns:
        transformed_df (pandas.DataFrame): The DataFrame after PCA transformation.
        """
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(df)

        # Create a DataFrame with the transformed data
        transformed_df = pd.DataFrame(transformed_data,
                                      columns=[f"PC{i + 1}" for i in range(n_components)])

        return transformed_df

    def run(self):
        """
        This function loads inputs and apply transformation to get X as input and y as target.

        Returns:
        a dict: X input and target dataframes.
        """
        ps = self.load_file_to_dataframe(self.file_path1)
        fs = self.load_file_to_dataframe(self.file_path2)
        df = self.load_file_to_dataframe(self.file_path3)
        target_df = self.encode_column(df, column_position=1)
        # Concatenate ps and fs column-wise to compute the X matrix
        X_all = pd.concat([ps, fs], axis=1)
        # sanity check: drop duplicated columns if any
        X = X_all.loc[:, ~X_all.columns.duplicated()]
        # sanity check: drop columns with zeroes values if any
        X = X.loc[:, (X != 0).any(axis=0)]
        self.compute_and_save_corr(ps=ps, fs=fs, X_all=X_all, X=X)
        # Standardization step for PCA
        self.logger.info("Standardization of input matrix")
        X = self.standardize_df(X)
        # PCA Step
        self.logger.info("PCA on input matrix to get rid off redundant dimensions")
        X = self.apply_pca(X, n_components=10)
        return {"X": X, "target_df": target_df}

    def compute_and_save_corr(self, analysis=False, **kwargs):
        """
        This function computes correlation matrix, saves it as a csv file

        Parameters:
        analysis is set to False: this option was only used for exploration purpose
        kwargs: dataframes to use.
        """
        if analysis:
            self.logger.info("Start Correlation analysis")
            for name, df in kwargs.items():  # Compute correlation matrix
                corr_matrix = df.corr()
                # Save correlation matrix to csv file
                corr_matrix.to_csv(os.path.join(self.output_dir, f'{name}_correlation_matrix.csv'))
        else:
            self.logger.info("Skip Correlation Analysis")

