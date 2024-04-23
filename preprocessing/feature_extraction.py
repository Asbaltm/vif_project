
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from utils.base import Base


class FeatureSelector(Base):

    def __init__(self, k: int = 5):
        self.k = k

    def run(self, X: pd.DataFrame, target_df: pd.DataFrame, target_column: str):
        """
        This function performs feature selection on a pandas DataFrame.

        Parameters:
        df (pandas.DataFrame): The input DataFrame.
        target_column (str): The name of the target column.
        k (int): The number of top features to select. The default is 10.

        Returns:
        df_selected (pandas.DataFrame): The DataFrame with only the selected features.
        """
        y = target_df[target_column]
        self.logger.info(f"Run Select K Best using anova with {self.k} params")
        selector = SelectKBest(f_classif, k=self.k)
        selector.fit(X, y)

        # Get a boolean array indicating which columns were selected
        mask = selector.get_support()

        # Use this mask to select only the columns from X that were selected by SelectKBest
        selected_columns = X.columns[mask]

        return X[selected_columns]
