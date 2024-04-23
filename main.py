
from utils.logger import LocalLogger
import os
from preprocessing.preprocessing import Inputs
from preprocessing.feature_extraction import FeatureSelector
from model.model import Model

if __name__ == "__main__":
    output_dir = './outputs'
    logger = LocalLogger().logger
    # Create folder outputs if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    logger.info("Start Preprocessing Inputs")
    preprocessed_inputs = Inputs(file_path1="data/data_subset/PS2.txt",
                                 file_path2="data/data_subset/FS1.txt",
                                 file_path3="data/data_subset/profile.txt").run()
    logger.info("End Preprocessing Inputs")
    # Run feature selection step
    X_new = FeatureSelector(k=5).run(X=preprocessed_inputs["X"],
                            target_df=preprocessed_inputs["target_df"],
                            target_column="target")
    logger.info("Start Model Choice, Fit and Predict step")
    accuracy = Model(x=X_new, y=preprocessed_inputs["target_df"]["target"]).run()
    logger.info(f"Final Accuracy on Test Set is {accuracy * 100} %")



