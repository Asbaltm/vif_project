## Project Purpose
 
The primary purpose of this project is to predict in a hydraulic test rig the binary target output which is the valve condition using raw sensors data PS: pressure and FS: volume flow.

## Main Steps
 
The project follows a set of main steps, detailed as follows:

### Input Preparation
 
In the initial phase, the input data is prepared by identifying and handling high correlation between features. This is important to ensure that our models do not overfit due to redundant information.

### Feature Selection
 
Next, Principal Component Analysis (PCA) is applied to the data. PCA is a dimensionality reduction technique that transforms the data into a new coordinate system in which the greatest variance by any projection of the data lies on the first coordinate (called the first principal component), the second greatest variance on the second coordinate, and so on. This helps in removing redundant features and reducing the dimensionality of the data.

After PCA, SelectKBest feature selection method is employed using ANOVA F-value to further select the most relevant features. This method works by selecting the best features based on univariate statistical tests.

### Model Training and Evaluation
 
With the prepared data, we then train our machine learning models. It's worth noting that using only PS data, a 100% accuracy can be easily achieved. However, this may not always be the case with other datasets or when the training and test datasets are different. Therefore, the application of PCA and SelectKBest is crucial to building a robust model that generalizes well to new, unseen data.

Two machine learning algorithms were used, Random Forest and XGboost.

Both these models were trained on the prepared data and their performance was evaluated using cross-validation. 
The model selected for the final prediction task is the one that performed the best on the cross-validation. This ensures that our final model has not only performed well on the given training data, but it's also expected to perform well on unseen data, thus ensuring the model's robustness and reliability.

# Results:

100% accuracy on test set.

# How to run

To run this project you need to create and activate a virtual environment with python3.11 (recommended).
then, Install dependencies with poetry.  
If python 3.11 isn't your default version you can use pyenv to set it.
```
   $pyenv local 3.11.7
```
Create virtual environment and install dependencies (update pip recommended)

  ```
  $ python -m venv venv
  $ source venv/bin/activate
  $ pip install --upgrade pip
  $ pip install poetry
  $ poetry install 
  ```
 
Great ! You are done, you can now run the main script.

