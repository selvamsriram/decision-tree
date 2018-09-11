#-----------------------------------------------------------------------------------------------------------------------
#   Decision Tree built using ID-3 algorithm.
#
#   This program does the following,
#   ---------------------------------
#   1. Uses the ID3 algorithm and constructs the decision tree with provided training data (train.csv).
#   2. Supports validating the tree with given test data. (test.csv).
#   3. Allows K-Fold cross validation testing, file names should be like (fold1.csv, fold2.csv ...).
#   4. Supports limiting the depth of the tree add_node () gives user an option to enter allowed depth in code.
#   5. Displays average accuracy and standard deviation of results using K-Fold cross validation and depth limiting.
#
#   Notes : 
#   -------
#   1. Files train.csv and test.csv should have the label in the column Zero.
#   2. To change testing depths modify "depths []" array. 
#   3. To change the number of folds change the "kfold" variable.
#------------------------------------------------------------------------------------------------------------------------

