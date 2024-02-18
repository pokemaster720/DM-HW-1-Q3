# DM-HW-1-Q3

The provided code defines a sequence of operations to handle a common machine learning task—assessing and visualizing the performance 
(in terms of the Mean Squared Error, MSE) of a regularized linear regression model (Ridge regression) trained on different subsets of dataset. 
The code is composed of Python functions performing specific tasks and uses libraries such as `pandas`, `numpy`, `sklearn`, and `matplotlib`. 
To go through this code step by step:

1. **Imports:**
   - `pandas` for data manipulation and analysis.
   - `numpy` for numerical operations.
   - `mean_squared_error` from `sklearn.metrics` to compute the MSE.
   - `Ridge` from `sklearn.linear_model` for Ridge regression.
   - `matplotlib.pyplot` for plotting graphs.
   - `resample` from `sklearn.utils` to randomly sample a subset of the data.
     
2. **Create Subset Training Files:**
   - Function `create_subset_training_files` takes an original CSV file containing the larger dataset, a list of subset sizes, and an optional prefix.
   - It reads the original data using Pandas, then creates smaller CSV files, each corresponding to a different subset of the data.
   - The subsets are determined by the `size` variable, which specifies how many rows to include.
   - Each new CSV files are named following the pattern `train-{size}(1000)-100.csv`.
   - 

3. **Compute Mean Squared Error (MSE) for Different Training Subsets and Lambda Values:**
   - Function `compute_mse_for_subsets` calculates the MSE of Ridge Regression models for varying training set sizes and regularization parameter λ (lambda).
   - The test set is loaded and prepared for evaluating the models.
   - The function loads a test dataset from a CSV file once to avoid reloading it multiple times within the loop.
   - It iterates over the specified `subset_sizes` and for each size, it takes a random subsample of the training dataset `repetitions` times (default `repetitions=10`).
   - For each subset, the function trains a Ridge regression model for each specified regularization strength (`lambdas`).
   - It computes the Mean Squared Error between the predictions on the test set and the actual test values, and it stores these values.
   - After performing the repetitions, it averages the MSE values to get a stable performance metric.   - For each subset size, multiple repetitions of training occur to average the results for better assessment. Each repetition involves:
     - Randomly sampling a training subset of given size (to inject randomness and better estimate model performance).
     - Training the Ridge Regression model on this subset with varying values of λ.
     - Predicting the target values on the test set.
     - Calculating the MSE of these predictions against the actual test target values.
   - The function stores these MSE values in a nested dictionary and computes the average MSE across all repetitions.

4. **Plot Learning Curves:**
   - Function `plot_learning_curves` is responsible for visualizing the effect of training set size and the regularization strength on the model's test MSE.
   - It generates a line plot with the x-axis as the size of the training set and the y-axis as the averaged test MSE.
   - Each line in the plot corresponds to a different value of λ (lambda), showing how the MSE varies with training set size.

5. **Execution of Defined Steps:**
   - The parameters such as the paths to the original and test dataset files, the desired subset sizes, and the values of lambda to be considered are predefined.
   - The `create_subset_training_files` function is called to generate smaller training sets.
   - The `full_train_file` variable is set, although it seems redundant as it is the same as `original_file`.
   - The `compute_mse_for_subsets` function is invoked with the full training file, the test file, an extended list of subset sizes (including the size 1000, the full training set size), and the predefined lambda values to compute the average MSE.
   - Finally, the `plot_learning_curves` function is called to visualize the results.

5. **`plot_learning_curves` function:**
   -This function takes the average MSE results and the list of lambdas to plot learning curves. Learning curves are graphs that show the performance of
   the model on the vertical axis (MSE in this case) versus the experience of the model on the horizontal axis (training set size). Each curve corresponds
   to a different lambda.        
   - This function plots learning curves that visualize the average test MSE against the size of the training set for each regularization strength (`lambda`).
   - `matplotlib` is used to handle the plotting, with separately labeled lines for each lambda showing how the model's performance changes with different amounts of training data.

6. **Execution of functions:**
   - The paths for `original_file` and `test_file` are defined, which are expected to be CSV files.
   - `subset_sizes` and `lambdas` are specified as parameters:
      - `subset_sizes` refers to the number of samples to consider for creating subsets of the training data.
      - `lambdas` are the regularization strength parameters for the Ridge regression model.
   - The `create_subset_training_files` function is called to create CSV files of subset training data.
   - The `compute_mse_for_subsets` function is called to train models and calculate MSE for various subsets and lambda values.
   - The `plot_learning_curves` function is then called to visualize the learning curves of the Ridge regression model.

**Overall**, the script prepares training subsets, evaluates how different sizes of training sets and regularization strengths affect model performance, 
and visualizes the results to aid in selecting the optimal model complexity and training set size. This approach can help avoid overfitting by demonstrating 
how the model generalizes to unseen data.
