### This project demonstrates a full end-to-end process from data cleaning, visualization, feature engineering, to model training, hyperparameter tuning, and evaluation, ultimately predicting customer churn.

### 1. **Data Loading and Preprocessing**
   - **Dataset**: The `Customer-Churn.csv` file is loaded into a pandas DataFrame.
   - **Data Cleaning**: The `TotalCharges` column, initially detected as an object type, is converted to a numeric type. Missing values are handled by dropping rows with null values.
   - **Feature Adjustment**: The `customerID` column is dropped as it doesn’t contribute to predictions. The `PaymentMethod` column is modified to remove the term "automatic" from entries.

### 2. **Data Visualization**
   - **Churn Distribution**: A bar plot shows the proportion of customers who have churned vs. those who haven’t, visualizing the class imbalance.
   - **Demographic Information**: A stacked bar plot visualizes how demographic variables (gender, senior citizen status, partner, dependents) relate to churn.
   - **Customer Account and Services Information**: Similar stacked plots explore variables like contract type, payment method, internet services, and support options.

### 3. **Feature Engineering**
   - **Label Encoding**: Binary categorical features such as `gender`, `Partner`, `PhoneService`, and `Churn` are label-encoded (i.e., converted to numerical values).
   - **One-Hot Encoding**: Categorical variables with more than two levels (e.g., `InternetService`, `PaymentMethod`, `Contract`) are transformed using one-hot encoding.
   - **Normalization**: Min-max normalization is applied to continuous variables like `tenure`, `MonthlyCharges`, and `TotalCharges` to scale them to a range of [0,1], improving model performance.

### 4. **Model Training and Evaluation**
   - **Train-Test Split**: The dataset is split into training (80%) and testing (20%) sets using `train_test_split`.
   - **Model Selection**: Six machine learning models are evaluated: 
     - Dummy Classifier
     - k-Nearest Neighbors
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest Classifier
     - Gradient Boosting Classifier
   - **Accuracy Measurement**: Each model is trained and evaluated on the test set using accuracy as the performance metric.

### 5. **Hyperparameter Tuning**
   - **Gradient Boosting Tuning**: `RandomizedSearchCV` is used to optimize the hyperparameters of the Gradient Boosting Classifier, searching through various combinations of estimators, depth, features, and sample splits.
   - **Best Parameters**: The best-performing combination of hyperparameters is identified and displayed.

### 6. **Confusion Matrix**
   - The confusion matrix is generated to further evaluate the performance of the optimized Gradient Boosting model by showing the number of true positives, true negatives, false positives, and false negatives.
