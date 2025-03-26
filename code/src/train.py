import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
import joblib
import json
from sklearn.model_selection import GridSearchCV

# Step 1: Generate Synthetic Customer Data with Time-Series Information
def time_series_dataset_generation(no_users=100000):
    # Define customer attributes
    names = [f"Customer_{i}" for i in range(1, no_users + 1)]
    salaries = np.random.randint(1500, 10000, size=no_users)  # Wider salary range
    preferred_discount_type = np.random.choice(["Dineout", "Shopping", "Travel", "Entertainment"], size=no_users)
    age=np.random.randint(18,60,size=no_users)
    gender = np.random.choice(["Male","Female"], size=no_users)
    no_of_dependents=np.random.randint(0,3,size=no_users)
    credit_score=np.random.randint(300,850,size=no_users)
    loan_history=np.random.choice(["Good","Bad","No_History"], size=no_users)
    social_media_engagement=np.random.choice(["Ignored","Clicked"], size=no_users)
    # Define spending categories
    categories = {
        "Restaurants": ["Taco Bell", "McDonald's", "KFC", "Subway", "Starbucks", "Burger King", "Dominos", "Pizza Hut"],
        "Shopping": ["Clothes", "Electronics", "Groceries", "Books"],
        "Entertainment": ["Movies", "Concerts", "Sports", "Streaming"],
        "Cashouts": ["ATM", "Online Transfers"],
        "Others": ["Utilities", "Healthcare", "Education"]
    }

    # Simulate time-series data: Monthly spending across categories over the last 3 months
    monthly_spending = []
    favorite_categories = []
    for _ in range(no_users):
        # Randomly assign a favorite category
        fav_category = np.random.choice(list(categories.keys()))
        favorite_categories.append(fav_category)

        # Simulate spending history
        spending = {}
        for month in ["Month_1", "Month_2", "Month_3"]:
            month_spending = {}
            for category, items in categories.items():
                if category == fav_category:
                    # Spend more in the favorite category
                    month_spending[category] = {item: np.random.randint(50, 200) for item in items}
                else:
                    # Spend less in other categories
                    month_spending[category] = {item: np.random.randint(0, 50) for item in items}
            spending[month] = month_spending
        monthly_spending.append(spending)

    # Simulate target variable: Will the customer respond positively? (1 = Yes, 0 = No)
    response_probability = (
            0.3 * (credit_score>650) +  # Higher overall credit score increases likelihood
            0.2 * (loan_history=="Good") + # Good loan history increases likelihood
            0.2 * (salaries / 10000) +  # Higher salary increases likelihood
            0.1 * (social_media_engagement=="Clicked") +
            0.05 * (preferred_discount_type == "Dineout") +  # Dineout preference increases likelihood
            0.05 * (favorite_categories == "Restaurants") +  # Preference for Restaurants increases likelihood
            0.05 * ((age>23) & (age<45)) +
            0.05 * (gender=="Male") +
            np.random.uniform(0, 0.3, size=no_users)  # Random noise
    )
    will_respond = (response_probability > 0.5).astype(int)  # Adjust threshold

    # Create a DataFrame
    data = {
        "Name": names,
        "Age": age,
        "Gender": gender,
        "Number_of_Dependents": no_of_dependents,
        "Salary": salaries,
        "Favorite_Category": favorite_categories,
        "Preferred_Discount_Type": preferred_discount_type,
        "Monthly_Spending": monthly_spending,
        "Credit_Score": credit_score,
        "Loan_History": loan_history,
        "Social_Media_Engagement": social_media_engagement,
        "Will_Respond": will_respond
    }
    return pd.DataFrame(data)


# Step 2: Preprocess Data with Time-Series Aggregation
def preprocess_data_with_time_series(data):
    # Extract time-series features
    total_spending_last_3_months = []
    spending_trend = []
    favorite_item_by_spending = []

    for spending_history in data["Monthly_Spending"]:
        # Total spending over the last 3 months
        total_spending = sum(
            sum(amount for item, amount in month[category].items())
            for month in spending_history.values()
            for category in spending_history["Month_1"].keys()
        )
        total_spending_last_3_months.append(total_spending)

        # Spending trend (Month_3 - Month_1)
        month_1_total = sum(
            sum(amount for item, amount in spending_history["Month_1"][category].items())
            for category in spending_history["Month_1"].keys()
        )
        month_3_total = sum(
            sum(amount for item, amount in spending_history["Month_3"][category].items())
            for category in spending_history["Month_3"].keys()
        )
        trend = month_3_total - month_1_total
        spending_trend.append(trend)

        # Favorite item by total spending
        total_spending_per_item = {}
        for month in spending_history.values():
            for category, items in month.items():
                for item, amount in items.items():
                    total_spending_per_item[item] = total_spending_per_item.get(item, 0) + amount
        favorite_item = max(total_spending_per_item, key=total_spending_per_item.get)
        favorite_item_by_spending.append(favorite_item)

    # Add new features to the dataset
    data["Total_Spending_Last_3_Months"] = total_spending_last_3_months
    data["Spending_Trend"] = spending_trend
    data["Favorite_Item_By_Spending"] = favorite_item_by_spending

    # Normalize spending by salary
    data["Normalized_Spending"] = data["Total_Spending_Last_3_Months"] / data["Salary"]

    # Drop the raw time-series data
    data = data.drop(columns=["Monthly_Spending"])

    # Target encode categorical variables
    def target_encode(df, column, target):
        mean_encoding = df.groupby(column)[target].mean()
        df[f"{column}_Encoded"] = df[column].map(mean_encoding)
        return df

    data = target_encode(data, "Favorite_Category", "Will_Respond")
    data = target_encode(data, "Preferred_Discount_Type", "Will_Respond")
    data = target_encode(data, "Favorite_Item_By_Spending", "Will_Respond")
    data = target_encode(data, "Age", "Will_Respond")
    data = target_encode(data, "Gender", "Will_Respond")
    data = target_encode(data, "Social_Media_Engagement", "Will_Respond")
    data = target_encode(data, "Loan_History", "Will_Respond")

    # Drop original categorical columns
    data = data.drop(columns=["Name", "Favorite_Category", "Preferred_Discount_Type", "Favorite_Item_By_Spending", "Age", "Gender","Loan_History","Social_Media_Engagement"])

    # Features and target
    X = data.drop(columns=["Will_Respond"])
    y = data["Will_Respond"]

    return X, y


# Step 3: Train and Evaluate the Model


def tune_hyperparameters(X, y):
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # SMOTE to balance data
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    # Define parameter grid
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9]
    }

    # Initialize XGBoost model
    model = CatBoostClassifier(random_state=42, eval_metric="logloss")

    # Grid search
    grid_search = GridSearchCV(model, param_grid, scoring='accuracy', cv=3, verbose=2, n_jobs=-1)
    grid_search.fit(X_resampled, y_resampled)

    # Get best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Evaluate
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Best Parameters: {best_params}")
    print(f"Tuned Model Accuracy: {accuracy:.2f}")

    return best_model, accuracy
# Step 4: Main Execution
if __name__ == "__main__":
    # Generate synthetic customer data with time-series information
    customer_data = time_series_dataset_generation(no_users=100000)
    # Preprocess the data
    X, y = preprocess_data_with_time_series(customer_data)

    # Train and evaluate the model
    model, accuracy = tune_hyperparameters(X, y)


    print(f"Final Model Accuracy: {accuracy:.2f}")

    # Save the synthetic data
    customer_data.to_csv("../dataset/customer_dataset.csv", index=False)
    print("Saved high-accuracy synthetic data to 'customer_dataset.csv'")

    # Save the trained model
    joblib.dump(model, "../model/trained_model.joblib")
    print("Saved high-accuracy trained model to 'trained_model.joblib'")

