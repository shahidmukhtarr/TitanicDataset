import pandas as pd
import numpy as np


data = {
    'PassengerId': [21, 22, 23, 24, 25],
    'Survived': [0, 1, 1, 0, 0],
    'Pclass': [2, 1, 3, 3, 2],
    'Name': [
        "Anderson, Mr. Michael",
        "Thomas, Mrs. Laura",
        "Robinson, Miss. Grace",
        "Walker, Mr. Henry",
        "Young, Miss. Lily"
    ],
    'Sex': ['male', 'female', 'female', 'male', 'female'],
    'Age': [37, np.nan, 22, 41, 17],
    'SibSp': [1, 0, 0, 2, 1],
    'Parch': [1, 0, 0, 1, 0],
    'Ticket': ['998877', '334455', '221100', '667788', '559900'],
    'Fare': [30.0, 85.4, 9.1, 25.0, 14.5],
    'Cabin': [np.nan, "D12", "Unknown", np.nan, "C56"],
    'Embarked': ['S', 'C', np.nan, 'Q', 'S']
}

df = pd.DataFrame(data)

# Check for missing values before processing
print("Initial Missing Values:\n", df.isnull().sum(), "\n")

# Step 1: Fill missing values
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Cabin'].fillna("Not Assigned", inplace=True)

# Step 2: Binning Age
def categorize_age(age):
    if age <= 12:
        return "Child"
    elif age <= 19:
        return "Teen"
    elif age <= 35:
        return "Adult"
    elif age <= 60:
        return "Middle-Aged"
    else:
        return "Senior"

df['AgeCategory'] = df['Age'].apply(categorize_age)

# Step 3: Integrate family names
df['Surname'] = df['Name'].apply(lambda x: x.split(',')[0])

# Final cleaned data
print("Missing Values After Cleaning:\n", df.isnull().sum(), "\n")
print("Processed Data Sample:\n", df[['PassengerId', 'Survived', 'Age', 'AgeCategory', 'Embarked', 'Cabin', 'Surname']])
