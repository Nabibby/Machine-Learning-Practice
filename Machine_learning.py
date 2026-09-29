import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


df = pd.read_csv("student_practice.csv")

print(df.head()) #first 5 rows of dataset
print(df.shape)  # number of rows + columns
print(df.isna().sum())  # number of missing values in each column
print(df.isna().mean() * 100) # precentage of missing values 


allowedInfo = df.drop(columns=["passed"])  
answer = df["passed"]  


# making the testing and training split for the dataset
allowedInfo_train, allowedInfo_test, answer_train, answer_test = train_test_split(
    allowedInfo,
    answer,
    test_size = 0.2,
    random_state = 42,
    stratify = answer
)

numeric_columns = allowedInfo.select_dtypes(include = "number").columns
text_columns = allowedInfo.select_dtypes(exclude = "number").columns

numeric_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="median")),
    ("scale", StandardScaler())
])

text_steps = Pipeline([
    ("fill_missing", SimpleImputer(strategy="most_frequent")),
    ("encode", OneHotEncoder(handle_unknown="ignore"))
])

preprocessing = ColumnTransformer([
    ("numbers", numeric_steps, numeric_columns),
    ("text", text_steps, text_columns)
])

model = Pipeline([
    ("preprocessing", preprocessing),
    ("classifier", LogisticRegression(max_iter=1000))
])

model.fit(allowedInfo_train, answer_train)

predictions = model.predict(allowedInfo_test)

print(confusion_matrix(answer_test, predictions))
print(classification_report(answer_test, predictions))





