from sklearn import tree
import pandas as pd

# Load the data into a pandas DataFrame
df = pd.read_csv("kerala.csv")

# Split the data into features and target variable
X = df.drop("FLOODS", axis=1)
Y = df["FLOODS"]

# Create the decision tree classifier
clf = tree.DecisionTreeClassifier()

# Train the classifier on the data
clf = clf.fit(X, Y)
