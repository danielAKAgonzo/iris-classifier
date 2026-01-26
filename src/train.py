from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data      # shape (150, 4)
y = iris.target    # shape (150,)
print(iris.feature_names, iris.target_names)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
# --- Start of new code for Confusion Matrix ---
# Calculate the confusion matrix using true labels (y_test) and predictions (y_pred)
cm = confusion_matrix(y_test, y_pred)

# Create a ConfusionMatrixDisplay object for plotting
# We use iris.target_names so the labels (setosa, versicolor, virginica) show up
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=iris.target_names)

# Plot the confusion matrix
# cmap=plt.cm.Blues adds the blue color scheme from your example image
disp.plot(cmap=plt.cm.Blues)

# Set the title of the plot
plt.title("Confusion Matrix - Decision Tree Classifier")

# Display the plot
plt.show()
# --- End of new code ---
print("Predictions:", y_pred[:5])
print("True labels:", y_test[:5])
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
model2 = KNeighborsClassifier(n_neighbors=5)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)
print("k-NN accuracy:", accuracy_score(y_test, y_pred2))
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)
import joblib  # Make sure this is at the very top!
import os

# ... after your model2.fit() code ...

# Create the folder if it's missing
os.makedirs('outputs', exist_ok=True) 

# This is the line that actually creates the file!
joblib.dump(model2, 'outputs/model.joblib') 
print("SUCCESS: model.joblib has been created in the outputs folder.")
