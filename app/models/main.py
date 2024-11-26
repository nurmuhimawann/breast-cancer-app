import pickle
import pandas as pd
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# Create a model using the breast cancer dataset
def create_model(data):
    # split the data into features and target
    X = data.drop(["class"], axis=1)
    y = data["class"]

    # scale the data
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    X = pd.DataFrame(X, columns=load_breast_cancer().feature_names)

    # split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # train the model
    model = SVC(probability=True)
    model.fit(X_train, y_train)

    # test model
    y_pred = model.predict(X_test)
    print("Accuracy of our model:", accuracy_score(y_test, y_pred))
    print("Classification report: \n", classification_report(y_test, y_pred))

    # return the model
    return model, scaler


# Load the breast cancer dataset from sklearn's datasets module
def get_data(data=load_breast_cancer()):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['class'] = data.target
    return df


# Create and save the model
def main():
    model, scaler = create_model(data=get_data())
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


# Run the main function
if __name__ == "__main__":
    main()
