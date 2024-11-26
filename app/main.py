import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from sklearn.datasets import load_breast_cancer


# Load the breast cancer dataset from sklearn's datasets module
def get_data(data=load_breast_cancer()):
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["class"] = data.target
    return df


# Add a sidebar to the app
def add_sidebar(data=get_data()):
    # title and description
    st.sidebar.header("Cell Nuclei Measurements")

    # create a dictionary of labels and keys for the sliders
    slider_labels = [
        ("Radius (mean)", "mean radius"),
        ("Texture (mean)", "mean texture"),
        ("Perimeter (mean)", "mean perimeter"),
        ("Area (mean)", "mean area"),
        ("Smoothness (mean)", "mean smoothness"),
        ("Compactness (mean)", "mean compactness"),
        ("Concavity (mean)", "mean concavity"),
        ("Concave points (mean)", "mean concave points"),
        ("Symmetry (mean)", "mean symmetry"),
        ("Fractal dimension (mean)", "mean fractal dimension"),
        ("Radius (se)", "radius error"),
        ("Texture (se)", "texture error"),
        ("Perimeter (se)", "perimeter error"),
        ("Area (se)", "area error"),
        ("Smoothness (se)", "smoothness error"),
        ("Compactness (se)", "compactness error"),
        ("Concavity (se)", "concavity error"),
        ("Concave points (se)", "concave points error"),
        ("Symmetry (se)", "symmetry error"),
        ("Fractal dimension (se)", "fractal dimension error"),
        ("Radius (worst)", "worst radius"),
        ("Texture (worst)", "worst texture"),
        ("Perimeter (worst)", "worst perimeter"),
        ("Area (worst)", "worst area"),
        ("Smoothness (worst)", "worst smoothness"),
        ("Compactness (worst)", "worst compactness"),
        ("Concavity (worst)", "worst concavity"),
        ("Concave points (worst)", "worst concave points"),
        ("Symmetry (worst)", "worst symmetry"),
        ("Fractal dimension (worst)", "worst fractal dimension"),
    ]

    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean()),
        )
    return input_dict


# Scale the input values
def get_scaled_values(input_dict, data=get_data()):
    X = data.drop(["class"], axis=1)

    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict


# Create a radar chart
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = [
        "Radius",
        "Texture",
        "Perimeter",
        "Area",
        "Smoothness",
        "Compactness",
        "Concavity",
        "Concave Points",
        "Symmetry",
        "Fractal Dimension",
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["mean radius"],
                input_data["mean texture"],
                input_data["mean perimeter"],
                input_data["mean area"],
                input_data["mean smoothness"],
                input_data["mean compactness"],
                input_data["mean concavity"],
                input_data["mean concave points"],
                input_data["mean symmetry"],
                input_data["mean fractal dimension"],
            ],
            theta=categories,
            fill="toself",
            name="Mean Value",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["radius error"],
                input_data["texture error"],
                input_data["perimeter error"],
                input_data["area error"],
                input_data["smoothness error"],
                input_data["compactness error"],
                input_data["concavity error"],
                input_data["concave points error"],
                input_data["symmetry error"],
                input_data["fractal dimension error"],
            ],
            theta=categories,
            fill="toself",
            name="Standard Error",
        )
    )
    fig.add_trace(
        go.Scatterpolar(
            r=[
                input_data["worst radius"],
                input_data["worst texture"],
                input_data["worst perimeter"],
                input_data["worst area"],
                input_data["worst smoothness"],
                input_data["worst compactness"],
                input_data["worst concavity"],
                input_data["worst concave points"],
                input_data["worst symmetry"],
                input_data["worst fractal dimension"],
            ],
            theta=categories,
            fill="toself",
            name="Worst Value",
        )
    )

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        width=650,
        height=650,
        legend=dict(
            orientation="h",
            y=1.15,
            x=0,
            xanchor="center",
            font=dict(size=16)
        ),
    )
    return fig


# Add the predictions to the app
def add_predictions(input_data):
    model = pickle.load(open("app/models/model.pkl", "rb"))
    scaler = pickle.load(open("app/models/scaler.pkl", "rb"))

    input_array = np.array(list(input_data.values())).reshape(1, -1)
    scaled_input = scaler.transform(input_array)

    prediction = model.predict(scaled_input)
    st.subheader("Cell cluster prediction")
    st.write("The cell cluster is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis malignant'>Malignant</span>", unsafe_allow_html=True)
    else:
        st.write(
            "<span class='diagnosis benign'>Benign</span>", unsafe_allow_html=True
        )

    st.write(
        f"Probability of being benign: ", round(model.predict_proba(scaled_input)[0][1], 3),
    )

    st.write(
        f"Probability of being malignant: ", round(model.predict_proba(scaled_input)[0][0], 3)
    )

    st.write("Algorithm:", "Support Vector Machine")
    st.write("Model Accuracy:", 0.98)

    st.write(
        "This app is just a prototype designed to assist medical professionals and students in making a diagnosis, but it should not be used as a substitute for a professional diagnosis."
    )


# Main function
def main():
    st.set_page_config(
        page_title="Breast Cancer Prediction",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    with open("app/assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)

    input_data = add_sidebar()

    with st.container():
        st.title("Prototype Model for Predicting Breast Cancer")
        st.write(
            "This app is a simple prototype created for educational purposes "
            "and should not be used as a strict diagnostic tool. Connect to "
            "your cytology lab to help determine whether a breast mass is "
            "likely benign or malignant using predictions from a machine "
            "learning model. The model generates its predictions based on "
            "measurements provided by your cytology lab, which can also be "
            "manually adjusted using the sliders in the sidebar. The radar "
            "chart displays the input data, and the predictions are displayed "
            "below."
        )

    col1, col2 = st.columns([4, 1.8])

    with col1:
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart)

    with col2:
        add_predictions(input_data)


if __name__ == "__main__":
    main()
