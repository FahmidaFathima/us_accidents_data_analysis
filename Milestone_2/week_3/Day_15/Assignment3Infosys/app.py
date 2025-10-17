import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score

# -------------------- Load Trained Model --------------------
with open("iris_model.pkl", "rb") as f:
    model = pickle.load(f)

# -------------------- Page Config --------------------
st.set_page_config(page_title="🌸 Iris Classifier", page_icon="🌺", layout="wide")

st.title("🌸 Iris Flower Prediction Dashboard")
st.markdown(
    """
    Welcome to the **Iris Classifier App** 🌱🌿🌸  
    Use the sidebar to input flower features and get predictions, or explore the dataset visually.
    """
)

# -------------------- Load Dataset --------------------
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df["species"] = pd.Categorical.from_codes(iris.target, iris.target_names)

# -------------------- Sidebar Inputs --------------------
st.sidebar.header("🔧 Input Flower Measurements")
def user_input_features():
    sepal_length = st.sidebar.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
    sepal_width  = st.sidebar.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider("Petal length (cm)", 1.0, 7.0, 1.4)
    petal_width  = st.sidebar.slider("Petal width (cm)", 0.1, 2.5, 0.2)
    return np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)

input_data = user_input_features()

st.sidebar.header("📊 Explore Dataset")
show_hist = st.sidebar.checkbox("Show Histograms", True)
show_scatter = st.sidebar.checkbox("Show Scatterplots", False)
show_heatmap = st.sidebar.checkbox("Show Correlation Heatmap", False)

# -------------------- Tabs --------------------
tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Visualization", "📑 Dataset Info"])

# -------------------- Prediction Tab --------------------
with tab1:
    st.subheader("🌼 Predict Flower Species")

    if st.button("✨ Run Prediction"):
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        species = ["Setosa 🌱", "Versicolor 🌿", "Virginica 🌸"]

        st.success(f"Predicted Class: **{species[prediction[0]]}**")

        st.subheader("📊 Prediction Probabilities")
        col1, col2, col3 = st.columns(3)
        col1.metric(species[0], f"{prediction_proba[0][0]*100:.2f}%")
        col2.metric(species[1], f"{prediction_proba[0][1]*100:.2f}%")
        col3.metric(species[2], f"{prediction_proba[0][2]*100:.2f}%")

# -------------------- Visualization Tab --------------------
with tab2:
    if show_hist:
        st.subheader("📊 Histograms of Features")
        fig, axs = plt.subplots(2, 2, figsize=(10, 6))
        features = iris.feature_names
        for i, ax in enumerate(axs.flatten()):
            for label, group in df.groupby("species", observed=True):
                ax.hist(group[features[i]], alpha=0.6, label=label, bins=10)
            ax.set_title(features[i])
        axs[0, 0].legend()
        st.pyplot(fig)

    if show_scatter:
        st.subheader("📈 Scatterplot of Feature Pairs")
        feature_x = st.selectbox("X-axis feature", iris.feature_names, index=0)
        feature_y = st.selectbox("Y-axis feature", iris.feature_names, index=2)

        fig, ax = plt.subplots(figsize=(6, 4))
        for label, group in df.groupby("species", observed=True):
            ax.scatter(group[feature_x], group[feature_y], label=label, alpha=0.7)
        ax.set_xlabel(feature_x)
        ax.set_ylabel(feature_y)
        ax.legend()
        st.pyplot(fig)

    if show_heatmap:
        st.subheader("🔥 Correlation Heatmap of Features")
        corr = df.iloc[:, :-1].corr()
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# -------------------- Dataset Info Tab --------------------
with tab3:
    st.subheader("📑 Iris Dataset Preview")
    st.dataframe(df.head())

    st.download_button(
        label="💾 Download Full Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="iris_dataset.csv",
        mime="text/csv",
    )

    # Show model accuracy
    X, y = iris.data, iris.target
    y_pred = model.predict(X)
    acc = accuracy_score(y, y_pred)
    st.info(f"Model Accuracy on Full Dataset: **{acc:.2%}** ✅")
