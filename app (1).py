import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF
import io

# ===============================
# PAGE CONFIG
# ===============================

st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="wide"
)

# ===============================
# LOAD MODEL
# ===============================

model = pickle.load(open("model.pkl", "rb"))

# ===============================
# TITLE
# ===============================

st.title("🌸 Iris Flower Species Predictor")

st.markdown("""
Predict the species of an Iris flower using Machine Learning.
""")

# ===============================
# SIDEBAR
# ===============================

st.sidebar.header("📊 Model Information")

st.sidebar.success("Random Forest Classifier")

st.sidebar.write("Input Features:")
st.sidebar.write("""
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width
""")

# ===============================
# FLOWER IMAGE
# ===============================

st.image(
    "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_versicolor_3.jpg",
    caption="Iris Flower",
    width=500
)

# ===============================
# INPUT SECTION
# ===============================

col1, col2 = st.columns(2)

with col1:

    sl = st.number_input(
        "Sepal Length",
        min_value=4.0,
        max_value=8.0,
        value=5.1
    )

    sw = st.number_input(
        "Sepal Width",
        min_value=2.0,
        max_value=5.0,
        value=3.5
    )

with col2:

    pl = st.number_input(
        "Petal Length",
        min_value=1.0,
        max_value=7.0,
        value=1.4
    )

    pw = st.number_input(
        "Petal Width",
        min_value=0.1,
        max_value=3.0,
        value=0.2
    )

# ===============================
# SESSION STATE
# ===============================

if "pred" not in st.session_state:
    st.session_state.pred = None

if "prob" not in st.session_state:
    st.session_state.prob = None

# ===============================
# PREDICTION BUTTON
# ===============================

if st.button("🔍 Predict Species"):

    sample = np.array([[sl, sw, pl, pw]])

    prediction = model.predict(sample)[0]

    probability = model.predict_proba(sample)[0]

    st.session_state.pred = prediction
    st.session_state.prob = probability

    st.success(f"🌼 Predicted Species: {prediction}")

    st.subheader("Prediction Confidence")

    confidence = int(max(probability) * 100)

    st.progress(confidence)

    st.write(f"Confidence: {confidence}%")

    # Probability Table

    prob_df = pd.DataFrame({
        "Species": ["Setosa", "Versicolor", "Virginica"],
        "Probability (%)": [
            round(probability[0] * 100, 2),
            round(probability[1] * 100, 2),
            round(probability[2] * 100, 2)
        ]
    })

    st.dataframe(prob_df)

    # Probability Chart

    fig, ax = plt.subplots()

    ax.bar(
        prob_df["Species"],
        prob_df["Probability (%)"]
    )

    ax.set_title("Prediction Probabilities")
    ax.set_ylabel("Probability (%)")

    st.pyplot(fig)

# ===============================
# FEATURE IMPORTANCE
# ===============================

st.subheader("📈 Feature Importance")

feature_names = [
    "Sepal Length",
    "Sepal Width",
    "Petal Length",
    "Petal Width"
]

# Example values
# Replace with:
# model.feature_importances_
# if your model supports it

importance = model.feature_importances_

fig2, ax2 = plt.subplots()

ax2.bar(
    feature_names,
    importance
)

ax2.set_title("Feature Importance")

st.pyplot(fig2)

# ===============================
# PDF REPORT
# ===============================

st.subheader("📄 Generate Report")

if st.button("Generate PDF Report"):

    if st.session_state.pred is None:

        st.error(
            "Please make a prediction first."
        )

    else:

        pdf = FPDF()

        pdf.add_page()

        pdf.set_font(
            "Arial",
            "B",
            16
        )

        pdf.cell(
            200,
            10,
            txt="Iris Flower Prediction Report",
            ln=True
        )

        pdf.ln(10)

        pdf.set_font(
            "Arial",
            size=12
        )

        pdf.cell(
            200,
            10,
            txt=f"Prediction: {st.session_state.pred}",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Sepal Length: {sl}",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Sepal Width: {sw}",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Petal Length: {pl}",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Petal Width: {pw}",
            ln=True
        )

        pdf.ln(10)

        pdf.cell(
            200,
            10,
            txt="Prediction Probabilities",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Setosa: {round(st.session_state.prob[0]*100,2)}%",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Versicolor: {round(st.session_state.prob[1]*100,2)}%",
            ln=True
        )

        pdf.cell(
            200,
            10,
            txt=f"Virginica: {round(st.session_state.prob[2]*100,2)}%",
            ln=True
        )

        pdf_bytes = pdf.output(dest="S").encode("latin-1")

        buffer = io.BytesIO(pdf_bytes)

        st.download_button(
            label="⬇️ Download PDF Report",
            data=buffer,
            file_name="iris_report.pdf",
            mime="application/pdf"
        )

# ===============================
# FOOTER
# ===============================

st.markdown("---")

st.markdown(
    "🚀 Built using Streamlit, Scikit-Learn, Random Forest, Matplotlib and Python"
)