import streamlit as st
import pickle
import re

# Load model 
model = pickle.load(open("model.pkl", "rb"))

# Text cleaning (must match training)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

# Page config
st.set_page_config(
    page_title="Sentiment Analyzer 💬",
    page_icon="💜",
    layout="centered"
)

# UI Header

st.title("💬 Review Sentiment Analyzer")
st.markdown("Enter a product review and get instant sentiment prediction.")

# Input
user_input = st.text_area("Enter review text here:")

# Prediction
if st.button("Analyze ✨"):
    if not user_input.strip():
        st.warning("Type something first.")
    else:
        cleaned = clean_text(user_input)

        prediction = model.predict([cleaned])[0]

        # probability handling 
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([cleaned])[0]
            confidence = max(proba) * 100
        else:
            confidence = None

        # Output
        if prediction == 1:
            if confidence:
                st.success(f"Positive 😊 ({confidence:.2f}% confidence)")
            else:
                st.success("Positive 😊")
        else:
            if confidence:
                st.error(f"Negative 😠 ({confidence:.2f}% confidence)")
            else:
                st.error("Negative 😠")

# Examples
st.divider()
st.subheader("💡 Try examples")

col1, col2 = st.columns(2)

with col1:
    if st.button("Good review"):
        st.write("I absolutely love this product. Works perfectly and exceeded expectations.")

with col2:
    if st.button("Bad review"):
        st.write("Terrible quality. Stopped working after two days and very disappointing.")

# Footer

st.markdown("---")
st.markdown("Built with ❤️ using Machine Learning")