import streamlit as st
from evaluate_model import TextEvaluator
import pandas as pd

# Initialize the evaluator
evaluator = TextEvaluator()

# Streamlit app
st.title("Text Evaluation App")

# Input fields
question = st.text_input("Question")
response = st.text_area("Response")
reference = st.text_area("Reference")

# Evaluate button
if st.button("Evaluate"):
    if question and response and reference:
        metrics = evaluator.evaluate_all(question, response, reference)
        st.write("Evaluation Metrics:")
        st.json(metrics)
        metrics_df = pd.DataFrame(metrics.items(), columns=['Metric', 'Value'])
        st.table(metrics_df)
        st.bar_chart(metrics_df.set_index('Metric'))
    else:
        st.error("Please provide a question, response, and reference.")