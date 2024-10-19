import streamlit as st
from PIL import Image
from utils.utils_classification_image import predict_vehicle_type
from utils.utils_object_detection import predict_image_yolo
from utils.utils_qa import get_answer
from utils.utils_text_summarization import abstarctive_summarization, extractive_summarization

# Title
st.title("Machine Learning Web App")

# Sidebar navigation
st.sidebar.title("Select Project")
project = st.sidebar.selectbox("Choose a project", ["Classification Image", "Image Recognition", "QA NLP", "Text Summarization"])

if project == "Classification Image":
    st.header("Image Classification")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify"):
            label = predict_vehicle_type(image)
            st.write(f"Classification result: {label}")

elif project == "Image Recognition":
    st.header("Image Recognition")
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Recognize"):
            recognized_objects = predict_image_yolo(image)
            st.write(f"Recognition result: {recognized_objects}")

elif project == "QA NLP":
    st.header("Question Answering")
    question = st.text_input("Enter your question")
    # context = st.text_area("Enter the context")
    if st.button("Get Answer"):
        answer = get_answer(question)
        st.write(f"Answer: {answer}")

elif project == "Text Summarization":
    st.header("Text Summarization")
    text = st.text_area("Enter the text to summarize")
    if st.button("Summarize"):
        summary = extractive_summarization(text)
        st.write(f"Summary: {summary}")
