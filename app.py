import streamlit as st
from main import process_resume
import tempfile

st.title("Resume Tailoring AI Agent")

uploaded_file = st.file_uploader("Upload your resume (PDF)", type=["pdf"])
job_description = st.text_area("Paste the job description here")

tailor_button = st.button("Tailor My Resume")

if tailor_button and uploaded_file and job_description:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    tailored_resume = process_resume(tmp_path, job_description)
    st.subheader("Tailored Resume:")
    st.text_area("", tailored_resume, height=400) 