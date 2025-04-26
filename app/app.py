import streamlit as st
import os
import pandas as pd
import json
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage

load_dotenv()

# GitHub Models / Azure AI Inference setup
endpoint = "https://models.github.ai/inference"
model_name = "meta/Meta-Llama-3-8B-Instruct"
token = os.getenv("GITHUB_TOKEN")
client = ChatCompletionsClient(
    endpoint=endpoint,
    credential=AzureKeyCredential(token),
)

def extract_text_from_pdf(file):
    endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
    doc_client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    model_id = "prebuilt-read"
    text = ""
    poller = doc_client.begin_analyze_document(model_id, file)
    result = poller.result()
    for page in result.pages:
        for line in page.lines:
            text += line.content + "\n"
    return text

def perform_resume_matching(resume_text, job_desc_text):
    prompt = f"""
    You are an HR expert evaluating resume suitability for a job. Given a resume and job description, determine if the resume is suitable for the role. Return a JSON object with 'suitability' ('Suitable' or 'Not Suitable') and 'score' (a float between 0 and 1 representing confidence). Ensure the response is strictly JSON, enclosed in curly braces, with no additional text.

    Example response:
    {{
        "suitability": "Suitable",
        "score": 0.85
    }}

    Resume: {resume_text[:512]}
    Job Description: {job_desc_text[:512]}
    """
    response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant."),
            UserMessage(prompt),
        ],
        temperature=0.5,
        top_p=1.0,
        max_tokens=100,
        model=model_name
    )
    content = response.choices[0].message.content.strip()
    try:
        result = json.loads(content)
        score = result["score"]
        suitability = result["suitability"]
    except Exception:
        score = 0.0
        suitability = "Not Suitable"
    return score, suitability

def analyze_sentiment(feedback):
    prompt = f"""
    You are a sentiment analysis expert. Given a piece of feedback, classify its sentiment as 'Positive', 'Negative', or 'Neutral'. Return only the sentiment class as a single word.

    Feedback: {feedback}
    """
    response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant."),
            UserMessage(prompt),
        ],
        temperature=0.3,
        top_p=1.0,
        max_tokens=10,
        model=model_name
    )
    return response.choices[0].message.content.strip()

def github_attrition_summary_and_recommendations(feedback_list, attrition_rate):
    feedback_str = "\n".join([f"- {f['text']} ({f['sentiment']})" for f in feedback_list])
    prompt = f"""
    You are an HR analytics expert. Given the following employee feedback and their classified sentiments (positive, negative, neutral), and an attrition rate of {attrition_rate:.1f}%, do the following:
    1. Summarize the overall sentiment distribution.
    2. Assess the attrition risk as 'Low', 'Medium', or 'High' based on the attrition rate.
    3. Provide 2-3 actionable, data-driven recommendations to improve employee engagement and reduce attrition, tailored to the issues reflected in the feedback and the attrition rate.
    Format your response as:
    **Summary**: [Your summary]
    **Attrition Risk**: [Low/Medium/High]
    **Recommendations**:
    - [Recommendation 1]
    - [Recommendation 2]
    - [Recommendation 3]

    Feedback List:
    {feedback_str}
    """
    response = client.complete(
        messages=[
            SystemMessage("You are a helpful assistant."),
            UserMessage(prompt),
        ],
        temperature=0.7,
        top_p=1.0,
        max_tokens=300,
        model=model_name
    )
    return response.choices[0].message.content.strip()

st.title("ResuMetrics")

tab1, tab2 = st.tabs(["Resume Screening", "Employee Sentiment Analysis"])

with tab1:
    st.header("Resume Screening")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume")
    job_description = st.text_area("Enter Job Description", placeholder="e.g., Data analyst with minimum 2 years of experience.")

    if st.button("Analyze Resume"):
        if resume_file and job_description:
            with st.spinner("Analyzing Resume..."):
                resume_text = extract_text_from_pdf(resume_file)
                score, suitability = perform_resume_matching(resume_text, job_description)
                st.write("Suitability:")
                st.write(f"- Status: {suitability}")
                st.write(f"- Score: {score:.2f}")
        else:
            st.warning("Please upload a resume and enter a job description.")

with tab2:
    st.header("Employee Sentiment Analysis")
    feedback_file = st.file_uploader("Upload Feedback (CSV)", type=["csv"], key="feedback")
    st.write("CSV format: Column 'feedback' with employee feedback text.")

    if st.button("Analyze Sentiment"):
        if feedback_file:
            try:
                with st.spinner("Reading Feedback CSV..."):
                    # Read CSV directly using pandas
                    df = pd.read_csv(feedback_file)

                if "feedback" not in df.columns:
                    st.error("CSV must contain a 'feedback' column.")
                else:
                    feedback_list = []
                    for _, row in df.iterrows():
                        text = row["feedback"]
                        sentiment = analyze_sentiment(text)
                        feedback_list.append({"text": text, "sentiment": sentiment})

                    total = len(feedback_list)
                    num_positive = sum(1 for f in feedback_list if f['sentiment'].lower() == 'positive')
                    num_negative = sum(1 for f in feedback_list if f['sentiment'].lower() == 'negative')
                    num_neutral = sum(1 for f in feedback_list if f['sentiment'].lower() == 'neutral')
                    avg_sentiment = (num_positive - num_negative) / total if total else 0
                    attrition_rate = (num_negative / total) * 100 if total else 0

                    st.write("Feedback Analysis:")
                    for item in feedback_list:
                        st.write(f"- Feedback: {item['text'][:100]}... Sentiment: {item['sentiment']}")

                    summary = github_attrition_summary_and_recommendations(feedback_list, attrition_rate)

                    st.write("Summary and Recommendations:")
                    st.write(summary)
                    st.write(f"Attrition Rate: {attrition_rate:.1f}%")
                    st.write(f"Average Sentiment Score: {avg_sentiment:.2f}")
            except Exception as e:
                st.error(f"Error processing CSV: {str(e)}")
        else:
            st.warning("Please upload a feedback CSV file.")
