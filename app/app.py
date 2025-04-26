import streamlit as st
import os
import glob
import pandas as pd
import json
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from groq import Groq
from azure.search.documents import SearchClient
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse

load_dotenv()

app = FastAPI()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def extract_text_from_pdf(file):
    endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
    key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    model_id = "prebuilt-read"
    text = ""
    poller = client.begin_analyze_document(model_id, file)
    result = poller.result()
    for page in result.pages:
        for line in page.lines:
            text += line.content + "\n"
    return text

def extract_resume_keywords(resume_text):
    prompt = f"""
    You are an HR expert specializing in resume analysis for Software Engineer roles. Given a resume, extract 5-10 key technical skills, certifications, or tools mentioned, focusing on those relevant to software engineering (e.g., programming languages, frameworks, cloud platforms). Return the keywords as a comma-separated list. If no relevant keywords are found, return an empty string.

    Resume: {resume_text}
    """
    completion = client.chat.completions.create(
        model="compound-beta",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=100,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content.strip()

def populate_resume_index():
    doc_intel_client = DocumentIntelligenceClient(
        endpoint=os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT"),
        credential=AzureKeyCredential(os.getenv("DOCUMENT_INTELLIGENCE_KEY"))
    )
    service_name = os.getenv("SEARCH_SERVICE_NAME")
    index_name = "resume-index"
    api_key = os.getenv("SEARCH_KEY")
    endpoint = f"https://{service_name}.search.windows.net"
    search_client = SearchClient(endpoint=endpoint, index_name=index_name, credential=AzureKeyCredential(api_key))
    
    resume_dir = "data/resumes/"
    documents = []
    for idx, pdf_path in enumerate(glob.glob(os.path.join(resume_dir, "*.pdf"))):
        with open(pdf_path, "rb") as f:
            text = extract_text_from_pdf(f)
        document = {
            "id": f"resume_{idx+1}",
            "content": text
        }
        documents.append(document)
    if documents:
        search_client.upload_documents(documents)
        return len(documents)
    return 0

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
    completion = client.chat.completions.create(
        model="compound-beta",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=100,
        top_p=1,
        stream=False,
        stop=None,
    )
    response = completion.choices[0].message.content.strip()
    try:
        result = json.loads(response)
        score = result["score"]
        suitability = result["suitability"]
    except json.JSONDecodeError:
        score = 0.0
        suitability = "Not Suitable"
    return score, suitability

def generate_resume_analysis(resume_text, job_desc_text):
    prompt = f"""
    You are an expert HR consultant specializing in resume optimization for Software Engineer roles. Given a candidate's resume and a job description, perform a detailed analysis of the resume's strengths and weaknesses relative to the job requirements. Identify specific skill gaps, missing qualifications, or experience deficiencies. Provide 3-5 actionable suggestions to improve the resume, focusing on technical skills, certifications, and project experience. Ensure suggestions are concise, prioritized by impact, and tailored to the job description. Format the response as follows:

    **Analysis**:
    - Strengths: [List specific strengths]
    - Weaknesses: [List specific gaps or deficiencies]

    **Suggestions**:
    - [Suggestion 1]
    - [Suggestion 2]
    - [Suggestion 3]

    Resume: {resume_text}
    Job Description: {job_desc_text}
    """
    completion = client.chat.completions.create(
        model="compound-beta",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=300,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content.strip()

def analyze_sentiment(feedback):
    prompt = f"""
    You are a sentiment analysis expert. Given a piece of feedback, classify its sentiment as 'Positive', 'Negative', or 'Neutral'. Return only the sentiment class as a single word.

    Feedback: {feedback}
    """
    completion = client.chat.completions.create(
        model="compound-beta",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.5,
        max_completion_tokens=10,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content.strip()

def analyze_feedback_with_gpt4(feedback):
    prompt = f"""
    You are an HR specialist analyzing employee feedback. Given a piece of feedback, assess its sentiment (positive, negative, neutral) and explain the key emotional or thematic elements driving the sentiment. Highlight specific phrases or themes (e.g., teamwork, workload, recognition). Format the response as follows:

    **Sentiment**: [Positive/Negative/Neutral]
    **Analysis**: [Detailed explanation of emotional/thematic elements]

    Feedback: {feedback}
    """
    completion = client.chat.completions.create(
        model="compound-beta",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=150,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content.strip()

def generate_sentiment_summary(feedback_list):
    feedback_str = "\n".join([f"- {f['text']} ({f['sentiment']})" for f in feedback_list])
    prompt = f"""
    You are an HR specialist analyzing employee feedback to assess sentiment and recommend engagement strategies. Given a list of feedback and their predicted sentiments (positive, negative, neutral), summarize the overall nature of the responses (e.g., predominantly positive, mixed, mostly negative). Assess the attrition risk (low, medium, high) based on the proportion of negative feedback. Provide 2-3 targeted recommendations to improve employee engagement, addressing specific issues in the feedback. Ensure recommendations are actionable, aligned with HR best practices, and prioritized by impact. Format the response as follows:

    **Summary**:
    - Overall Nature: [e.g., Predominantly positive]
    - Attrition Risk: [Low/Medium/High]

    **Recommendations**:
    - [Recommendation 1]
    - [Recommendation 2]

    Feedback List:
    {feedback_str}
    """
    completion = client.chat.completions.create(
        model="compound-beta",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_completion_tokens=200,
        top_p=1,
        stream=False,
        stop=None,
    )
    return completion.choices[0].message.content.strip()

@app.post("/api/resume-screen")
async def resume_screen(job_description: str, resume: UploadFile = File(...)):
    resume_text = extract_text_from_pdf(resume.file)
    keywords = extract_resume_keywords(resume_text)
    score, suitability = perform_resume_matching(resume_text, job_description)
    analysis = generate_resume_analysis(resume_text, job_description)
    return JSONResponse({
        "suitability": suitability,
        "score": score,
        "keywords": keywords,
        "analysis": analysis
    })

@app.post("/api/sentiment-analysis")
async def sentiment_analysis(feedback: UploadFile = File(...)):
    df = pd.read_csv(feedback.file)
    feedback_list = []
    for _, row in df.iterrows():
        text = row["feedback"]
        sentiment = analyze_sentiment(text)
        gpt4_analysis = analyze_feedback_with_gpt4(text)
        feedback_list.append({"text": text, "sentiment": sentiment, "gpt4_analysis": gpt4_analysis})
    summary = generate_sentiment_summary(feedback_list)
    return JSONResponse({
        "feedback_list": feedback_list,
        "summary": summary
    })

st.title("HR-Tech Solution")

tab1, tab2 = st.tabs(["Resume Screening", "Employee Sentiment Analysis"])

with tab1:
    st.header("Resume Screening")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], key="resume")
    job_description = st.text_area("Enter Job Description", placeholder="e.g., Software Engineer with expertise in Python and Azure.")
    
    if st.button("Populate Resume Index"):
        num_uploaded = populate_resume_index()
        st.success(f"Uploaded {num_uploaded} resumes to resume-index.")

    if resume_file and job_description:
        resume_text = extract_text_from_pdf(resume_file)
        keywords = extract_resume_keywords(resume_text)
        score, suitability = perform_resume_matching(resume_text, job_description)
        analysis = generate_resume_analysis(resume_text, job_description)

        st.write("Suitability:")
        st.write(f"- Status: {suitability}")
        st.write(f"- Score: {score:.2f}")

        st.write("Extracted Keywords:")
        st.write(keywords)

        st.write("Resume Analysis:")
        st.write(analysis)

with tab2:
    st.header("Employee Sentiment Analysis")
    feedback_file = st.file_uploader("Upload Feedback (CSV)", type=["csv"], key="feedback")
    st.write("CSV format: Column 'feedback' with employee feedback text.")

    if feedback_file:
        try:
            df = pd.read_csv(feedback_file)
            if "feedback" not in df.columns:
                st.error("CSV must contain a 'feedback' column.")
                st.stop()
            feedback_list = []
            for _, row in df.iterrows():
                text = row["feedback"]
                sentiment = analyze_sentiment(text)
                gpt4_analysis = analyze_feedback_with_gpt4(text)
                feedback_list.append({"text": text, "sentiment": sentiment, "gpt4_analysis": gpt4_analysis})
            summary = generate_sentiment_summary(feedback_list)

            st.write("Feedback Analysis:")
            for item in feedback_list:
                st.write(f"- Feedback: {item['text'][:100]}... Sentiment: {item['sentiment']}")
                st.write(f"  Analysis: {item['gpt4_analysis']}")

            st.write("Summary and Recommendations:")
            st.write(summary)
        except Exception as e:
            st.error(f"Error processing CSV: {str(e)}")

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8501))
    st.run(["streamlit", "run", "app.py", f"--server.port={port}", "--server.address=0.0.0.0"])