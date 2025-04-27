# ResuMetrics: Automating Recruitment and Employee Engagement
Overview
The HR-Tech AI project is an AI-powered solution designed to streamline HR processes by automating resume screening and employee sentiment analysis. Built for the HR-Tech Innovation Challenge, it addresses two key challenges:

Resume Screening: Manually reviewing resumes is time-consuming and inconsistent.
Employee Feedback: Analyzing feedback to gauge morale or turnover risk is slow and subjective.

Our solution uses Azure AI services, fine-tuned machine learning models (BERT, DistilBERT), and Llama-3 8B (GitHub Models) to deliver a user-friendly Streamlit app that:

Screens resumes for job fit, extracts keywords, and suggests improvements.
Analyzes employee feedback for sentiment and provides HR recommendations.

Features

Resume Screening:
Extracts text from PDF resumes using Azure AI Document Intelligence.
Identifies key skills with Llama-3 8B.
Predicts job suitability using a fine-tuned BERT model.
Suggests resume enhancements.


Sentiment Analysis:
Processes feedback CSVs with a fine-tuned DistilBERT model.
Analyzes sentiments and themes with Llama-3 8B.
Summarizes feedback and recommends HR actions.


Streamlit App:
Two tabs: Resume Screening and Sentiment Analysis.
Deployed on Azure App Service with FastAPI endpoints (/resume-screen, /sentiment-analysis).



Technologies

Azure Services: Azure AI Document Intelligence, Azure AI Search, Azure ML Studio, Azure App Service.
Models: BERT (hrtech-resume-model), DistilBERT (hrtech-sentiment-model), Llama-3 8B (GitHub Models).
Frameworks: Python 3.12, Streamlit, FastAPI, Transformers, PyTorch.
Environment: Azure ML (Resume_Analysis workspace, ML_Tasks resource group).

Project Structure
HR-Tech-AI/
├── app/
│   ├── app.py               # Streamlit app with FastAPI endpoints
│   └── requirements.txt     # Python dependencies
├── data/
│   ├── resumes/            # Sample resumes (e.g., resume1.pdf)
│   └── feedback/           # Sample feedback (e.g., feedback.csv)
├── notebooks/
│   ├── resume_screening.ipynb    # Resume processing pipeline
│   └── sentiment_analysis.ipynb  # Feedback processing pipeline
├── deploy/
│   ├── sentiment_env.yml    # Azure ML environment
│   ├── score_resume.py      # Scoring script for resume model
│   └── score_sentiment.py   # Scoring script for sentiment model
├── deliverables/
│   ├── technical_report.pdf # Project report
│   └── presentation.pdf     # Presentation slides
├── .env                     # Environment variables (not tracked)
└── README.md                # This file

Prerequisites

Python: 3.12
Azure Subscription: Azure for Students (or equivalent)
GitHub Account: For GitHub Models API access
PowerShell: For Windows users (or Bash for Linux/macOS)
Azure CLI: Installed and logged in
Sample Data:
Resumes (PDFs) in data/resumes/
Feedback CSV (feedback.csv) in data/feedback/ with a feedback column



Setup Instructions
1. Clone the Repository
Clone the project to your local machine:
git clone https://github.com/incognito-unlimited/ResuMetrics.git
cd ResuMetrics

git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/incognito-unlimited/ResuMetrics.git
git push -u origin main

2. Set Up Python Environment
Create and activate a virtual environment:
cd D:\Python\ResuMetrics
python -m venv hrtech-venv
.\hrtech-venv\Scripts\Activate.ps1

Install dependencies:
pip install -r app\requirements.txt
python -m nltk.downloader punkt stopwords

requirements.txt:
streamlit==1.42.0
azure-ai-documentintelligence==1.0.0
azure-search-documents==11.5.1
openai==1.63.2
python-dotenv==1.0.1
fastapi==0.115.0
uvicorn==0.30.6
pandas>=2.0.0
transformers>=4.44.2
torch>=2.4.1
datasets>=3.0.1
nltk>=3.8.1
python-multipart==0.0.12
matplotlib>=3.9.2

3. Configure Environment Variables
Create a .env file in the project root (D:\Python\ResuMetrics\.env) and add the following keys:
# Azure AI Document Intelligence
DOCUMENT_INTELLIGENCE_ENDPOINT=https://<your-endpoint>.cognitiveservices.azure.com/
DOCUMENT_INTELLIGENCE_KEY=<your-key>

# Azure AI Search
SEARCH_SERVICE_NAME=HRTechSearch
SEARCH_KEY=<your-key>

# Azure ML Endpoints
RESUME_MATCHER_ENDPOINT=https://<your-endpoint>.eastus2.inference.ml.azure.com/score
RESUME_MATCHER_KEY=<your-key>
SENTIMENT_CLASSIFIER_ENDPOINT=https://<your-endpoint>.eastus2.inference.ml.azure.com/score
SENTIMENT_CLASSIFIER_KEY=<your-key>

# GitHub Models (Llama-3 8B)
GITHUB_MODELS_TOKEN=<your-github-token>

How to Obtain Keys:

Azure Keys:
Document Intelligence: Azure Portal > HRTechDocIntel > Keys and Endpoint.
Search: Azure Portal > HRTechSearch > Keys.
ML Endpoints: Azure ML Studio > Resume_Analysis > Endpoints > resume-matcher-endpoint, sentiment-classifier-endpoint > Consume.
If AI_STUDIO_KEY is pending, contact Azure Support with subscription ID: 7880aa39-f3d5-4c17-9cee-71f972d79989.


GitHub Models Token:
Go to GitHub > Settings > Developer settings > Personal access tokens > Generate new token.
Select read:models scope.
Copy the token to GITHUB_MODELS_TOKEN.

If errors occur, check .env or share output for debugging.

4. Prepare Sample Data
Ensure sample data is in place:

Resumes: Place PDF files (e.g., resume1.pdf) in data/resumes/.
Feedback: Create data/feedback/feedback.csv with a feedback column:feedback
"I love working here! The team is great."
"I feel undervalued and overworked."


If missing, download sample resumes from Kaggle Resume Dataset or use provided datasets (resume_data.csv, sentiment_data.csv).

5. Run the Streamlit App Locally
Start the Streamlit app:
cd D:\Python\ResuMetrics
.\hrtech-venv\Scripts\Activate.ps1
streamlit run app\app.py


Access: Open http://localhost:8501 in your browser.
Usage:
Resume Screening Tab:
Upload a resume PDF (e.g., resume1.pdf).
Enter a job description (e.g., “Software Engineer with expertise in Python and Azure”).
View suitability score, keywords, and suggestions.


Sentiment Analysis Tab:
Upload a feedback CSV (e.g., feedback.csv).
View sentiment labels, analysis, and HR recommendations.



Troubleshooting:
Port Conflict: If 8501 is in use, specify another port:streamlit run app\app.py --server.port 8502


ModuleNotFoundError: Reinstall dependencies:pip install -r app\requirements.txt


API Errors: Verify .env keys and test with curl:curl -X POST https://<your-endpoint>.eastus2.inference.ml.azure.com/score -H "Authorization: Bearer <your-key>" -H "Content-Type: application/json" -d "{\"feedback\": \"Test\"}"