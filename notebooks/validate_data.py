import pandas as pd

# Validate resume matching data
resume_df = pd.read_csv(r"D:\Python\ResuMetrics\data\training_data\resume_training.csv")
print("Resume Matching Data:")
print(resume_df.isnull().sum())
print(resume_df["label"].value_counts())

# Validate sentiment data
sentiment_df = pd.read_csv(r"D:\Python\ResuMetrics\data\training_data\sentiment_training.csv")
print("\nSentiment Data:")
print(sentiment_df.isnull().sum())
print(sentiment_df["sentiment"].value_counts())