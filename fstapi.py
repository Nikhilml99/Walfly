from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
from data_extraction import *
from data_preprocessing import *
from recommendation import *
from chatbot_app import *
app = FastAPI()
stopword = nltk.corpus.stopwords.words('english')
# import functions for candidate recommendation
app = FastAPI()

@app.get("/")
async def read_root():
    return {"msg": "testing api!"}
@app.post("/recommend_jobs/")
async def recommend_jobs(resume_text:str, num: int):
    # Input format for resume_df should be a dictionary with keys for each column
    # Output format should be a list of recommended job titles
    jobs = givejob2([resume_text], jobs_df, num)
    return {"recommended_jobs": jobs}
@app.post("/recommend_candidates/")
async def recommend_candidates(job_text: str, num: int):
    # Input format for resume_df_input should be a dictionary with keys for each column
    # Output format should be a dictionary of recommended candidates with their details
    candidate = cr(resume_df, job_text, num)
    return {"recommended_candidates": candidate}
