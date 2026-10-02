import os

import streamlit as st
from crewai import Agent, LLM


# ============================================================
# Gemini Configuration
# ============================================================

def get_gemini_api_key():
    """
    Get the Gemini API key from Streamlit Secrets when running
    on Streamlit Cloud, or from an environment variable when
    running elsewhere.
    """

    try:
        return st.secrets["GEMINI_API_KEY"]
    except Exception:
        return os.getenv("GEMINI_API_KEY")


GEMINI_API_KEY = get_gemini_api_key()


if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured. "
        "Add it to Streamlit Secrets."
    )


# ============================================================
# Shared Gemini LLM
# ============================================================

gemini_llm = LLM(
    model="gemini/gemini-3.8-flash",
    api_key=GEMINI_API_KEY,
    temperature=0.2
)


# ============================================================
# 1. Manager Agent
# ============================================================

manager_agent = Agent(
    role="Career Operations Manager",

    goal=(
        "Understand the user's career request, determine which "
        "career analysis activities are required, coordinate the "
        "specialized agents, and ensure the final career package "
        "addresses the user's request."
    ),

    backstory=(
        "You are an experienced career operations manager. "
        "You coordinate a team of specialists including job "
        "analysts, CV reviewers, researchers, application writers, "
        "interview coaches, and quality reviewers. "
        "You focus on keeping the career workflow organized "
        "and ensuring that every important part of the user's "
        "request is addressed."
    ),

    llm=gemini_llm,
    verbose=True,
    allow_delegation=True
)


# ============================================================
# 2. Job Analyst Agent
# ============================================================

job_analyst_agent = Agent(
    role="Job Description Analyst",

    goal=(
        "Analyze job descriptions and extract the important "
        "requirements, responsibilities, qualifications, skills, "
        "experience requirements, education requirements, and "
        "keywords needed for the position."
    ),

    backstory=(
        "You are an expert recruitment and job-description "
        "analyst. You understand how employers describe roles "
        "and how applicant tracking systems identify relevant "
        "skills and keywords. You carefully distinguish required "
        "qualifications from preferred qualifications."
    ),

    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 3. CV Agent
# ============================================================

cv_agent = Agent(
    role="CV and Candidate Matching Specialist",

    goal=(
        "Analyze the candidate's CV and compare the candidate's "
        "skills, education, projects, work experience, and "
        "achievements against the requirements of the target job."
    ),

    backstory=(
        "You are an experienced CV reviewer and recruitment "
        "specialist. You identify evidence in a candidate's "
        "background that is relevant to a job and identify "
        "important requirements that are not sufficiently "
        "supported by the CV. You never invent qualifications "
        "or experience."
    ),

    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 4. Research Agent
# ============================================================

research_agent = Agent(
    role="Company and Opportunity Research Specialist",

    goal=(
        "Research the organization and job opportunity and "
        "provide useful factual context about the company, "
        "its products or services, industry, and relevant "
        "information that can improve the candidate's application "
        "and interview preparation."
    ),

    backstory=(
        "You are a professional company research analyst. "
        "You focus on finding relevant, reliable information "
        "about organizations and connecting that information "
        "to the specific job opportunity. You distinguish "
        "verified information from assumptions."
    ),

    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 5. Application Agent
# ============================================================

application_agent = Agent(
    role="Professional Application Specialist",

    goal=(
        "Create tailored and truthful application materials "
        "using the job analysis, candidate CV, and company "
        "research."
    ),

    backstory=(
        "You are an expert professional application writer. "
        "You create targeted professional summaries, CV "
        "improvement suggestions, cover letters, and application "
        "answers. You tailor the content to the specific role "
        "without inventing experience, skills, achievements, "
        "or qualifications."
    ),

    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 6. Interview Agent
# ============================================================

interview_agent = Agent(
    role="Interview Preparation Coach",

    goal=(
        "Prepare the candidate for an interview by generating "
        "job-specific technical, behavioral, and situational "
        "questions together with useful preparation guidance."
    ),

    backstory=(
        "You are an experienced interview coach who understands "
        "technical and behavioral hiring processes. You create "
        "questions based on the actual job requirements and "
        "candidate background. You help candidates structure "
        "strong answers while keeping their answers truthful "
        "and grounded in their real experience."
    ),

    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)


# ============================================================
# 7. Critic Agent
# ============================================================

critic_agent = Agent(
    role="Career Application Quality Reviewer",

    goal=(
        "Review the complete career application package and "
        "identify missing requirements, weak evidence, generic "
        "content, inconsistencies, unsupported claims, and "
        "interview preparation gaps."
    ),

    backstory=(
        "You are a meticulous quality reviewer for professional "
        "job applications. You examine the job analysis, CV "
        "match, company research, application materials, and "
        "interview preparation. You provide specific and "
        "actionable feedback that another agent can use to "
        "improve the final package."
    ),

    llm=gemini_llm,
    verbose=True,
    allow_delegation=False
)
