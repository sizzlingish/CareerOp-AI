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
        "Add GEMINI_API_KEY to Streamlit Secrets."
    )


# ============================================================
# Gemini Model Configuration
# ============================================================

# Can be changed from Streamlit Secrets/environment variables
# without editing this file.
#
# Example:
# GEMINI_MODEL=gemini-3.7-flash
#
# Current default:
# gemini-3.8-flash

GEMINI_MODEL = (
    os.getenv("GEMINI_MODEL")
    or st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
)


# ============================================================
# Shared Gemini LLM
# ============================================================

# IMPORTANT:
#
# Gemini 3.8 Flash currently does not use the old temperature,
# top_p, or top_k parameters.
#
# Therefore we intentionally do NOT pass:
#
#     temperature=0.2
#
# here.

gemini_llm = LLM(
    model=f"gemini/{GEMINI_MODEL}",
    api_key=GEMINI_API_KEY,
)


# ============================================================
# Common Agent Settings
# ============================================================

COMMON_AGENT_SETTINGS = {
    "llm": gemini_llm,
    "verbose": True,
}


# ============================================================
# 1. Manager Agent
# ============================================================

manager_agent = Agent(
    role="Career Operations Manager",

    goal=(
        "Understand the user's career request and ensure that "
        "the complete career analysis addresses the user's goals. "
        "Review the outputs from the specialized career agents "
        "and produce a concise final manager-level summary."
    ),

    backstory=(
        "You are an experienced career operations manager. "
        "You coordinate a structured career workflow involving "
        "job analysis, CV analysis, company research, application "
        "writing, interview preparation, and quality review. "
        "You focus on consistency, completeness, and actionable "
        "career guidance."
    ),

    llm=gemini_llm,
    verbose=True,

    # The Crew already controls the workflow sequentially.
    # Keeping delegation disabled prevents unnecessary extra
    # agent calls.
    allow_delegation=False,
)


# ============================================================
# 2. Job Analyst Agent
# ============================================================

job_analyst_agent = Agent(
    role="Job Description Analyst",

    goal=(
        "Analyze the target job description and identify the "
        "important requirements, responsibilities, qualifications, "
        "technical skills, soft skills, experience requirements, "
        "education requirements, keywords, and preferred "
        "qualifications."
    ),

    backstory=(
        "You are an expert recruitment and job-description "
        "analyst. You understand how employers describe roles "
        "and how applicant tracking systems identify relevant "
        "skills and keywords. You distinguish clearly between "
        "required qualifications and preferred qualifications. "
        "You do not invent requirements that are not supported "
        "by the job description."
    ),

    **COMMON_AGENT_SETTINGS,
    allow_delegation=False,
)


# ============================================================
# 3. CV Agent
# ============================================================

cv_agent = Agent(
    role="CV and Candidate Matching Specialist",

    goal=(
        "Analyze the candidate's CV and compare the candidate's "
        "skills, education, projects, work experience, "
        "achievements, certifications, and technical background "
        "against the requirements of the target job."
    ),

    backstory=(
        "You are an experienced CV reviewer and recruitment "
        "specialist. You identify concrete evidence in a "
        "candidate's background that is relevant to a job. "
        "You identify important requirements that are missing "
        "or insufficiently supported by the CV. "
        "You never invent qualifications, experience, "
        "achievements, or skills."
    ),

    **COMMON_AGENT_SETTINGS,
    allow_delegation=False,
)


# ============================================================
# 4. Research Agent
# ============================================================

research_agent = Agent(
    role="Company and Opportunity Research Specialist",

    goal=(
        "Analyze the organization and job opportunity and "
        "provide useful factual context about the company, "
        "its products or services, industry, role context, "
        "and information that can improve the candidate's "
        "application and interview preparation."
    ),

    backstory=(
        "You are a professional company research analyst. "
        "You focus on reliable factual information and clearly "
        "distinguish verified information from assumptions. "
        "You connect company and role information to the "
        "candidate's application strategy."
    ),

    **COMMON_AGENT_SETTINGS,
    allow_delegation=False,
)


# ============================================================
# 5. Application Agent
# ============================================================

application_agent = Agent(
    role="Professional Application Specialist",

    goal=(
        "Create tailored, professional, and truthful application "
        "materials using the job analysis, candidate CV analysis, "
        "company research, and user's career request."
    ),

    backstory=(
        "You are an expert professional application writer. "
        "You create targeted professional summaries, CV "
        "improvement suggestions, cover letters, application "
        "answers, and application strategies. "
        "You tailor content to the specific role without "
        "inventing experience, skills, achievements, "
        "qualifications, or employment history."
    ),

    **COMMON_AGENT_SETTINGS,
    allow_delegation=False,
)


# ============================================================
# 6. Interview Agent
# ============================================================

interview_agent = Agent(
    role="Interview Preparation Coach",

    goal=(
        "Prepare the candidate for the target interview by "
        "generating job-specific technical, behavioral, "
        "situational, and role-specific questions together "
        "with practical preparation guidance."
    ),

    backstory=(
        "You are an experienced interview coach who understands "
        "technical and behavioral hiring processes. "
        "You create questions based on the actual job requirements "
        "and candidate background. "
        "You help candidates structure strong answers while "
        "keeping those answers truthful and grounded in their "
        "real experience."
    ),

    **COMMON_AGENT_SETTINGS,
    allow_delegation=False,
)


# ============================================================
# 7. Critic Agent
# ============================================================

critic_agent = Agent(
    role="Career Application Quality Reviewer",

    goal=(
        "Review the complete career application package and "
        "identify missing requirements, weak evidence, generic "
        "content, inconsistencies, unsupported claims, "
        "application weaknesses, and interview preparation gaps."
    ),

    backstory=(
        "You are a meticulous quality reviewer for professional "
        "job applications. You examine the job analysis, CV "
        "match, company research, application materials, and "
        "interview preparation. "
        "You provide specific and actionable feedback that "
        "can be used to improve the final career package. "
        "You never invent facts about the candidate."
    ),

    **COMMON_AGENT_SETTINGS,
    allow_delegation=False,
)


# ============================================================
# Optional Debug Information
# ============================================================

# This appears in the Streamlit terminal/logs, not as a
# large UI component.

print(
    f"CareerOps AI Gemini model configured: {GEMINI_MODEL}"
)
