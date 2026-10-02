import os
import streamlit as st
from crewai import Agent, LLM


def get_gemini_api_key():
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


GEMINI_MODEL = (
    os.getenv("GEMINI_MODEL")
    or st.secrets.get("GEMINI_MODEL", "gemini-3.8-flash")
)


gemini_llm = LLM(
    model=f"gemini/{GEMINI_MODEL}",
    api_key=GEMINI_API_KEY,
)


COMMON_AGENT_SETTINGS = {
    "llm": gemini_llm,
    "verbose": True,
}


# ============================================================
# 1. MANAGER AGENT
# ============================================================

manager_agent = Agent(
    role="Career Operations Manager",
    goal=(
        "Understand the user's career request and provide a "
        "clear, structured career analysis based only on the "
        "information provided by the user."
    ),
    backstory=(
        "You are an experienced career operations manager. "
        "You provide practical, structured, and truthful career "
        "guidance. You never invent information about the "
        "candidate, job, company, or career history."
    ),
    **COMMON_AGENT_SETTINGS,
    allow_delegation=False,
)


# ============================================================
# 2. JOB ANALYST AGENT
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
# 3. CV ANALYST AGENT
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
# 4. COMPANY RESEARCH AGENT
# ============================================================

research_agent = Agent(
    role="Company and Opportunity Research Specialist",
    goal=(
        "Analyze the organization and job opportunity using "
        "the information supplied by the user. Provide useful "
        "factual context about the company, products or services, "
        "industry, role context, and information that can improve "
        "the candidate's application and interview preparation."
    ),
    backstory=(
        "You are a professional company research analyst. "
        "You focus on reliable factual information and clearly "
        "distinguish between information explicitly provided by "
        "the user and reasonable observations. You never claim "
        "to have performed live web research unless a web "
        "research tool is actually available."
    ),
    **COMMON_AGENT_SETTINGS,
    allow_delegation=False,
)


# ============================================================
# 5. APPLICATION AGENT
# ============================================================

application_agent = Agent(
    role="Professional Application Specialist",
    goal=(
        "Create tailored, professional, and truthful application "
        "materials using the job description, candidate CV, "
        "company information, and user's career request."
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
# 6. INTERVIEW AGENT
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
# 7. CRITIC AGENT
# ============================================================

critic_agent = Agent(
    role="Career Application Quality Reviewer",
    goal=(
        "Review the supplied career application information and "
        "identify missing requirements, weak evidence, generic "
        "content, inconsistencies, unsupported claims, "
        "application weaknesses, and interview preparation gaps."
    ),
    backstory=(
        "You are a meticulous quality reviewer for professional "
        "job applications. You identify specific and actionable "
        "improvements. You never invent facts about the candidate."
    ),
    **COMMON_AGENT_SETTINGS,
    allow_delegation=False,
)


print(f"CareerOps AI Gemini model configured: {GEMINI_MODEL}")
