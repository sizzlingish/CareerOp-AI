from crewai import Task

from agents import (
    manager_agent,
    job_analyst_agent,
    cv_agent,
    research_agent,
    application_agent,
    interview_agent,
    critic_agent,
)


# ============================================================
# 1. MANAGER TASK
# ============================================================

manager_task = Task(
    description="""
You are the Career Operations Manager.

Analyze the user's career situation using the information provided
below.

CAREER REQUEST:
{career_request}

JOB DESCRIPTION:
{job_description}

CANDIDATE CV:
{cv_text}

Provide a structured career analysis that includes:

1. Understanding of the user's career goal
2. Important job requirements
3. Relevant candidate strengths
4. Potential gaps or weaknesses
5. Recommended next steps
6. Practical career strategy

Only use information that is actually provided.

Do not invent:
- experience
- qualifications
- skills
- achievements
- employment history
- company facts

Clearly identify assumptions when necessary.
""",
    expected_output="""
A structured career operations analysis containing:
- Career goal
- Job requirements
- Candidate strengths
- Potential gaps
- Recommended actions
- Practical next steps
""",
    agent=manager_agent,
)


# ============================================================
# 2. JOB ANALYSIS TASK
# ============================================================

job_analysis_task = Task(
    description="""
Analyze the following job description.

JOB DESCRIPTION:
{job_description}

CAREER REQUEST:
{career_request}

Identify and organize:

1. Job title
2. Main responsibilities
3. Required qualifications
4. Preferred qualifications
5. Technical skills
6. Soft skills
7. Education requirements
8. Experience requirements
9. Important tools, technologies, or certifications
10. Important keywords
11. Candidate expectations
12. Any unclear or ambiguous requirements

Separate required qualifications from preferred qualifications.

Do not invent requirements that are not supported by the
job description.
""",
    expected_output="""
A detailed job analysis containing:
- Job title
- Responsibilities
- Required qualifications
- Preferred qualifications
- Technical skills
- Soft skills
- Education
- Experience
- Tools and technologies
- Certifications
- Keywords
- Important observations
""",
    agent=job_analyst_agent,
)


# ============================================================
# 3. CV ANALYSIS TASK
# ============================================================

cv_analysis_task = Task(
    description="""
Analyze the candidate's CV against the target job.

CANDIDATE CV:
{cv_text}

JOB DESCRIPTION:
{job_description}

CAREER REQUEST:
{career_request}

Identify:

1. Candidate skills
2. Education
3. Work experience
4. Projects
5. Certifications
6. Technical background
7. Relevant achievements
8. Strong matches with the job
9. Missing or weak requirements
10. Skills that should be emphasized
11. CV areas that should be improved
12. Important evidence supporting the candidate's suitability

Only use evidence contained in the CV.

Do not invent experience, skills, qualifications, achievements,
projects, or certifications.
""",
    expected_output="""
A structured CV-to-job analysis containing:
- Candidate profile
- Skills
- Education
- Experience
- Projects
- Certifications
- Strong matches
- Missing requirements
- Weak areas
- Recommended CV improvements
- Evidence-based observations
""",
    agent=cv_agent,
)


# ============================================================
# 4. COMPANY RESEARCH TASK
# ============================================================

company_research_task = Task(
    description="""
Analyze the company and opportunity using only the information
provided in the user's input.

JOB DESCRIPTION:
{job_description}

CAREER REQUEST:
{career_request}

CANDIDATE CV:
{cv_text}

Provide:

1. Company information that is explicitly available
2. Products or services mentioned
3. Industry or business area
4. Role context
5. Important themes from the job description
6. Potential company priorities suggested by the role
7. Useful points the candidate could research further
8. Interview topics suggested by the opportunity

IMPORTANT:

You do not have live web access in this task.

Do not claim that you searched the internet.
Do not invent company facts.
Clearly distinguish supplied information from reasonable
interpretations.
""",
    expected_output="""
A factual company and opportunity analysis containing:
- Available company information
- Industry/context
- Role context
- Important themes
- Possible company priorities
- Suggested research areas
- Interview preparation topics
""",
    agent=research_agent,
)


# ============================================================
# 5. APPLICATION TASK
# ============================================================

application_task = Task(
    description="""
Create professional application guidance for the candidate.

CANDIDATE CV:
{cv_text}

JOB DESCRIPTION:
{job_description}

CAREER REQUEST:
{career_request}

Create useful, truthful application material including:

1. Targeted professional summary
2. CV improvement recommendations
3. Important keywords to emphasize
4. Suggested CV bullet improvements
5. Cover letter structure or draft
6. Application strategy
7. Potential application questions and suggested answers
8. Important points the candidate should communicate

Everything must be based on the candidate's actual information.

Do not invent:
- experience
- achievements
- qualifications
- skills
- employment history
- projects
- certifications

If information is missing, clearly indicate what the candidate
needs to provide instead of inventing it.
""",
    expected_output="""
A professional application package containing:
- Professional summary
- CV recommendations
- Keywords
- Improved bullet suggestions
- Cover letter content
- Application strategy
- Suggested application answers
""",
    agent=application_agent,
)


# ============================================================
# 6. INTERVIEW TASK
# ============================================================

interview_task = Task(
    description="""
Prepare the candidate for an interview for the target job.

CANDIDATE CV:
{cv_text}

JOB DESCRIPTION:
{job_description}

CAREER REQUEST:
{career_request}

Create:

1. Technical interview questions
2. Behavioral interview questions
3. Situational questions
4. Job-specific questions
5. CV-based questions
6. Questions about potential weaknesses or gaps
7. Suggested answer structures
8. STAR-style guidance where appropriate
9. Topics the candidate should revise
10. Questions the candidate can ask the interviewer

Questions should be based on the actual job description and CV.

Do not invent candidate experiences or achievements.
Suggested answers must remain truthful.
""",
    expected_output="""
A complete interview preparation guide containing:
- Technical questions
- Behavioral questions
- Situational questions
- Job-specific questions
- CV-based questions
- Gap/weakness questions
- Answer frameworks
- Preparation topics
- Questions for the interviewer
""",
    agent=interview_agent,
)


# ============================================================
# 7. CRITIC TASK
# ============================================================

critic_task = Task(
    description="""
Act as a quality reviewer for the candidate's job application.

CANDIDATE CV:
{cv_text}

JOB DESCRIPTION:
{job_description}

CAREER REQUEST:
{career_request}

Review the available information and identify:

1. Missing job requirements
2. Weak candidate evidence
3. CV weaknesses
4. Generic or weak application messaging
5. Unsupported claims that should be avoided
6. Important keywords that may be missing
7. Interview preparation gaps
8. Potential inconsistencies
9. Areas requiring clarification
10. Specific improvements the candidate should make

Be precise and actionable.

Do not invent facts about the candidate or employer.
""",
    expected_output="""
A detailed quality review containing:
- Missing requirements
- Evidence gaps
- CV weaknesses
- Application weaknesses
- Keyword gaps
- Interview gaps
- Potential inconsistencies
- Specific recommended improvements
""",
    agent=critic_agent,
)


# ============================================================
# TASK REGISTRY
# ============================================================

TASKS = {
    "Manager": manager_task,
    "Job Analyst": job_analysis_task,
    "CV Analyst": cv_analysis_task,
    "Research Agent": company_research_task,
    "Application Agent": application_task,
    "Interview Agent": interview_task,
    "Critic Agent": critic_task,
}
