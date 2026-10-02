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
# 1. JOB ANALYSIS
# ============================================================

job_analysis_task = Task(
    description="""
    Analyze the following job opportunity.

    JOB DESCRIPTION:
    {job_description}

    Extract and organize:

    1. Job title
    2. Required technical skills
    3. Required soft skills
    4. Required education
    5. Required experience
    6. Preferred qualifications
    7. Main responsibilities
    8. Important technologies and tools
    9. Important keywords
    10. Certifications or other requirements

    Clearly distinguish between required and preferred
    qualifications.

    Do not invent information that is not present in the
    job description.
    """,

    expected_output="""
    A structured job analysis containing:

    - Job title
    - Required skills
    - Preferred skills
    - Education requirements
    - Experience requirements
    - Responsibilities
    - Technologies/tools
    - Certifications
    - Important keywords
    - Other relevant requirements
    """,

    agent=job_analyst_agent,
)


# ============================================================
# 2. CV ANALYSIS
# ============================================================

cv_analysis_task = Task(
    description="""
    Analyze the candidate's CV and compare it against the
    job analysis produced by the previous task.

    CANDIDATE CV:
    {cv_text}

    The previous task has already analyzed the job description.
    Use that task's output as the job analysis.

    Identify:

    1. Relevant technical skills
    2. Relevant soft skills
    3. Relevant education
    4. Relevant work experience
    5. Relevant projects
    6. Relevant achievements
    7. Requirements clearly supported by the CV
    8. Requirements that are only partially supported
    9. Requirements with no supporting evidence
    10. Skills or experience that should be emphasized

    Do not invent experience or qualifications.
    """,

    expected_output="""
    A structured CV-to-job matching report containing:

    - Strong matches
    - Partial matches
    - Missing or unsupported requirements
    - Relevant CV evidence
    - Skills to emphasize
    - Areas where the CV could be improved
    """,

    agent=cv_agent,

    context=[
        job_analysis_task
    ],
)


# ============================================================
# 3. COMPANY RESEARCH
# ============================================================

company_research_task = Task(
    description="""
    Research the organization associated with the job.

    JOB DESCRIPTION:
    {job_description}

    Identify the organization if it is provided in the
    job description.

    Research relevant factual information such as:

    1. Company background
    2. Products or services
    3. Industry
    4. Main business areas
    5. Relevant technologies or areas of work
    6. Recent publicly available developments
    7. Information relevant to the specific role

    Focus only on information that could help the candidate
    understand the organization or prepare for the application
    and interview.

    Clearly distinguish factual information from uncertainty.

    Do not invent company information.
    """,

    expected_output="""
    A concise company research report containing:

    - Organization overview
    - Products/services
    - Industry
    - Relevant work
    - Relevant technologies
    - Recent developments when available
    - Useful interview/application context
    """,

    agent=research_agent,

    context=[
        job_analysis_task
    ],
)


# ============================================================
# 4. APPLICATION
# ============================================================

application_task = Task(
    description="""
    Create tailored application materials using the outputs
    produced by the previous career analysis tasks.

    CANDIDATE CV:
    {cv_text}

    JOB DESCRIPTION:
    {job_description}

    USER REQUEST:
    {career_request}

    The previous tasks provide:

    - Job analysis
    - CV-to-job matching
    - Company research

    Use those outputs when preparing the application.

    Create:

    1. Professional summary tailored to the role
    2. Specific CV improvement recommendations
    3. Skills that should be emphasized
    4. Suggested improvements to relevant experience/project
       descriptions
    5. A customized cover letter
    6. Suggested answers for important application questions
       when appropriate

    All content must remain truthful to the candidate's actual
    background.

    Never invent experience, qualifications, achievements,
    employers, education, certifications, or skills.
    """,

    expected_output="""
    A complete application package containing:

    - Tailored professional summary
    - CV improvement recommendations
    - Skills to emphasize
    - Experience/project recommendations
    - Customized cover letter
    - Suggested application answers where appropriate
    """,

    agent=application_agent,

    context=[
        job_analysis_task,
        cv_analysis_task,
        company_research_task,
    ],
)


# ============================================================
# 5. INTERVIEW PREPARATION
# ============================================================

interview_task = Task(
    description="""
    Prepare the candidate for an interview for the target job.

    JOB DESCRIPTION:
    {job_description}

    CANDIDATE CV:
    {cv_text}

    USER REQUEST:
    {career_request}

    The previous tasks provide:

    - Job analysis
    - CV-to-job matching
    - Company research
    - Application materials

    Use all of those outputs.

    Create:

    1. Ten likely interview questions
    2. Technical questions relevant to the role
    3. Behavioral questions
    4. Situational questions
    5. Suggested answer structures
    6. Topics the candidate should revise
    7. Questions the candidate could ask the interviewer

    The preparation should be specific to the actual role and
    candidate background.

    Do not invent candidate experience.
    """,

    expected_output="""
    A structured interview preparation guide containing:

    - Likely interview questions
    - Technical questions
    - Behavioral questions
    - Situational questions
    - Answer preparation guidance
    - Topics to revise
    - Questions for the interviewer
    """,

    agent=interview_agent,

    context=[
        job_analysis_task,
        cv_analysis_task,
        company_research_task,
        application_task,
    ],
)


# ============================================================
# 6. CRITIC / FINAL REVIEW
# ============================================================

critic_task = Task(
    description="""
    Perform a final quality review of the complete career
    application package.

    The previous tasks provide:

    - Job analysis
    - CV match
    - Company research
    - Application materials
    - Interview preparation

    Review the complete package for:

    1. Missing job requirements
    2. Weak evidence of required skills
    3. Generic application content
    4. Missing important keywords
    5. Unsupported claims
    6. Inconsistencies between the CV and application
    7. Weak cover letter personalization
    8. Interview preparation gaps
    9. Missing important questions
    10. Areas requiring revision

    Do not invent problems that are not supported by the
    provided information.

    End the review with one of:

    APPROVED

    or

    NEEDS_REVISION
    """,

    expected_output="""
    A final quality review containing:

    - Overall review status
    - Strengths
    - Missing requirements
    - Weak areas
    - Application issues
    - Interview preparation issues
    - Specific revision recommendations
    - Final status: APPROVED or NEEDS_REVISION
    """,

    agent=critic_agent,

    context=[
        job_analysis_task,
        cv_analysis_task,
        company_research_task,
        application_task,
        interview_task,
    ],
)


# ============================================================
# 7. MANAGER SUMMARY
# ============================================================

manager_task = Task(
    description="""
    Coordinate and summarize the complete career analysis
    workflow for the user's request.

    USER REQUEST:
    {career_request}

    JOB DESCRIPTION:
    {job_description}

    CANDIDATE CV:
    {cv_text}

    The previous tasks provide all specialized analysis.

    Review their outputs and produce a final career operations
    summary that addresses the user's request.

    Identify:

    1. Main job requirements
    2. Candidate's strongest relevant evidence
    3. Important gaps or unsupported requirements
    4. Application preparation status
    5. Interview preparation status
    6. Critic/review status
    7. Recommended next actions

    Do not invent candidate information.

    Do not claim that the candidate has experience or skills
    that are not supported by the CV.
    """,

    expected_output="""
    A final career operations summary containing:

    - User's objective
    - Summary of job requirements
    - Summary of candidate match
    - Application status
    - Interview preparation status
    - Critic status
    - Recommended next actions
    """,

    agent=manager_agent,

    context=[
        job_analysis_task,
        cv_analysis_task,
        company_research_task,
        application_task,
        interview_task,
        critic_task,
    ],
)

