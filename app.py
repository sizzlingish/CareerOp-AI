import os
import time
from datetime import datetime

import streamlit as st
from crewai import Crew, Process

from agents import (
    manager_agent,
    job_analyst_agent,
    cv_agent,
    research_agent,
    application_agent,
    interview_agent,
    critic_agent,
)

from tasks import (
    job_analysis_task,
    cv_analysis_task,
    company_research_task,
    application_task,
    interview_task,
    critic_task,
    manager_task,
)

from tools import extract_cv_text
from memory import CareerMemory


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="CareerOps AI",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.7rem;
            font-weight: 800;
            margin-bottom: 0.2rem;
        }

        .subtitle {
            font-size: 1.05rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }

        .section-title {
            font-size: 1.4rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.7rem;
        }

        .success-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #ecfdf5;
            border: 1px solid #a7f3d0;
        }

        .warning-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #fffbeb;
            border: 1px solid #fde68a;
        }

        .info-box {
            padding: 1rem;
            border-radius: 10px;
            background-color: #eff6ff;
            border: 1px solid #bfdbfe;
        }

        .metric-card {
            padding: 1rem;
            border-radius: 12px;
            border: 1px solid #e5e7eb;
            background: #ffffff;
            text-align: center;
        }

        .small-text {
            font-size: 0.85rem;
            color: #6b7280;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

DEFAULT_STATE = {
    "cv_text": "",
    "job_description": "",
    "career_request": "",
    "analysis_result": None,
    "memory": None,
    "analysis_complete": False,
    "error_message": None,
}


for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def reset_analysis():
    """Clear the previous analysis result."""
    st.session_state.analysis_result = None
    st.session_state.memory = None
    st.session_state.analysis_complete = False
    st.session_state.error_message = None


def reset_everything():
    """Reset the complete application."""
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value


def extract_task_output(task_output):
    """
    Safely convert CrewAI task output into plain text.

    CrewAI versions can return slightly different output objects,
    so this function handles the common cases.
    """

    if task_output is None:
        return ""

    # CrewAI TaskOutput commonly has .raw
    if hasattr(task_output, "raw"):
        raw = task_output.raw
        if raw is not None:
            return str(raw)

    # Fallback
    return str(task_output)


def is_temporary_ai_error(error):
    """
    Detect temporary Gemini/API errors where retrying makes sense.
    """

    error_text = str(error).upper()

    temporary_signals = [
        "503",
        "UNAVAILABLE",
        "SERVICE UNAVAILABLE",
        "HIGH DEMAND",
        "SERVICEUNAVAILABLE",
        "429",
        "RESOURCE_EXHAUSTED",
        "RATE LIMIT",
        "TOO MANY REQUESTS",
        "INTERNAL SERVER ERROR",
    ]

    return any(signal in error_text for signal in temporary_signals)


def kickoff_with_retry(crew, inputs, max_attempts=4):
    """
    Run CrewAI with retry handling for temporary Gemini failures.

    Delays:
        attempt 1 -> immediate
        attempt 2 -> 5 seconds
        attempt 3 -> 15 seconds
        attempt 4 -> 30 seconds
    """

    delays = [5, 15, 30]

    for attempt in range(max_attempts):

        try:
            return crew.kickoff(inputs=inputs)

        except Exception as error:

            # Don't retry permanent errors.
            if not is_temporary_ai_error(error):
                raise

            # No attempts remaining.
            if attempt >= max_attempts - 1:
                raise

            delay = delays[attempt]

            st.warning(
                f"Gemini is temporarily busy or rate-limited. "
                f"Retrying in {delay} seconds "
                f"(attempt {attempt + 2}/{max_attempts})..."
            )

            time.sleep(delay)

    raise RuntimeError(
        "The AI service could not be reached after multiple attempts."
    )


def create_crew():
    """
    Create the CareerOps Crew.

    The process is sequential because later tasks depend
    on outputs from earlier tasks.
    """

    crew = Crew(
        agents=[
            job_analyst_agent,
            cv_agent,
            research_agent,
            application_agent,
            interview_agent,
            critic_agent,
            manager_agent,
        ],
        tasks=[
            job_analysis_task,
            cv_analysis_task,
            company_research_task,
            application_task,
            interview_task,
            critic_task,
            manager_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def run_career_analysis(cv_text, job_description, career_request):
    """
    Run the complete CareerOps workflow.
    """

    # --------------------------------------------------------
    # Create memory
    # --------------------------------------------------------

    memory = CareerMemory()

    try:
        memory.set_cv(cv_text)
    except Exception:
        pass

    try:
        memory.set_job(job_description)
    except Exception:
        pass

    # --------------------------------------------------------
    # Create crew
    # --------------------------------------------------------

    crew = create_crew()

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # These are the ONLY direct inputs used by our corrected
    # tasks.py.
    #
    # Do NOT add job_analysis, cv_analysis, etc. here.
    # Those are provided through CrewAI task context.
    # --------------------------------------------------------

    inputs = {
        "cv_text": cv_text,
        "job_description": job_description,
        "career_request": career_request,
    }

    # --------------------------------------------------------
    # Run CrewAI with retry handling
    # --------------------------------------------------------

    result = kickoff_with_retry(
        crew,
        inputs,
        max_attempts=4,
    )

    # --------------------------------------------------------
    # Extract task outputs
    # --------------------------------------------------------

    outputs = {}

    task_names = [
        "job_analysis",
        "cv_analysis",
        "company_research",
        "application_materials",
        "interview_preparation",
        "critic_review",
        "manager_summary",
    ]

    for index, task_name in enumerate(task_names):

        try:
            if index < len(crew.tasks):
                task_output = crew.tasks[index].output
                outputs[task_name] = extract_task_output(task_output)
            else:
                outputs[task_name] = ""

        except Exception:
            outputs[task_name] = ""

    # --------------------------------------------------------
    # Fallback:
    # If task outputs aren't available, use final result.
    # --------------------------------------------------------

    if not any(outputs.values()):
        outputs["manager_summary"] = extract_task_output(result)

    # --------------------------------------------------------
    # Store results in memory when supported
    # --------------------------------------------------------

    memory_methods = {
        "job_analysis": "set_job_analysis",
        "cv_analysis": "set_cv_analysis",
        "company_research": "set_company_research",
        "application_materials": "set_application_materials",
        "interview_preparation": "set_interview_preparation",
        "critic_review": "set_critic_review",
        "manager_summary": "set_manager_summary",
    }

    for result_key, method_name in memory_methods.items():

        value = outputs.get(result_key, "")

        if not value:
            continue

        try:
            method = getattr(memory, method_name, None)

            if method:
                method(value)

        except Exception:
            # Memory should never crash the main workflow.
            pass

    return outputs, memory


def create_report(outputs, job_description, career_request):
    """
    Build a plain-text report that users can download.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")

    report_parts = [
        "CAREEROPS AI - CAREER ANALYSIS REPORT",
        "=" * 60,
        "",
        f"Generated: {timestamp}",
        "",
        "CAREER REQUEST",
        "-" * 60,
        career_request,
        "",
        "JOB DESCRIPTION",
        "-" * 60,
        job_description,
        "",
        "JOB ANALYSIS",
        "-" * 60,
        outputs.get("job_analysis", ""),
        "",
        "CV ANALYSIS",
        "-" * 60,
        outputs.get("cv_analysis", ""),
        "",
        "COMPANY RESEARCH",
        "-" * 60,
        outputs.get("company_research", ""),
        "",
        "APPLICATION MATERIALS",
        "-" * 60,
        outputs.get("application_materials", ""),
        "",
        "INTERVIEW PREPARATION",
        "-" * 60,
        outputs.get("interview_preparation", ""),
        "",
        "CRITIC REVIEW",
        "-" * 60,
        outputs.get("critic_review", ""),
        "",
        "MANAGER SUMMARY",
        "-" * 60,
        outputs.get("manager_summary", ""),
        "",
        "=" * 60,
        "Generated by CareerOps AI",
    ]

    return "\n".join(report_parts)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("## 💼 CareerOps AI")

    st.markdown(
        """
        **AI-powered career operations assistant**

        Your workflow:

        1. 📄 Add your CV
        2. 🎯 Add the job description
        3. 🤖 Analyze job fit
        4. 🏢 Research the company
        5. ✍️ Prepare application materials
        6. 🎤 Prepare for interviews
        7. 🔍 Review everything
        """
    )

    st.divider()

    st.markdown("### System Status")

    gemini_key_exists = bool(
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
    )

    if gemini_key_exists:
        st.success("Gemini API key detected")
    else:
        st.warning(
            "No Gemini API key detected in environment variables."
        )

    st.divider()

    if st.button(
        "🗑️ Reset Application",
        use_container_width=True,
    ):
        reset_everything()
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">💼 CareerOps AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="subtitle">'
    "Your AI-powered career operations assistant"
    "</div>",
    unsafe_allow_html=True,
)


# ============================================================
# INPUT SECTION
# ============================================================

st.markdown(
    '<div class="section-title">1. Your CV</div>',
    unsafe_allow_html=True,
)

cv_input_method = st.radio(
    "How would you like to provide your CV?",
    options=[
        "Upload CV",
        "Paste CV text",
    ],
    horizontal=True,
)


# ------------------------------------------------------------
# UPLOAD CV
# ------------------------------------------------------------

if cv_input_method == "Upload CV":

    uploaded_file = st.file_uploader(
        "Upload your CV",
        type=["pdf", "txt"],
        help="Supported formats: PDF and TXT",
    )

    if uploaded_file is not None:

        file_name = uploaded_file.name.lower()

        try:

            # TXT
            if file_name.endswith(".txt"):

                cv_text = uploaded_file.read().decode(
                    "utf-8",
                    errors="ignore",
                )

            # PDF
            else:

                # Write uploaded file to a temporary file
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getvalue()
                    )

                    temp_path = temp_file.name

                try:
                    cv_text = extract_cv_text(temp_path)

                finally:

                    try:
                        os.remove(temp_path)
                    except Exception:
                        pass

            st.session_state.cv_text = cv_text

            if cv_text.strip():

                st.success(
                    f"CV loaded successfully: {uploaded_file.name}"
                )

                with st.expander(
                    "👀 Preview CV text",
                    expanded=False,
                ):

                    st.text_area(
                        "Extracted CV",
                        value=cv_text,
                        height=250,
                        disabled=True,
                        label_visibility="collapsed",
                    )

            else:

                st.error(
                    "The CV was uploaded, but no text could be extracted."
                )

        except Exception as error:

            st.error(
                "I couldn't read this CV file."
            )

            with st.expander(
                "Technical details"
            ):
                st.exception(error)


# ------------------------------------------------------------
# PASTE CV
# ------------------------------------------------------------

else:

    pasted_cv = st.text_area(
        "Paste your CV here",
        value=st.session_state.cv_text,
        height=350,
        placeholder=(
            "Paste your complete CV here...\n\n"
            "Example:\n"
            "John Doe\n"
            "Senior Data Scientist\n"
            "john@example.com\n\n"
            "Experience...\n"
            "Education...\n"
            "Skills..."
        ),
    )

    st.session_state.cv_text = pasted_cv


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.markdown(
    '<div class="section-title">2. Target Job</div>',
    unsafe_allow_html=True,
)

job_description = st.text_area(
    "Paste the job description",
    value=st.session_state.job_description,
    height=300,
    placeholder=(
        "Paste the complete job description here...\n\n"
        "Include responsibilities, qualifications, "
        "requirements, skills, location, and other details."
    ),
)

st.session_state.job_description = job_description


# ============================================================
# CAREER REQUEST
# ============================================================

st.markdown(
    '<div class="section-title">3. What do you want CareerOps AI to do?</div>',
    unsafe_allow_html=True,
)

career_request = st.text_area(
    "Career request",
    value=st.session_state.career_request,
    height=120,
    placeholder=(
        "Example: Help me apply for this job. "
        "Analyze my fit, identify gaps, create an application strategy, "
        "prepare interview questions, and suggest improvements."
    ),
)

st.session_state.career_request = career_request


# ============================================================
# INPUT VALIDATION
# ============================================================

st.markdown(
    '<div class="section-title">Input Check</div>',
    unsafe_allow_html=True,
)

cv_ready = bool(
    st.session_state.cv_text
    and st.session_state.cv_text.strip()
)

job_ready = bool(
    st.session_state.job_description
    and st.session_state.job_description.strip()
)

request_ready = bool(
    st.session_state.career_request
    and st.session_state.career_request.strip()
)


col1, col2, col3 = st.columns(3)

with col1:

    if cv_ready:
        st.success("✅ CV ready")
    else:
        st.warning("⚠️ CV missing")


with col2:

    if job_ready:
        st.success("✅ Job description ready")
    else:
        st.warning("⚠️ Job description missing")


with col3:

    if request_ready:
        st.success("✅ Career request ready")
    else:
        st.warning("⚠️ Career request missing")


# ============================================================
# ANALYZE BUTTON
# ============================================================

st.divider()

analyze_button = st.button(
    "🚀 Analyze Job & Build Application Strategy",
    type="primary",
    use_container_width=True,
)


if analyze_button:

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not cv_ready:

        st.error(
            "Please upload or paste your CV before starting."
        )
        st.stop()

    if not job_ready:

        st.error(
            "Please provide the job description before starting."
        )
        st.stop()

    if not request_ready:

        st.error(
            "Please tell CareerOps AI what you want it to do."
        )
        st.stop()

    # --------------------------------------------------------
    # Reset old results
    # --------------------------------------------------------

    reset_analysis()

    # --------------------------------------------------------
    # Progress UI
    # --------------------------------------------------------

    progress_bar = st.progress(0)

    status_text = st.empty()

    status_text.info(
        "🚀 Starting CareerOps AI..."
    )

    try:

        # ----------------------------------------------------
        # Stage 1
        # ----------------------------------------------------

        progress_bar.progress(5)

        status_text.info(
            "🔎 Preparing job, CV, and career request..."
        )

        time.sleep(0.5)

        # ----------------------------------------------------
        # Stage 2
        # ----------------------------------------------------

        progress_bar.progress(10)

        status_text.info(
            "🤖 Running the CareerOps AI team..."
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # This is where all CrewAI tasks execute.
        # The actual task chain is controlled by tasks.py.
        # ----------------------------------------------------

        outputs, memory = run_career_analysis(
            cv_text=st.session_state.cv_text,
            job_description=st.session_state.job_description,
            career_request=st.session_state.career_request,
        )

        # ----------------------------------------------------
        # Complete
        # ----------------------------------------------------

        progress_bar.progress(100)

        status_text.success(
            "✅ Career analysis completed successfully."
        )

        st.session_state.analysis_result = outputs
        st.session_state.memory = memory
        st.session_state.analysis_complete = True

    except Exception as error:

        progress_bar.empty()

        status_text.error(
            "❌ Career analysis failed."
        )

        st.session_state.error_message = str(error)

        error_text = str(error)

        # ----------------------------------------------------
        # Friendly Gemini 503 message
        # ----------------------------------------------------

        if is_temporary_ai_error(error):

            st.warning(
                """
                Gemini is temporarily unavailable or overloaded.

                This is usually a temporary API/service issue rather
                than a problem with your CV or job description.

                The application already retried automatically.
                Please wait a little and try again.
                """
            )

        else:

            st.error(
                "Something went wrong while running the AI workflow."
            )

        # ----------------------------------------------------
        # Technical details
        # ----------------------------------------------------

        with st.expander(
            "🔧 Technical error details"
        ):

            st.code(
                error_text,
                language="text",
            )

            st.caption(
                "If you are debugging the application, "
                "the full traceback is shown below."
            )

            st.exception(error)


# ============================================================
# RESULTS
# ============================================================

if st.session_state.analysis_complete:

    outputs = st.session_state.analysis_result

    st.divider()

    st.markdown(
        '<div class="section-title">📊 CareerOps Analysis</div>',
        unsafe_allow_html=True,
    )

    st.success(
        "Your career analysis has been completed."
    )

    # --------------------------------------------------------
    # Summary metrics
    # --------------------------------------------------------

    result_items = [
        (
            "Job Analysis",
            outputs.get("job_analysis", ""),
        ),
        (
            "CV Analysis",
            outputs.get("cv_analysis", ""),
        ),
        (
            "Company Research",
            outputs.get("company_research", ""),
        ),
        (
            "Application",
            outputs.get("application_materials", ""),
        ),
        (
            "Interview Prep",
            outputs.get("interview_preparation", ""),
        ),
        (
            "Review",
            outputs.get("critic_review", ""),
        ),
    ]

    completed_count = sum(
        1
        for _, value in result_items
        if value and value.strip()
    )

    metric1, metric2, metric3 = st.columns(3)

    with metric1:

        st.metric(
            "Workflow",
            "Complete",
        )

    with metric2:

        st.metric(
            "AI Tasks",
            f"{completed_count}/6",
        )

    with metric3:

        st.metric(
            "Status",
            "Ready",
        )

    st.divider()

    # --------------------------------------------------------
    # Result tabs
    # --------------------------------------------------------

    tabs = st.tabs(
        [
            "🎯 Job Analysis",
            "📄 CV Analysis",
            "🏢 Company Research",
            "✍️ Application",
            "🎤 Interview Prep",
            "🔍 Review",
            "👔 Manager Summary",
        ]
    )

    # --------------------------------------------------------
    # Job Analysis
    # --------------------------------------------------------

    with tabs[0]:

        st.subheader(
            "🎯 Job Analysis"
        )

        content = outputs.get(
            "job_analysis",
            "",
        )

        if content:
            st.markdown(content)
        else:
            st.info(
                "No job analysis output was returned."
            )

    # --------------------------------------------------------
    # CV Analysis
    # --------------------------------------------------------

    with tabs[1]:

        st.subheader(
            "📄 CV Analysis"
        )

        content = outputs.get(
            "cv_analysis",
            "",
        )

        if content:
            st.markdown(content)
        else:
            st.info(
                "No CV analysis output was returned."
            )

    # --------------------------------------------------------
    # Company Research
    # --------------------------------------------------------

    with tabs[2]:

        st.subheader(
            "🏢 Company Research"
        )

        content = outputs.get(
            "company_research",
            "",
        )

        if content:
            st.markdown(content)
        else:
            st.info(
                "No company research output was returned."
            )

    # --------------------------------------------------------
    # Application Materials
    # --------------------------------------------------------

    with tabs[3]:

        st.subheader(
            "✍️ Application Materials"
        )

        content = outputs.get(
            "application_materials",
            "",
        )

        if content:
            st.markdown(content)
        else:
            st.info(
                "No application material output was returned."
            )

    # --------------------------------------------------------
    # Interview Preparation
    # --------------------------------------------------------

    with tabs[4]:

        st.subheader(
            "🎤 Interview Preparation"
        )

        content = outputs.get(
            "interview_preparation",
            "",
        )

        if content:
            st.markdown(content)
        else:
            st.info(
                "No interview preparation output was returned."
            )

    # --------------------------------------------------------
    # Critic Review
    # --------------------------------------------------------

    with tabs[5]:

        st.subheader(
            "🔍 Final Review"
        )

        content = outputs.get(
            "critic_review",
            "",
        )

        if content:
            st.markdown(content)
        else:
            st.info(
                "No review output was returned."
            )

    # --------------------------------------------------------
    # Manager Summary
    # --------------------------------------------------------

    with tabs[6]:

        st.subheader(
            "👔 Manager Summary"
        )

        content = outputs.get(
            "manager_summary",
            "",
        )

        if content:
            st.markdown(content)
        else:
            st.info(
                "No manager summary output was returned."
            )

    # ========================================================
    # DOWNLOAD REPORT
    # ========================================================

    st.divider()

    st.subheader(
        "📥 Export Your Career Analysis"
    )

    report = create_report(
        outputs=outputs,
        job_description=st.session_state.job_description,
        career_request=st.session_state.career_request,
    )

    st.download_button(
        label="📄 Download Full CareerOps Report",
        data=report,
        file_name="careerops_analysis_report.txt",
        mime="text/plain",
        use_container_width=True,
    )


# ============================================================
# EMPTY STATE
# ============================================================

elif not analyze_button:

    st.divider()

    st.markdown(
        """
        ### 👋 Ready to get started?

        Provide your **CV**, the **job description**, and tell
        CareerOps AI what you want help with.

        CareerOps will then work through the application workflow:
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            "**1. 🎯 Fit**\n\n"
            "Analyze the job and compare it with your CV."
        )

    with col2:
        st.markdown(
            "**2. 🏢 Research**\n\n"
            "Analyze the company and role context."
        )

    with col3:
        st.markdown(
            "**3. ✍️ Apply**\n\n"
            "Prepare application materials."
        )

    with col4:
        st.markdown(
            "**4. 🎤 Interview**\n\n"
            "Prepare for the interview and review."
        )
