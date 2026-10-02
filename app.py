import os
import tempfile

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
# Streamlit Configuration
# ============================================================

st.set_page_config(
    page_title="CareerOps AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# Session State
# ============================================================

DEFAULT_STATE = {
    "cv_text": "",
    "cv_filename": "",
    "cv_input_method": "Upload CV",
    "job_description": "",
    "career_request": "",
    "analysis_started": False,
    "analysis_complete": False,
    "analysis_error": "",
    "crew_result": None,
    "career_memory": None,
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

if st.session_state.career_memory is None:
    st.session_state.career_memory = CareerMemory()


# ============================================================
# Helper Functions
# ============================================================

def result_to_text(result):
    """
    Convert a CrewAI task result into readable text.
    """

    if result is None:
        return ""

    try:
        return str(result)
    except Exception:
        return repr(result)


def extract_task_output(result, task):
    """
    Safely retrieve the output of a specific CrewAI task.
    """

    # Try task.output first
    try:
        if hasattr(task, "output") and task.output is not None:
            return result_to_text(task.output)
    except Exception:
        pass

    # Try result.tasks_output
    try:
        if hasattr(result, "tasks_output"):

            outputs = result.tasks_output

            for output in outputs:

                if getattr(output, "task", None) == task:
                    return result_to_text(output)

            if outputs:
                return result_to_text(outputs[-1])

    except Exception:
        pass

    return ""


def save_uploaded_cv(uploaded_file):
    """
    Save an uploaded PDF/TXT temporarily and extract its text.
    """

    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        suffix = ".pdf"

    elif filename.endswith(".txt"):
        suffix = ".txt"

    else:
        raise ValueError(
            "Unsupported file type. Please upload PDF or TXT."
        )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(
                uploaded_file.getvalue()
            )

            temp_path = temp_file.name

        # TXT files don't need PDF extraction
        if suffix == ".txt":

            cv_text = uploaded_file.getvalue().decode(
                "utf-8",
                errors="ignore"
            )

        else:

            cv_text = extract_cv_text(temp_path)

        return cv_text

    finally:

        if temp_path and os.path.exists(temp_path):

            try:
                os.remove(temp_path)

            except Exception:
                pass


def reset_analysis():

    st.session_state.analysis_started = False
    st.session_state.analysis_complete = False
    st.session_state.analysis_error = ""
    st.session_state.crew_result = None


def reset_everything():

    st.session_state.cv_text = ""
    st.session_state.cv_filename = ""
    st.session_state.cv_input_method = "Upload CV"
    st.session_state.job_description = ""
    st.session_state.career_request = ""
    st.session_state.analysis_started = False
    st.session_state.analysis_complete = False
    st.session_state.analysis_error = ""
    st.session_state.crew_result = None
    st.session_state.career_memory = CareerMemory()


# ============================================================
# Crew Creation
# ============================================================

def create_crew():
    """
    Create the CareerOps Crew.

    IMPORTANT:
    Task dependencies should be defined in tasks.py using
    CrewAI's context=[previous_task] mechanism.
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


# ============================================================
# Run Career Analysis
# ============================================================

def run_career_analysis(
    cv_text,
    job_description,
    career_request
):

    memory = CareerMemory()

    memory.set_candidate_cv(cv_text)

    memory.set_job(job_description)

    memory.update_stage("Starting")

    memory.update_status("Running")

    crew = create_crew()

    # --------------------------------------------------------
    # IMPORTANT:
    # These are the ONLY initial variables supplied to CrewAI.
    #
    # Therefore tasks.py should only use:
    #
    # {cv_text}
    # {job_description}
    # {career_request}
    #
    # Any result such as job_analysis must come from
    # a previous task through context=[...].
    # --------------------------------------------------------

    inputs = {
        "cv_text": cv_text,
        "job_description": job_description,
        "career_request": career_request,
    }

    result = crew.kickoff(
        inputs=inputs
    )

    # --------------------------------------------------------
    # Extract individual task results
    # --------------------------------------------------------

    job_result = extract_task_output(
        result,
        job_analysis_task
    )

    cv_result = extract_task_output(
        result,
        cv_analysis_task
    )

    research_result = extract_task_output(
        result,
        company_research_task
    )

    application_result = extract_task_output(
        result,
        application_task
    )

    interview_result = extract_task_output(
        result,
        interview_task
    )

    critic_result = extract_task_output(
        result,
        critic_task
    )

    manager_result = extract_task_output(
        result,
        manager_task
    )

    # --------------------------------------------------------
    # Store results in CareerMemory
    # --------------------------------------------------------

    memory.save_job_analysis({
        "content": job_result
    })

    memory.save_cv_analysis({
        "content": cv_result
    })

    memory.save_company_research({
        "content": research_result
    })

    memory.save_application_materials({
        "content": application_result
    })

    memory.save_interview_preparation({
        "content": interview_result
    })

    memory.save_critic_review({
        "content": critic_result
    })

    memory.save_manager_summary({
        "content": manager_result
    })

    memory.update_stage("Completed")

    memory.update_status("Completed")

    return result, memory


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:

    st.title("🤖 CareerOps AI")

    st.markdown(
        """
        ### AI Career Assistant

        CareerOps AI analyzes your CV and target job,
        researches the organization, prepares application
        materials, and generates interview preparation.
        """
    )

    st.divider()

    st.markdown("### Workflow")

    st.markdown(
        """
        1. 📄 CV Analysis
        2. 🔎 Job Analysis
        3. 🏢 Company Research
        4. ✍️ Application
        5. 🎤 Interview Preparation
        6. 🧐 Quality Review
        7. 📋 Manager Summary
        """
    )

    st.divider()

    if st.button(
        "🔄 Reset Session",
        use_container_width=True
    ):

        reset_everything()

        st.rerun()


# ============================================================
# Main Header
# ============================================================

st.title("🤖 CareerOps AI")

st.markdown(
    """
    ### Your AI-powered career operations assistant

    Provide your CV, paste a job description, and tell
    CareerOps AI what you need help with.
    """
)

st.divider()


# ============================================================
# CV INPUT
# ============================================================

st.subheader("📄 Your CV")

st.markdown(
    "Choose how you want to provide your CV:"
)

cv_input_method = st.radio(
    "CV input method",
    [
        "Upload CV",
        "Paste CV text",
    ],
    horizontal=True,
    label_visibility="collapsed",
)

st.session_state.cv_input_method = cv_input_method


# ============================================================
# Upload CV
# ============================================================

if cv_input_method == "Upload CV":

    uploaded_cv = st.file_uploader(
        "Upload your CV",
        type=["pdf", "txt"],
        help="Upload your CV as a PDF or TXT file.",
    )

    if uploaded_cv is not None:

        # Avoid unnecessarily re-processing the same file
        if (
            st.session_state.cv_filename
            != uploaded_cv.name
        ):

            try:

                with st.spinner(
                    "📄 Reading your CV..."
                ):

                    cv_text = save_uploaded_cv(
                        uploaded_cv
                    )

                st.session_state.cv_text = cv_text

                st.session_state.cv_filename = (
                    uploaded_cv.name
                )

                # Reset old analysis because CV changed
                reset_analysis()

                if not cv_text.strip():

                    st.warning(
                        "The uploaded file does not contain "
                        "readable CV text."
                    )

                elif cv_text.startswith(
                    "Unable to extract"
                ):

                    st.error(cv_text)

                elif cv_text.startswith(
                    "No readable text"
                ):

                    st.warning(cv_text)

                else:

                    st.success(
                        f"CV uploaded successfully: "
                        f"{uploaded_cv.name}"
                    )

            except Exception as error:

                st.error(
                    f"Could not process the CV: {error}"
                )

        else:

            st.success(
                f"CV ready: {uploaded_cv.name}"
            )

    # Preview uploaded CV
    if st.session_state.cv_text.strip():

        with st.expander(
            "👀 Preview CV text"
        ):

            preview = (
                st.session_state.cv_text[:5000]
            )

            st.text(preview)

            if len(
                st.session_state.cv_text
            ) > 5000:

                st.caption(
                    "Showing the first 5,000 characters."
                )


# ============================================================
# Paste CV
# ============================================================

else:

    pasted_cv = st.text_area(
        "Paste your CV text here",
        value=st.session_state.cv_text,
        height=400,
        placeholder=(
            "Paste the complete text of your CV here...\n\n"
            "Example:\n"
            "John Smith\n"
            "Senior Data Engineer\n\n"
            "Experience\n"
            "...\n\n"
            "Skills\n"
            "Python, SQL, AWS, Databricks..."
        ),
        help=(
            "You can paste the full text of your CV "
            "directly into this box."
        ),
    )

    # Only update if the text changed
    if pasted_cv != st.session_state.cv_text:

        st.session_state.cv_text = pasted_cv

        st.session_state.cv_filename = (
            "Pasted CV"
        )

        reset_analysis()

    if pasted_cv.strip():

        st.success(
            f"CV text ready — "
            f"{len(pasted_cv):,} characters"
        )

    else:

        st.info(
            "Paste your CV text above to continue."
        )


# ============================================================
# Job Description
# ============================================================

st.subheader("💼 Target Job")

job_description = st.text_area(
    "Paste the job description",
    value=st.session_state.job_description,
    height=350,
    placeholder=(
        "Paste the complete job description here...\n\n"
        "Include the job title, responsibilities, "
        "requirements, qualifications, skills, and "
        "company information if available."
    ),
)

if job_description != st.session_state.job_description:

    st.session_state.job_description = (
        job_description
    )

    reset_analysis()


# ============================================================
# Career Request
# ============================================================

st.subheader("🎯 Career Request")

career_request = st.text_area(
    "What would you like CareerOps AI to help with?",
    value=st.session_state.career_request,
    placeholder=(
        "Example:\n"
        "Help me apply for this job. Identify my strengths "
        "and gaps, tailor my application, write a cover "
        "letter, and prepare me for the interview."
    ),
    height=150,
)

st.session_state.career_request = career_request


# ============================================================
# Input Summary
# ============================================================

if (
    st.session_state.cv_text.strip()
    or job_description.strip()
):

    st.divider()

    st.subheader("✅ Application Input Check")

    col1, col2, col3 = st.columns(3)

    with col1:

        if st.session_state.cv_text.strip():

            st.success(
                f"📄 CV ready\n\n"
                f"{len(st.session_state.cv_text):,} "
                f"characters"
            )

        else:

            st.warning(
                "📄 CV missing"
            )

    with col2:

        if job_description.strip():

            st.success(
                f"💼 Job description ready\n\n"
                f"{len(job_description):,} "
                f"characters"
            )

        else:

            st.warning(
                "💼 Job description missing"
            )

    with col3:

        if career_request.strip():

            st.success(
                "🎯 Career request ready"
            )

        else:

            st.info(
                "🎯 Career request optional"
            )


# ============================================================
# Start Analysis
# ============================================================

st.divider()

start_analysis = st.button(
    "🚀 Start Career Analysis",
    type="primary",
    use_container_width=True,
)


if start_analysis:

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    if not st.session_state.cv_text.strip():

        st.warning(
            "📄 Please upload your CV or paste your CV "
            "text first."
        )

    elif not job_description.strip():

        st.warning(
            "💼 Please provide a job description."
        )

    else:

        st.session_state.analysis_started = True

        st.session_state.analysis_complete = False

        st.session_state.analysis_error = ""

        st.session_state.crew_result = None

        progress = st.progress(0)

        status_placeholder = st.empty()

        try:

            status_placeholder.info(
                "🤖 Starting CareerOps AI workflow..."
            )

            progress.progress(10)

            status_placeholder.info(
                "🔎 Analyzing the job description..."
            )

            progress.progress(20)

            status_placeholder.info(
                "📄 Matching your CV against the role..."
            )

            progress.progress(35)

            status_placeholder.info(
                "🏢 Researching the organization..."
            )

            progress.progress(50)

            status_placeholder.info(
                "✍️ Preparing application materials..."
            )

            progress.progress(65)

            status_placeholder.info(
                "🎤 Preparing interview questions..."
            )

            progress.progress(80)

            status_placeholder.info(
                "🧐 Performing final quality review..."
            )

            result, memory = run_career_analysis(
                cv_text=st.session_state.cv_text,
                job_description=job_description,
                career_request=career_request,
            )

            st.session_state.crew_result = result

            st.session_state.career_memory = memory

            st.session_state.analysis_complete = True

            progress.progress(100)

            status_placeholder.success(
                "✅ Career analysis completed successfully!"
            )

        except Exception as error:

            st.session_state.analysis_error = (
                str(error)
            )

            st.session_state.analysis_complete = False

            progress.empty()

            status_placeholder.error(
                "❌ Career analysis failed."
            )

            st.error(
                f"Error: {error}"
            )

            with st.expander(
                "🔧 Technical error details"
            ):

                st.exception(error)


# ============================================================
# Results Dashboard
# ============================================================

if st.session_state.analysis_started:

    st.divider()

    if st.session_state.analysis_complete:

        memory = st.session_state.career_memory

        st.success(
            "🎉 Your CareerOps AI analysis is ready."
        )

        # ----------------------------------------------------
        # Workflow Summary
        # ----------------------------------------------------

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "Workflow",
                memory.workflow_status
            )

        with col2:

            st.metric(
                "Current Stage",
                memory.current_stage
            )

        with col3:

            st.metric(
                "Revision Count",
                memory.revision_count
            )

        with col4:

            st.metric(
                "CV Characters",
                f"{len(memory.cv_text):,}"
            )

        st.divider()

        # ----------------------------------------------------
        # Tabs
        # ----------------------------------------------------

        (
            tab_job,
            tab_cv,
            tab_research,
            tab_application,
            tab_interview,
            tab_review,
            tab_manager,
        ) = st.tabs(
            [
                "🔎 Job Analysis",
                "📄 CV Match",
                "🏢 Research",
                "✍️ Application",
                "🎤 Interview",
                "🧐 Review",
                "📋 Manager Summary",
            ]
        )

        # ----------------------------------------------------
        # Job Analysis
        # ----------------------------------------------------

        with tab_job:

            st.subheader(
                "🔎 Job Analysis"
            )

            job_content = (
                memory.job_analysis.get(
                    "content",
                    ""
                )
            )

            if job_content:

                st.markdown(job_content)

            else:

                st.warning(
                    "No job analysis result was returned."
                )

        # ----------------------------------------------------
        # CV Match
        # ----------------------------------------------------

        with tab_cv:

            st.subheader(
                "📄 CV-to-Job Match"
            )

            cv_content = (
                memory.cv_analysis.get(
                    "content",
                    ""
                )
            )

            if cv_content:

                st.markdown(cv_content)

            else:

                st.warning(
                    "No CV analysis result was returned."
                )

        # ----------------------------------------------------
        # Company Research
        # ----------------------------------------------------

        with tab_research:

            st.subheader(
                "🏢 Company Research"
            )

            research_content = (
                memory.company_research.get(
                    "content",
                    ""
                )
            )

            if research_content:

                st.markdown(research_content)

            else:

                st.warning(
                    "No company research result was returned."
                )

        # ----------------------------------------------------
        # Application
        # ----------------------------------------------------

        with tab_application:

            st.subheader(
                "✍️ Application Materials"
            )

            application_content = (
                memory.application_materials.get(
                    "content",
                    ""
                )
            )

            if application_content:

                st.markdown(
                    application_content
                )

            else:

                st.warning(
                    "No application materials were returned."
                )

        # ----------------------------------------------------
        # Interview
        # ----------------------------------------------------

        with tab_interview:

            st.subheader(
                "🎤 Interview Preparation"
            )

            interview_content = (
                memory.interview_preparation.get(
                    "content",
                    ""
                )
            )

            if interview_content:

                st.markdown(
                    interview_content
                )

            else:

                st.warning(
                    "No interview preparation was returned."
                )

        # ----------------------------------------------------
        # Critic Review
        # ----------------------------------------------------

        with tab_review:

            st.subheader(
                "🧐 Final Quality Review"
            )

            critic_content = (
                memory.critic_review.get(
                    "content",
                    ""
                )
            )

            if critic_content:

                st.markdown(
                    critic_content
                )

            else:

                st.warning(
                    "No critic review was returned."
                )

        # ----------------------------------------------------
        # Manager Summary
        # ----------------------------------------------------

        with tab_manager:

            st.subheader(
                "📋 Career Operations Summary"
            )

            manager_content = (
                memory.manager_summary.get(
                    "content",
                    ""
                )
            )

            if manager_content:

                st.markdown(
                    manager_content
                )

            else:

                st.warning(
                    "No manager summary was returned."
                )

    elif st.session_state.analysis_error:

        st.error(
            "❌ The CareerOps workflow could not complete."
        )

        st.markdown(
            """
            The application encountered an error while
            running the CrewAI workflow.

            The technical details below can be used to
            identify the problem.
            """
        )

        with st.expander(
            "🔧 Show technical error"
        ):

            st.code(
                st.session_state.analysis_error
            )

    else:

        st.info(
            "⏳ Career analysis is running. "
            "Results will appear here when processing completes."
        )
