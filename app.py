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

if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

if "cv_filename" not in st.session_state:
    st.session_state.cv_filename = ""

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "career_request" not in st.session_state:
    st.session_state.career_request = ""

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "analysis_error" not in st.session_state:
    st.session_state.analysis_error = ""

if "crew_result" not in st.session_state:
    st.session_state.crew_result = None

if "career_memory" not in st.session_state:
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
    Try to retrieve the output of a specific task from the
    final CrewAI result.

    CrewAI versions can expose task outputs slightly
    differently, so this function uses safe fallbacks.
    """

    try:
        if hasattr(task, "output") and task.output is not None:
            return result_to_text(task.output)
    except Exception:
        pass

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
    Save the uploaded CV temporarily and extract its text.
    """

    suffix = ".pdf"

    if uploaded_file.name.lower().endswith(".txt"):
        suffix = ".txt"

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=suffix
        ) as temp_file:

            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        if suffix == ".pdf":
            cv_text = extract_cv_text(temp_path)
        else:
            cv_text = uploaded_file.getvalue().decode(
                "utf-8",
                errors="ignore"
            )

        return cv_text

    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception:
                pass


def create_crew():
    """
    Create the CareerOps Crew.

    The tasks already contain their dependencies through
    the `context` arguments in tasks.py.
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


def run_career_analysis(
    cv_text,
    job_description,
    career_request
):
    """
    Run the complete CareerOps workflow.
    """

    memory = CareerMemory()

    memory.set_candidate_cv(cv_text)
    memory.set_job(job_description)

    memory.update_stage("Starting")
    memory.update_status("Running")

    crew = create_crew()

    inputs = {
        "cv_text": cv_text,
        "job_description": job_description,
        "career_request": career_request,
    }

    result = crew.kickoff(inputs=inputs)

    # --------------------------------------------------------
    # Save individual task results
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
    # Store results
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
        then prepares a complete application and
        interview package.
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
        st.session_state.cv_text = ""
        st.session_state.cv_filename = ""
        st.session_state.job_description = ""
        st.session_state.career_request = ""
        st.session_state.analysis_started = False
        st.session_state.analysis_complete = False
        st.session_state.analysis_error = ""
        st.session_state.crew_result = None
        st.session_state.career_memory = CareerMemory()

        st.rerun()


# ============================================================
# Main Header
# ============================================================

st.title("🤖 CareerOps AI")

st.markdown(
    """
    ### Your AI-powered career operations assistant

    Upload your CV, provide a job description, and let
    CareerOps AI analyze the opportunity, match your
    experience, research the organization, prepare your
    application, and generate interview preparation.
    """
)

st.divider()


# ============================================================
# CV Upload
# ============================================================

st.subheader("📄 Your CV")

uploaded_cv = st.file_uploader(
    "Upload your CV",
    type=["pdf", "txt"],
    help="Upload your CV as a PDF or text file.",
)

if uploaded_cv is not None:

    st.session_state.cv_filename = uploaded_cv.name

    try:

        with st.spinner("Reading your CV..."):

            cv_text = save_uploaded_cv(uploaded_cv)

        st.session_state.cv_text = cv_text

        if cv_text.startswith("Unable to extract"):
            st.error(cv_text)

        elif cv_text.startswith("No readable text"):
            st.warning(cv_text)

        else:
            st.success(
                f"CV uploaded and extracted: "
                f"{uploaded_cv.name}"
            )

            with st.expander("Preview extracted CV text"):

                preview = cv_text[:5000]

                st.text(preview)

                if len(cv_text) > 5000:
                    st.caption(
                        "Showing the first 5,000 characters."
                    )

    except Exception as error:

        st.error(
            f"Could not process the CV: {error}"
        )


# ============================================================
# Job Description
# ============================================================

st.subheader("💼 Target Job")

job_description = st.text_area(
    "Paste the job description",
    value=st.session_state.job_description,
    height=300,
    placeholder=(
        "Paste the complete job description here..."
    ),
)

st.session_state.job_description = job_description


# ============================================================
# Career Request
# ============================================================

st.subheader("🎯 Career Request")

career_request = st.text_area(
    "Describe what you want CareerOps AI to help with",
    value=st.session_state.career_request,
    placeholder=(
        "Example: Help me apply for this Security Analyst "
        "position and prepare me for the interview."
    ),
    height=120,
)

st.session_state.career_request = career_request


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
            "Please upload a readable CV first."
        )

    elif not job_description.strip():

        st.warning(
            "Please provide a job description."
        )

    else:

        st.session_state.analysis_started = True
        st.session_state.analysis_complete = False
        st.session_state.analysis_error = ""
        st.session_state.crew_result = None

        # ----------------------------------------------------
        # Run CrewAI
        # ----------------------------------------------------

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

            st.session_state.analysis_error = str(error)
            st.session_state.analysis_complete = False

            progress.empty()

            status_placeholder.error(
                "❌ Career analysis failed."
            )

            st.error(
                f"Error: {error}"
            )

            with st.expander(
                "Technical error details"
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
            "Your CareerOps AI analysis is ready."
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

        # ====================================================
        # Job Analysis
        # ====================================================

        with tab_job:

            st.subheader("🔎 Job Analysis")

            job_content = memory.job_analysis.get(
                "content",
                ""
            )

            if job_content:

                st.markdown(job_content)

            else:

                st.warning(
                    "No job analysis result was returned."
                )

        # ====================================================
        # CV Match
        # ====================================================

        with tab_cv:

            st.subheader("📄 CV-to-Job Match")

            cv_content = memory.cv_analysis.get(
                "content",
                ""
            )

            if cv_content:

                st.markdown(cv_content)

            else:

                st.warning(
                    "No CV analysis result was returned."
                )

        # ====================================================
        # Company Research
        # ====================================================

        with tab_research:

            st.subheader("🏢 Company Research")

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

        # ====================================================
        # Application
        # ====================================================

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

                st.markdown(application_content)

            else:

                st.warning(
                    "No application materials were returned."
                )

        # ====================================================
        # Interview
        # ====================================================

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

                st.markdown(interview_content)

            else:

                st.warning(
                    "No interview preparation was returned."
                )

        # ====================================================
        # Critic Review
        # ====================================================

        with tab_review:

            st.subheader(
                "🧐 Final Quality Review"
            )

            critic_content = memory.critic_review.get(
                "content",
                ""
            )

            if critic_content:

                st.markdown(critic_content)

            else:

                st.warning(
                    "No critic review was returned."
                )

        # ====================================================
        # Manager Summary
        # ====================================================

        with tab_manager:

            st.subheader(
                "📋 Career Operations Summary"
            )

            manager_content = memory.manager_summary.get(
                "content",
                ""
            )

            if manager_content:

                st.markdown(manager_content)

            else:

                st.warning(
                    "No manager summary was returned."
                )

    elif st.session_state.analysis_error:

        st.error(
            "The CareerOps workflow could not complete."
        )

        st.markdown(
            """
            Check the error message above. Common causes
            include an incorrect Gemini API key, a CrewAI
            configuration issue, or an agent/task failure.
            """
        )

    else:

        st.info(
            "⏳ Career analysis has started. "
            "Results will appear here when processing completes."
        )
