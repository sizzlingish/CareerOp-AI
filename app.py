import streamlit as st


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="CareerOps AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)


# --------------------------------------------------
# Session State
# --------------------------------------------------

if "cv_text" not in st.session_state:
    st.session_state.cv_text = ""

if "job_description" not in st.session_state:
    st.session_state.job_description = ""

if "analysis_started" not in st.session_state:
    st.session_state.analysis_started = False


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:
    st.title("🤖 CareerOps AI")

    st.markdown(
        """
        **AI Career Assistant**

        Your personal AI team for:

        - 🔎 Job analysis
        - 📄 CV matching
        - 🌐 Company research
        - ✍️ Applications
        - 🎤 Interview preparation
        - 🧐 Application review
        """
    )

    st.divider()

    st.info(
        "Upload your CV and provide a job description "
        "to start your career analysis."
    )


# --------------------------------------------------
# Main Header
# --------------------------------------------------

st.title("🤖 CareerOps AI")
st.subheader("Your AI-powered career operations team")

st.markdown(
    """
    CareerOps AI uses specialized AI agents to analyze job
    opportunities, match your experience, create tailored
    application materials, and prepare you for interviews.
    """
)

st.divider()


# --------------------------------------------------
# User Input
# --------------------------------------------------

st.header("📋 Start Your Application")


col1, col2 = st.columns(2)


# --------------------------------------------------
# CV Upload
# --------------------------------------------------

with col1:

    st.subheader("📄 Your CV")

    uploaded_cv = st.file_uploader(
        "Upload your CV",
        type=["pdf", "txt"],
        help="Upload your CV as a PDF or text file."
    )

    if uploaded_cv is not None:

        st.success(
            f"CV uploaded: {uploaded_cv.name}"
        )

        if uploaded_cv.type == "text/plain":
            st.session_state.cv_text = (
                uploaded_cv.read()
                .decode("utf-8")
            )

        else:
            st.session_state.cv_text = (
                f"PDF uploaded: {uploaded_cv.name}"
            )


# --------------------------------------------------
# Job Description
# --------------------------------------------------

with col2:

    st.subheader("💼 Job Description")

    job_description = st.text_area(
        "Paste the job description",
        value=st.session_state.job_description,
        height=250,
        placeholder=(
            "Paste the complete job description here..."
        )
    )

    st.session_state.job_description = job_description


st.divider()


# --------------------------------------------------
# Career Request
# --------------------------------------------------

st.header("🎯 What do you want CareerOps AI to do?")

career_request = st.text_area(
    "Describe your request",
    placeholder=(
        "Example: Help me apply for this Software "
        "Engineering internship and prepare me for the interview."
    ),
    height=120
)


# --------------------------------------------------
# Start Analysis
# --------------------------------------------------

if st.button(
    "🚀 Start Career Analysis",
    type="primary",
    use_container_width=True
):

    if uploaded_cv is None:
        st.warning("Please upload your CV first.")

    elif not job_description.strip():
        st.warning("Please provide a job description.")

    else:
        st.session_state.analysis_started = True

        st.success(
            "Your application is ready for analysis!"
        )


# --------------------------------------------------
# Analysis Dashboard
# --------------------------------------------------

if st.session_state.analysis_started:

    st.divider()

    st.header("📊 CareerOps Analysis")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
        [
            "🔎 Job Analysis",
            "📄 CV Match",
            "🌐 Research",
            "✍️ Application",
            "🎤 Interview",
            "🧐 Review"
        ]
    )


    # ----------------------------------------------
    # Job Analysis
    # ----------------------------------------------

    with tab1:

        st.subheader("🔎 Job Analysis")

        st.info(
            "The Job Analyst Agent will extract job "
            "requirements, skills, qualifications, "
            "responsibilities, and keywords here."
        )

        st.write("Status: ⏳ Waiting for Job Analyst Agent")


    # ----------------------------------------------
    # CV Match
    # ----------------------------------------------

    with tab2:

        st.subheader("📄 CV Match")

        st.info(
            "The CV Agent will compare your experience "
            "and skills against the job requirements here."
        )

        st.write("Status: ⏳ Waiting for CV Agent")


    # ----------------------------------------------
    # Research
    # ----------------------------------------------

    with tab3:

        st.subheader("🌐 Company Research")

        st.info(
            "The Research Agent will research the "
            "organization and opportunity here."
        )

        st.write(
            "Status: ⏳ Waiting for Research Agent"
        )


    # ----------------------------------------------
    # Application
    # ----------------------------------------------

    with tab4:

        st.subheader("✍️ Application Materials")

        st.info(
            "The Application Agent will create "
            "tailored application materials here."
        )

        st.write(
            "Status: ⏳ Waiting for Application Agent"
        )


    # ----------------------------------------------
    # Interview
    # ----------------------------------------------

    with tab5:

        st.subheader("🎤 Interview Preparation")

        st.info(
            "The Interview Agent will generate "
            "job-specific interview preparation here."
        )

        st.write(
            "Status: ⏳ Waiting for Interview Agent"
        )


    # ----------------------------------------------
    # Critic
    # ----------------------------------------------

    with tab6:

        st.subheader("🧐 Final Review")

        st.info(
            "The Critic Agent will review the complete "
            "application and identify weaknesses."
        )

        st.write(
            "Status: ⏳ Waiting for Critic Agent"
        )
