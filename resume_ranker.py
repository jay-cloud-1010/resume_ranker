import streamlit as st
import pandas as pd
import tempfile
import os
import PyPDF2
from typing import List, Dict
from langchain_openai import AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class ResumeScore(BaseModel):
    score: float = Field(description="Score between 0 and 100")
    reasoning: str = Field(description="Explanation for the score")


class ResumeParser:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from a PDF file."""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text() or ""
                return text.strip()
        except Exception as e:
            print(f"Error extracting text from {pdf_path}: {str(e)}")
            return ""

    @staticmethod
    def process_resumes(resume_files: List[str]) -> Dict[str, str]:
        """Process multiple resume PDFs and return a dictionary of filename to content."""
        resume_contents = {}
        for resume_file in resume_files:
            if resume_file.lower().endswith('.pdf'):
                content = ResumeParser.extract_text_from_pdf(resume_file)
                filename = os.path.basename(resume_file)
                resume_contents[filename] = content
        return resume_contents


class ResumeRanker:
    def _init_(self):
        # Check if required environment variables are set
        required_vars = ["AZURE_OPENAI_DEPLOYMENT_NAME", "AZURE_OPENAI_API_VERSION", "AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY"]
        missing_vars = [var for var in required_vars if not os.getenv(var)]

        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

        self.llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            temperature=0.7,
        )

        self.parser = PydanticOutputParser(pydantic_object=ResumeScore)

        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", f"""You are an expert resume reviewer. Your task is to evaluate how well a resume matches a job description.
Consider factors like:
- Required skills and experience
- Education and qualifications
- Relevant projects and achievements
- Overall fit for the role

Provide a score between 0-100 and explain your reasoning.
{self.parser.get_format_instructions()}"""),
            ("human", """Job Description:
{job_description}

Resume:
{resume_content}

Evaluate this resume against the job description.""")
        ])

    def score_resume(self, job_description: str, resume_content: str) -> ResumeScore:
        """Score a single resume against the job description."""
        try:
            formatted_prompt = self.prompt_template.format_messages(
                job_description=job_description,
                resume_content=resume_content
            )
            result = self.llm.invoke(formatted_prompt)
            return self.parser.parse(result.content)
        except Exception as e:
            print(f"Error scoring resume: {e}")
            return ResumeScore(score=0.0, reasoning="Error occurred while scoring.")

    def rank_resumes(self, job_description: str, resume_contents: Dict[str, str]) -> List[Dict]:
        """Rank multiple resumes against the job description."""
        ranked_resumes = []

        for filename, content in resume_contents.items():
            score_result = self.score_resume(job_description, content)
            ranked_resumes.append({
                "filename": filename,
                "score": score_result.score,
                "reasoning": score_result.reasoning
            })

        # Sort by score in descending order
        ranked_resumes.sort(key=lambda x: x["score"], reverse=True)
        return ranked_resumes


# Streamlit UI
st.set_page_config(
    page_title="Resume Ranker",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Resume Ranker")
st.write("Upload a job description and multiple resumes to get them ranked based on relevance.")

# Initialize session state
if 'ranked_resumes' not in st.session_state:
    st.session_state.ranked_resumes = None

# File uploaders
job_description = st.text_area("Paste the Job Description here:", height=200)

uploaded_resumes = st.file_uploader(
    "Upload Resumes (PDF files)",
    type=['pdf'],
    accept_multiple_files=True
)

if st.button("Rank Resumes") and job_description and uploaded_resumes:
    with st.spinner("Processing resumes..."):
        # Save uploaded files temporarily
        temp_files = []
        for uploaded_file in uploaded_resumes:
            if uploaded_file.name.lower().endswith('.pdf'):
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_files.append(tmp_file.name)

        try:
            # Parse resumes
            resume_parser = ResumeParser()
            resume_contents = resume_parser.process_resumes(temp_files)

            # Rank resumes
            ranker = ResumeRanker()
            ranked_results = ranker.rank_resumes(job_description, resume_contents)
            st.session_state.ranked_resumes = ranked_results

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                os.unlink(temp_file)

# Display results
if st.session_state.ranked_resumes:
    st.subheader("Ranked Resumes")

    # Create DataFrame for better display
    df = pd.DataFrame(st.session_state.ranked_resumes)
    df['score'] = df['score'].round(2)

    # Display as a table
    st.dataframe(
        df,
        column_config={
            "filename": "Resume",
            "score": st.column_config.NumberColumn(
                "Score",
                help="Match score (0-100)",
                format="%.2f"
            ),
            "reasoning": "Reasoning"
        },
        hide_index=True
    )

    # Display detailed view
    st.subheader("Detailed Analysis")
    for resume in st.session_state.ranked_resumes:
        with st.expander(f"{resume['filename']} - Score: {resume['score']:.2f}"):
            st.write(resume['reasoning'])