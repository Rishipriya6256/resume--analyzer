import streamlit as st
import pdfplumber
import re
import spacy

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------
# LOAD NLP MODEL
# -----------------------------
nlp = spacy.load("en_core_web_sm")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="AI Resume Analyzer", layout="wide")
st.title("🚀 AI Resume Analyzer (NLP + ATS)")

# -----------------------------
# FILE INPUT
# -----------------------------
resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
job_desc = st.text_area("Paste Job Description")

# -----------------------------
# EXTRACT TEXT FROM PDF
# -----------------------------
def extract_text(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# -----------------------------
# CLEAN TEXT
# -----------------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    return text

# -----------------------------
# NLP SECTION EXTRACTION
# -----------------------------
def extract_sections_nlp(text):
    doc = nlp(text)

    skills = []
    projects = []
    internships = []
    certificates = []

    for sent in doc.sents:
        s = sent.text.lower()

        if any(word in s for word in ["python", "java", "sql", "html", "css", "javascript", "machine learning", "ai", "mongodb"]):
            skills.append(sent.text)

        if any(word in s for word in ["project", "developed", "built", "application"]):
            projects.append(sent.text)

        if any(word in s for word in ["intern", "internship", "worked", "company"]):
            internships.append(sent.text)

        if any(word in s for word in ["certified", "certificate", "course", "training", "completed"]):
            certificates.append(sent.text)

    return {
        "skills": skills,
        "projects": projects,
        "internships": internships,
        "certificates": certificates
    }

# -----------------------------
# ATS SCORE FUNCTION
# -----------------------------
def calculate_ats_score(resume_text, job_desc):

    resume_text = clean_text(resume_text)
    job_desc = clean_text(job_desc)

    # Cosine Similarity
    cv = CountVectorizer(stop_words='english')
    vectors = cv.fit_transform([resume_text, job_desc]).toarray()
    cosine_score = cosine_similarity(vectors)[0][1]

    # Keyword Match
    resume_words = set(resume_text.split())
    jd_words = set(job_desc.split())

    matched = resume_words.intersection(jd_words)

    if len(jd_words) == 0:
        skill_score = 0
    else:
        skill_score = len(matched) / len(jd_words)

    final_score = (cosine_score * 0.6) + (skill_score * 0.4)

    return final_score, matched, jd_words

# -----------------------------
# MAIN EXECUTION
# -----------------------------
if resume_file and job_desc:

    resume_text = extract_text(resume_file)

    st.subheader("📄 Extracted Resume Text")
    st.write(resume_text[:1000])

    # NLP Extraction
    sections = extract_sections_nlp(resume_text)

    st.subheader("🧠 Resume Insights")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🛠 Skills")
        for s in sections["skills"][:5]:
            st.write("✔", s)

        st.markdown("### 💼 Projects")
        for p in sections["projects"][:5]:
            st.write("✔", p)

    with col2:
        st.markdown("### 🏢 Internships")
        for i in sections["internships"][:5]:
            st.write("✔", i)

        st.markdown("### 📜 Certificates")
        for c in sections["certificates"][:5]:
            st.write("✔", c)

    # ATS Score
    score, matched, jd_words = calculate_ats_score(resume_text, job_desc)
    final_score = round(score * 100, 2)

    st.subheader(f"📊 ATS Match Score: {final_score}%")
    st.progress(final_score / 100)

    # Result Message
    if final_score < 50:
        st.error("❌ Your resume needs improvement!")
    elif final_score < 75:
        st.warning("⚠️ Moderate match. Improve keywords.")
    else:
        st.success("✅ Strong match! Ready to apply.")

    # Matched Skills
    st.subheader("✅ Matched Skills")
    st.write(list(matched)[:20])

    # Missing Skills
    missing_skills = jd_words - matched

    st.subheader("⚠️ Missing Skills")
    st.write(list(missing_skills)[:20])

    # Suggestions
    if final_score < 50:
        st.subheader("💡 Suggestions to Improve")
        st.write("- Add more relevant skills from job description")
        st.write("- Include real projects using required technologies")
        st.write("- Add certifications related to the role")