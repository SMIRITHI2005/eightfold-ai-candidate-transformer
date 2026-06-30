"""Simple Streamlit UI for the candidate transformer."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import streamlit as st

if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[2]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

from candidate_transformer.pipeline import CandidateTransformer
from candidate_transformer.projection import ProjectionConfig, ProjectionEngine


DEFAULT_PROJECTION_CONFIG = """{
  "fields": [
    { "path": "full_name", "type": "string", "required": true },
    { "path": "primary_email", "from": "emails[0]", "type": "string", "required": true },
    { "path": "phone", "from": "phones[0]", "type": "string", "normalize": "E164" },
    { "path": "skills", "from": "skills", "type": "string[]", "normalize": "canonical" }
  ],
  "include_confidence": true,
  "include_provenance": true,
  "on_missing": "null"
}"""


def _read_uploaded_file(uploaded_file: Any, output_dir: Path) -> Path:
    destination = output_dir / uploaded_file.name
    destination.write_bytes(uploaded_file.getbuffer())
    return destination


def _render_json(title: str, payload: Any) -> None:
    st.subheader(title)
    st.code(json.dumps(payload, indent=2, default=str), language="json")


def main() -> None:
    """Render the interactive candidate transformer UI."""

    st.set_page_config(page_title="Candidate Profile Builder", layout="wide", initial_sidebar_state="expanded")

    # Enhanced humanized styling
    st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&family=Merriweather:ital,wght@0,300;0,400;0,700;1,300;1,400&display=swap');
        :root{
            --bg:#eef4ff;
            --surface:#ffffff;
            --surface-soft:#f9fbff;
            --muted:#52606d;
            --accent:#4f67f2;
            --accent-soft:#dbe6ff;
            --border:rgba(79,103,242,0.18);
            --card:#ffffff;
            --glass: rgba(255,255,255,0.88);
        }
        * { font-family: 'Inter', 'Merriweather', 'Helvetica Neue', Arial, sans-serif; }
        body { background: linear-gradient(180deg, #f7faff 0%, #ffffff 70%); }

        .header-section { 
            background: linear-gradient(90deg, #f8fbff 0%, #ffffff 100%);
            color: #0f1724; 
            padding: 28px; 
            border-radius: 18px; 
            margin-bottom: 20px;
            box-shadow: 0 10px 30px rgba(16,24,40,0.06);
            border: 1px solid var(--border);
            display:flex; align-items:center; gap:20px;
        }
        .brand { width:88px; height:88px; background:linear-gradient(135deg,var(--accent),#7c93ff); border-radius:14px; display:flex;align-items:center;justify-content:center; color:white; font-weight:700; font-size:28px; }
        .header-text h1 { margin: 0; font-size:1.6rem; letter-spacing: -0.5px; }
        .header-text p { margin: 6px 0 0 0; font-size:0.98rem; color:var(--muted); }

        .input-card {
            background: var(--surface);
            border: 1px solid rgba(79,103,242,0.08);
            padding: 22px;
            border-radius: 14px;
            margin: 14px 0;
            box-shadow: 0 6px 18px rgba(79,103,242,0.06);
        }

        .hint-text { color: var(--muted); font-size:0.94rem; margin-top:8px; }

        .metric-card { background: linear-gradient(180deg,#ffffff, #fbfdff); color:#1f2937; border-radius:12px; padding:18px; min-height:110px; box-shadow:0 6px 14px rgba(16,24,40,0.04); }
        .metric-card strong{ display:block; color:#0f1724; margin-bottom:8px; }

        .profile-detail{ background:var(--card); color:#1f2937; border-radius:12px; padding:16px; margin:12px 0; box-shadow:0 6px 14px rgba(16,24,40,0.03);} 
        .profile-detail h4{ color:#0f1724; }

        .stButton>button{ background: linear-gradient(90deg,var(--accent),#60a5fa); color:white; border-radius:999px; padding:12px 20px; border:none; }
        .stButton>button:hover{ filter:brightness(0.98); }

        .avatar-circle{ width:84px; height:84px; border-radius:14px; overflow:hidden; display:inline-block; box-shadow:0 6px 14px rgba(16,24,40,0.06); }
        .small-muted{ color:var(--muted); font-size:0.88rem; }

        /* Responsive tweaks */
        @media (max-width: 800px){ .header-section{flex-direction:column; align-items:flex-start;} }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("""
        <div class="header-section">
            <div class="brand">CP</div>
            <div class="header-text">
                <h1>Candidate Profile Builder</h1>
                <p class="small-muted">Merge resumes, LinkedIn, GitHub and ATS data into a single human-friendly profile — quick, private, and evidence-backed.</p>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Input Section
    st.markdown("## 📥 Input Sources")
    st.markdown("Upload files or enter URLs to add candidate information. We keep everything local and private.")
    
    uploaded_files = []
    input_sources = []
    
    # LinkedIn Input (File or URL)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 💼 LinkedIn")
    st.markdown("Share a LinkedIn export file or paste a public profile URL. We only read the public HTML and extract basic profile info.")
    linkedin_tab1, linkedin_tab2 = st.tabs(["Upload File", "Enter URL"])
    
    with linkedin_tab1:
        linkedin_file = st.file_uploader("LinkedIn export file (JSON)", type=["json"], key="linkedin_file")
        if linkedin_file:
            uploaded_files.append(linkedin_file)
    
    with linkedin_tab2:
        linkedin_url = st.text_input("LinkedIn Profile URL", placeholder="https://www.linkedin.com/in/johndoe", key="linkedin_url")
        st.markdown('<div class="hint-text">Example: https://www.linkedin.com/in/johndoe</div>', unsafe_allow_html=True)
        if linkedin_url and linkedin_url.strip():
            input_sources.append(linkedin_url.strip())
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # GitHub Input (File or URL)
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 💻 GitHub")
    st.markdown("Upload a GitHub profile export or paste the profile URL. We extract public bio, repos, and contact links.")
    github_tab1, github_tab2 = st.tabs(["Upload File", "Enter URL"])
    
    with github_tab1:
        github_file = st.file_uploader("GitHub export file (JSON)", type=["json"], key="github_file")
        if github_file:
            uploaded_files.append(github_file)
    
    with github_tab2:
        github_url = st.text_input("GitHub Profile URL", placeholder="https://github.com/johndoe", key="github_url")
        st.markdown('<div class="hint-text">Example: https://github.com/johndoe</div>', unsafe_allow_html=True)
        if github_url and github_url.strip():
            input_sources.append(github_url.strip())
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Other Files
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    st.markdown("### 📋 Additional Documents")
    st.markdown("Drag and drop resumes, ATS CSVs, recruiter notes, or recruiter-exported JSON files. Supported: PDF, DOCX, TXT, CSV, JSON.")
    other_files = st.file_uploader(
        "Upload files",
        type=["pdf", "docx", "txt", "csv", "json"],
        accept_multiple_files=True,
        key="other_files"
    )
    if other_files:
        uploaded_files.extend(other_files)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Configuration Section
    st.markdown("## ⚙️ Configuration")
    with st.expander("Advanced Projection Config", expanded=False):
        config_text = st.text_area("JSON Config", value=DEFAULT_PROJECTION_CONFIG, height=200)
    
    # Transform Button
    col_btn = st.columns([1, 3, 1])
    with col_btn[1]:
        run_clicked = st.button("🚀 Transform & Analyze", type="primary", use_container_width=True)
    
    if not run_clicked:
        st.info("📌 Upload files or enter URLs, then click Transform to merge and analyze candidate data.")
        return
    
    if not uploaded_files and not input_sources:
        st.error("❌ Please upload at least one file or enter at least one URL.")
        return
    
    try:
        projection_config = ProjectionConfig.from_text(config_text)
    except Exception as exc:
        st.error(f"❌ Invalid configuration: {exc}")
        return
    
    # Process files and URLs
    with st.spinner("🔄 Processing candidate data from all sources..."):
        with tempfile.TemporaryDirectory() as temp_dir_name:
            temp_dir = Path(temp_dir_name)
            input_paths = [_read_uploaded_file(uploaded_file, temp_dir) for uploaded_file in uploaded_files]
            input_paths.extend([Path(source) for source in input_sources])
            
            try:
                result = CandidateTransformer().transform_paths(input_paths)
                projection_result = ProjectionEngine(projection_config).project(result.profile)
            except Exception as exc:
                st.error(f"❌ Transformation failed: {exc}")
                return
    
    st.success("✅ Transformation complete!")
    
    # Output Section
    st.markdown("## 📊 Results")
    st.markdown("We'll show the merged, human-reviewed profile below along with the evidence supporting each item.")

    # Full Candidate Profile
    st.markdown("### 👤 Complete Candidate Profile")
    profile_data = result.profile.model_dump(mode="json")
    
    # Personal Information
    st.markdown("#### 📋 Personal Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card'><strong>Full Name</strong><br>{profile_data.get('full_name', 'N/A')}</div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card'><strong>Headline</strong><br>{profile_data.get('headline', 'N/A')}</div>", unsafe_allow_html=True)
    with col3:
        st.markdown(f"<div class='metric-card'><strong>Location</strong><br>{profile_data.get('location', 'N/A')}</div>", unsafe_allow_html=True)
    with col4:
        st.markdown(f"<div class='metric-card'><strong>Summary</strong><br>{(profile_data.get('summary', 'N/A')[:120] + '...') if profile_data.get('summary') else 'N/A'}</div>", unsafe_allow_html=True)
    
    # Avatar and contact quick view
    left, right = st.columns([1, 3])
    with left:
        avatar_html = f"<div class='avatar-circle'><img src='https://ui-avatars.com/api/?name={profile_data.get('full_name','Candidate')}&background=60a5fa&color=fff&size=256' style='width:100%;height:100%;object-fit:cover;'></div>"
        st.markdown(avatar_html, unsafe_allow_html=True)
        if profile_data.get('emails'):
            st.markdown(f"**Email**<br><span class='small-muted'>{', '.join(profile_data.get('emails'))}</span>", unsafe_allow_html=True)
        if profile_data.get('phones'):
            st.markdown(f"**Phone**<br><span class='small-muted'>{', '.join(profile_data.get('phones'))}</span>", unsafe_allow_html=True)
    with right:
        st.markdown("<div class='profile-detail'><h4>About this profile</h4><div class='small-muted'>This view presents a merged profile across uploaded sources. Use the evidence toggle to inspect sources and reasoning.</div></div>", unsafe_allow_html=True)

    # Evidence toggle
    show_evidence = st.checkbox("Show raw evidence and provenance", value=False)
    if show_evidence:
        st.markdown("### Evidence")
        _render_json("Evidence", [e.model_dump(mode='json') for e in result.evidence])
        st.markdown("### Provenance")
        _render_json("Provenance", {k: v.model_dump(mode='json') for k, v in result.profile.provenance.items()})
    
    # Contact Information
    st.markdown("#### 📞 Contact Information")
    contact_cols = st.columns(3)
    
    email_list = profile_data.get("emails", [])
    with contact_cols[0]:
        st.markdown('<div class="profile-detail">', unsafe_allow_html=True)
        st.write("**📧 Emails**")
        if email_list:
            for email in email_list:
                st.write(f"• {email}")
        else:
            st.write("_No emails found_")
        st.markdown('</div>', unsafe_allow_html=True)
    
    phone_list = profile_data.get("phones", [])
    with contact_cols[1]:
        st.markdown('<div class="profile-detail">', unsafe_allow_html=True)
        st.write("**📱 Phones**")
        if phone_list:
            for phone in phone_list:
                st.write(f"• {phone}")
        else:
            st.write("_No phones found_")
        st.markdown('</div>', unsafe_allow_html=True)
    
    url_list = profile_data.get("urls", [])
    with contact_cols[2]:
        st.markdown('<div class="profile-detail">', unsafe_allow_html=True)
        st.write("**🔗 URLs**")
        if url_list:
            for url in url_list:
                st.write(f"• {url}")
        else:
            st.write("_No URLs found_")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Skills
    skills = profile_data.get("skills", [])
    if skills:
        st.markdown("#### 🎯 Skills")
        skills_list = [s.get("title", "") if isinstance(s, dict) else str(s) for s in skills]
        skills_text = ", ".join(skills_list)
        with st.container():
            st.markdown(f"<div class='profile-detail'>{skills_text}</div>", unsafe_allow_html=True)
    
    # Experience
    experience = profile_data.get("experience", [])
    if experience:
        st.markdown("#### 💼 Work Experience")
        for idx, exp in enumerate(experience, 1):
            with st.expander(f"📍 {exp.get('title', 'Position')} at {exp.get('company', 'Company')} ({exp.get('start_date', 'N/A')} - {exp.get('end_date', 'Present')})", expanded=(idx == 1)):
                col_exp1, col_exp2 = st.columns(2)
                with col_exp1:
                    st.write(f"**Company:** {exp.get('company', 'N/A')}")
                    st.write(f"**Title:** {exp.get('title', 'N/A')}")
                with col_exp2:
                    st.write(f"**Start:** {exp.get('start_date', 'N/A')}")
                    st.write(f"**End:** {exp.get('end_date', 'N/A')}")
                if exp.get("summary"):
                    st.write(f"**Summary:** {exp.get('summary')}")
    
    # Education
    education = profile_data.get("education", [])
    if education:
        st.markdown("#### 🎓 Education")
        for idx, edu in enumerate(education, 1):
            with st.expander(f"🏫 {edu.get('degree', 'Degree')} in {edu.get('field', 'Field')}", expanded=(idx == 1)):
                col_edu1, col_edu2 = st.columns(2)
                with col_edu1:
                    st.write(f"**Institution:** {edu.get('institution', 'N/A')}")
                    st.write(f"**Field:** {edu.get('field', 'N/A')}")
                with col_edu2:
                    st.write(f"**Degree:** {edu.get('degree', 'N/A')}")
                    st.write(f"**Period:** {edu.get('start_date', 'N/A')} - {edu.get('end_date', 'N/A')}")
    
    # Companies
    companies = profile_data.get("companies", [])
    if companies:
        st.markdown("#### 🏢 Companies")
        st.write(", ".join(companies))
    
    # Certifications/Licenses
    certifications = profile_data.get("certifications", [])
    if certifications:
        st.markdown("#### 📜 Certifications")
        for cert in certifications:
            st.write(f"✓ {cert}")
    
    # Languages
    languages = profile_data.get("languages", [])
    if languages:
        st.markdown("#### 🌐 Languages")
        st.write(", ".join(languages))
    
    # Projected Output
    st.markdown("### 📤 Structured Output (Projection)")
    projected_data = projection_result.model_dump(mode="json")
    with st.expander("View JSON", expanded=False):
        st.json(projected_data)
    
    # Provenance
    st.markdown("### 🔍 Data Confidence & Sources")
    provenance_rows = []
    for field_name, decision in result.profile.provenance.items():
        selected_value = decision.selected
        if isinstance(selected_value, (list, dict)):
            selected_value = json.dumps(selected_value, ensure_ascii=False)
        selected_text = str(selected_value) if selected_value not in (None, []) else "(empty)"
        if len(selected_text) > 80:
            selected_text = selected_text[:77] + "..."

        confidence_text = f"{decision.score:.2f}" if decision.score is not None else "N/A"
        evidence_count = len(decision.evidence) if decision.evidence else 0
        provenance_rows.append({
            "Field": field_name,
            "Selected": selected_text,
            "Confidence": confidence_text,
            "Evidence count": evidence_count,
        })

    if provenance_rows:
        st.table(provenance_rows)
    else:
        st.info("No provenance data available")


if __name__ == "__main__":
    main()
