#EVIDENTIA


An evidence-first, multi-source candidate data transformer.

This project ingests resumes, ATS exports, LinkedIn/GitHub profiles, and recruiter notes; extracts structured facts with provenance; normalizes values; resolves conflicts via weighted consensus; and emits a canonical profile plus a configurable projection schema.

Features
- Merge data from local files and public profile URLs (LinkedIn, GitHub).
- Preserve provenance for every extracted datum (source, confidence, raw text).
- Hybrid extraction: structured JSON/CSV parsing, regex, optional spaCy NER, and optional local Ollama semantic parsing.
- Normalization for emails, phones (E.164), dates, skills, and URLs.
- Consensus resolution with source weighting and evidence graphs (NetworkX).
- Runtime projection engine to map canonical fields to custom output schemas.
- CLI (`typer`) and a Streamlit UI for interactive use.

Requirements
- Python 3.11 or newer
- Recommended: create and use a virtual environment

Core dependencies (installed by default):
- `networkx`, `pydantic`, `PyYAML`, `streamlit`, `typer`, `phonenumbers`, `python-dateutil`

Optional (document & NLP):
- `pypdf` (PDF resume parsing)
- `python-docx` (DOCX parsing)
- `spacy` + model (NER)

Quick start (Windows)
1. Clone the repo and open a terminal in the project root.

```powershell
cd "e:\sm temp\SM projt temp\EightFold.ai"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

2. (Optional) install document and NLP support:

```powershell
python -m pip install -e .[all]
# or individually
python -m pip install pypdf python-docx spacy
python -m spacy download en_core_web_sm
```

Run the Streamlit UI

```powershell
python -m streamlit run src/candidate_transformer/ui.py --server.port 8503
# Then open http://localhost:8503 in your browser
```

Run the CLI

```powershell
candidate-transformer transform sample_data\candidate1.json --projection examples\projection.json
# or via python module if the script isn't on PATH
python -m candidate_transformer.cli transform sample_data\candidate1.json --projection examples\projection.json
```

Save output to file

```powershell
candidate-transformer transform sample_data\candidate1.json --projection examples\projection.json --output result.json
```

Notes and troubleshooting
- If you see import errors for `pypdf` or `python-docx`, install the optional `documents` extras or the packages individually.
- URL scraping uses `requests`; install it if missing: `python -m pip install requests`.
- Ollama support is local-only. If you want semantic extraction, run an Ollama-compatible server at `http://localhost:11434` and set `OLLAMA_HOST` in your environment (or update `AppSettings`).
- If `streamlit run` fails with port in use, pass `--server.port` with an available port.



