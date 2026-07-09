import os
import requests
import fitz  # PyMuPDF
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional
from groq import Groq
import groq as groq_sdk
import base64
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI()
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Fallback model when Groq is rate-limited. Flash is the free-tier workhorse —
# cheap/fast and not the heavily-capped Pro tier.
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"

# =========================================================================
# Helpers
# =========================================================================
def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect a Groq 429 (RPM/RPD/TPM/TPD exceeded) so we know when to fall
    back to Gemini rather than surfacing every error type as a failover."""
    if isinstance(exc, groq_sdk.RateLimitError):
        return True
    status = getattr(exc, "status_code", None)
    return status == 429


def _chat_completion_json(system_prompt: str, user_content: str, max_tokens: int) -> str:
    """Try Groq first; on a rate-limit error, fall back to Gemini so a single
    provider's quota doesn't take the endpoint down. Both paths are told to
    return raw JSON only, so the caller can validate against the same
    Pydantic schema regardless of which provider answered."""
    try:
        chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model="llama-3.3-70b-versatile",
            temperature=0.0,
            response_format={"type": "json_object"},
            max_tokens=max_tokens,
        )
        return chat_completion.choices[0].message.content
    except Exception as exc:
        if not (GEMINI_API_KEY and _is_rate_limit_error(exc)):
            raise

        gemini_model = genai.GenerativeModel(
            GEMINI_FALLBACK_MODEL,
            system_instruction=system_prompt,
        )
        response = gemini_model.generate_content(
            user_content,
            generation_config={
                "response_mime_type": "application/json",
                "max_output_tokens": max_tokens,
                "temperature": 0.0,
            },
        )
        return response.text


def _truncate(text: Optional[str], max_chars: int, min_ratio: float = 0.4) -> Optional[str]:
    """Safety net: trim text fields in case the model ignores length
    instructions in the prompt. Prefers cutting at a sentence boundary
    (no trailing "...", since the sentence is complete) and only falls
    back to a hard word-boundary cut with "..." if no sentence end is
    found reasonably close to the limit — this avoids lopping off
    mid-sentence, which reads as broken output."""
    if not text:
        return text
    text = text.strip()
    if len(text) <= max_chars:
        return text

    window = text[:max_chars]
    last_end = max(window.rfind(". "), window.rfind(".\n"), window.rfind("? "), window.rfind("! "))
    if last_end == -1 and window.endswith((".", "?", "!")):
        last_end = len(window) - 1
    if last_end >= max_chars * min_ratio:
        return window[: last_end + 1].strip()

    # No sentence boundary close enough — fall back to a word cut with ellipsis
    cut = window.rsplit(" ", 1)[0]
    return (cut or window).rstrip(",.;: ") + "..."


def _truncate_list(items: List[str], max_items: int, max_chars: int) -> List[str]:
    return [_truncate(i, max_chars) for i in items[:max_items]]


# =========================================================================
# 1. CV Parser Models & Endpoint
# =========================================================================
class ParseRequest(BaseModel):
    cvId: str
    url: str

class Skill(BaseModel):
    name: str = Field(..., min_length=1)
    level: Optional[str] = None

class SkillsContainer(BaseModel):
    technical: List[Skill] = []
    nonTechnical: List[Skill] = []

class Experience(BaseModel):
    company: Optional[str] = Field(None, min_length=1)
    title: Optional[str] = Field(None, min_length=1)
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    description: Optional[str] = None

class Project(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    technologies: List[str] = []
    url: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None

class Education(BaseModel):
    institution: Optional[str] = Field(None, min_length=1)
    degree: Optional[str] = None
    field: Optional[str] = None
    major: Optional[str] = None
    startDate: Optional[str] = None
    endDate: Optional[str] = None

class Certification(BaseModel):
    name: Optional[str] = None
    issuer: Optional[str] = None
    date: Optional[str] = None

class Links(BaseModel):
    github: Optional[str] = None
    linkedin: Optional[str] = None
    portfolio: Optional[str] = None

class ParsedCVData(BaseModel):
    fullName: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: SkillsContainer = Field(default_factory=SkillsContainer)
    experience: List[Experience] = []
    projects: List[Project] = []
    education: List[Education] = []
    certifications: List[Certification] = []
    languages: List[str] = []
    links: Links = Field(default_factory=Links)

class SuccessResponse(BaseModel):
    cvId: str
    status: str = "completed"
    parsedData: ParsedCVData

def download_and_extract_text(url: str) -> str:
    response = requests.get(url, stream=True)
    response.raise_for_status()
    pdf_document = fitz.open(stream=response.content, filetype="pdf")
    return "".join(page.get_text() for page in pdf_document).strip()

def extract_text_from_bytes(file_bytes: bytes) -> str:
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join(page.get_text() for page in pdf_document).strip()

@app.post("/parse")
def parse_cv(request: ParseRequest):
    try:
        cv_text = download_and_extract_text(request.url)
        if not cv_text:
            raise ValueError("No text extracted from PDF.")

        system_prompt = f"""
        You are an expert ATS CV parser. Extract information from the CV text into this exact JSON structure:
        {ParsedCVData.model_json_schema()}

        RULES:
        1. Use null for missing/unknown values. Never use empty strings.
        2. All strings must be at least 1 character, or null.
        3. `links` object must always be present, even with all-null fields.
        4. Copy values as they appear in the CV — do not paraphrase, summarize, embellish, or add commentary.
        5. For `summary`, extract the CV's existing summary/objective verbatim if present; otherwise null. Do not generate a new one.
        6. `description` fields (experience/projects): condense to 1-2 short sentences max, using the CV's own wording. Do not invent details not in the source text.
        """

        response_text = _chat_completion_json(
            system_prompt=system_prompt,
            user_content=f"Extract details from this CV:\n\n{cv_text}",
            max_tokens=1500,
        )

        parsed_data = ParsedCVData.model_validate_json(response_text)

        # Safety net: enforce concise descriptions/summary regardless of model behavior
        parsed_data.summary = _truncate(parsed_data.summary, 400)
        for exp in parsed_data.experience:
            exp.description = _truncate(exp.description, 300)
        for proj in parsed_data.projects:
            proj.description = _truncate(proj.description, 300)

        return SuccessResponse(cvId=request.cvId, parsedData=parsed_data).model_dump()
    except Exception as e:
        return {"cvId": request.cvId, "status": "failed", "errorMessage": str(e)}

# =========================================================================
# 2. CV ATS Evaluation Models & Endpoint
# =========================================================================
class ScoreDetail(BaseModel):
    score: int
    max: int = 100
    evidence: str

class Scores(BaseModel):
    parseability_formatting: ScoreDetail
    section_structure: ScoreDetail
    content_quality: ScoreDetail
    keyword_optimization: ScoreDetail

class Deductions(BaseModel):
    total: int
    reasons: str

class AtsReport(BaseModel):
    has_multi_column_layout: bool
    has_tables: bool
    has_text_in_images: bool
    is_scanned_pdf: bool
    missing_sections: List[str] = []
    contact_info_in_header_footer: bool
    font_count: int
    page_count: int
    word_count: int

class AtsScoreOutput(BaseModel):
    scores: Scores
    deductions: Deductions
    ats_report: AtsReport
    key_strengths: List[str] = []
    areas_for_improvement: List[str] = []
    total_score: int
    total_max: int = 100

@app.post("/evaluate")
async def evaluate_cv(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        cv_text = extract_text_from_bytes(file_bytes)

        system_prompt = f"""
        You are an expert ATS (Applicant Tracking System). Evaluate the CV text strictly based on ATS parseability, formatting, and content quality.
        Return the exact JSON structure defined by this schema:
        {AtsScoreOutput.model_json_schema()}

        LENGTH CONSTRAINTS (enforce strictly):
        - `evidence` fields: exactly one sentence, 15-20 words, stating only the specific fact that justifies the score. No preamble like "The CV shows...".
        - `deductions.reasons`: comma-separated short phrases, not full sentences.
        - `key_strengths` / `areas_for_improvement`: max 4 items each, max 8 words per item, no punctuation-heavy explanations.

        Deduct points for missing sections, poor formatting, or lack of keywords. Populate all fields.
        """

        response_text = _chat_completion_json(
            system_prompt=system_prompt,
            user_content=f"Evaluate this CV:\n\n{cv_text}",
            max_tokens=900,
        )

        result = AtsScoreOutput.model_validate_json(response_text)

        # Safety net: enforce concise evidence/reasons/lists regardless of model behavior
        for detail in (
            result.scores.parseability_formatting,
            result.scores.section_structure,
            result.scores.content_quality,
            result.scores.keyword_optimization,
        ):
            detail.evidence = _truncate(detail.evidence, 200)
        result.deductions.reasons = _truncate(result.deductions.reasons, 200)
        result.key_strengths = _truncate_list(result.key_strengths, 4, 60)
        result.areas_for_improvement = _truncate_list(result.areas_for_improvement, 4, 60)

        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =========================================================================
# 3. Skill Matching Engine Models & Endpoint
# =========================================================================
class ScoreDetailsMatch(BaseModel):
    hard_skills_score: int
    experience_score: int
    soft_skills_score: int
    logistics_score: int

class MatchResult(BaseModel):
    match_analysis: str
    score_details: ScoreDetailsMatch
    total_score: int
    explanation: str
    key_matched_skills: List[str] = []
    missing_skills: List[str] = []
    recommendation: str

class ScoreMatchOutput(BaseModel):
    match_result: MatchResult

@app.post("/match/upload")
async def match_cv(
    cv_file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    job_description_image: Optional[UploadFile] = File(None)
):
    try:
        cv_bytes = await cv_file.read()
        cv_text = extract_text_from_bytes(cv_bytes)

        jd_text = job_description or ""

        # Use Vision Model if JD Image is provided
        if job_description_image:
            image_bytes = await job_description_image.read()
            base64_image = base64.b64encode(image_bytes).decode('utf-8')

            vision_completion = client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all the text from this job description image exactly as written. Do not summarize or add commentary."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                            },
                        ],
                    }
                ],
                model="llama-3.2-90b-vision-preview",
                temperature=0.0,
                max_tokens=1200,
            )
            jd_text = vision_completion.choices[0].message.content

        if not jd_text:
            raise ValueError("No job description text could be extracted.")

        system_prompt = f"""
        You are an expert technical recruiter matching a candidate's CV against a Job Description.
        Compare the CV to the JD and return the exact JSON structure defined by this schema:
        {ScoreMatchOutput.model_json_schema()}

        LENGTH CONSTRAINTS (enforce strictly):
        - `match_analysis`: max 2 sentences, no repetition of information found elsewhere in the JSON.
        - `explanation`: max 2 sentences, must add new information not already in match_analysis.
        - `recommendation`: ONE short sentence, max 20 words, ending in a period. Format: verdict + brief reason (e.g. "Strong fit — proceed to interview." / "Missing key cloud skills; screen further before proceeding.").
        - `key_matched_skills` / `missing_skills`: skill names only, max 8 items each, no descriptions or justifications inline.

        Give honest, strict scores out of 100 for each category. Be thorough in identifying missing skills, but concise in stating them.
        """

        response_text = _chat_completion_json(
            system_prompt=system_prompt,
            user_content=f"CV TEXT:\n{cv_text}\n\nJOB DESCRIPTION:\n{jd_text}",
            max_tokens=700,
        )

        result = ScoreMatchOutput.model_validate_json(response_text)

        # Safety net: enforce concise analysis/skills lists regardless of model behavior
        mr = result.match_result
        mr.match_analysis = _truncate(mr.match_analysis, 400)
        mr.explanation = _truncate(mr.explanation, 400)
        mr.recommendation = _truncate(mr.recommendation, 220)
        mr.key_matched_skills = _truncate_list(mr.key_matched_skills, 8, 40)
        mr.missing_skills = _truncate_list(mr.missing_skills, 8, 40)

        return result.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))