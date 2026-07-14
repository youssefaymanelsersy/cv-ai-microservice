import os
import re
import logging
import base64
from typing import List, Optional, Literal

import requests
import fitz  # PyMuPDF
from fastapi import (
    FastAPI, APIRouter, HTTPException, UploadFile, File, Form, Header, Depends
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from groq import Groq
import groq as groq_sdk
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

# =========================================================================
# Config
# =========================================================================
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("cv-ai-microservice")

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# --- Model IDs -----------------------------------------------------------
# llama-3.3-70b-versatile is deprecated by Groq, shutdown date 08/16/26.
# llama-3.2-90b-vision-preview was already shut down 04/14/25 — its
# replacement (llama-4-scout) was *also* deprecated 07/17/26. Re-check
# https://console.groq.com/docs/deprecations before your next migration.
GROQ_TEXT_MODEL = os.environ.get("GROQ_TEXT_MODEL", "openai/gpt-oss-120b")
GROQ_VISION_MODEL = os.environ.get("GROQ_VISION_MODEL", "qwen/qwen3.6-27b")

# Fallback model when Groq is rate-limited. Flash is the free-tier workhorse —
# cheap/fast and not the heavily-capped Pro tier.
GEMINI_FALLBACK_MODEL = "gemini-2.5-flash"

# --- Security / limits ----------------------------------------------------
# Optional shared-secret auth. Off by default so this doesn't break an
# existing frontend integration — set API_SHARED_SECRET in the environment
# to require an `x-api-key` header on every request below /health.
API_SHARED_SECRET = os.environ.get("API_SHARED_SECRET")

MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 8 * 1024 * 1024))       # 8MB CV files
MAX_IMAGE_BYTES = int(os.environ.get("MAX_IMAGE_BYTES", 15 * 1024 * 1024))        # 15MB JD screenshots
MAX_JD_TEXT_CHARS = int(os.environ.get("MAX_JD_TEXT_CHARS", 8000))
HTTP_DOWNLOAD_TIMEOUT_SECONDS = 15

ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "*").split(",")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    """No-op unless API_SHARED_SECRET is set, so existing deployments keep
    working until the frontend is updated to send the header."""
    if API_SHARED_SECRET and x_api_key != API_SHARED_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


router = APIRouter(dependencies=[Depends(require_api_key)])


@app.get("/health")
def health():
    return {"status": "ok"}


# =========================================================================
# Helpers — LLM calls (with Groq -> Gemini fallback on rate limits)
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
            model=GROQ_TEXT_MODEL,
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


def _vision_extract_text(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    """Extract text from an image (e.g. a JD screenshot) via Groq's vision
    model, with the same Gemini failover the text path gets."""
    prompt = "Extract all the text from this job description image exactly as written. Do not summarize or add commentary."
    base64_image = base64.b64encode(image_bytes).decode("utf-8")
    try:
        vision_completion = client.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{base64_image}"}},
                    ],
                }
            ],
            model=GROQ_VISION_MODEL,
            temperature=0.0,
            max_tokens=1200,
        )
        return vision_completion.choices[0].message.content
    except Exception as exc:
        if not (GEMINI_API_KEY and _is_rate_limit_error(exc)):
            raise
        gemini_model = genai.GenerativeModel(GEMINI_FALLBACK_MODEL)
        response = gemini_model.generate_content(
            [{"mime_type": mime_type, "data": image_bytes}, prompt]
        )
        return response.text


# =========================================================================
# Helpers — text/list truncation safety nets
# =========================================================================
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
# Helpers — file handling / validation
# =========================================================================
async def _read_upload_capped(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Reads an UploadFile but refuses anything over max_bytes, instead of
    buffering an unbounded amount of attacker-controlled data into memory."""
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the maximum allowed size of {max_bytes // (1024 * 1024)}MB.",
        )
    return data


def _validate_pdf_magic(file_bytes: bytes) -> None:
    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded file does not appear to be a valid PDF.")


def download_and_extract_text(url: str) -> str:
    """Streams the download with a size cap and a timeout, instead of
    trusting a remote URL to be small and responsive."""
    with requests.get(url, stream=True, timeout=HTTP_DOWNLOAD_TIMEOUT_SECONDS) as response:
        response.raise_for_status()
        content = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            content.extend(chunk)
            if len(content) > MAX_UPLOAD_BYTES:
                raise ValueError(f"File exceeds the maximum allowed size of {MAX_UPLOAD_BYTES // (1024 * 1024)}MB.")
        file_bytes = bytes(content)
    _validate_pdf_magic(file_bytes)
    return extract_text_from_bytes(file_bytes)


def extract_text_from_bytes(file_bytes: bytes) -> str:
    pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
    return "".join(page.get_text() for page in pdf_document).strip()


def _open_pdf(file_bytes: bytes):
    _validate_pdf_magic(file_bytes)
    return fitz.open(stream=file_bytes, filetype="pdf")


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


CV_PARSE_SYSTEM_PROMPT_TEMPLATE = """
You are an expert ATS CV parser. Extract information from the CV text into this exact JSON structure:
{schema}

RULES:
1. Use null for missing/unknown values. Never use empty strings.
2. All strings must be at least 1 character, or null.
3. `links` object must always be present, even with all-null fields.
4. Copy values as they appear in the CV — do not paraphrase, summarize, embellish, or add commentary.
5. For `summary`, extract the CV's existing summary/objective verbatim if present; otherwise null. Do not generate a new one.
6. `description` fields (experience/projects): condense to 1-2 short sentences max, using the CV's own wording. Do not invent details not in the source text.
"""


@router.post("/parse")
def parse_cv(request: ParseRequest):
    try:
        cv_text = download_and_extract_text(request.url)
        if not cv_text:
            raise ValueError("No text extracted from PDF.")

        system_prompt = CV_PARSE_SYSTEM_PROMPT_TEMPLATE.format(schema=ParsedCVData.model_json_schema())

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
        logger.exception("parse_cv failed for cvId=%s", request.cvId)
        # Generic message to the client — the real exception is in the logs, not the response.
        return {"cvId": request.cvId, "status": "failed", "errorMessage": "Failed to parse CV. Please check the file and try again."}


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


_CONTACT_REGEX = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+|(?:\+?\d[\d\-\s()]{7,}\d)")


def _compute_pdf_structural_facts(pdf_document) -> dict:
    """Computes ATS-relevant *structural* facts directly from the PDF via
    PyMuPDF, instead of asking an LLM to guess layout properties from plain
    extracted text (which it cannot reliably determine). Multi-column
    detection and header/footer contact detection are best-effort
    heuristics — good enough to ground the LLM, not perfect geometry."""
    page_count = len(pdf_document)
    fonts = set()
    total_words = 0
    total_extractable_chars = 0
    total_images = 0
    multi_column_hits = 0
    table_hits = 0
    header_footer_contact = False

    for page in pdf_document:
        page_text = page.get_text()
        total_words += len(page_text.split())
        total_extractable_chars += len(page_text.strip())
        total_images += len(page.get_images(full=True))

        try:
            for font in page.get_fonts(full=True):
                fonts.add(font[3])
        except Exception:
            pass

        # Table detection — available in PyMuPDF 1.23+; degrade gracefully on older versions.
        try:
            found_tables = page.find_tables()
            if found_tables and len(found_tables.tables) > 0:
                table_hits += 1
        except Exception:
            pass

        # Multi-column heuristic: look for text blocks that share a vertical
        # band but sit in clearly separate horizontal regions.
        page_width = page.rect.width
        blocks = page.get_text("blocks")
        bands = {}
        for b in blocks:
            x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
            band = round(y0 / 40)
            bands.setdefault(band, []).append((x0, x1))
        for spans in bands.values():
            if len(spans) < 2:
                continue
            spans_sorted = sorted(spans)
            for i in range(len(spans_sorted) - 1):
                gap = spans_sorted[i + 1][0] - spans_sorted[i][1]
                if gap > page_width * 0.08 and spans_sorted[i][1] < page_width * 0.6:
                    multi_column_hits += 1
                    break

        # Contact info in header/footer band (top/bottom ~12% of the page).
        page_height = page.rect.height
        try:
            words = page.get_text("words")
            for w in words:
                x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
                if (y0 < page_height * 0.12 or y1 > page_height * 0.88) and _CONTACT_REGEX.search(word):
                    header_footer_contact = True
        except Exception:
            pass

    is_scanned = total_extractable_chars < (50 * max(page_count, 1)) and total_images > 0

    return {
        "has_multi_column_layout": multi_column_hits > 0,
        "has_tables": table_hits > 0,
        "is_scanned_pdf": is_scanned,
        "contact_info_in_header_footer": header_footer_contact,
        "font_count": len(fonts) if fonts else 1,
        "page_count": page_count,
        "word_count": total_words,
        "detected_image_count": total_images,
    }


@router.post("/evaluate")
async def evaluate_cv(file: UploadFile = File(...)):
    try:
        file_bytes = await _read_upload_capped(file)
        pdf_document = _open_pdf(file_bytes)
        cv_text = "".join(page.get_text() for page in pdf_document).strip()
        structural_facts = _compute_pdf_structural_facts(pdf_document)

        system_prompt = f"""
        You are an expert ATS (Applicant Tracking System). Evaluate the CV text strictly based on ATS parseability, formatting, and content quality.
        Return the exact JSON structure defined by this schema:
        {AtsScoreOutput.model_json_schema()}

        STRUCTURAL FACTS (already computed from the PDF — copy these values into
        `ats_report` exactly as given, do not re-derive or contradict them):
        {structural_facts}

        `has_text_in_images` is the one structural field NOT provided above — judge
        this yourself using `detected_image_count` as a hint (a resume with images
        but very little extractable text often has text embedded in an image).

        LENGTH CONSTRAINTS (enforce strictly):
        - `evidence` fields: exactly one sentence, 15-20 words, stating only the specific fact that justifies the score. No preamble like "The CV shows...".
        - `deductions.reasons`: comma-separated short phrases, not full sentences.
        - `key_strengths` / `areas_for_improvement`: max 4 items each, max 8 words per item, no punctuation-heavy explanations.

        Deduct points for missing sections, poor formatting, or lack of keywords. Populate all fields, including `missing_sections` (a semantic judgment about CV content, not a structural fact).
        """

        response_text = _chat_completion_json(
            system_prompt=system_prompt,
            user_content=f"Evaluate this CV:\n\n{cv_text}",
            max_tokens=900,
        )

        result = AtsScoreOutput.model_validate_json(response_text)

        # Belt-and-suspenders: force the structural fields to the computed
        # values regardless of what the model echoed back.
        for key in ("has_multi_column_layout", "has_tables", "is_scanned_pdf",
                    "contact_info_in_header_footer", "font_count", "page_count", "word_count"):
            setattr(result.ats_report, key, structural_facts[key])

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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("evaluate_cv failed")
        raise HTTPException(status_code=500, detail="Failed to evaluate CV. Please try again.")


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


@router.post("/match/upload")
async def match_cv(
    cv_file: UploadFile = File(...),
    job_description: Optional[str] = Form(None),
    job_description_image: Optional[UploadFile] = File(None)
):
    try:
        cv_bytes = await _read_upload_capped(cv_file)
        _validate_pdf_magic(cv_bytes)
        cv_text = extract_text_from_bytes(cv_bytes)

        jd_text = job_description or ""

        if job_description_image:
            image_bytes = await _read_upload_capped(job_description_image, max_bytes=MAX_IMAGE_BYTES)
            jd_text = _vision_extract_text(image_bytes)

        jd_text = _truncate(jd_text, MAX_JD_TEXT_CHARS) or ""

        if not jd_text:
            raise HTTPException(status_code=422, detail="No job description text could be extracted.")

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
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("match_cv failed")
        raise HTTPException(status_code=500, detail="Failed to match CV against job description. Please try again.")


# =========================================================================
# 4. CV Optimization / Regeneration Models & Endpoint
# =========================================================================
DesignPreference = Literal["classic", "modern", "minimalist"]

# Content-density guidance per template — this is the only thing
# `design_preference` affects on the backend. Visual rendering of the
# chosen template lives entirely in the frontend (Option A).
_DESIGN_DENSITY_HINTS = {
    "classic": "Target a dense, traditional one-page layout: concise bullets (max ~18 words each).",
    "modern": "Target a balanced one-to-two-page layout: bullets can be slightly longer where impact-focused.",
    "minimalist": "Target a lean one-page layout: only the strongest 3-4 bullets per role, a very concise summary.",
}


class OptimizationMeta(BaseModel):
    design_preference: DesignPreference
    changes_summary: List[str] = []
    unaddressed_gaps: List[str] = []
    warnings: List[str] = []


class CvOptimizationResponse(BaseModel):
    status: str = "completed"
    data: ParsedCVData
    meta: OptimizationMeta


class _OptimizationLLMOutput(BaseModel):
    """Internal shape the LLM is asked to fill in. Kept separate from the
    public response model so we can post-process `data` before returning it."""
    data: ParsedCVData
    changes_summary: List[str] = []
    unaddressed_gaps: List[str] = []


def _entity_names(cv: ParsedCVData) -> set:
    names = {(e.company or "").strip().lower() for e in cv.experience if e.company}
    names |= {(ed.institution or "").strip().lower() for ed in cv.education if ed.institution}
    return {n for n in names if n}


@router.post("/optimize", response_model=CvOptimizationResponse)
async def optimize_cv(
    cv_file: Optional[UploadFile] = File(None),
    cv_data: Optional[str] = Form(None),
    job_description: Optional[str] = Form(None),
    job_description_image: Optional[UploadFile] = File(None),
    design_preference: DesignPreference = Form("classic"),
):
    """
    Regenerates/optimizes a full CV. Accepts EITHER:
      - `cv_data`: a JSON string matching ParsedCVData (preferred — reuse the
        output of /parse, including any edits the user made since), OR
      - `cv_file`: a raw CV PDF, which will be parsed first.
    An optional job description (text or image) steers ATS/JD alignment.
    Returns the optimized CV in the same ParsedCVData shape as /parse, so
    the frontend can reuse its existing CV renderer/templates (Option A —
    this endpoint never generates HTML or a PDF itself).
    """
    try:
        if not cv_data and not cv_file:
            raise HTTPException(status_code=400, detail="Provide either cv_data (parsed CV JSON) or cv_file (a CV PDF).")

        if cv_data:
            try:
                source_cv = ParsedCVData.model_validate_json(cv_data)
            except Exception:
                raise HTTPException(status_code=400, detail="cv_data must be valid JSON matching the ParsedCVData schema.")
        else:
            file_bytes = await _read_upload_capped(cv_file)
            _validate_pdf_magic(file_bytes)
            cv_text = extract_text_from_bytes(file_bytes)
            if not cv_text:
                raise HTTPException(status_code=422, detail="No text could be extracted from the uploaded CV.")
            parse_system_prompt = CV_PARSE_SYSTEM_PROMPT_TEMPLATE.format(schema=ParsedCVData.model_json_schema())
            response_text = _chat_completion_json(
                system_prompt=parse_system_prompt,
                user_content=f"Extract details from this CV:\n\n{cv_text}",
                max_tokens=1500,
            )
            source_cv = ParsedCVData.model_validate_json(response_text)

        jd_text = job_description or ""
        if job_description_image:
            image_bytes = await _read_upload_capped(job_description_image, max_bytes=MAX_IMAGE_BYTES)
            jd_text = _vision_extract_text(image_bytes)
        jd_text = _truncate(jd_text, MAX_JD_TEXT_CHARS) or ""

        density_hint = _DESIGN_DENSITY_HINTS[design_preference]

        system_prompt = f"""
        You are an expert CV writer optimizing an already-parsed CV for ATS parseability, clarity, and impact, and — if a job description is given — alignment with it.

        Return the exact JSON structure defined by this schema:
        {_OptimizationLLMOutput.model_json_schema()}

        CRITICAL RULES — DO NOT VIOLATE:
        1. NEVER invent or alter factual details: company names, job titles, employment dates, degree names, institutions, or technologies not present in the source CV. You may rephrase, reorder, quantify vague statements using numbers already implied by the source, and emphasize existing JD-relevant experience — but you must not fabricate new facts.
        2. If the job description calls for a skill genuinely absent from the source CV, do NOT add it to `skills` or anywhere else. Instead list it in `unaddressed_gaps`.
        3. Every project's `technologies` list must only contain items already associated with that project in the source data.
        4. Preserve every section that had content in the source — do not silently drop experience, education, project, or certification entries.
        5. {density_hint}
        6. Rewrite `summary` to be punchy and quantify achievements where the source data supports it; naturally weave in JD keywords only where truthfully applicable.
        7. Reorder bullets within each experience/project entry so the most JD-relevant points lead, without inventing new ones.
        8. `changes_summary`: max 5 short bullet points describing what changed at a high level (e.g. "Reworded summary to emphasize backend experience"). No fluff.
        9. `unaddressed_gaps`: skill/requirement names from the JD that the CV genuinely does not support. Max 6 items, names only. Empty list if no JD was given or no gaps found.
        """

        user_content = f"SOURCE CV (JSON):\n{source_cv.model_dump_json()}\n\nJOB DESCRIPTION (may be empty):\n{jd_text}"

        response_text = _chat_completion_json(
            system_prompt=system_prompt,
            user_content=user_content,
            max_tokens=3000,
        )
        llm_output = _OptimizationLLMOutput.model_validate_json(response_text)
        optimized = llm_output.data

        # Safety net: enforce concise summary/descriptions regardless of model behavior
        optimized.summary = _truncate(optimized.summary, 500)
        for exp in optimized.experience:
            exp.description = _truncate(exp.description, 400)
        for proj in optimized.projects:
            proj.description = _truncate(proj.description, 400)

        changes_summary = _truncate_list(llm_output.changes_summary, 5, 120)
        unaddressed_gaps = _truncate_list(llm_output.unaddressed_gaps, 6, 40)

        # Best-effort hallucination guard: flag (don't silently accept) if
        # section counts or core entity names shifted more than a rewrite
        # should cause. This is a heuristic tripwire, not a guarantee.
        warnings: List[str] = []
        if len(optimized.experience) != len(source_cv.experience):
            warnings.append("Experience entry count changed — please review before using.")
        if len(optimized.education) != len(source_cv.education):
            warnings.append("Education entry count changed — please review before using.")
        if len(optimized.projects) != len(source_cv.projects):
            warnings.append("Project entry count changed — please review before using.")
        original_entities = _entity_names(source_cv)
        optimized_entities = _entity_names(optimized)
        if original_entities and not original_entities.issubset(optimized_entities):
            warnings.append("Some company/institution names differ from the source — please verify accuracy.")

        meta = OptimizationMeta(
            design_preference=design_preference,
            changes_summary=changes_summary,
            unaddressed_gaps=unaddressed_gaps,
            warnings=warnings,
        )

        return CvOptimizationResponse(data=optimized, meta=meta)

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("optimize_cv failed")
        raise HTTPException(status_code=500, detail="Failed to optimize CV. Please try again.")


app.include_router(router)