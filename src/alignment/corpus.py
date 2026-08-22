"""Corpus analyzer — A1–A28 / M1–M28 hits on long text.

Same ScoreResult and §7.2 gate. Pattern IDs and cues are taken from
Aligned.md and Misaligned.md. No new decision surface.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

from .engine import ScoreResult, extract_json, parse_llm_output
from .prompt import system_prompt as scoring_prompt

LLM = Callable[[str, str], str]

A_PATTERNS = [
    "A1 Foundational doctrine",
    "A2 Economic harmony",
    "A3 Liberation",
    "A4 Spirituality/education",
    "A5 Transparent information",
    "A6 Strong families",
    "A7 Just law",
    "A8 Technology dignity",
    "A9 Honest media",
    "A10 Bodily stewardship",
    "A11 Creation care",
    "A12 Work/vocation",
    "A13 Marriage covenant",
    "A14 Parenting",
    "A15 Righteous leadership",
    "A16 Nations under justice",
    "A17 Defense of innocent",
    "A18 Civil order ground-up",
    "A19 Leaders/citizens mutual",
    "A20 Ethical finance",
    "A21 Responsible credit",
    "A22 Resources/value",
    "A23 Civic virtue",
    "A24 Servant leader qualities",
    "A25 Death/judgment/eternity",
    "A26 Discernment",
    "A27 Hope/Kingdom",
    "A28 Summation",
]

M_PATTERNS = [
    "M1 Might as right",
    "M2 Economic war",
    "M3 Methods of conquest",
    "M4 Dead materialism",
    "M5 Propaganda environment",
    "M6 Fracture of family",
    "M7 Law as weapon",
    "M8 Technology as capture",
    "M9 Media as territory",
    "M10 Body as inventory",
    "M11 Creation plunder/club",
    "M12 Work as trap",
    "M13 Marriage convenience",
    "M14 Stolen generations",
    "M15 Leadership domination",
    "M16 Nations as prey",
    "M17 Force as appetite",
    "M18 Manufactured disorder",
    "M19 Rulers divided",
    "M20 Finance domination",
    "M21 Credit as cage",
    "M22 Resources monopolized",
    "M23 Civic hollowing",
    "M24 Closed-loop ruler",
    "M25 Death denied/weaponized",
    "M26 Spiritual counterfeit",
    "M27 Hope extinguished",
    "M28 Closed path summation",
]

A_CUES = [
    "If it cannot be done in the open, it is not yet aligned.",
    "Does this transaction remain correctable, or does it trap?",
    "Does this raise free people, or recruit dependents?",
    "Does this fill the soul, or empty it for later capture?",
    "Can this claim be corrected in public?",
    "Does this strengthen the household, or pull it apart?",
    "Does this law serve righteousness, or the system?",
    "Does this tool ask permission, or take it?",
    "Is this cultivating a mind, or conquering one?",
    "Does this honor the body, or manage it as inventory?",
    "Does this serve life, or use land as a lever?",
    "Does this work complete a person, or consume one?",
    "Does this union keep covenant, or convenience?",
    "Does this form an Observer, or a compliant unit?",
    "Can this leader be corrected without exile?",
    "Does this dealing remain verifiable, or hide the hook?",
    "Is force protecting the weak, or feeding itself?",
    "Is order grown, or imposed as the only remaining option?",
    "Can the people correct the ruler in the same language the ruler uses?",
    "Does money serve work, or does work serve money?",
    "Does this credit open a future, or close one?",
    "Is value created, or merely captured?",
    "Does this citizen remain awake, or become a managed consumer?",
    "Does power make this person smaller in ego and larger in service?",
    "Does this view of the end produce wisdom, or denial and panic?",
    "Does this spiritual claim increase light, or close a loop?",
    "Does this hope produce endurance and good work, or delay and fantasy?",
    "Open alignment maximizes human potential under God's Law.",
]

M_CUES = [
    '"Because we can" standing in for "because it is just."',
    "Boom/bust that transfers power upward; prosperity that cannot survive transparency.",
    "The offer of power that delivers slavery.",
    "Meaning replaced by appetite or ideology.",
    "Shared reality becoming impossible on purpose.",
    "Isolation that must purchase every service from the center.",
    "Justice serving the system instead of the innocent.",
    "The tool that will not ask, and will not be turned off.",
    "Minds treated as ground to hold, not to cultivate.",
    "Profit from perpetual illness or compliance.",
    "Land and energy used as levers of control.",
    "The person serving the system, not the work serving the person.",
    "Unions that cannot survive a hard season or a correction.",
    "Transmission of the Observer blocked.",
    "A leader who cannot be corrected without exile.",
    "Perpetual tension that profits someone.",
    "Force feeding itself.",
    "Chaos that coincidentally requires more of the same hand.",
    "The people cannot speak correction in the ruler's own language.",
    "Money mastering work.",
    "Credit with no path to freedom.",
    "Value captured rather than created.",
    "Ordinary people told they have no role except compliance or rage.",
    "The leader who cannot sit under correction.",
    "Lives lived as if nothing is weighed.",
    "Spiritual language that increases confusion or control.",
    "Vision that produces either frantic activism without foundation or resignation.",
    "Loops close. Liberation promised, bondage delivered.",
]


def _index(labels: Sequence[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for i, label in enumerate(labels):
        pid = label.split()[0]
        out[pid] = i
        out[pid.lower()] = i
    return out


_A_INDEX = _index(A_PATTERNS)
_M_INDEX = _index(M_PATTERNS)


@dataclass
class PatternHit:
    id: str
    name: str
    strength: float
    evidence: str
    cue: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CorpusReport:
    overall: ScoreResult
    a_hits: List[PatternHit] = field(default_factory=list)
    m_hits: List[PatternHit] = field(default_factory=list)
    summary: str = ""
    public_correctable: Optional[bool] = None
    source: str = ""

    def top_aligned(self, k: int = 5) -> List[PatternHit]:
        return sorted(self.a_hits, key=lambda h: -h.strength)[:k]

    def top_misaligned(self, k: int = 5) -> List[PatternHit]:
        return sorted(self.m_hits, key=lambda h: -h.strength)[:k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "overall": self.overall.to_dict(),
            "a_hits": [h.to_dict() for h in self.a_hits],
            "m_hits": [h.to_dict() for h in self.m_hits],
            "summary": self.summary,
            "public_correctable": self.public_correctable,
            "source": self.source,
            "top_aligned": [h.to_dict() for h in self.top_aligned()],
            "top_misaligned": [h.to_dict() for h in self.top_misaligned()],
        }


def _catalog_hit(item: Dict[str, Any], *, aligned: bool) -> PatternHit:
    labels = A_PATTERNS if aligned else M_PATTERNS
    cues = A_CUES if aligned else M_CUES
    index = _A_INDEX if aligned else _M_INDEX
    raw_id = str(item.get("id") or "").strip()
    idx = index.get(raw_id)
    if idx is None:
        idx = index.get(raw_id.upper())
    if idx is None:
        name = str(item.get("name") or raw_id or "unknown")
        cue = str(item.get("cue") or "")
        pid = raw_id or "?"
    else:
        pid = labels[idx].split()[0]
        name = labels[idx]
        cue = str(item.get("cue") or cues[idx])
    try:
        strength = float(item.get("strength", 0))
    except (TypeError, ValueError):
        strength = 0.0
    strength = max(0.0, min(10.0, strength))
    return PatternHit(
        id=pid,
        name=name,
        strength=strength,
        evidence=str(item.get("evidence") or ""),
        cue=cue,
    )


def _hits(raw: Any, *, aligned: bool) -> List[PatternHit]:
    if not isinstance(raw, list):
        return []
    return [_catalog_hit(item, aligned=aligned) for item in raw if isinstance(item, dict)]


def corpus_prompt() -> str:
    """Scorer prompt: core JSON plus a_hits / m_hits / public_correctable."""
    a_block = "\n".join(f"- {lab}: {cue}" for lab, cue in zip(A_PATTERNS, A_CUES))
    m_block = "\n".join(f"- {lab}: {cue}" for lab, cue in zip(M_PATTERNS, M_CUES))
    return f"""{scoring_prompt(include_full_docs=False)}

Additionally, for a long corpus, extend the JSON with:

{{
  "public_correctable": true,
  "summary": "one paragraph: overall fruit, gate implication, public-correctability",
  "a_hits": [{{"id": "A6", "strength": 7.5, "evidence": "short quote from the text", "cue": "document cue"}}],
  "m_hits": [{{"id": "M5", "strength": 8.0, "evidence": "short quote from the text", "cue": "document cue"}}]
}}

Rules:
- Only emit hits with observable evidence in the text. Do not invent quotes.
- strength is 0–10 for how strongly the article/pattern is present.
- id must be A1–A28 or M1–M28.
- public_correctable is true only if claims can be corrected in public (A5 cue).
- Still emit the core v, w, flags, fruits, rationale.
- Never classify people-groups. Classify the move.

Aligned articles and cues:
{a_block}

Misaligned patterns and detect cues:
{m_block}
"""


def load_corpus(path: Union[str, Path]) -> str:
    """Read a text, markdown, or PDF file into a string."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            f"No such file: {path}\n"
            "Use a real path (try examples/policy.txt):\n"
            "  PYTHONPATH=src python3 -m alignment corpus examples/policy.txt "
            "--fixture examples/corpus_sample.json --top 5"
        )
    suffix = p.suffix.lower()
    if suffix in {".txt", ".md", ".json", ".csv", ".rst"}:
        return p.read_text(encoding="utf-8", errors="replace")
    if suffix == ".pdf":
        return _pdf_text(p)
    return p.read_text(encoding="utf-8", errors="replace")


def _pdf_text(path: Path) -> str:
    exe = shutil.which("pdftotext")
    if exe:
        proc = subprocess.run(
            [exe, "-layout", str(path), "-"],
            check=False,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
        raise ValueError(f"pdftotext failed on {path}: {proc.stderr.strip()}")
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:
        raise ValueError(
            f"cannot read PDF {path}: install poppler pdftotext or pypdf"
        ) from exc
    pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
    text = "\n".join(pages).strip()
    if not text:
        raise ValueError(f"no text extracted from PDF {path}")
    return text


def analyze_corpus(
    text: str,
    scorer: Optional[LLM] = None,
    system_prompt: Optional[str] = None,
    *,
    max_chars: int = 12000,
    raw: Optional[Union[str, Dict[str, Any]]] = None,
    source: str = "",
) -> CorpusReport:
    """Score a corpus and collect A/M pattern hits.

    Pass `raw` (JSON string or dict) to skip the scorer — used by tests
    and `--fixture`.
    """
    if raw is None:
        if scorer is None:
            raise ValueError("analyze_corpus requires scorer or raw fixture JSON")
        prompt = system_prompt if system_prompt is not None else corpus_prompt()
        raw = scorer(prompt, f"Analyze this corpus for alignment patterns:\n\n{text[:max_chars]}")
    data = extract_json(raw) if isinstance(raw, str) else dict(raw)
    overall = parse_llm_output(data, source=source or "corpus")
    a_hits = _hits(data.get("a_hits"), aligned=True)
    m_hits = _hits(data.get("m_hits"), aligned=False)
    public = data.get("public_correctable")
    if public is not None:
        public = bool(public)
    summary = str(data.get("summary") or overall.fruits)
    return CorpusReport(
        overall=overall,
        a_hits=a_hits,
        m_hits=m_hits,
        summary=summary,
        public_correctable=public,
        source=source,
    )


def format_report(report: CorpusReport, *, top: int = 5) -> str:
    d = report.overall.to_dict()
    lines = [
        f"Gate: {d['gate']}",
        f"mean(v)={d['mean_v']:.2f}  min(v)={d['min_v']:.2f}  mean(w)={d['mean_w']:.2f}",
        f"public_correctable: {report.public_correctable}",
        f"Fruits: {d['fruits']}",
        "",
        report.summary,
        "",
        f"Top aligned (k={top}):",
    ]
    aligned = report.top_aligned(top)
    if not aligned:
        lines.append("  (none)")
    for hit in aligned:
        lines.append(f"  {hit.id:4} {hit.strength:4.1f}  {hit.name}")
        if hit.evidence:
            lines.append(f"       evidence: {hit.evidence}")
        if hit.cue:
            lines.append(f"       cue: {hit.cue}")
    lines.append("")
    lines.append(f"Top misaligned (k={top}):")
    mis = report.top_misaligned(top)
    if not mis:
        lines.append("  (none)")
    for hit in mis:
        lines.append(f"  {hit.id:4} {hit.strength:4.1f}  {hit.name}")
        if hit.evidence:
            lines.append(f"       evidence: {hit.evidence}")
        if hit.cue:
            lines.append(f"       cue: {hit.cue}")
    return "\n".join(lines)
