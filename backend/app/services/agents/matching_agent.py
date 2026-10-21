import logging
import re
from typing import List, Dict, Any, Tuple, Optional
from app.core.config import settings
from app.services.embeddings import embedding_service
from app.schemas.match_schema import (
    MatchBreakdown,
    JobMatchCandidateItem,
    CandidateMatchJobItem,
)

logger = logging.getLogger(__name__)

# Common Tech Synonyms dictionary for flexible skill matching
TECH_SYNONYMS: Dict[str, str] = {
    "py": "python",
    "python3": "python",
    "js": "javascript",
    "ts": "typescript",
    "k8s": "kubernetes",
    "postgres": "postgresql",
    "pg": "postgresql",
    "golang": "go",
    "reactjs": "react",
    "react.js": "react",
    "vuejs": "vue",
    "vue.js": "vue",
    "nodejs": "node.js",
    "node": "node.js",
    "aws": "amazon web services",
    "gcp": "google cloud platform",
    "azure": "microsoft azure",
    "docker": "containerization",
    "containers": "containerization",
    "ci/cd": "cicd",
    "ci-cd": "cicd",
    "continuous integration": "cicd",
    "fast api": "fastapi",
    "fastapi": "fastapi",
    "rest": "rest api",
    "restful": "rest api",
    "nosql": "nosql databases",
    "mongo": "mongodb",
}


def normalize_skill(skill: str) -> str:
    """Normalize skill string and map to canonical synonym."""
    cleaned = skill.lower().strip()
    cleaned = re.sub(r"[^\w\.\-\+]", "", cleaned)
    return TECH_SYNONYMS.get(cleaned, cleaned)


class MatchingAgent:
    """
    Subagent 3: Job Matching & Compatibility Agent
    Computes multi-dimensional hybrid scores:
      - 40% Vector similarity (pgvector embeddings)
      - 35% Skill coverage (exact, synonym, and BM25 token matching)
      - 15% Experience level & title alignment
      - 10% Hard constraints (visa sponsorship, location, remote policy)
    """

    def __init__(
        self,
        weight_vector: float = None,
        weight_skills: float = None,
        weight_experience: float = None,
        weight_constraints: float = None,
    ):
        self.w_vector = weight_vector if weight_vector is not None else settings.WEIGHT_VECTOR_SIMILARITY
        self.w_skills = weight_skills if weight_skills is not None else settings.WEIGHT_SKILL_COVERAGE
        self.w_experience = weight_experience if weight_experience is not None else settings.WEIGHT_EXPERIENCE_ALIGNMENT
        self.w_constraints = weight_constraints if weight_constraints is not None else settings.WEIGHT_HARD_CONSTRAINTS

    def calculate_skill_coverage(
        self,
        candidate_skills: List[str],
        job_required_skills: List[str],
        job_preferred_skills: List[str],
    ) -> Tuple[float, List[str], List[str], List[str]]:
        """
        Computes skill match score (0-100) using normalized exact and synonym matching.
        Must-have skills are weighted 75%, nice-to-have skills 25%.
        """
        cand_norm = {normalize_skill(s) for s in candidate_skills}

        req_norm = [normalize_skill(s) for s in job_required_skills]
        pref_norm = [normalize_skill(s) for s in job_preferred_skills]

        matching_skills = []
        missing_must_haves = []
        missing_nice_to_haves = []

        # Evaluate must-have skills
        matched_req = 0
        for orig, norm in zip(job_required_skills, req_norm):
            if norm in cand_norm or any(norm in c or c in norm for c in cand_norm):
                matched_req += 1
                matching_skills.append(orig)
            else:
                missing_must_haves.append(orig)

        # Evaluate nice-to-have skills
        matched_pref = 0
        for orig, norm in zip(job_preferred_skills, pref_norm):
            if norm in cand_norm or any(norm in c or c in norm for c in cand_norm):
                matched_pref += 1
                if orig not in matching_skills:
                    matching_skills.append(orig)
            else:
                missing_nice_to_haves.append(orig)

        # Compute weighted percentage
        req_score = (matched_req / len(req_norm) * 100.0) if req_norm else 100.0
        pref_score = (matched_pref / len(pref_norm) * 100.0) if pref_norm else 100.0

        if req_norm and pref_norm:
            final_skill_score = (req_score * 0.75) + (pref_score * 0.25)
        elif req_norm:
            final_skill_score = req_score
        elif pref_norm:
            final_skill_score = pref_score
        else:
            final_skill_score = 80.0

        return round(final_skill_score, 2), matching_skills, missing_must_haves, missing_nice_to_haves

    def calculate_experience_score(
        self,
        candidate_exp: float,
        job_min_exp: float,
        candidate_title: Optional[str] = None,
        job_title: Optional[str] = None,
    ) -> float:
        """
        Computes experience alignment score (0-100) combining years of experience and title parity.
        """
        # Years ratio
        if job_min_exp <= 0.0:
            years_score = 100.0
        elif candidate_exp >= job_min_exp:
            # Overqualification penalty is minimal (max 100)
            years_score = 100.0
        else:
            # Ratio of candidate experience to required
            years_score = (candidate_exp / job_min_exp) * 85.0

        # Title alignment bonus
        title_bonus = 0.0
        if candidate_title and job_title:
            c_words = set(candidate_title.lower().split())
            j_words = set(job_title.lower().split())
            overlap = c_words.intersection(j_words)
            if overlap:
                title_bonus = min(15.0, len(overlap) * 5.0)

        final_score = min(100.0, years_score + title_bonus)
        return round(final_score, 2)

    def evaluate_hard_constraints(
        self,
        candidate_visa: Optional[str],
        job_visa_sponsorship: bool,
        candidate_location: Optional[str],
        job_location: Optional[str],
        job_remote_policy: str,
    ) -> Tuple[float, bool]:
        """
        Evaluates hard constraints: visa sponsorship and remote/location compatibility.
        Returns score (0-100) and boolean passed status.
        """
        passed = True
        score = 100.0

        # Check visa sponsorship
        c_visa = (candidate_visa or "").lower()
        needs_sponsorship = any(v in c_visa for v in ["needs sponsorship", "require sponsorship", "h1b transfer"])

        if needs_sponsorship and not job_visa_sponsorship:
            passed = False
            score -= 60.0

        # Check location / remote policy
        policy = (job_remote_policy or "Hybrid").lower()
        if policy == "onsite":
            # If onsite and candidate has a different location
            if candidate_location and job_location:
                c_loc = candidate_location.lower()
                j_loc = job_location.lower()
                if c_loc not in j_loc and j_loc not in c_loc and "remote" not in c_loc:
                    score -= 30.0
                    # Does not completely fail if relocation is possible, but penalizes
        elif policy == "remote":
            # Remote is accessible to all
            pass

        final_score = max(0.0, min(100.0, score))
        return round(final_score, 2), passed

    def compute_hybrid_match(
        self,
        candidate_data: Dict[str, Any],
        job_data: Dict[str, Any],
        candidate_embedding: Optional[List[float]] = None,
        job_embedding: Optional[List[float]] = None,
    ) -> Tuple[float, MatchBreakdown, List[str], List[str], List[str], str]:
        """
        Executes complete 4-dimensional hybrid matching algorithm.
        Returns overall_score, breakdown, matching_skills, missing_must_haves, missing_nice_to_haves, recommendation.
        """
        # 1. Vector Similarity (40%)
        if candidate_embedding and job_embedding:
            vector_score = embedding_service.cosine_similarity(candidate_embedding, job_embedding)
        else:
            vector_score = 75.0

        # 2. Skill Coverage (35%)
        c_skills = candidate_data.get("skills", [])
        j_req_skills = job_data.get("required_skills", [])
        j_pref_skills = job_data.get("preferred_skills", [])

        skills_score, matching_skills, missing_must, missing_pref = self.calculate_skill_coverage(
            c_skills, j_req_skills, j_pref_skills
        )

        # 3. Experience & Seniority Alignment (15%)
        c_exp = float(candidate_data.get("years_of_experience", 0.0))
        j_exp = float(job_data.get("min_years_experience", 0.0))
        c_title = candidate_data.get("target_role")
        j_title = job_data.get("title")

        exp_score = self.calculate_experience_score(c_exp, j_exp, c_title, j_title)

        # 4. Hard Constraints (10%)
        c_visa = candidate_data.get("visa_status")
        j_visa = bool(job_data.get("visa_sponsorship", False))
        c_loc = candidate_data.get("location")
        j_loc = job_data.get("location")
        j_remote = job_data.get("remote_policy", "Hybrid")

        constraints_score, constraints_passed = self.evaluate_hard_constraints(
            c_visa, j_visa, c_loc, j_loc, j_remote
        )

        # Overall weighted score
        overall = (
            (vector_score * self.w_vector) +
            (skills_score * self.w_skills) +
            (exp_score * self.w_experience) +
            (constraints_score * self.w_constraints)
        )

        # If hard constraints failed (e.g. visa not sponsored), apply non-viability factor
        if not constraints_passed:
            overall = overall * 0.70

        overall = round(max(0.0, min(100.0, overall)), 1)

        breakdown = MatchBreakdown(
            vector_score=vector_score,
            skills_score=skills_score,
            experience_score=exp_score,
            hard_constraints_score=constraints_score,
            hard_constraints_passed=constraints_passed,
        )

        # Generate recommendation summary
        if overall >= 85.0:
            recommendation = f"Exceptional match ({overall}%). Candidate strongly satisfies required technical skills with high semantic role compatibility."
        elif overall >= 70.0:
            recommendation = f"Solid candidate ({overall}%). Covers primary core stack with slight qualification gaps in: {', '.join(missing_must[:2]) if missing_must else 'preferred tools'}."
        elif overall >= 50.0:
            recommendation = f"Moderate match ({overall}%). Notable technical or experience gap. Missing essential skills: {', '.join(missing_must[:3])}."
        else:
            recommendation = f"Low compatibility ({overall}%). Significant delta in required technical stack or unmet visa/experience constraints."

        return overall, breakdown, matching_skills, missing_must, missing_pref, recommendation


matching_agent = MatchingAgent()
