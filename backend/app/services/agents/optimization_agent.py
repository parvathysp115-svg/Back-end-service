import json
import logging
import re
from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.schemas.coaching_schema import (
    ResumeOptimizationResponse,
    WeakBulletPoint,
    STARBulletRewrite,
)
from app.schemas.resume_schema import CandidateProfileResponse

logger = logging.getLogger(__name__)

# Passive voice / weak action indicators
PASSIVE_MARKERS = [
    "responsible for", "assisted with", "helped to", "worked on", "duties included",
    "was involved in", "participated in", "tasked with", "handled", "supported"
]

# Action verbs for strong bullet points
STRONG_ACTION_VERBS = [
    "Architected", "Engineered", "Spearheaded", "Optimized", "Scaled", "Automated",
    "Streamlined", "Accelerated", "Delivered", "Overhauled", "Formulated", "Championed"
]


class OptimizationAgent:
    """
    Subagent 2: Resume Optimization & Training Agent
    Evaluates resumes against industry benchmarks and target roles, scores ATS compliance,
    detects weak bullet points (passive voice, missing metrics), and rewrites them into STAR format.
    """

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model = model or settings.GEMINI_MODEL
        self._genai_client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key or self.api_key.startswith("your-gemini"):
            return

        try:
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
        except Exception:
            try:
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self._genai_client = gai
            except Exception as e:
                logger.warning(f"Could not init GenAI in OptimizationAgent: {e}")

    async def optimize_resume(
        self,
        candidate_data: Dict[str, Any],
        target_role: Optional[str] = None,
        job_description: Optional[str] = None
    ) -> ResumeOptimizationResponse:
        """
        Runs comprehensive resume evaluation and STAR rewrites using Gemini LLM
        with rule-based fallback analyzer.
        """
        effective_target = target_role or candidate_data.get("target_role") or "Senior Software Engineer"

        if self._genai_client:
            try:
                prompt = f"""You are an elite Executive Career Coach and ATS Algorithm Specialist.
Analyze the following candidate resume for the target role: "{effective_target}".
Job Description Context (if provided): "{job_description or 'None'}"

Candidate Details:
Name: {candidate_data.get('full_name')}
Skills: {candidate_data.get('skills')}
Experience: {json.dumps(candidate_data.get('work_history', []))}
Education: {json.dumps(candidate_data.get('education', []))}

Your task:
1. Identify 2-4 weak bullet points (flag passive voice, missing quantifiable metrics, or vague impact).
2. For each weak bullet, rewrite into a high-impact STAR bullet point (Situation, Task, Action, Result).
3. Compute an ATS compliance score (0-100) based on role keyword density, action verbs, and structure.
4. Identify 3-5 missing high-value technical keywords for "{effective_target}".
5. Suggest 2-3 relevant industry certifications.
6. Provide actionable next steps.

Output strictly in JSON conforming to this schema:
{{
  "candidate_id": "{candidate_data.get('id', 'temp-id')}",
  "target_role_evaluated": "{effective_target}",
  "ats_score": 82.5,
  "ats_summary": "Strong technical foundations, but lacking quantified financial/system performance outcomes in several recent positions.",
  "weak_bullet_points": [
    {{
      "original_text": "Responsible for maintaining backend APIs",
      "role_or_context": "Software Engineer at Tech Corp",
      "issue_type": "PASSIVE_VOICE",
      "critique": "Uses passive marker 'Responsible for' and lacks metrics showing scale, latency, or throughput."
    }}
  ],
  "star_rewrites": [
    {{
      "original_bullet": "Responsible for maintaining backend APIs",
      "situation": "High-traffic microservice architecture experiencing periodic latency spikes during peak usage.",
      "task": "Redesign database access patterns and introduce caching to stabilize p99 response times.",
      "action": "Engineered asynchronous FastAPI endpoints with Redis tiered caching and connection pooling.",
      "result": "Reduced p99 API latency from 450ms to 65ms and supported 4x increase in concurrent active users.",
      "rewritten_bullet": "Architected asynchronous FastAPI endpoints with Redis caching and connection pooling, slashing p99 latency by 85% (450ms to 65ms) and supporting 4x traffic surges."
    }}
  ],
  "missing_high_value_keywords": ["Distributed Systems", "Kubernetes", "Kafka", "PostgreSQL Query Optimization"],
  "recommended_certifications": ["AWS Certified Solutions Architect - Associate", "CKA: Certified Kubernetes Administrator"],
  "actionable_next_steps": [
    "Replace passive phrases with strong engineering action verbs like Spearheaded, Scaled, and Architected.",
    "Quantify throughput, percentage reductions in cost/latency, and user scale for each bullet point."
  ]
}}
"""
                response_text = await self._call_llm(prompt)
                parsed_json = self._parse_json_from_response(response_text)
                parsed_json["candidate_id"] = candidate_data.get("id", "temp-id")
                return ResumeOptimizationResponse(**parsed_json)
            except Exception as e:
                logger.warning(f"OptimizationAgent LLM call failed ({e}). Running rule-based optimizer.")

        # Rule-based fallback optimizer
        return self._rule_based_optimize(candidate_data, effective_target, job_description)

    async def _call_llm(self, prompt: str) -> str:
        """Call LLM with JSON response format."""
        if hasattr(self._genai_client, "models"):
            response = self._genai_client.models.generate_content(
                model=self.model,
                contents=prompt,
                config={"response_mime_type": "application/json"}
            )
            return response.text
        elif hasattr(self._genai_client, "GenerativeModel"):
            model = self._genai_client.GenerativeModel(
                self.model,
                generation_config={"response_mime_type": "application/json"}
            )
            res = model.generate_content(prompt)
            return res.text
        raise RuntimeError("No LLM client configured")

    def _parse_json_from_response(self, text: str) -> Dict[str, Any]:
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return json.loads(cleaned.strip())

    def _rule_based_optimize(
        self,
        candidate_data: Dict[str, Any],
        target_role: str,
        job_description: Optional[str] = None
    ) -> ResumeOptimizationResponse:
        """
        Rule-based heuristic evaluator for detecting passive voice, missing metrics,
        and producing STAR rewrites.
        """
        work_history = candidate_data.get("work_history", [])
        weak_bullets: List[WeakBulletPoint] = []
        star_rewrites: List[STARBulletRewrite] = []

        total_bullets = 0
        metric_bullets = 0
        active_bullets = 0

        # Scan work history bullet points
        for work in work_history:
            role = work.get("title", "Software Engineer")
            company = work.get("company", "Company")
            context = f"{role} at {company}"

            # Check responsibilities and achievements
            bullets = work.get("responsibilities", []) + work.get("achievements", [])
            for bullet in bullets:
                total_bullets += 1
                b_low = bullet.lower()

                has_metric = bool(re.search(r"\b\d+([.,]\d+)?%?|\$\d+", bullet))
                if has_metric:
                    metric_bullets += 1

                has_passive = any(marker in b_low for marker in PASSIVE_MARKERS)
                if not has_passive:
                    active_bullets += 1

                if has_passive or not has_metric:
                    issue = "PASSIVE_VOICE" if has_passive else "MISSING_METRICS"
                    critique = (
                        "Contains passive marker ('" +
                        next((m for m in PASSIVE_MARKERS if m in b_low), "weak verb") +
                        "') and lacks clear ownership."
                        if has_passive
                        else "Lacks quantifiable numbers, percentage changes, or measurable performance benchmarks."
                    )
                    weak_bullets.append(
                        WeakBulletPoint(
                            original_text=bullet,
                            role_or_context=context,
                            issue_type=issue,
                            critique=critique
                        )
                    )

                    # Generate STAR rewrite
                    verb = STRONG_ACTION_VERBS[len(star_rewrites) % len(STRONG_ACTION_VERBS)]
                    clean_core = re.sub(r"^(responsible for|worked on|assisted with|helped to|duties included)\s*", "", bullet, flags=re.IGNORECASE).strip()
                    star_rewrites.append(
                        STARBulletRewrite(
                            original_bullet=bullet,
                            situation=f"Complex legacy workflows in {context} requiring architectural elevation and performance tuning.",
                            task="Modernize system throughput and automate repetitive pipelines.",
                            action=f"{verb} {clean_core.lower()} utilizing cloud-native patterns and asynchronous pipelines.",
                            result="Achieved a 38% reduction in latency and improved operational efficiency across 5 engineering squads.",
                            rewritten_bullet=f"{verb} {clean_core.lower()} using distributed patterns, improving system throughput by 38% and reducing operational overhead."
                        )
                    )

                if len(weak_bullets) >= 4:
                    break
            if len(weak_bullets) >= 4:
                break

        # Calculate ATS score
        ats_score = 70.0
        if total_bullets > 0:
            metric_ratio = metric_bullets / total_bullets
            active_ratio = active_bullets / total_bullets
            ats_score = round(50.0 + (metric_ratio * 30.0) + (active_ratio * 20.0), 1)

        # Missing keywords based on target role
        target_low = target_role.lower()
        missing_kw = []
        if "backend" in target_low:
            missing_kw = ["Distributed Systems", "PostgreSQL Indexing", "Redis Caching", "gRPC", "Docker & Kubernetes"]
        elif "data" in target_low:
            missing_kw = ["Apache Spark", "Airflow", "ETL Pipelines", "Data Warehousing", "Snowflake"]
        elif "cloud" in target_low or "devops" in target_low:
            missing_kw = ["Terraform", "CI/CD Pipelines", "AWS EKS", "Observability (Prometheus/Grafana)"]
        else:
            missing_kw = ["System Architecture", "Microservices", "RESTful APIs", "CI/CD Automation", "Cloud Infrastructure"]

        # Filter out existing skills
        candidate_skills = [s.lower() for s in candidate_data.get("skills", [])]
        final_missing = [kw for kw in missing_kw if kw.lower() not in candidate_skills][:4]

        return ResumeOptimizationResponse(
            candidate_id=str(candidate_data.get("id", "temp-id")),
            target_role_evaluated=target_role,
            ats_score=ats_score,
            ats_summary=f"ATS compliance score of {ats_score}/100. Resume shows solid foundational competencies. Key improvement areas: eliminate passive markers and inject concrete business/system metrics into every bullet.",
            weak_bullet_points=weak_bullets[:4],
            star_rewrites=star_rewrites[:4],
            missing_high_value_keywords=final_missing,
            recommended_certifications=[
                "AWS Certified Solutions Architect - Associate",
                "CKA: Certified Kubernetes Administrator"
            ],
            actionable_next_steps=[
                "Rewrite passive experience items into active STAR bullets starting with impact action verbs.",
                "Incorporate exact industry terms from the target job description (e.g., Redis, Kafka, Distributed Systems).",
                "Add measurable KPIs (throughput, latency, error rate, dollar savings, user scale) to demonstrate tangible ROI."
            ]
        )


optimization_agent = OptimizationAgent()
