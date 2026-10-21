import json
import logging
from typing import Dict, Any, Optional, List
from app.core.config import settings
from app.schemas.coaching_schema import (
    CareerRoadmapResponse,
    RoadmapMilestone,
    InterviewPrepPointer,
    CoverLetterResponse,
)

logger = logging.getLogger(__name__)


class CoachingAgent:
    """
    Subagent 4: Career Coach & Action Plan Agent
    Generates personalized learning roadmaps, targeted interview prep pointers,
    and tailored cover letters based on identified qualification gaps.
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
                logger.warning(f"Could not init GenAI in CoachingAgent: {e}")

    # -------------------------------------------------------------
    # Action Plan & Learning Roadmap
    # -------------------------------------------------------------
    async def generate_career_roadmap(
        self,
        candidate_data: Dict[str, Any],
        target_role: str,
        missing_skills: List[str],
        experience_gap_years: float = 0.0,
    ) -> CareerRoadmapResponse:
        """
        Creates a structured learning roadmap and targeted interview prep pointers.
        """
        candidate_id = str(candidate_data.get("id", "temp-id"))
        cand_name = candidate_data.get("full_name", "Candidate")
        skills_str = ", ".join(missing_skills) if missing_skills else "advanced system design patterns"

        if self._genai_client:
            try:
                prompt = f"""You are a Principal Engineering Career Coach and Tech Lead Mentor.
Candidate: {cand_name}
Target Role: {target_role}
Current Skills: {candidate_data.get('skills')}
Identified Technical/Skill Gaps: {skills_str}
Experience Delta: {experience_gap_years} years

Generate:
1. A structured 4-phase learning roadmap (milestones, resources, hands-on portfolio projects, expected outcomes).
2. 3-4 deep-dive interview preparation pointers covering System Design, Coding, and Behavioral/STAR.

Output strictly in JSON conforming to this schema:
{{
  "candidate_id": "{candidate_id}",
  "target_role": "{target_role}",
  "gap_summary": "Summary of primary qualification deltas in {skills_str}",
  "learning_roadmap": [
    {{
      "week_or_phase": "Phase 1: Weeks 1-2",
      "focus_topic": "Foundations & Architecture",
      "learning_resources": ["Designing Data-Intensive Applications", "Official Documentation"],
      "hands_on_project": "Build an async distributed event-pipeline with dead-letter queue",
      "expected_outcome": "Demonstrated mastery of distributed state and event streaming"
    }}
  ],
  "interview_prep_pointers": [
    {{
      "category": "System Design",
      "question": "How would you design a low-latency job matching search engine using vector embeddings?",
      "key_points_to_cover": ["pgvector indexing with HNSW/IVFFlat", "Hybrid search with BM25", "Redis cache eviction"],
      "sample_answer_outline": "1. Clarify scale & QPS. 2. Define schema. 3. Detail vector index + cosine metrics. 4. Address failover."
    }}
  ]
}}
"""
                response_text = await self._call_llm(prompt)
                parsed_json = self._parse_json_from_response(response_text)
                parsed_json["candidate_id"] = candidate_id
                return CareerRoadmapResponse(**parsed_json)
            except Exception as e:
                logger.warning(f"CoachingAgent LLM call failed ({e}). Using rule-based fallback roadmap.")

        # Rule-based fallback roadmap
        return self._rule_based_roadmap(candidate_id, target_role, missing_skills)

    # -------------------------------------------------------------
    # Customized Cover Letter
    # -------------------------------------------------------------
    async def generate_cover_letter(
        self,
        candidate_data: Dict[str, Any],
        target_company: str,
        target_role: str,
        job_description: Optional[str] = None,
        tone: str = "Confident and Impact-driven",
    ) -> CoverLetterResponse:
        """
        Synthesizes candidate background with job requirements into a compelling cover letter.
        """
        cand_id = str(candidate_data.get("id", "temp-id"))
        cand_name = candidate_data.get("full_name", "Candidate")
        cand_skills = ", ".join(candidate_data.get("skills", [])[:8])

        if self._genai_client:
            try:
                prompt = f"""You are an elite Executive Career Strategist.
Write a high-converting, tailored cover letter for:
Candidate: {cand_name}
Target Company: {target_company}
Target Role: {target_role}
Candidate Strengths & Skills: {cand_skills}
Tone: {tone}
Job Description Context: {job_description or 'High-scale backend engineering role'}

Requirements:
- Emphasize quantifiable engineering achievements and architectural ownership.
- Connect candidate's specific background directly to {target_company}'s engineering challenges.
- Avoid generic filler clichés (e.g. 'I am writing to express my passion').
- Output strictly in JSON conforming to:
{{
  "candidate_id": "{cand_id}",
  "target_company": "{target_company}",
  "target_role": "{target_role}",
  "cover_letter_markdown": "# Dear Engineering Leadership at {target_company},\\n\\n...",
  "key_selling_points_highlighted": ["Distributed systems latency reduction", "Cloud infrastructure scalability"]
}}
"""
                response_text = await self._call_llm(prompt)
                parsed_json = self._parse_json_from_response(response_text)
                parsed_json["candidate_id"] = cand_id
                return CoverLetterResponse(**parsed_json)
            except Exception as e:
                logger.warning(f"CoachingAgent cover letter LLM failed ({e}). Using rule-based generator.")

        return self._rule_based_cover_letter(candidate_data, target_company, target_role, tone)

    # -------------------------------------------------------------
    # Helpers & Fallback Generators
    # -------------------------------------------------------------
    async def _call_llm(self, prompt: str) -> str:
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

    def _rule_based_roadmap(
        self,
        candidate_id: str,
        target_role: str,
        missing_skills: List[str],
    ) -> CareerRoadmapResponse:
        """Deterministic roadmap generator."""
        skills_str = ", ".join(missing_skills[:3]) if missing_skills else "Distributed Microservices"

        milestones = [
            RoadmapMilestone(
                week_or_phase="Phase 1: Weeks 1-2 (Deep Foundations)",
                focus_topic=f"Mastery of {skills_str}",
                learning_resources=[
                    "Designing Data-Intensive Applications by Martin Kleppmann",
                    f"Official Documentation & Architectural RFCs for {skills_str}",
                    "System Design Primer by Donne Martin"
                ],
                hands_on_project=f"Build an end-to-end benchmark service demonstrating high concurrency with {skills_str}.",
                expected_outcome=f"Deep architectural familiarity with internals and performance patterns in {skills_str}."
            ),
            RoadmapMilestone(
                week_or_phase="Phase 2: Weeks 3-4 (Production System Architecture)",
                focus_topic="Scalability, Caching, and Vector Search",
                learning_resources=[
                    "PostgreSQL Query Optimization & pgvector indexing docs",
                    "Redis Enterprise Patterns & Cache Invalidation Strategies"
                ],
                hands_on_project="Implement hybrid BM25 + dense vector semantic search service handling 10k items/second.",
                expected_outcome="Production-ready portfolio repository with benchmark telemetry and Docker Compose harness."
            ),
            RoadmapMilestone(
                week_or_phase="Phase 3: Weeks 5-6 (Interview Mastery & Mock Execution)",
                focus_topic="System Design & Behavioral STAR Delivery",
                learning_resources=[
                    "Alex Xu System Design Interview Volume 1 & 2",
                    "STAR Framework Story Matrix for Leadership Principles"
                ],
                hands_on_project="Conduct 4 timed system design drills under whiteboard constraints.",
                expected_outcome="Fluent, structured delivery of complex distributed systems trade-offs."
            ),
        ]

        pointers = [
            InterviewPrepPointer(
                category="System Design",
                question=f"How would you design a scalable recommendation pipeline integrating {skills_str}?",
                key_points_to_cover=[
                    "Data ingestion bottlenecks and asynchronous decoupling",
                    "Vector indexing strategies (HNSW vs IVFFlat) and memory footprints",
                    "Consistent hashing, failover mechanisms, and p99 SLA guarantees"
                ],
                sample_answer_outline="1. Define requirements (RPS, read-to-write ratio). 2. High-level architecture. 3. Deep-dive into indexing & caching layer. 4. Fault-tolerance analysis."
            ),
            InterviewPrepPointer(
                category="Algorithmic & Concurrency",
                question="How do you mitigate database race conditions and connection pool exhaustion under sudden traffic spikes?",
                key_points_to_cover=[
                    "Optimistic vs pessimistic locking in SQLAlchemy / PostgreSQL",
                    "Connection pool sizing (HikariCP / asyncpg pool bounds)",
                    "Circuit breakers and backoff retry algorithms"
                ],
                sample_answer_outline="Explain connection pool lifecycle, queue timeout thresholds, and asynchronous non-blocking worker pools."
            ),
            InterviewPrepPointer(
                category="Behavioral / STAR",
                question="Describe a critical production incident where system latency degraded, and how you diagnosed and resolved it.",
                key_points_to_cover=[
                    "Root cause identification via APM metrics and slow query logs",
                    "Immediate remediation (circuit breaker, cache warm-up)",
                    "Long-term post-mortem and automated regression guardrails"
                ],
                sample_answer_outline="Structure using STAR: Situation (p99 latency spike), Task (restore SLA), Action (indexed hot tables, enabled Redis caching), Result (85% latency drop)."
            )
        ]

        return CareerRoadmapResponse(
            candidate_id=candidate_id,
            target_role=target_role,
            gap_summary=f"Key technical areas for immediate career leveling: {skills_str}. Closing these gaps will elevate compatibility above the 90th percentile.",
            learning_roadmap=milestones,
            interview_prep_pointers=pointers
        )

    def _rule_based_cover_letter(
        self,
        candidate_data: Dict[str, Any],
        target_company: str,
        target_role: str,
        tone: str,
    ) -> CoverLetterResponse:
        """Deterministic cover letter generator."""
        cand_name = candidate_data.get("full_name", "Candidate")
        skills = candidate_data.get("skills", ["Distributed Systems", "Python", "FastAPI"])
        top_skills = ", ".join(skills[:4])
        years = candidate_data.get("years_of_experience", 4.0)

        cover_letter = f"""Dear Engineering Leadership Team at {target_company},

I am writing to express my strong interest in the **{target_role}** opportunity at {target_company}. With over {years} years of dedicated engineering experience architecting scalable backend services and high-throughput data pipelines, I have consistently focused on engineering resilient, latency-sensitive distributed systems that directly drive business outcomes.

Throughout my career, I have specialized in **{top_skills}**, with a continuous track record of transforming complex product requirements into battle-tested cloud-native microservices. In previous positions, I have spearheaded architectural overhauls that reduced p99 service latency by over 40%, streamlined database query access patterns, and automated mission-critical ingestion pipelines processing millions of daily records.

What excites me most about {target_company} is your commitment to engineering excellence, technical velocity, and solving demanding architectural challenges at scale. My core strengths in clean system design, asynchronous concurrency, and performance optimization align closely with the responsibilities of the {target_role}.

I look forward to the possibility of discussing how my engineering background and drive for architectural rigor can contribute immediately to {target_company}'s engineering objectives.

Sincerely,  
**{cand_name}**
"""
        return CoverLetterResponse(
            candidate_id=str(candidate_data.get("id", "temp-id")),
            target_company=target_company,
            target_role=target_role,
            cover_letter_markdown=cover_letter.strip(),
            key_selling_points_highlighted=[
                f"{years}+ years of backend engineering experience",
                f"Core proficiency in {top_skills}",
                "Proven track record of quantifiable latency reduction and cloud scalability"
            ]
        )


coaching_agent = CoachingAgent()
