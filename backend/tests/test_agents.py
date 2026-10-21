import pytest
from app.services.agents.extraction_agent import extraction_agent
from app.services.agents.optimization_agent import optimization_agent
from app.services.agents.coaching_agent import coaching_agent


@pytest.mark.asyncio
async def test_extraction_agent_resume(sample_resume_text):
    profile = await extraction_agent.extract_candidate_profile(sample_resume_text)
    assert profile.full_name is not None
    assert "john.doe@example.com" in (profile.email or "")
    assert profile.years_of_experience >= 1.0
    # Technical skills extracted
    norm_skills = [s.lower() for s in profile.skills]
    assert "python" in norm_skills
    assert "fastapi" in norm_skills
    assert len(profile.work_history) > 0


@pytest.mark.asyncio
async def test_extraction_agent_job_description():
    jd_raw = """
    Staff Backend Engineer - Payments Team
    FinTech Global Inc.
    Location: New York, NY (Remote available)
    Experience required: 6+ years
    We are seeking a Staff Backend Engineer to design high-throughput transaction pipelines.
    Required: Python, PostgreSQL, Kafka, Redis, Docker, System Design
    Preferred: Kubernetes, AWS, Go
    Visa sponsorship is provided for qualified applicants.
    """
    job = await extraction_agent.extract_job_posting(jd_raw, title_hint="Staff Backend Engineer")
    assert "Staff Backend Engineer" in job.title
    assert "FinTech Global" in job.company or "Enterprise" in job.company
    assert job.min_years_experience >= 5.0
    norm_skills = [s.lower() for s in job.required_skills + job.preferred_skills]
    assert "python" in norm_skills
    assert "postgresql" in norm_skills
    assert job.visa_sponsorship is True


@pytest.mark.asyncio
async def test_optimization_agent():
    cand_data = {
        "id": "cand-123",
        "full_name": "Test Candidate",
        "target_role": "Backend Engineer",
        "skills": ["Python", "FastAPI"],
        "work_history": [
            {
                "title": "Software Engineer",
                "company": "Beta Tech",
                "responsibilities": ["Responsible for maintaining internal APIs"],
                "achievements": ["Assisted with database migration"],
            }
        ]
    }

    result = await optimization_agent.optimize_resume(cand_data, target_role="Senior Backend Engineer")
    assert result.candidate_id == "cand-123"
    assert 0.0 <= result.ats_score <= 100.0
    assert len(result.weak_bullet_points) > 0
    # Verify passive voice detected
    assert any("PASSIVE" in w.issue_type or "METRICS" in w.issue_type for w in result.weak_bullet_points)
    # Verify STAR rewrites
    assert len(result.star_rewrites) > 0
    first_rewrite = result.star_rewrites[0]
    assert first_rewrite.situation != ""
    assert first_rewrite.task != ""
    assert first_rewrite.action != ""
    assert first_rewrite.result != ""
    assert first_rewrite.rewritten_bullet != ""
    assert len(result.missing_high_value_keywords) > 0


@pytest.mark.asyncio
async def test_coaching_agent():
    cand_data = {
        "id": "cand-456",
        "full_name": "Devin Coder",
        "skills": ["Python", "Docker"],
        "years_of_experience": 3.0,
    }

    # 1. Test career roadmap
    roadmap = await coaching_agent.generate_career_roadmap(
        cand_data,
        target_role="Lead Backend Engineer",
        missing_skills=["Kafka", "Kubernetes", "PostgreSQL Query Tuning"],
    )
    assert roadmap.candidate_id == "cand-456"
    assert len(roadmap.learning_roadmap) >= 2
    assert len(roadmap.interview_prep_pointers) >= 2
    assert any("System Design" in p.category for p in roadmap.interview_prep_pointers)

    # 2. Test cover letter synthesis
    cover = await coaching_agent.generate_cover_letter(
        cand_data,
        target_company="Acme Aerospace",
        target_role="Lead Backend Engineer",
    )
    assert cover.candidate_id == "cand-456"
    assert "Acme Aerospace" in cover.cover_letter_markdown
    assert len(cover.key_selling_points_highlighted) > 0
