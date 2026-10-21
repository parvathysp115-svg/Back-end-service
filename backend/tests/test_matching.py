import pytest
from app.services.agents.matching_agent import matching_agent, normalize_skill


def test_normalize_skill_synonyms():
    assert normalize_skill("k8s") == "kubernetes"
    assert normalize_skill("postgres") == "postgresql"
    assert normalize_skill("golang") == "go"
    assert normalize_skill("py") == "python"


def test_matching_agent_skill_coverage():
    candidate_skills = ["Python", "FastAPI", "K8s", "Postgres", "Redis"]
    required_skills = ["Python", "Kubernetes", "PostgreSQL", "Docker"]
    preferred_skills = ["Redis", "Kafka"]

    score, matching, missing_must, missing_pref = matching_agent.calculate_skill_coverage(
        candidate_skills, required_skills, preferred_skills
    )

    # Python, K8s (Kubernetes), Postgres (PostgreSQL) should match
    assert "Python" in matching
    assert "Kubernetes" in matching
    assert "PostgreSQL" in matching
    assert "Docker" in missing_must
    assert "Kafka" in missing_pref
    assert score > 60.0


def test_matching_agent_experience_score():
    # Meets or exceeds
    score_full = matching_agent.calculate_experience_score(6.0, 5.0, "Senior Engineer", "Senior Engineer")
    assert score_full == 100.0

    # Underqualified
    score_partial = matching_agent.calculate_experience_score(2.0, 5.0, "Junior Developer", "Senior Engineer")
    assert score_partial < 60.0


def test_matching_agent_hard_constraints():
    # Case 1: Needs sponsorship but job doesn't provide
    score, passed = matching_agent.evaluate_hard_constraints(
        candidate_visa="Needs Sponsorship",
        job_visa_sponsorship=False,
        candidate_location="London, UK",
        job_location="San Francisco, CA",
        job_remote_policy="Onsite",
    )
    assert passed is False
    assert score < 50.0

    # Case 2: Citizen and remote job
    score_ok, passed_ok = matching_agent.evaluate_hard_constraints(
        candidate_visa="US Citizen",
        job_visa_sponsorship=False,
        candidate_location="Austin, TX",
        job_location="San Francisco, CA",
        job_remote_policy="Remote",
    )
    assert passed_ok is True
    assert score_ok == 100.0


def test_matching_agent_hybrid_weights():
    cand_data = {
        "id": "c1",
        "full_name": "Test User",
        "skills": ["Python", "PostgreSQL", "FastAPI"],
        "years_of_experience": 5.0,
        "visa_status": "Citizen",
        "location": "San Francisco, CA",
    }
    job_data = {
        "id": "j1",
        "title": "Backend Engineer",
        "min_years_experience": 4.0,
        "required_skills": ["Python", "FastAPI", "PostgreSQL"],
        "preferred_skills": ["Redis"],
        "visa_sponsorship": True,
        "location": "San Francisco, CA",
        "remote_policy": "Hybrid",
    }

    dummy_vec = [0.1] * 768
    overall, breakdown, matching, missing_must, missing_pref, rec = (
        matching_agent.compute_hybrid_match(cand_data, job_data, dummy_vec, dummy_vec)
    )

    assert 80.0 <= overall <= 100.0
    assert breakdown.hard_constraints_passed is True
    assert breakdown.vector_score > 0.0
    assert breakdown.skills_score >= 75.0
    assert len(matching) == 3


@pytest.mark.asyncio
async def test_full_api_matching_pipeline(client, sample_resume_text, sample_job_data):
    # 1. Upload Resume
    upload_res = await client.post(
        "/api/v1/resumes/upload",
        data={"raw_text": sample_resume_text},
    )
    assert upload_res.status_code == 201
    cand_json = upload_res.json()
    candidate_id = cand_json["candidate_id"]
    assert candidate_id is not None

    # 2. Create Job Posting
    job_res = await client.post(
        "/api/v1/jobs/",
        json=sample_job_data,
    )
    assert job_res.status_code == 201
    job_json = job_res.json()
    job_id = job_json["id"]
    assert job_id is not None

    # 3. Match Resume against Jobs
    match_resume_res = await client.post(f"/api/v1/matching/match-resume/{candidate_id}")
    assert match_resume_res.status_code == 200
    match_data = match_resume_res.json()
    assert match_data["candidate_id"] == candidate_id
    assert match_data["total_jobs_evaluated"] >= 1
    top_matches = match_data["top_matches"]
    assert len(top_matches) >= 1
    first_job_match = top_matches[0]
    assert first_job_match["job_id"] == job_id
    assert first_job_match["overall_score"] > 60.0
    assert "breakdown" in first_job_match
    assert first_job_match["breakdown"]["vector_score"] > 0

    # 4. Match Job against Candidates
    match_job_res = await client.post(f"/api/v1/matching/match-job/{job_id}")
    assert match_job_res.status_code == 200
    job_matches = match_job_res.json()
    assert job_matches["job_id"] == job_id
    assert len(job_matches["top_matches"]) >= 1
    assert job_matches["top_matches"][0]["candidate_id"] == candidate_id

    # 5. Direct Compare Endpoint
    compare_res = await client.post(
        "/api/v1/matching/compare",
        json={"candidate_id": candidate_id, "job_id": job_id},
    )
    assert compare_res.status_code == 200
    comp_data = compare_res.json()
    assert comp_data["candidate_id"] == candidate_id
    assert comp_data["job_id"] == job_id
    assert comp_data["overall_score"] > 60.0

    # 6. Resume Optimization & Training Endpoint
    train_res = await client.post(
        f"/api/v1/resumes/{candidate_id}/train",
        json={"target_role": "Lead Backend Engineer", "job_id": job_id},
    )
    assert train_res.status_code == 200
    train_data = train_res.json()
    assert train_data["candidate_id"] == candidate_id
    assert 0.0 <= train_data["ats_score"] <= 100.0
    assert len(train_data["star_rewrites"]) > 0

    # 7. Career Roadmap Endpoint
    roadmap_res = await client.post(
        f"/api/v1/resumes/{candidate_id}/coaching-roadmap",
        params={"target_role": "Staff Backend Engineer"},
    )
    assert roadmap_res.status_code == 200
    roadmap_data = roadmap_res.json()
    assert roadmap_data["candidate_id"] == candidate_id
    assert len(roadmap_data["learning_roadmap"]) >= 2

    # 8. Cover Letter Endpoint
    cover_res = await client.post(
        f"/api/v1/resumes/{candidate_id}/cover-letter",
        json={"job_id": job_id, "tone": "Confident and Technical"},
    )
    assert cover_res.status_code == 200
    cover_data = cover_res.json()
    assert cover_data["candidate_id"] == candidate_id
    assert "cover_letter_markdown" in cover_data

    # 9. Health Check
    health_res = await client.get("/health")
    assert health_res.status_code == 200
    assert health_res.json()["status"] == "healthy"

