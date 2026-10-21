# Job Description Matcher Skill

## Overview
The `jd-matcher` skill implements a multi-dimensional, hybrid matching and gap analysis engine to evaluate compatibility between a candidate profile and a job description.

## Matching Architecture
The matching engine computes a final compatibility score (0-100) using a 4-dimensional weighted model:

1. **Semantic Vector Similarity (40%)**:
   - Cosine similarity between candidate embedding and job embedding in PostgreSQL `pgvector` (Vector(768)).
2. **Skill Coverage (35%)**:
   - Hybrid match combining BM25 keyword matching with exact and synonym matching.
   - Distinct weight for `must-have` vs `nice-to-have` skills.
3. **Experience & Seniority Alignment (15%)**:
   - Compares candidate's total years of experience against the vacancy requirement.
   - Evaluates title progression and seniority level (Junior, Mid, Senior, Lead, Principal).
4. **Hard Constraints Filter (10%)**:
   - Evaluates visa sponsorship requirements, remote/hybrid/onsite policies, and geographic constraints.
   - Acts as a penalty factor or pass/fail indicator.

## Gap Analysis & Recommendations
- Identifies critical missing skills required for the role.
- Highlights transferable experience and adjacent strengths.
- Provides actionable recommendations for resume optimization and candidate upskilling.
