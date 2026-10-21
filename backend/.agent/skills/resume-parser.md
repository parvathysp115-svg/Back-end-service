# Resume Parser & Extraction Skill

## Overview
The `resume-parser` skill extracts and structures unstructured text from candidate resumes (PDF, DOCX, TXT) into a strongly-typed Pydantic schema (`CandidateProfileCreate`). It identifies contact information, educational credentials, certifications, normalized work history, and a fine-grained technical skill taxonomy.

## Capabilities
1. **Multi-Format Ingestion**: Extracts text using PyMuPDF (`fitz`), fallback to `pdfplumber`, and `python-docx` for Word documents.
2. **Schema Normalization**:
   - Cleans formatting artifacts, ligatures, and unconventional bullet points.
   - Normalizes roles and dates into standard ISO formats or structured duration objects.
   - Breaks down achievements into quantifiable metrics and core technologies used.
3. **Skill Taxonomy Extraction**:
   - Differentiates programming languages, cloud platforms, frameworks, databases, and soft skills.
   - Flags skill proficiencies where explicitly mentioned.
4. **Vector Embedding**: Generates a 768-dimensional dense vector representing the candidate profile for semantic retrieval.

## Validation Rules
- Extracted JSON must strictly adhere to `CandidateProfileCreate`.
- Empty or uncertain fields should default to empty lists or `null` rather than hallucinations.
- Skills must be deduplicated and lowercased for index consistency.
