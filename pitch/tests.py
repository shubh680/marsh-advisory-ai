import json
from pathlib import Path
from django.test import TestCase, Client
from django.urls import reverse
from django.conf import settings
from django.core.files.uploadedfile import SimpleUploadedFile

from pitch.models import PolicyDocument
from services.qdrant_service import check_qdrant_connection
from services.policy_ingestion import extract_pdf_pages, chunk_text_by_paragraphs, ingest_policy_document
from services.policy_retrieval import search_policy


class Phase0And1And2Tests(TestCase):
    def setUp(self):
        self.client = Client()
        policies_dir = Path(settings.MEDIA_ROOT) / "policies"
        if policies_dir.exists():
            for pdf_file in policies_dir.glob("*.pdf"):
                if "care_supreme" in pdf_file.name.lower():
                    clean_name = "Care Supreme.pdf"
                    doc_id = "b2dc0971-e73e-457a-8ac8-df421247b6be"
                elif "hdfc" in pdf_file.name.lower():
                    clean_name = "HDFC Product Brochure.pdf"
                    doc_id = "8263fab2-37be-4d88-85ac-9c081a5bfddb"
                else:
                    clean_name = pdf_file.name
                    doc_id = None

                defaults = {
                    'file': f'policies/{pdf_file.name}',
                    'processing_status': PolicyDocument.STATUS_INDEXED,
                    'page_count': 4
                }
                if doc_id:
                    defaults['document_id'] = doc_id

                PolicyDocument.objects.get_or_create(
                    document_name=clean_name,
                    defaults=defaults
                )


    def test_settings_loaded(self):
        """Ensure settings load core variables from environment"""
        self.assertIsNotNone(settings.SECRET_KEY)
        self.assertIn(settings.LLM_PROVIDER, ['groq', 'openai', 'gcp'])
        self.assertEqual(settings.DATABASES['default']['ENGINE'], 'django.db.backends.sqlite3')
        self.assertTrue(settings.MEDIA_ROOT.exists())

    def test_qdrant_connection_helper(self):
        """Verify Qdrant client connection test runs and connects"""
        status_info = check_qdrant_connection()
        self.assertEqual(status_info.get('status'), 'connected')
        self.assertIn('url', status_info)

    def test_policy_document_model(self):
        """Phase 1: Test PolicyDocument model creation and fields"""
        doc = PolicyDocument.objects.create(
            document_name="Test Policy.pdf",
            processing_status=PolicyDocument.STATUS_PENDING
        )
        self.assertTrue(len(str(doc.document_id)) > 10)
        self.assertEqual(doc.processing_status, PolicyDocument.STATUS_PENDING)
        self.assertIn("Test Policy.pdf", str(doc))


    def test_pdf_extraction_and_page_number_preservation(self):
        """Phase 2: Verify extract_pdf_pages preserves 1-indexed page numbers"""
        sample_pdf = Path(settings.MEDIA_ROOT) / "scratch" / "Care Supreme.pdf"
        if sample_pdf.exists():
            pages = extract_pdf_pages(str(sample_pdf))
            self.assertEqual(len(pages), 4)
            self.assertEqual(pages[0]['page_number'], 1)
            self.assertIn("INPATIENT", pages[0]['text'].upper())
            self.assertEqual(pages[2]['page_number'], 3)
            self.assertIn("AIR AMBULANCE", pages[2]['text'].upper())

    def test_policy_search_retrieval(self):
        """Phase 2: Verify search_policy returns grounded evidence with exact page numbers"""
        doc = PolicyDocument.objects.filter(document_name="Care Supreme.pdf").first()
        doc_ids = [doc.document_id] if doc and doc.document_id else None
        results = search_policy("air ambulance coverage", document_ids=doc_ids, limit=3)
        if not results:
            results = search_policy("air ambulance coverage", limit=3)
        self.assertTrue(len(results) > 0)
        top_hit = results[0]
        self.assertEqual(top_hit['page_number'], 3)
        self.assertIn("Air Ambulance", top_hit['text'])
        self.assertGreater(top_hit['score'], 0.5)

    def test_api_documents_list(self):
        """Phase 1 API: Verify documents list JSON endpoint"""
        response = self.client.get(reverse('pitch:documents_list'))
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('documents', data)

    def test_api_search_policy(self):
        """Phase 2 API: Verify search policy endpoint"""
        response = self.client.get(reverse('pitch:search_policy'), {'query': 'hospitalization'})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('results', data)

    def test_research_tools_search_web(self):
        """Phase 3 Tool: Verify search_web returns factual snippets and URLs"""
        from services.company_research import search_web
        res = search_web("Tata Motors", max_results=2)
        self.assertTrue(len(res) > 20)
        self.assertIn("URL: ", res)

    def test_llm_generate_structured(self):
        """Phase 3 LLM: Verify generate_structured produces valid dict from LLM"""
        from services.llm_client import generate_structured
        res = generate_structured("Output a JSON object with key 'status' equal to 'ready'")
        self.assertIsInstance(res, dict)
        self.assertEqual(res.get('status'), 'ready')

    def test_api_research_company(self):
        """Phase 3 API: Verify research company endpoint validation and execution"""
        # Test empty validation
        bad_res = self.client.get(reverse('pitch:research_company'))
        self.assertEqual(bad_res.status_code, 400)

        # Test valid company call
        good_res = self.client.get(reverse('pitch:research_company'), {'company_name': 'Tata Motors'})
        self.assertEqual(good_res.status_code, 200)
        data = good_res.json()
        self.assertEqual(data.get('company_name'), 'Tata Motors')
        self.assertIn('industry', data)
        self.assertIn('business_activities', data)
        self.assertIn('key_risks', data)
        self.assertIn('sources', data)

    def test_workforce_health_formulator_strictly_medical(self):
        """Pipeline Test: Verify workforce health queries are strictly medical/hospitalization"""
        from services.workforce_health import formulate_workforce_health_exposures
        sample_profile = {
            "company_name": "Tata Motors",
            "industry_sector": "automotive manufacturing",
            "company_size": "75,000+ employees",
            "business_activities": ["vehicle manufacturing"],
            "operational_footprint": ["Pune", "Sanand", "Jamshedpur"],
            "key_risks_and_exposures": ["factory machinery hazards", "supply chain disruptions"]
        }
        res = formulate_workforce_health_exposures(sample_profile)
        self.assertEqual(res.get("company_name"), "Tata Motors")
        self.assertTrue(len(res.get("medical_search_queries", [])) >= 3)
        # Ensure no property/casualty terms exist in generated queries
        for q in res["medical_search_queries"]:
            q_lower = q.lower()
            self.assertNotIn("machinery breakdown", q_lower)
            self.assertNotIn("marine cargo", q_lower)

    def test_pitch_generator_structure_and_pptx(self):
        """Pipeline Test: Verify medical pitch generates 3-5 slides and valid .pptx"""
        from services.pitch_generator import generate_medical_pitch
        sample_profile = {
            "company_name": "Acme Corp",
            "industry_sector": "technology services",
            "company_size": "5,000 employees"
        }
        workforce_health = {
            "medical_search_queries": ["inpatient hospitalization ICU room rent", "emergency air ambulance"],
            "priority_healthcare_exposures": [{"exposure_name": "Hospitalization", "policy_coverage_focus": "ICU"}]
        }
        res = generate_medical_pitch(sample_profile, workforce_health)
        slides = res.get("slides", [])
        self.assertTrue(3 <= len(slides) <= 5)
        self.assertTrue(Path(res.get("pptx_path", "")).exists())
        self.assertIn(".pptx", res.get("pptx_url", ""))

    def test_4_tier_claim_audit_taxonomy(self):
        """Pipeline Test: Verify 4-tier audit classifies claims against PDF text"""
        from services.audit_service import audit_single_claim
        # Test claim: Road ambulance has Rs 10,000 cap; claiming no cap is contradicted
        res_contradicted = audit_single_claim(
            claim_text="Road ambulance is covered without any sub-limit or cap",
            cited_policy="Care Supreme.pdf",
            cited_page=3
        )
        self.assertEqual(res_contradicted.get("verdict"), "CONTRADICTED")

        # Test claim: Spiritual retreats have zero grounding
        res_unsupported = audit_single_claim(
            claim_text="Provides 200,000 allowance for international spiritual retreats",
            cited_policy="Care Supreme.pdf",
            cited_page=1
        )
        self.assertEqual(res_unsupported.get("verdict"), "UNSUPPORTED")

    def test_api_generate_and_audit_pipeline(self):
        """API Test: Verify POST /api/generate-and-audit/ executes pipeline end-to-end"""
        payload = {
            "company_name": "HDFC Bank",
            "company_profile": {
                "company_name": "HDFC Bank",
                "industry_sector": "banking",
                "company_size": "170,000+ employees",
                "business_activities": ["commercial and retail banking"],
                "operational_footprint": ["8000+ branches"],
                "key_risks_and_exposures": ["ergonomic stress", "executive travel"]
            }
        }
        response = self.client.post(
            reverse('pitch:generate_and_audit'),
            data=json.dumps(payload),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data.get("success"))
        self.assertIn("pitch", data)
        self.assertIn("audit", data)
    def test_audit_report_pdf_generation(self):
        """Pipeline Test: Verify audit report PDF is compiled and saved with valid path and URL"""
        from services.audit_service import compile_audit_report_pdf
        sample_audit = {
            "company_name": "Test Enterprise",
            "grounding_score": 90.0,
            "audit_status": "PASSED",
            "total_claims": 2,
            "supported_count": 2,
            "partially_supported_count": 0,
            "contradicted_count": 0,
            "unsupported_count": 0,
            "audited_claims": [
                {
                    "slide_number": 3,
                    "claim_text": "Inpatient care covered up to Sum Insured.",
                    "cited_policy": "Care Supreme.pdf",
                    "cited_page": 1,
                    "verdict": "SUPPORTED",
                    "evidence_quote": "Room Rent, Boarding and Nursing charges up to the Sum Insured.",
                    "explanation": "Substantiated in Section 1.1."
                }
            ]
        }
        pdf_path, pdf_url = compile_audit_report_pdf(sample_audit)
        self.assertTrue(Path(pdf_path).exists())
        self.assertGreater(Path(pdf_path).stat().st_size, 500)
        self.assertIn("audit_report.pdf", pdf_url)

    def test_filter_pitch_claims_by_audit(self):
        """Pipeline Test: Verify PPT claims only include SUPPORTED and PARTIALLY_SUPPORTED terms"""
        from services.audit_service import filter_pitch_claims_by_audit
        sample_pitch = {
            "company_name": "Filter Test Corp",
            "slides": [
                {
                    "slide_number": 3,
                    "title": "Coverage Mapping",
                    "bullets": ["Bullet 1"],
                    "policy_claims": [
                        {"claim_text": "Good Claim 1", "cited_policy": "Policy A.pdf", "cited_page": 1},
                        {"claim_text": "Bad Claim 2", "cited_policy": "Policy A.pdf", "cited_page": 2},
                        {"claim_text": "Partial Claim 3", "cited_policy": "Policy A.pdf", "cited_page": 3}
                    ]
                }
            ]
        }
        sample_audit = {
            "audited_claims": [
                {"slide_number": 3, "claim_text": "Good Claim 1", "verdict": "SUPPORTED", "cited_page": 1},
                {"slide_number": 3, "claim_text": "Bad Claim 2", "verdict": "UNSUPPORTED", "cited_page": 2},
                {"slide_number": 3, "claim_text": "Partial Claim 3", "verdict": "PARTIALLY_SUPPORTED", "cited_page": 3}
            ]
        }
        filtered = filter_pitch_claims_by_audit(sample_pitch, sample_audit)
        slide_3_claims = filtered["slides"][0]["policy_claims"]
        claim_texts = [c["claim_text"] for c in slide_3_claims]

        self.assertIn("Good Claim 1", claim_texts)
        self.assertIn("Partial Claim 3", claim_texts)
        self.assertNotIn("Bad Claim 2", claim_texts)
        self.assertEqual(len(slide_3_claims), 2)
        self.assertTrue(Path(filtered["pptx_path"]).exists())


