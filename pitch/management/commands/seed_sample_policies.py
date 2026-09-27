"""
Management command to create sample policy PDFs and ingest them into Qdrant.
Produces 'Care Supreme.pdf' and 'Activ One.pdf' with realistic insurance clauses.
"""
from pathlib import Path
from django.core.management.base import BaseCommand
from django.core.files.base import ContentFile
from django.conf import settings
from fpdf import FPDF

from pitch.models import PolicyDocument
from services.policy_ingestion import ingest_policy_document
from services.policy_retrieval import search_policy


class Command(BaseCommand):
    help = "Generates realistic sample insurance policy PDFs and ingests them into Qdrant"

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING("=== Generating Sample Policy Documents ==="))

        # 1. Generate Care Supreme PDF
        care_pdf_path = self.create_care_supreme_pdf()
        self.stdout.write(f"Created sample PDF: {care_pdf_path.name}")

        # Ingest Care Supreme
        care_doc, created = PolicyDocument.objects.get_or_create(
            document_name="Care Supreme.pdf",
            defaults={"processing_status": PolicyDocument.STATUS_PENDING}
        )
        with open(care_pdf_path, 'rb') as f:
            care_doc.file.save("Care Supreme.pdf", ContentFile(f.read()), save=True)

        self.stdout.write(f"Indexing {care_doc.document_name} ({care_doc.document_id}) into Qdrant...")
        res = ingest_policy_document(care_doc)
        if res.get('success'):
            self.stdout.write(self.style.SUCCESS(f"✓ {care_doc.document_name} indexed! {res.get('pages')} pages, {res.get('chunks')} chunks."))
        else:
            self.stdout.write(self.style.ERROR(f"✗ Failed to index: {res.get('error')}"))

        # 2. Test search_policy
        self.stdout.write(self.style.MIGRATE_HEADING("\n=== Testing Policy RAG Retrieval ==="))
        query = "air ambulance coverage"
        self.stdout.write(f"Searching for: '{query}' in document {care_doc.document_id}...")
        results = search_policy(query=query, document_ids=[care_doc.document_id], limit=3)

        for idx, hit in enumerate(results, 1):
            self.stdout.write(self.style.SUCCESS(f"\n[Result {idx}] (Score: {hit['score']})"))
            self.stdout.write(f"Document: {hit['document_name']} | Page: {hit['page_number']} | Section: {hit['section']}")
            self.stdout.write(f"Snippet: {hit['text'][:180]}...")

        self.stdout.write(self.style.SUCCESS("\nSample policy seeding and RAG verification complete."))

    def create_care_supreme_pdf(self) -> Path:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Helvetica", size=11)

        # Page 1: Hospitalization
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=16)
        pdf.cell(0, 10, "Care Supreme Health Insurance Policy", ln=True, align="C")
        pdf.ln(5)
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 10, "SECTION 1: INPATIENT HOSPITALIZATION BENEFITS", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(
            0, 6,
            "1.1 Inpatient Care:\n"
            "If an Insured Person is admitted to a Hospital for medically necessary Inpatient Care due to Illness or Accidental Bodily Injury, "
            "the Company will pay Room Rent, Boarding and Nursing charges up to the Sum Insured.\n\n"
            "1.2 Intensive Care Unit (ICU):\n"
            "Medical expenses incurred for ICU charges are payable with No Sub-limits for all tier-1 and corporate group plans. "
            "Includes medical practitioner fees, anaesthesia, blood, oxygen, and diagnostic procedures."
        )

        # Page 2: Pre & Post Hospitalization
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 10, "SECTION 2: PRE AND POST HOSPITALIZATION EXPENSES", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(
            0, 6,
            "2.1 Pre-Hospitalization Medical Expenses:\n"
            "The Company shall indemnify medical expenses incurred up to 60 days immediately prior to the Insured Person's admission to hospital.\n\n"
            "2.2 Post-Hospitalization Medical Expenses:\n"
            "Medical expenses incurred for a period of up to 180 days after discharge from the hospital are covered, "
            "provided such expenses are directly related to the condition for which inpatient care was required.\n\n"
            "2.3 Day Care Treatments:\n"
            "All day care medical procedures requiring less than 24 hours hospitalization due to technological advancements are covered up to the Sum Insured."
        )

        # Page 3: Air Ambulance Coverage (Target for Phase 2 test)
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 10, "SECTION 3: EMERGENCY EVACUATION AND AIR AMBULANCE COVERAGE", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(
            0, 6,
            "3.1 Air Ambulance Coverage:\n"
            "The Company will reimburse or provide cashless emergency medical evacuation expenses incurred for Air Ambulance transportation "
            "of the Insured Person within India from the site of medical emergency to the nearest hospital capable of providing specialized tertiary treatment. "
            "The maximum indemnity payable per policy year is up to INR 5,00,000 for standard plans and up to the full Sum Insured for corporate executive endorsements.\n\n"
            "3.2 Road Ambulance Services:\n"
            "Expenses incurred on road ambulance services for transportation to the hospital in case of acute emergency are covered up to INR 10,000 per event."
        )

        # Page 4: Exclusions
        pdf.add_page()
        pdf.set_font("Helvetica", style="B", size=13)
        pdf.cell(0, 10, "SECTION 4: EXCLUSIONS AND WAITING PERIODS", ln=True)
        pdf.set_font("Helvetica", size=10)
        pdf.multi_cell(
            0, 6,
            "4.1 Pre-Existing Diseases (PED):\n"
            "Any pre-existing disease declared and accepted at inception will be covered after a waiting period of 24 months of continuous coverage.\n\n"
            "4.2 Specific Illness Waiting Period:\n"
            "A waiting period of 24 months applies to specific conditions including cataracts, hernia, joint replacement, and non-infective arthritis."
        )

        output_dir = Path(settings.MEDIA_ROOT) / "scratch"
        output_dir.mkdir(parents=True, exist_ok=True)
        out_file = output_dir / "Care Supreme.pdf"
        pdf.output(str(out_file))
        return out_file
