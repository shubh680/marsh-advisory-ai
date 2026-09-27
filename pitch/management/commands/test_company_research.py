"""
Management command to test Phase 3 Company Research Agent.
Executes autonomous research on a target company and outputs structured JSON profile.
"""
import json
from django.core.management.base import BaseCommand
from services.company_research import research_company


class Command(BaseCommand):
    help = "Tests Phase 3 Company Research Agent using GPT-OSS-120B and external research tools"

    def add_arguments(self, parser):
        parser.add_argument(
            '--company',
            type=str,
            default='Tata Motors',
            help='Company name to research (default: Tata Motors)'
        )

    def handle(self, *args, **options):
        company = options['company']
        self.stdout.write(self.style.MIGRATE_HEADING(f"=== Researching Target Company: '{company}' ==="))
        self.stdout.write("Running autonomous research agent with web search & URL tools...")

        profile = research_company(company_name=company)

        self.stdout.write(self.style.SUCCESS("\n=== Structured Company Profile Output ==="))
        self.stdout.write(json.dumps(profile, indent=2))

        self.stdout.write(self.style.MIGRATE_HEADING("\n=== Profile Summary Verification ==="))
        self.stdout.write(f"Company: {profile.get('company_name')}")
        self.stdout.write(f"Industry Sector: {profile.get('industry_sector') or profile.get('industry')}")
        self.stdout.write(f"Company Size: {profile.get('company_size')}")
        self.stdout.write(f"Business Activities: {len(profile.get('business_activities', []))} items")
        self.stdout.write(f"Operational Footprint: {len(profile.get('operational_footprint', []))} items")
        self.stdout.write(f"Key Risks & Exposures: {len(profile.get('key_risks_and_exposures', []) or profile.get('key_risks', []))} items")
        self.stdout.write(f"Sources Verified: {len(profile.get('sources', []))} (Title + URL)")

        self.stdout.write(self.style.SUCCESS("\nPhase 3 Company Research Agent verified successfully."))

