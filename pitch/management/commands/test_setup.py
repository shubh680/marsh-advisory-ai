from django.core.management.base import BaseCommand
from django.conf import settings
from services.qdrant_service import check_qdrant_connection


class Command(BaseCommand):
    help = 'Tests Phase 0 environment setup and Qdrant connectivity'

    def handle(self, *args, **options):
        self.stdout.write(self.style.MIGRATE_HEADING('=== Checking Phase 0 Project Setup ==='))

        # Check SQLite
        db_engine = settings.DATABASES['default']['ENGINE']
        db_name = settings.DATABASES['default']['NAME']
        self.stdout.write(f'Database Engine: {db_engine}')
        self.stdout.write(f'Database File: {db_name}')

        # Check LLM Config
        llm_provider = getattr(settings, 'LLM_PROVIDER', 'unset')
        llm_model = getattr(settings, 'LLM_MODEL', 'unset')
        llm_api_key_set = bool(getattr(settings, 'LLM_API_KEY', ''))
        self.stdout.write(f'LLM Provider: {llm_provider}')
        self.stdout.write(f'LLM Model: {llm_model}')
        self.stdout.write(f'LLM API Key configured: {"Yes" if llm_api_key_set else "No (set in .env when ready)"}')

        # Check Qdrant Connection
        self.stdout.write(self.style.MIGRATE_HEADING('\n=== Testing Qdrant Connection ==='))
        qdrant_info = check_qdrant_connection()
        if qdrant_info.get('status') == 'connected':
            self.stdout.write(
                self.style.SUCCESS(
                    f'✓ Qdrant connected successfully! (Mode: {qdrant_info.get("mode")}, URL: {qdrant_info.get("url")})'
                )
            )
        else:
            self.stdout.write(
                self.style.ERROR(
                    f'✗ Qdrant connection failed: {qdrant_info.get("error")}'
                )
            )

        self.stdout.write(self.style.SUCCESS('\nPhase 0 Setup check complete.'))
