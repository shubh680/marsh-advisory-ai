from django.contrib import admin
from .models import PolicyDocument


@admin.register(PolicyDocument)
class PolicyDocumentAdmin(admin.ModelAdmin):
    list_display = ('document_name', 'document_id', 'processing_status', 'page_count', 'chunk_count', 'uploaded_at')
    list_filter = ('processing_status', 'uploaded_at')
    search_fields = ('document_name', 'document_id')
    readonly_fields = ('document_id', 'uploaded_at', 'page_count', 'chunk_count', 'error_message')
