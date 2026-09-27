from django.urls import path
from . import views
from services.inngest_orchestrator import inngest_django_pattern

app_name = 'pitch'

urlpatterns = [
    path('', views.index, name='index'),
    path('api/upload/', views.upload_document_view, name='upload_document'),
    path('api/documents/', views.documents_list_view, name='documents_list'),
    path('api/search-policy/', views.search_policy_view, name='search_policy'),
    path('api/research-company/', views.research_company_view, name='research_company'),
    path('api/qdrant-status/', views.qdrant_status_view, name='qdrant_status'),
    path('api/generate-and-audit/', views.generate_and_audit_view, name='generate_and_audit'),
    path('api/audit-claim/', views.audit_single_claim_view, name='audit_single_claim'),
    path('api/documents/<str:document_id>/delete/', views.delete_document_view, name='delete_document'),
    inngest_django_pattern,
]

