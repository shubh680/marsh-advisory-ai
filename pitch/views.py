import json
from django.shortcuts import render, get_object_or_404
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings

from .models import PolicyDocument
from .forms import PolicyUploadForm
from services.qdrant_service import check_qdrant_connection
from services.policy_ingestion import ingest_policy_document
from services.policy_retrieval import search_policy
from services.company_research import research_company


def index(request):
    """
    Renders the main advisor pitch generator interface.
    Allows advisor to enter company name, upload PDFs, and select policies.
    """
    documents = PolicyDocument.objects.all()
    qdrant_status = check_qdrant_connection()
    context = {
        'phase': 'End-to-End Advisory Engine',
        'documents': documents,
        'llm_provider': getattr(settings, 'LLM_PROVIDER', 'Not set'),
        'llm_model': getattr(settings, 'LLM_MODEL', 'Not set'),
        'embedding_model': getattr(settings, 'EMBEDDING_MODEL', 'BAAI/bge-small-en-v1.5'),
        'qdrant_status': qdrant_status,
        'debug_mode': settings.DEBUG,
    }
    return render(request, 'pitch/index.html', context)



@require_http_methods(["POST"])
def upload_document_view(request):
    """
    Endpoint for uploading policy PDFs.
    Saves document to disk (media/policies/) and triggers Qdrant ingestion.
    """
    files = request.FILES.getlist('file')
    if not files:
        return JsonResponse({'error': 'No file uploaded'}, status=400)

    uploaded_docs = []
    errors = []

    for uploaded_file in files:
        form = PolicyUploadForm(files={'file': uploaded_file})
        if form.is_valid():
            doc = form.save(commit=False)
            doc.document_name = uploaded_file.name
            doc.save()

            # Ingest into Qdrant pipeline (Phase 2)
            ingest_result = ingest_policy_document(doc)

            uploaded_docs.append({
                'id': doc.id,
                'document_id': doc.document_id,
                'document_name': doc.document_name,
                'status': doc.processing_status,
                'pages': doc.page_count,
                'chunks': doc.chunk_count,
                'error': ingest_result.get('error')
            })
        else:
            errors.append({
                'filename': uploaded_file.name,
                'errors': form.errors.get('file', ['Invalid file'])
            })

    return JsonResponse({
        'success': len(uploaded_docs) > 0,
        'documents': uploaded_docs,
        'errors': errors
    })


def documents_list_view(request):
    """
    Returns JSON list of all available policy documents.
    """
    docs = PolicyDocument.objects.all().values(
        'id', 'document_id', 'document_name', 'processing_status',
        'page_count', 'chunk_count', 'uploaded_at'
    )
    return JsonResponse({'documents': list(docs)})


def search_policy_view(request):
    """
    API endpoint to test policy search / retrieval grounded in Qdrant.
    Query parameters:
    - query: search text
    - document_ids: comma-separated list of document_id strings
    """
    query = request.GET.get('query', '').strip()
    if not query:
        return JsonResponse({'error': 'Query parameter is required'}, status=400)

    doc_ids_param = request.GET.get('document_ids', '')
    doc_ids = [d.strip() for d in doc_ids_param.split(',') if d.strip()] if doc_ids_param else None

    results = search_policy(query=query, document_ids=doc_ids)
    return JsonResponse({
        'query': query,
        'filters': {'document_ids': doc_ids},
        'count': len(results),
        'results': results
    })


def qdrant_status_view(request):
    """
    API endpoint to check Qdrant connectivity.
    """
    status_info = check_qdrant_connection()
    return JsonResponse(status_info)


def research_company_view(request):
    """
    API endpoint for running the autonomous company research agent (Phase 3).
    Query parameter:
    - company_name: Name of company to research
    """
    company_name = request.GET.get('company_name', '').strip()
    if not company_name:
        return JsonResponse({'error': 'company_name query parameter is required'}, status=400)

    try:
        profile = research_company(company_name=company_name)
        return JsonResponse(profile)
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)





@csrf_exempt
@require_http_methods(["POST"])
def generate_and_audit_view(request):
    """
    End-to-End Pipeline Endpoint:
    1. Researches target company (or uses pre-researched profile).
    2. Formulates workforce health and medical coverage priorities.
    3. Retrieves candidate policy chunks from Qdrant and generates a 3-5 slide presentation (.pptx).
    4. Audits every policy claim against actual PDF text into 4-tier taxonomy (SUPPORTED, PARTIALLY_SUPPORTED, UNSUPPORTED, CONTRADICTED).
    """
    try:
        data = json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    company_name = data.get('company_name', '').strip() or request.POST.get('company_name', '').strip()
    if not company_name:
        return JsonResponse({'error': 'company_name is required'}, status=400)

    selected_docs = data.get('selected_docs') or request.POST.getlist('selected_docs')
    if isinstance(selected_docs, str):
        selected_docs = [d.strip() for d in selected_docs.split(',') if d.strip()]

    # Execute pipeline through Inngest orchestrator with detailed observability
    try:
        from services.inngest_orchestrator import run_orchestrated_pipeline
        result = run_orchestrated_pipeline(
            company_name=company_name,
            selected_docs=selected_docs,
            cached_profile=data.get('company_profile')
        )
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': f'Pipeline orchestration error: {str(e)}'}, status=500)


@require_http_methods(["POST"])
def audit_single_claim_view(request):
    """
    API endpoint to audit an individual claim against PDF text on demand.
    """
    from services.audit_service import audit_single_claim

    try:
        data = json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        data = {}

    claim_text = data.get('claim_text', '').strip()
    cited_policy = data.get('cited_policy', '').strip()
    cited_page = int(data.get('cited_page', 1))

    if not claim_text or not cited_policy:
        return JsonResponse({'error': 'claim_text and cited_policy are required'}, status=400)

    try:
        result = audit_single_claim(claim_text=claim_text, cited_policy=cited_policy, cited_page=cited_page)
        return JsonResponse(result)
    except Exception as e:
        return JsonResponse({'error': f'Audit LLM interpretation failed: {str(e)}'}, status=500)


@csrf_exempt
@require_http_methods(["DELETE", "POST"])
def delete_document_view(request, document_id):
    """
    Deletes a policy document: removes Qdrant chunks, DB record, and PDF file.
    """
    from django.conf import settings as django_settings
    from services.qdrant_service import get_qdrant_client
    from qdrant_client.http import models as qmodels
    from services.policy_retrieval import invalidate_bm25_cache

    doc = get_object_or_404(PolicyDocument, document_id=document_id)

    # 1. Delete chunks from Qdrant
    try:
        client = get_qdrant_client()
        collection_name = getattr(django_settings, 'QDRANT_COLLECTION_NAME', 'policy_documents')
        existing = [c.name for c in client.get_collections().collections]
        if collection_name in existing:
            client.delete(
                collection_name=collection_name,
                points_selector=qmodels.FilterSelector(
                    filter=qmodels.Filter(
                        must=[qmodels.FieldCondition(
                            key="document_id",
                            match=qmodels.MatchValue(value=str(doc.document_id)),
                        )]
                    )
                ),
            )
        invalidate_bm25_cache()
    except Exception as e:
        return JsonResponse({'error': f'Qdrant deletion failed: {e}'}, status=500)

    # 2. Delete PDF file from disk
    if doc.file:
        try:
            import os
            if os.path.isfile(doc.file.path):
                os.remove(doc.file.path)
        except Exception:
            pass  # File may already be gone

    # 3. Delete DB record
    doc_name = doc.document_name
    doc.delete()

    return JsonResponse({
        'success': True,
        'deleted_document': doc_name,
    })
