"""
Qdrant Vector Database service helper.
Provides client initialization and connection health check.
"""
from django.conf import settings
from qdrant_client import QdrantClient


def get_qdrant_client() -> QdrantClient:
    """
    Returns a configured QdrantClient instance based on Django settings.
    Supports:
    - In-memory testing: QDRANT_URL=":memory:"
    - HTTP/HTTPS instance (local Docker or Qdrant Cloud): QDRANT_URL="http://localhost:6333"
    - Local disk path: QDRANT_URL="./qdrant_data"
    """
    url = getattr(settings, 'QDRANT_URL', ':memory:')
    api_key = getattr(settings, 'QDRANT_API_KEY', '') or None

    if url == ':memory:':
        return QdrantClient(location=':memory:')
    elif url.startswith(('http://', 'https://')):
        return QdrantClient(url=url, api_key=api_key)
    else:
        return QdrantClient(path=url)


def check_qdrant_connection() -> dict:
    """
    Tests connection to Qdrant and returns status information.
    """
    try:
        client = get_qdrant_client()
        collections_response = client.get_collections()
        collection_names = [col.name for col in collections_response.collections]
        return {
            'status': 'connected',
            'mode': 'in-memory' if getattr(settings, 'QDRANT_URL', '') == ':memory:' else 'network/persistent',
            'url': getattr(settings, 'QDRANT_URL', ':memory:'),
            'collections_count': len(collection_names),
            'collections': collection_names,
        }
    except Exception as e:
        return {
            'status': 'error',
            'url': getattr(settings, 'QDRANT_URL', ':memory:'),
            'error': str(e),
        }
