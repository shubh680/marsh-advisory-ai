import uuid
from django.db import models


def generate_document_id() -> str:
    return str(uuid.uuid4())


class PolicyDocument(models.Model):
    """
    Model representing an uploaded insurance policy PDF.
    The original PDF is stored on the filesystem under media/policies/.
    Vectors and extracted chunk payloads are stored in Qdrant.
    """
    STATUS_PENDING = 'pending'
    STATUS_PROCESSING = 'processing'
    STATUS_INDEXED = 'indexed'
    STATUS_FAILED = 'failed'

    STATUS_CHOICES = [
        (STATUS_PENDING, 'Pending'),
        (STATUS_PROCESSING, 'Processing'),
        (STATUS_INDEXED, 'Indexed in Qdrant'),
        (STATUS_FAILED, 'Failed'),
    ]

    document_id = models.CharField(
        max_length=64,
        unique=True,
        default=generate_document_id,
        editable=False,
        help_text='Unique identifier for the policy document (used for vector payload filtering)'
    )

    document_name = models.CharField(
        max_length=255,
        help_text='Display name or original filename of the document'
    )
    file = models.FileField(
        upload_to='policies/',
        help_text='Original PDF file stored on disk under media/policies/'
    )
    uploaded_at = models.DateTimeField(
        auto_now_add=True
    )
    processing_status = models.CharField(
        max_length=20,
        choices=STATUS_CHOICES,
        default=STATUS_PENDING
    )
    page_count = models.PositiveIntegerField(
        default=0,
        help_text='Total number of pages extracted from the PDF'
    )
    chunk_count = models.PositiveIntegerField(
        default=0,
        help_text='Total number of text chunks generated and indexed in Qdrant'
    )
    error_message = models.TextField(
        blank=True,
        null=True,
        help_text='Error details if processing failed'
    )

    class Meta:
        ordering = ['-uploaded_at']
        verbose_name = 'Policy Document'
        verbose_name_plural = 'Policy Documents'

    def __str__(self):
        return f"{self.document_name} ({self.processing_status})"
