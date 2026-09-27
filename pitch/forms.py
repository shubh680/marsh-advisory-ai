from django import forms
from .models import PolicyDocument


class PolicyUploadForm(forms.ModelForm):
    """
    Form for validating and saving uploaded policy PDFs.
    """
    class Meta:
        model = PolicyDocument
        fields = ['file']

    def clean_file(self):
        uploaded_file = self.cleaned_data.get('file')
        if not uploaded_file:
            raise forms.ValidationError('No file provided.')

        if not uploaded_file.name.lower().endswith('.pdf'):
            raise forms.ValidationError('Only PDF documents (.pdf) are supported.')

        # Limit file size to 25MB for safety
        max_size_mb = 25
        if uploaded_file.size > max_size_mb * 1024 * 1024:
            raise forms.ValidationError(f'File size exceeds {max_size_mb}MB limit.')

        return uploaded_file
