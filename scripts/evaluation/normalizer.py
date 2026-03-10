import re
from whisper.normalizers import BasicTextNormalizer

class UnifiedNormalizer:
    def __init__(self, use_whisper_basic=True, custom_mappings=None):
        self.whisper_normalizer = BasicTextNormalizer() if use_whisper_basic else None
        self.custom_mappings = custom_mappings or {}

    def __call__(self, text):
        if not text:
            return ""
        
        # 1. Whisper's basic normalization (lowercasing, punctuation removal, etc.)
        if self.whisper_normalizer:
            text = self.whisper_normalizer(text)
        
        # 2. Custom project-specific mapping
        for original, replacement in self.custom_mappings.items():
            text = text.replace(original, replacement)
        
        # 3. Extra cleanup (e.g., redundant spacing)
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
