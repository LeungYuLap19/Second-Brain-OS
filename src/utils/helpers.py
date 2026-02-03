import os
import re
from typing import Optional

# Vector DB helpers moved to `src.services.vectordb` — import needed symbols from there
from src.services.vectordb import (
    get_loader,
    ingest_documents_generic,
    ingest_professor_document,
    delete_professor_document,
    EMBEDDINGS,
)

def clean_html_content(html_text: str) -> str:
    """
    Thoroughly clean HTML content to extract only readable text.
    
    Removes:
    - HTML tags
    - CSS styles (both <style> tags and inline styles)
    - JavaScript (<script> tags)
    - HTML comments
    - CSS block patterns and selectors
    - Common HTML entities (&nbsp;, &amp;, etc.)
    
    Args:
        html_text (str): Raw HTML content to clean
        
    Returns:
        str: Clean, readable text with all HTML/CSS/JS removed
        
    Example:
        >>> html = "<p>Hello <strong>world</strong></p>"
        >>> clean_html_content(html)
        'Hello world'
    """
    if not html_text:
        return ""
    
    # First decode HTML entities
    import html
    text = html.unescape(html_text)
    
    # Remove CSS content (style tags and inline style attributes)
    # Remove <style> tags and their content
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove inline style attributes
    text = re.sub(r'style="[^"]*"', '', text, flags=re.IGNORECASE)
    text = re.sub(r"style='[^']*'", '', text, flags=re.IGNORECASE)
    
    # Remove script tags and their content
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove HTML comments
    text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    
    # Remove all other HTML tags but preserve text content
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove CSS block patterns (content between { and })
    text = re.sub(r'\{[^}]*\}', '', text)
    
    # Remove common CSS selectors that might appear in email body
    css_selectors = [
        r'\.\w+\s*{[^}]*}',
        r'#\w+\s*{[^}]*}',
        r'@media[^{]+\{[^}]*\}',
        r'@font-face[^{]+\{[^}]*\}',
        r'body\s*{[^}]*}',
        r'div\s*{[^}]*}',
        r'span\s*{[^}]*}',
        r'p\s*{[^}]*}',
        r'a\s*{[^}]*}',
        r'table\s*{[^}]*}',
        r'tr\s*{[^}]*}',
        r'td\s*{[^}]*}',
        r'th\s*{[^}]*}',
    ]
    for pattern in css_selectors:
        text = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove multiple spaces, newlines, tabs
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common email artifacts
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&zwnj;', '', text)
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    text = re.sub(r'&quot;', '"', text)
    text = re.sub(r'&#39;', "'", text)
    text = re.sub(r'&#\d+;', '', text)  # Remove numeric entities
    
    # Remove URLs (optional, but keeps text cleaner)
    # text = re.sub(r'https?://\S+', '', text)
    
    # Clean up whitespace again
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text