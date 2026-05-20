import re

class IoCMasker:
    """
    A utility class to mask Indicators of Compromise (IoCs) in Cyber Threat Intelligence (CTI) reports.
    Replaces raw IPs, hashes, URLs, emails, and domains with typed semantic tags to prevent
    data leakage or hallucination in embedding spaces and LLM prompts.
    """
    
    def __init__(self) -> None:
        # Regex patterns compiled for performance
        # URLs: Matches http, https, ftp
        self.url_pattern = re.compile(
            r'(?:https?|ftp)://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
            re.IGNORECASE
        )
        
        # Emails: Standard email address matching
        self.email_pattern = re.compile(
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            re.IGNORECASE
        )
        
        # IPv6: Basic matching for IPv6 addresses
        self.ipv6_pattern = re.compile(
            r'\b(?:[a-fA-F0-9]{1,4}:){7}[a-fA-F0-9]{1,4}\b',
            re.IGNORECASE
        )
        
        # IPv4: Matches valid IPv4 octets
        self.ipv4_pattern = re.compile(
            r'\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.'
            r'(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
        )
        
        # Hashes: MD5 (32), SHA1 (40), SHA256 (64)
        self.hash_pattern = re.compile(
            r'\b(?:[a-fA-F0-9]{32}|[a-fA-F0-9]{40}|[a-fA-F0-9]{64})\b'
        )
        
        # Domains: Matches standard domain formats
        # We apply this last or carefully so it doesn't mask parts of emails or valid prose
        self.domain_pattern = re.compile(
            r'\b(?:[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?\.)+[a-zA-Z]{2,}\b',
            re.IGNORECASE
        )

    def mask_text(self, text: str) -> str:
        """
        Masks all identified IoCs in the provided text with typed tags.
        
        Order of substitution is critical to avoid partial matches 
        (e.g., masking a domain inside an email or URL first).
        
        Args:
            text (str): The raw text from the CTI report.
            
        Returns:
            str: The sanitized text with IoCs masked.
        """
        if not text:
            return text
            
        # 1. URLs (most complex, contains domains/paths)
        masked_text = self.url_pattern.sub('<IoC_URL>', text)
        
        # 2. Emails (contains domains)
        masked_text = self.email_pattern.sub('<IoC_EMAIL>', masked_text)
        
        # 3. IPv6 and IPv4
        masked_text = self.ipv6_pattern.sub('<IoC_IPv6>', masked_text)
        masked_text = self.ipv4_pattern.sub('<IoC_IPv4>', masked_text)
        
        # 4. Hashes (MD5, SHA1, SHA256)
        masked_text = self.hash_pattern.sub('<IoC_HASH>', masked_text)
        
        # 5. Domains (done after URLs and Emails to prevent partial overrides)
        # Note: In a real prod env, we might want a stricter domain regex to avoid 
        # matching filenames like "report.pdf", but for CTI context this is standard.
        masked_text = self.domain_pattern.sub('<IoC_DOMAIN>', masked_text)
        
        return masked_text
