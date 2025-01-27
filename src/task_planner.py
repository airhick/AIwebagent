import re
from typing import Dict, Optional
import logging

logger = logging.getLogger('task_planner')

class TaskPlanner:
    """Plans and analyzes web tasks"""
    
    def __init__(self):
        self.common_actions = {
            'navigate': ['go to', 'visit', 'open', 'browse', 'navigate'],
            'search': ['search for', 'find', 'look up', 'search'],
            'click': ['click', 'press', 'select', 'choose'],
            'type': ['type', 'enter', 'input', 'write'],
            'scroll': ['scroll', 'move down', 'move up'],
            'wait': ['wait', 'pause', 'delay']
        }

    def analyze_request(self, request: str) -> Dict:
        """Analyze user request and create task plan"""
        request_lower = request.lower()
        logger.info(f"Analyzing request: {request}")
        
        # Check for navigation actions
        for action in self.common_actions['navigate']:
            if action in request_lower:
                url = self._extract_url(request)
                if url:
                    logger.info(f"Detected navigation task to: {url}")
                    return {
                        'action_type': 'navigate',
                        'url': url,
                        'original_request': request
                    }
        
        # Check for search actions
        for action in self.common_actions['search']:
            if action in request_lower:
                search_term = self._extract_search_term(request)
                if search_term:
                    logger.info(f"Detected search task for: {search_term}")
                    return {
                        'action_type': 'search',
                        'query': search_term,
                        'original_request': request
                    }
        
        # Default to navigation if it looks like a URL
        if self._looks_like_url(request):
            logger.info(f"Request looks like URL, treating as navigation: {request}")
            return {
                'action_type': 'navigate',
                'url': request,
                'original_request': request
            }
            
        # Default to search
        logger.info(f"Treating request as search query: {request}")
        return {
            'action_type': 'search',
            'query': request,
            'original_request': request
        }

    def _extract_url(self, text: str) -> Optional[str]:
        """Extract URL from text"""
        words = text.split()
        for word in words:
            if self._looks_like_url(word):
                return word
        return None

    def _looks_like_url(self, text: str) -> bool:
        """Check if text looks like a URL"""
        url_patterns = [
            r'https?://',
            r'www\.',
            r'[a-zA-Z0-9-]+\.[a-zA-Z]{2,}',
        ]
        return any(re.search(pattern, text) for pattern in url_patterns)

    def _extract_search_term(self, text: str) -> str:
        """Extract search term from text"""
        for action in self.common_actions['search']:
            if action in text.lower():
                # Get everything after the action word
                parts = text.lower().split(action, 1)
                if len(parts) > 1:
                    return parts[1].strip()
        return text.strip() 