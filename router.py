"""
Smart Router for Phase 3 Integration
Determines whether to use RAG or Web Search based on query analysis
"""

from typing import Literal
import re


class QueryRouter:
    """
    A smart router that decides the best source (RAG or Web) for answering a query.
    Uses keyword-based and contextual analysis to make intelligent routing decisions.
    """
    
    # Keywords that indicate need for current/real-time information
    WEB_KEYWORDS = {
        'latest', 'current', 'news', 'today', 'recent', 'breaking',
        'now', 'trend', 'trending', 'update', 'happening', 'real-time',
        'weather', 'stock', 'live', 'today\'s', 'this week', 'this month',
        'tomorrow', 'yesterday', 'recent news', 'latest news', 'current news',
        'what\'s new', 'what is new', '2024', '2025', '2026', 'covid', 'elections'
    }
    
    # Keywords that indicate document-based queries
    RAG_KEYWORDS = {
        'document', 'pdf', 'uploaded', 'paper', 'chapter', 'section',
        'according to', 'in the document', 'in the pdf', 'from the file',
        'our document', 'the material', 'the content', 'here', 'previously'
    }
    
    # Keywords that are neutral (can go either way)
    NEUTRAL_KEYWORDS = {
        'what', 'how', 'why', 'explain', 'describe', 'tell', 'define',
        'summarize', 'provide', 'give', 'list', 'show', 'detail'
    }
    
    def __init__(self, rag_database_exists: bool = False):
        """
        Initialize the QueryRouter.
        
        Args:
            rag_database_exists (bool): Whether RAG database is available
        """
        self.rag_database_exists = rag_database_exists
    
    def set_rag_availability(self, exists: bool):
        """
        Update RAG database availability status.
        
        Args:
            exists (bool): Whether RAG database exists
        """
        self.rag_database_exists = exists
    
    def _extract_keywords(self, query: str) -> set:
        """
        Extract lowercase keywords from query for analysis.
        
        Args:
            query (str): User's query
        
        Returns:
            set: Set of keywords found in query
        """
        query_lower = query.lower()
        words = set(query_lower.split())
        
        # Also check for phrase keywords
        phrases = [
            'latest', 'current', 'news', 'real-time', 'live', 'trending',
            'breaking', 'what\'s new', 'recent news', 'latest news'
        ]
        
        found_phrases = set()
        for phrase in phrases:
            if phrase in query_lower:
                found_phrases.add(phrase)
        
        return words | found_phrases
    
    def _calculate_web_score(self, keywords: set) -> int:
        """
        Calculate a score indicating likelihood that Web search is needed.
        
        Args:
            keywords (set): Keywords from query
        
        Returns:
            int: Score (higher = more likely to use web)
        """
        web_matches = len(keywords & self.WEB_KEYWORDS)
        return web_matches * 2
    
    def _calculate_rag_score(self, keywords: set) -> int:
        """
        Calculate a score indicating likelihood that RAG is needed.
        
        Args:
            keywords (set): Keywords from query
        
        Returns:
            int: Score (higher = more likely to use RAG)
        """
        if not self.rag_database_exists:
            return 0
        
        rag_matches = len(keywords & self.RAG_KEYWORDS)
        return rag_matches * 2
    
    def route(self, query: str, force_web: bool = False) -> Literal["rag", "web", "hybrid"]:
        """
        Determine the best source for answering the query.
        
        Args:
            query (str): User's query
            force_web (bool): Force web search (override routing logic)
        
        Returns:
            Literal["rag", "web", "hybrid"]: Routing decision (rag, web, or hybrid)
        """
        print(f" Routing query: {query[:50]}...")
        
        # Force web if requested
        if force_web:
            print("  Forced to WEB source")
            return "web"
        
        # If no RAG database, default to web
        if not self.rag_database_exists:
            print("  No RAG database available, routing to WEB")
            return "web"
        
        # Extract and analyze keywords
        keywords = self._extract_keywords(query)
        web_score = self._calculate_web_score(keywords)
        rag_score = self._calculate_rag_score(keywords)
        
        print(f" Scores - WEB: {web_score}, RAG: {rag_score}")
        
        # Decision logic
        if web_score > rag_score + 2:
            print("  Routing to WEB (high confidence)")
            return "web"
        elif rag_score > web_score + 2:
            print("  Routing to RAG (high confidence)")
            return "rag"
        else:
            # Neutral or close scores - consider hybrid or default to RAG
            # For this version, we'll default to RAG if available
            print("  Routing to RAG (neutral/close scores)")
            return "rag"
    
    def should_use_web(self, query: str) -> bool:
        """
        Quick check if web search should be used.
        
        Args:
            query (str): User's query
        
        Returns:
            bool: True if web should be used
        """
        return self.route(query) == "web"
    
    def should_use_rag(self, query: str) -> bool:
        """
        Quick check if RAG should be used.
        
        Args:
            query (str): User's query
        
        Returns:
            bool: True if RAG should be used
        """
        return self.route(query) == "rag"


def create_router(rag_available: bool = False) -> QueryRouter:
    """
    Factory function to create a QueryRouter instance.
    
    Args:
        rag_available (bool): Whether RAG database is available
    
    Returns:
        QueryRouter: New router instance
    """
    return QueryRouter(rag_database_exists=rag_available)
