"""
Phase 4: Intelligent Agent (Brain of the System)
Autonomous decision-making layer that dynamically selects the best tool
for each query using intelligent reasoning.
"""

from typing import Dict, List, Literal, Optional, Any
from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime
import json
import os


class ToolType(Enum):
    """Available tools for the agent"""
    RAG = "rag"
    WEB_SEARCH = "web_search"
    DIRECT_LLM = "direct_llm"
    SPORTS = "sports_api"
    WEATHER = "weather_api"
    STOCKS = "stocks_api"
    CRYPTO = "crypto_api"
    NEWS = "news_api"
    MAPS = "maps_api"
    TIME = "time_api"
    BROWSER_ACTION = "browser_action"  # Open websites, perform searches


@dataclass
class QueryAnalysis:
    """Analysis result for a query"""
    query: str
    intent: str  # what user wants
    requires_realtime: bool  # needs current data?
    requires_document: bool  # references documents?
    complexity: Literal["simple", "moderate", "complex"]  # query complexity
    suggested_tool: ToolType
    confidence: float  # 0.0 to 1.0
    reasoning: str  # why this tool was chosen


@dataclass
class ConversationMessage:
    """Single message in conversation history"""
    role: Literal["user", "assistant"]
    content: str
    timestamp: datetime
    tool_used: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """Context for managing multi-turn conversations"""
    session_id: str
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    summary: str = ""
    
    def add_message(self, role: Literal["user", "assistant"], content: str, 
                   tool_used: Optional[str] = None, metadata: Dict[str, Any] = None):
        """Add message to conversation history"""
        msg = ConversationMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            tool_used=tool_used,
            metadata=metadata or {}
        )
        self.messages.append(msg)
        self.last_updated = datetime.now()
    
    def get_summary(self) -> str:
        """Get conversation summary for context"""
        if not self.messages:
            return ""
        
        summary_parts = []
        for msg in self.messages[-5:]:  # Last 5 messages
            summary_parts.append(f"{msg.role}: {msg.content[:100]}...")
        
        return "\n".join(summary_parts)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "message_count": len(self.messages),
            "summary": self.get_summary()
        }


class AgentBrain:
    """
    The intelligent brain of the AURA system.
    Makes autonomous decisions about which tool to use for each query.
    """
    
    # Keywords for real-time information needs
    REALTIME_KEYWORDS = {
        'latest', 'current', 'news', 'today', 'recent', 'breaking',
        'now', 'trend', 'trending', 'update', 'happening', 'real-time',
        'weather', 'stock', 'live', 'price', 'rate', 'score', 'status',
        'what\'s new', 'recent news', 'latest news', 'current events'
    }
    
    # Keywords for document-based queries
    DOCUMENT_KEYWORDS = {
        'document', 'pdf', 'uploaded', 'paper', 'chapter', 'section',
        'according to', 'in the document', 'in the pdf', 'from the file',
        'our document', 'the material', 'the content', 'here', 'previously',
        'reference', 'cite', 'mention', 'state', 'written', 'what is in',
        'that', 'it', 'summarize', 'summary', 'about what i uploaded',
        'resume', 'cv', 'file', 'attachment', 'candidate', 'profile', 'bio',
        'projects mentioned', 'skills mentioned', 'in the uploaded'
    }

    
    # Keywords indicating simple/direct questions
    SIMPLE_KEYWORDS = {
        'what is', 'define', 'explain', 'who is', 'where is', 'when is'
    }
    
    def __init__(self, rag_available: bool = False, max_memory: int = 100):
        """
        Initialize the Agent Brain.
        
        Args:
            rag_available (bool): Whether RAG database is available
            max_memory (int): Maximum conversation contexts to keep
        """
        self.rag_available = rag_available
        self.max_memory = max_memory
        self.conversation_history: Dict[str, ConversationContext] = {}
        self.decision_log: List[Dict] = []  # Track decisions made
        
        print(" Agent Brain initialized")
    
    def set_rag_availability(self, available: bool):
        """Update RAG availability status"""
        self.rag_available = available
    
    # ===========================
    # 1. QUERY ANALYSIS
    # ===========================
    def analyze_query(self, query: str) -> QueryAnalysis:
        """
        Analyze a query to understand intent and requirements.
        
        Args:
            query (str): User's query
        
        Returns:
            QueryAnalysis: Detailed analysis of the query
        """
        query_lower = query.lower()
        
        # Check for real-time information needs
        requires_realtime = any(keyword in query_lower for keyword in self.REALTIME_KEYWORDS)
        
        # Check for document references
        requires_document = any(keyword in query_lower for keyword in self.DOCUMENT_KEYWORDS)
        
        # Determine query complexity
        complexity = self._determine_complexity(query)
        
        # Extract intent
        intent = self._extract_intent(query)
        
        # Make tool selection
        tool, confidence, reasoning = self._select_tool(
            query_lower, 
            requires_realtime, 
            requires_document, 
            complexity
        )
        
        analysis = QueryAnalysis(
            query=query,
            intent=intent,
            requires_realtime=requires_realtime,
            requires_document=requires_document,
            complexity=complexity,
            suggested_tool=tool,
            confidence=confidence,
            reasoning=reasoning
        )
        
        print(f" Query Analysis:")
        print(f"   Intent: {intent}")
        print(f"   Requires Real-time: {requires_realtime}")
        print(f"   Requires Document: {requires_document}")
        print(f"   Complexity: {complexity}")
        print(f"   Suggested Tool: {tool.value} (confidence: {confidence:.1%})")
        print(f"   Reasoning: {reasoning}")
        
        return analysis
    
    def _determine_complexity(self, query: str) -> Literal["simple", "moderate", "complex"]:
        """Determine query complexity"""
        words = query.split()
        
        if len(words) < 5:
            return "simple"
        elif len(words) < 15:
            return "moderate"
        else:
            return "complex"
    
    def _extract_intent(self, query: str) -> str:
        """Extract the user's intent from query"""
        query_lower = query.lower()
        
        # Intent categories
        if any(word in query_lower for word in ['what', 'which', 'who', 'where', 'when', 'why', 'how']):
            return "information_seeking"
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus']):
            return "comparison"
        elif any(word in query_lower for word in ['list', 'show', 'give', 'provide']):
            return "enumeration"
        elif any(word in query_lower for word in ['explain', 'describe', 'tell']):
            return "explanation"
        elif any(word in query_lower for word in ['analyze', 'analyze', 'evaluate', 'assess']):
            return "analysis"
        else:
            return "general_knowledge"
    
    def _select_tool(self, query_lower: str, requires_realtime: bool, 
                     requires_document: bool, complexity: str) -> tuple:
        """
        Intelligently select the best tool for the query.
        
        Returns:
            Tuple of (ToolType, confidence, reasoning)
        """
        
        # PHASE 6+: Browser Automation (highest priority for action-oriented commands)
        # Check for "Open" commands
        if query_lower.startswith(("open ", "visit ", "go to ", "launch ", "browse ")):
            return ToolType.BROWSER_ACTION, 0.99, "User wants to open a website - routing to Browser Automation"
        
        # Check for "Search" commands
        if query_lower.startswith(("search ", "google ", "find ", "look for ")):
            return ToolType.BROWSER_ACTION, 0.99, "User wants to perform a search - routing to Browser Automation"
        
        # Phase 6: API-specific routing (highest priority for specific APIs)
        if "ipl" in query_lower or ("cricket" in query_lower and "score" in query_lower):
            return ToolType.SPORTS, 0.95, "Query is about IPL/cricket scores - routing to Sports API"
        
        elif "weather" in query_lower:
            return ToolType.WEATHER, 0.95, "Query is about weather - routing to Weather API"
        
        elif "stock" in query_lower or "share" in query_lower or "price" in query_lower and "stock" in query_lower:
            return ToolType.STOCKS, 0.95, "Query is about stock prices - routing to Stocks API"
        
        elif "bitcoin" in query_lower or "crypto" in query_lower or "ethereum" in query_lower:
            return ToolType.CRYPTO, 0.95, "Query is about cryptocurrency - routing to Crypto API"
        
        elif "news" in query_lower or "headline" in query_lower or "breaking" in query_lower:
            return ToolType.NEWS, 0.95, "Query is about news - routing to News API"
        
        elif "hospital" in query_lower or "nearby" in query_lower or "location" in query_lower or "map" in query_lower:
            return ToolType.MAPS, 0.90, "Query is about nearby places/locations - routing to Maps API"
        
        elif "time" in query_lower or "date" in query_lower or "what time" in query_lower or "current time" in query_lower:
            return ToolType.TIME, 0.95, "Query is about current date/time - routing to Time API"
        
        # Rule 1: If query explicitly references documents and RAG is available
        elif requires_document and self.rag_available:
            return ToolType.RAG, 0.95, "Query references documents and RAG database is available"
        
        # Rule 2: If real-time information is needed
        elif requires_realtime:
            return ToolType.WEB_SEARCH, 0.90, "Query requires current/real-time information"
        
        # Rule 3: If document keywords present but RAG not available, use web search as fallback
        elif requires_document and not self.rag_available:
            return ToolType.WEB_SEARCH, 0.75, "Document reference detected, but RAG unavailable. Using web search fallback"
        
        # Rule 4: Complex queries - use direct LLM for reasoning
        elif complexity == "complex":
            return ToolType.DIRECT_LLM, 0.85, "Complex query requires reasoning and synthesis"
        
        # Rule 5: Simple information questions - use web search for freshness
        if complexity == "simple" and any(keyword in query_lower for keyword in ['latest', 'current', 'new']):
            return ToolType.WEB_SEARCH, 0.80, "Simple query about current information"
        
        # Default: Use direct LLM for general knowledge
        return ToolType.DIRECT_LLM, 0.70, "General knowledge question - using LLM for comprehensive answer"
    
    # ===========================
    # 2. DECISION MAKING
    # ===========================
    def decide(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Make a decision about how to process the query.
        
        Args:
            query (str): User's query
            session_id (str, optional): Conversation session ID for context
        
        Returns:
            Decision dict with tool, reasoning, and context
        """
        print("\n" + "="*70)
        print(" AGENT BRAIN - DECISION MAKING PHASE")
        print("="*70)
        
        # Analyze the query
        analysis = self.analyze_query(query)
        
        # Get conversation context if available
        context = None
        if session_id and session_id in self.conversation_history:
            context = self.conversation_history[session_id]
        
        # Log the decision
        decision = {
            "query": query,
            "session_id": session_id,
            "tool": analysis.suggested_tool.value,
            "confidence": analysis.confidence,
            "intent": analysis.intent,
            "requires_realtime": analysis.requires_realtime,
            "requires_document": analysis.requires_document,
            "complexity": analysis.complexity,
            "reasoning": analysis.reasoning,
            "timestamp": datetime.now().isoformat(),
            "conversation_context": context.to_dict() if context else None
        }
        
        self.decision_log.append(decision)
        
        print(f"\n Decision made: Use {analysis.suggested_tool.value.upper()}")
        print(f"{'='*70}\n")
        
        return decision
    
    # ===========================
    # 3. CONVERSATION MANAGEMENT
    # ===========================
    def get_or_create_session(self, session_id: str) -> ConversationContext:
        """Get existing session or create new one"""
        if session_id not in self.conversation_history:
            self.conversation_history[session_id] = ConversationContext(session_id=session_id)
            # Clean up old sessions if memory limit exceeded
            if len(self.conversation_history) > self.max_memory:
                oldest_key = min(self.conversation_history.keys(), 
                               key=lambda k: self.conversation_history[k].created_at)
                del self.conversation_history[oldest_key]
        
        return self.conversation_history[session_id]
    
    def add_to_history(self, session_id: str, role: Literal["user", "assistant"], 
                      content: str, tool_used: Optional[str] = None,
                      metadata: Optional[Dict] = None):
        """Add message to conversation history"""
        context = self.get_or_create_session(session_id)
        context.add_message(role, content, tool_used, metadata)
    
    def get_session_context(self, session_id: str) -> str:
        """Get conversation context for LLM"""
        if session_id not in self.conversation_history:
            return ""
        
        context = self.conversation_history[session_id]
        return context.get_summary()
    
    # ===========================
    # 4. AGENT WORKFLOW
    # ===========================
    def execute_workflow(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute the complete agent workflow.
        
        Args:
            query (str): User's query
            session_id (str, optional): Conversation session
        
        Returns:
            Workflow result with decision and next steps
        """
        # Step 1: Make decision
        decision = self.decide(query, session_id)
        
        # Step 2: Store user query in history
        if session_id:
            self.add_to_history(session_id, "user", query)
        
        # Step 3: Prepare execution context
        execution_context = {
            "decision": decision,
            "conversation_context": self.get_session_context(session_id) if session_id else "",
            "execution_ready": True
        }
        
        return execution_context
    
    # ===========================
    # 5. REPORTING & DIAGNOSTICS
    # ===========================
    def get_decision_statistics(self) -> Dict[str, Any]:
        """Get statistics about agent decisions"""
        if not self.decision_log:
            return {"total_decisions": 0}
        
        tool_usage = {}
        intent_distribution = {}
        complexity_distribution = {}
        
        for decision in self.decision_log:
            tool = decision["tool"]
            intent = decision["intent"]
            complexity = decision["complexity"]
            
            tool_usage[tool] = tool_usage.get(tool, 0) + 1
            intent_distribution[intent] = intent_distribution.get(intent, 0) + 1
            complexity_distribution[complexity] = complexity_distribution.get(complexity, 0) + 1
        
        avg_confidence = sum(d["confidence"] for d in self.decision_log) / len(self.decision_log)
        
        return {
            "total_decisions": len(self.decision_log),
            "average_confidence": avg_confidence,
            "tool_usage": tool_usage,
            "intent_distribution": intent_distribution,
            "complexity_distribution": complexity_distribution,
            "rag_available": self.rag_available,
            "active_sessions": len(self.conversation_history),
            "decision_log_size": len(self.decision_log)
        }
    
    def export_conversation(self, session_id: str) -> Dict[str, Any]:
        """Export conversation for analysis or backup"""
        if session_id not in self.conversation_history:
            return None
        
        context = self.conversation_history[session_id]
        return {
            "session_id": session_id,
            "created_at": context.created_at.isoformat(),
            "last_updated": context.last_updated.isoformat(),
            "message_count": len(context.messages),
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "tool_used": msg.tool_used,
                    "metadata": msg.metadata
                }
                for msg in context.messages
            ]
        }
    
    def print_decision_log(self, limit: int = 10):
        """Print recent decisions for debugging"""
        print("\n" + "="*70)
        print(" AGENT DECISION LOG (Recent)")
        print("="*70)
        
        for i, decision in enumerate(self.decision_log[-limit:], 1):
            print(f"\n[{i}] {decision['timestamp']}")
            print(f"    Query: {decision['query'][:50]}...")
            print(f"    Tool: {decision['tool']} (confidence: {decision['confidence']:.1%})")
            print(f"    Intent: {decision['intent']}")
            print(f"    Reasoning: {decision['reasoning']}")


# ===========================
# SINGLETON INSTANCE
# ===========================
_agent_brain = None


def get_agent_brain(rag_available: bool = False) -> AgentBrain:
    """Get or create singleton agent brain instance"""
    global _agent_brain
    if _agent_brain is None:
        _agent_brain = AgentBrain(rag_available=rag_available)
    return _agent_brain
