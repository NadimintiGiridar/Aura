import sys
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from llm import query_llm, query_llm_with_rag, query_llm_with_web, query_llm_with_web_fallback, query_llm_coding, query_llm_normal
from coding_mode import detect_mode, detect_coding_intent, get_coding_mode_info
from rag import process_pdf, load_vector_database, generate_rag_context, check_database_exists, delete_vector_database
from web_search import get_web_searcher
from router import create_router
from agent import get_agent_brain, ToolType
from web_automation import get_web_automation_manager
from realtime_tools import (
    get_weather, get_cricket_score, get_stock, get_crypto, get_news, get_datetime,
    extract_city_from_query, extract_stock_symbol_from_query, extract_crypto_from_query,
    extract_news_query_from_query
)

# Load environment variables
load_dotenv()

# API Keys from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CRIC_API_KEY = os.getenv("CRIC_API_KEY")
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Initialize FastAPI app
app = FastAPI(title="AURA", description="AI-powered User Response Assistant (Phase 7)")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# PHASE 7: AUTH + POSTGRESQL ROUTERS
# ==============================
try:
    import sys as _sys
    import os as _os
    _backend_dir = _os.path.dirname(_os.path.abspath(__file__))
    if _backend_dir not in _sys.path:
        _sys.path.insert(0, _backend_dir)

    from app.api.auth import router as auth_router
    from app.api.conversations import router as conversations_router
    from app.api.messages import router as messages_router
    from app.api.documents import router as documents_router
    from app.database.connection import create_all_tables

    app.include_router(auth_router)
    app.include_router(conversations_router)
    app.include_router(messages_router)
    app.include_router(documents_router)

    # Create PostgreSQL tables on startup
    @app.on_event("startup")
    async def startup_event():
        try:
            create_all_tables()
            print("[OK] PostgreSQL tables initialized")
        except Exception as e:
            print(f"[WARNING] PostgreSQL not available: {e}")
            print("   Backend still functional with existing AI endpoints.")

    print("[OK] Phase 7 (Auth + PostgreSQL) routers registered")
except Exception as _e:
    print(f"[WARNING] Phase 7 routers could not be loaded: {_e}")
    print("   Existing AURA endpoints remain fully operational.")

# Configuration
UPLOAD_DIR = "./uploads"
DB_PATH = "./faiss_index"

# Create uploads directory if it doesn't exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Define request model for POST endpoint
class QueryRequest(BaseModel):
    query: str

# Define request model for RAG query
class RAGQueryRequest(BaseModel):
    query: str

# Define request model for Smart Query (Phase 3)
class SmartQueryRequest(BaseModel):
    query: str
    force_web: bool = False  # Optional: Force web search instead of auto-routing

# Define request model for Agent Query (Phase 4)
class AgentQueryRequest(BaseModel):
    query: str
    session_id: str = None  # Optional: For conversation context
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the latest AI trends?",
                "session_id": "user_123"
            }
        }

# Define request model for Unified AURA Query (Phase 5)
class UnifiedAuraRequest(BaseModel):
    query: str
    session_id: str = None  # Optional: For conversation context
    mode: str = "intelligent"  # "intelligent" (agent decides) or specific tool: "rag", "web", "llm"
    include_analysis: bool = True  # Include agent analysis in response
    
    class Config:
        json_schema_extra = {
            "example": {
                "query": "What are the latest AI trends?",
                "session_id": "user_123",
                "mode": "intelligent",
                "include_analysis": True
            }
        }

# Root endpoint - Check if backend is running
@app.get("/")
def read_root():
    """
    Root endpoint to verify the backend is running.
    """
    return {
        "message": "AURA Backend is running!",
        "status": "active"
    }

# GET endpoint - Chat with query parameter
@app.get("/chat")
def chat_get(query: str):
    """
    Chat endpoint using GET method.
    
    Args:
        query (str): The user's query
    
    Returns:
        JSONResponse: Contains the LLM response
    """
    if not query:
        return JSONResponse(
            status_code=400,
            content={"error": "Query parameter is required"}
        )
    
    response = query_llm(query)
    return {"response": response}

# POST endpoint - Chat with JSON body (recommended for production)
@app.post("/chat")
def chat_post(request: QueryRequest):
    """
    Chat endpoint using POST method (recommended for better scalability).
    
    Args:
        request (QueryRequest): JSON body containing the query
    
    Returns:
        JSONResponse: Contains the LLM response
    """
    if not request.query:
        return JSONResponse(
            status_code=400,
            content={"error": "Query field is required"}
        )
    
    response = query_llm(request.query)
    return {"response": response}

# ===== RAG ENDPOINTS =====

# POST endpoint - Upload and process PDF
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Upload and process a PDF file for RAG.
    Creates embeddings and stores them in FAISS vector database.
    
    Args:
        file (UploadFile): PDF file to upload
    
    Returns:
        JSONResponse: Status of PDF processing
    """
    try:
        if not file.filename.endswith('.pdf'):
            return JSONResponse(
                status_code=400,
                content={"error": "Only PDF files are allowed"}
            )
        
        # Save uploaded file
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as f:
            contents = await file.read()
            f.write(contents)
        
        # Process PDF with RAG pipeline
        success, message, chunk_count = process_pdf(file_path, DB_PATH)
        
        if success:
            return {
                "status": "success",
                "message": message,
                "file_name": file.filename,
                "database_path": DB_PATH
            }
        else:
            return JSONResponse(
                status_code=400,
                content={"error": message}
            )
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error processing PDF: {str(e)}"}
        )

# POST endpoint - Query with RAG context
@app.post("/query-rag")
def query_rag(request: RAGQueryRequest):
    """
    Query the system using RAG (retrieval-augmented generation).
    Retrieves relevant document context and generates answer based on it.
    
    Args:
        request (RAGQueryRequest): JSON body containing the query
    
    Returns:
        JSONResponse: LLM response with retrieved context
    """
    try:
        if not request.query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query field is required"}
            )
        
        # Check if vector database exists
        if not check_database_exists(DB_PATH):
            return JSONResponse(
                status_code=400,
                content={"error": "No PDF loaded yet. Please upload a PDF first using /upload-pdf endpoint."}
            )
        
        # Load vector database
        vector_store = load_vector_database(DB_PATH)
        if vector_store is None:
            return JSONResponse(
                status_code=500,
                content={"error": "Failed to load vector database"}
            )
        
        # Generate context from retrieved documents
        context, _ = generate_rag_context(request.query, vector_store, k=4)
        
        # Query LLM with context
        response = query_llm_with_rag(request.query, context)
        
        return {
            "response": response,
            "context_used": True,
            "database": DB_PATH
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error querying RAG: {str(e)}"}
        )

# GET endpoint - Check database status
@app.get("/rag-status")
def rag_status():
    """
    Check if RAG vector database is initialized and ready.
    
    Returns:
        JSONResponse: Status of RAG system
    """
    exists = check_database_exists(DB_PATH)
    return {
        "database_initialized": exists,
        "database_path": DB_PATH,
        "status": "ready" if exists else "no PDF loaded"
    }

# DELETE endpoint - Clear database
@app.delete("/clear-database")
def clear_database():
    """
    Delete the FAISS vector database.
    Useful for starting over with a new PDF.
    
    Returns:
        JSONResponse: Status of deletion
    """
    try:
        success, message = delete_vector_database(DB_PATH)
        if success:
            return {"status": "success", "message": message}
        else:
            return JSONResponse(
                status_code=400,
                content={"error": message}
            )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error clearing database: {str(e)}"}
        )

# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


# ==============================
# PHASE 3: SMART QUERY ENDPOINT
# ==============================

# Initialize router and web searcher
query_router = None
web_searcher = None

def initialize_phase3():
    """Initialize Phase 3 components"""
    global query_router, web_searcher
    rag_available = check_database_exists(DB_PATH)
    query_router = create_router(rag_available=rag_available)
    web_searcher = get_web_searcher(max_results=5)
    print("[OK] Phase 3 (Smart Query System) initialized")

# Initialize on startup
initialize_phase3()

# ==============================
# PHASE 4: INTELLIGENT AGENT
# ==============================

# Initialize agent brain
agent_brain = None

def initialize_phase4():
    """Initialize Phase 4 components"""
    global agent_brain
    rag_available = check_database_exists(DB_PATH)
    agent_brain = get_agent_brain(rag_available=rag_available)
    print("[OK] Phase 4 (Intelligent Agent Brain) initialized")

# Initialize on startup
initialize_phase4()

# POST endpoint - Smart Query with Automatic Routing (Phase 3)
@app.post("/smart-query")
def smart_query(request: SmartQueryRequest):
    """
    Smart Query endpoint that automatically routes queries to RAG or Web Search.
    Implements Phase 3 functionality with intelligent decision-making.
    
    This endpoint:
    1. Analyzes the user's query
    2. Decides whether to use document-based knowledge (RAG) or web search
    3. Retrieves relevant information from the selected source
    4. Generates response using LLM
    5. Returns response with source indication
    
    Args:
        request (SmartQueryRequest): JSON body containing:
            - query (str): User's question
            - force_web (bool, optional): Force web search (default: False)
    
    Returns:
        JSONResponse: Response with content and source information
    """
    try:
        if not request.query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query field is required"}
            )
        
        print(f"\n{'='*60}")
        print(f" SMART QUERY REQUEST: {request.query}")
        print(f"{'='*60}")
        
        # Update router with current RAG status
        rag_available = check_database_exists(DB_PATH)
        query_router.set_rag_availability(rag_available)
        
        # Decide routing
        routing_decision = query_router.route(request.query, force_web=request.force_web)
        
        response_data = {
            "query": request.query,
            "routing_decision": routing_decision,
            "response": None,
            "source": None,
            "context": None,
            "status": "success"
        }
        
        # Execute based on routing decision
        if routing_decision == "web":
            print("\n[WEB] EXECUTING: Web Search")
            
            # Perform web search
            print(f"Searching for: {request.query}")
            web_results = web_searcher.search(request.query)
            
            if not web_results:
                # Even if no results, provide fallback response using LLM training data
                print("[WARNING] No web results - using LLM fallback response with training data")
                web_context = "No live web results available, but here is relevant information based on training data:"
                llm_response = query_llm_with_web_fallback(request.query, web_context)
                response_data["response"] = llm_response
                response_data["source"] = "web"
                response_data["context"] = "Some relevant web sources were found but content extraction failed."
            else:
                # Format web results for context
                web_context = web_searcher.generate_web_context(request.query, web_results)
                print(f"[OK] Found {len(web_results)} web results")
                
                # Query Groq LLM with web context
                llm_response = query_llm_with_web(request.query, web_context)
                response_data["response"] = llm_response
                response_data["source"] = "web"
                response_data["context"] = web_context
        
        elif routing_decision == "rag":
            print("\n EXECUTING: RAG (Document-based)")
            
            if not rag_available:
                response_data["response"] = "No PDF documents are currently loaded. Please upload a PDF using /upload-pdf endpoint to enable document-based queries."
                response_data["source"] = "none"
                response_data["status"] = "rag_unavailable"
            else:
                # Load vector database
                vector_store = load_vector_database(DB_PATH)
                if vector_store is None:
                    response_data["response"] = "Failed to load the document database."
                    response_data["source"] = "none"
                    response_data["status"] = "error"
                else:
                    # Generate RAG context
                    rag_context, _ = generate_rag_context(request.query, vector_store, k=4)
                    
                    # Generate response using LLM with RAG context
                    rag_response = query_llm_with_rag(request.query, rag_context)
                    response_data["response"] = rag_response
                    response_data["source"] = "rag"
                    response_data["context"] = rag_context
        
        print(f"\n[OK] SOURCE USED: {response_data['source'].upper()}")
        print(f"{'='*60}\n")
        
        return response_data
    
    except Exception as e:
        print(f"[ERROR] Error in smart query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error processing smart query: {str(e)}",
                "status": "error"
            }
        )

# GET endpoint - Smart Query status and information
@app.get("/smart-query-info")
def smart_query_info():
    """
    Get information about Phase 3 Smart Query system capabilities and current status.
    
    Returns:
        JSONResponse: System information including available sources and configuration
    """
    rag_available = check_database_exists(DB_PATH)
    
    return {
        "system": "AURA Phase 3 - Smart Query System",
        "description": "Intelligent routing between RAG (documents) and Web Search",
        "status": "active",
        "sources_available": {
            "rag": rag_available,
            "web": True
        },
        "rag_status": {
            "database_initialized": rag_available,
            "database_path": DB_PATH
        },
        "endpoints": {
            "smart_query": "/smart-query (POST)",
            "upload_pdf": "/upload-pdf (POST)",
            "rag_status": "/rag-status (GET)",
            "clear_database": "/clear-database (DELETE)"
        },
        "routing_logic": "Keyword-based decision engine",
        "supported_queries": {
            "document_based": "Questions about uploaded PDFs",
            "web_based": "Current news, latest updates, trending topics",
            "hybrid": "System intelligently selects the best source"
        }
    }


# ==============================
# PHASE 4: INTELLIGENT AGENT ENDPOINTS
# ==============================

# POST endpoint - Agent Query with Autonomous Decision-Making
@app.post("/agent-query")
def agent_query(request: AgentQueryRequest):
    """
    Query using the intelligent agent with autonomous decision-making.
    Phase 4 endpoint that uses the agent brain to decide the best tool.
    
    The agent analyzes:
    - Query intent
    - Real-time information needs
    - Document references
    - Query complexity
    
    Then selects the appropriate tool (RAG, Web Search, or Direct LLM).
    
    Args:
        request (AgentQueryRequest): JSON body containing:
            - query (str): User's question
            - session_id (str, optional): For conversation context
    
    Returns:
        JSONResponse: Response with decision reasoning and answer
    """
    try:
        if not request.query:
            return JSONResponse(
                status_code=400,
                content={"error": "Query field is required"}
            )
        
        # Generate session ID if not provided
        session_id = request.session_id or str(uuid.uuid4())
        
        print(f"\n{'='*70}")
        print(f"[AI] AGENT QUERY REQUEST")
        print(f"Query: {request.query}")
        print(f"Session ID: {session_id}")
        print(f"{'='*70}")
        
        # Update agent brain with current RAG status
        rag_available = check_database_exists(DB_PATH)
        agent_brain.set_rag_availability(rag_available)
        
        # Execute agent workflow
        execution_context = agent_brain.execute_workflow(request.query, session_id)
        decision = execution_context["decision"]
        tool = decision["tool"]
        
        response_data = {
            "query": request.query,
            "session_id": session_id,
            "agent_decision": {
                "tool": tool,
                "confidence": decision["confidence"],
                "intent": decision["intent"],
                "requires_realtime": decision["requires_realtime"],
                "requires_document": decision["requires_document"],
                "complexity": decision["complexity"],
                "reasoning": decision["reasoning"]
            },
            "response": None,
            "context": None,
            "status": "success"
        }
        
        # Execute based on agent decision
        if tool == "web_search":
            print(f"\n[WEB] EXECUTING: Web Search (Agent Decision)")
            web_results = web_searcher.search(request.query)
            
            if not web_results:
                print("[WARNING] No web results - using fallback")
                llm_response = query_llm_with_web_fallback(request.query, "No live results available")
                response_data["context"] = "Some relevant web sources were found but content extraction failed."
            else:
                print(f"[OK] Found {len(web_results)} web results")
                web_context = web_searcher.generate_web_context(request.query, web_results)
                llm_response = query_llm_with_web(request.query, web_context)
                response_data["context"] = web_context
            
            response_data["response"] = llm_response
            agent_brain.add_to_history(session_id, "assistant", llm_response, tool_used="web_search")
        
        elif tool == "rag":
            print(f"\n EXECUTING: RAG (Agent Decision)")
            
            if not rag_available:
                response_data["response"] = "No PDF documents are loaded. Please upload a PDF first."
                response_data["status"] = "rag_unavailable"
            else:
                vector_store = load_vector_database(DB_PATH)
                if vector_store is None:
                    response_data["response"] = "Failed to load document database."
                    response_data["status"] = "error"
                else:
                    rag_context, _ = generate_rag_context(request.query, vector_store, k=4)
                    rag_response = query_llm_with_rag(request.query, rag_context)
                    response_data["response"] = rag_response
                    response_data["context"] = rag_context
                    agent_brain.add_to_history(session_id, "assistant", rag_response, tool_used="rag")
        
        elif tool == "direct_llm":
            print(f"\n EXECUTING: Direct LLM (Agent Decision)")
            
            # Use direct LLM for reasoning
            conversation_context = agent_brain.get_session_context(session_id)
            if conversation_context:
                prompt = f"Previous context:\n{conversation_context}\n\nCurrent query: {request.query}"
            else:
                prompt = request.query
            
            llm_response = query_llm(prompt)
            response_data["response"] = llm_response
            response_data["context"] = "Direct LLM - No external tools required"
            agent_brain.add_to_history(session_id, "assistant", llm_response, tool_used="direct_llm")
        
        print(f"\n[OK] AGENT EXECUTION COMPLETE")
        print(f"Tool: {tool}")
        print(f"{'='*70}\n")
        
        return response_data
    
    except Exception as e:
        print(f"[ERROR] Error in agent query: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error processing agent query: {str(e)}",
                "status": "error"
            }
        )


# GET endpoint - Agent Analysis (Debug Tool)
@app.get("/agent-analyze")
def agent_analyze(query: str):
    """
    Analyze a query using the agent brain without executing it.
    Useful for debugging and understanding agent decisions.
    
    Args:
        query (str): Query to analyze
    
    Returns:
        JSONResponse: Analysis results with reasoning
    """
    try:
        print(f"\n[STATS] AGENT ANALYSIS: {query}")
        
        # Update RAG status
        rag_available = check_database_exists(DB_PATH)
        agent_brain.set_rag_availability(rag_available)
        
        # Analyze without executing
        analysis = agent_brain.analyze_query(query)
        
        return {
            "query": query,
            "analysis": {
                "intent": analysis.intent,
                "requires_realtime": analysis.requires_realtime,
                "requires_document": analysis.requires_document,
                "complexity": analysis.complexity,
                "suggested_tool": analysis.suggested_tool.value,
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning
            },
            "rag_status": {
                "available": rag_available,
                "database_path": DB_PATH
            }
        }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"Error analyzing query: {str(e)}"}
        )


# GET endpoint - Agent Statistics
@app.get("/agent-stats")
def agent_stats():
    """
    Get statistics about agent decisions and performance.
    
    Returns:
        JSONResponse: Agent statistics and metrics
    """
    stats = agent_brain.get_decision_statistics()
    rag_available = check_database_exists(DB_PATH)
    
    return {
        "system": "AURA Phase 4 - Intelligent Agent",
        "statistics": stats,
        "rag_status": rag_available,
        "active_sessions": len(agent_brain.conversation_history),
        "description": "Agent brain tracks all decisions, tool usage, and conversation contexts"
    }


# GET endpoint - Agent Info
@app.get("/agent-info")
def agent_info():
    """
    Get information about the Phase 4 Intelligent Agent system.
    
    Returns:
        JSONResponse: System information and capabilities
    """
    rag_available = check_database_exists(DB_PATH)
    
    return {
        "system": "AURA Phase 4 - Intelligent Agent Brain",
        "description": "Autonomous decision-making layer that selects tools intelligently",
        "status": "active",
        "capabilities": {
            "decision_making": "Autonomous tool selection based on query analysis",
            "conversation_management": "Multi-turn conversation with context awareness",
            "tool_access": ["rag", "web_search", "direct_llm"],
            "query_analysis": [
                "Intent detection",
                "Real-time information needs",
                "Document reference detection",
                "Complexity assessment"
            ]
        },
        "endpoints": {
            "agent_query": "/agent-query (POST) - Main agent endpoint",
            "agent_analyze": "/agent-analyze (GET) - Analyze query without executing",
            "agent_stats": "/agent-stats (GET) - Get agent statistics",
            "agent_info": "/agent-info (GET) - Get system information"
        },
        "decision_factors": {
            "real_time_keywords": "latest, current, news, trending, live, etc.",
            "document_keywords": "document, pdf, uploaded, paper, chapter, etc.",
            "complexity_levels": "simple, moderate, complex",
            "tool_selection_logic": "Intelligent multi-factor analysis with confidence scoring"
        },
        "rag_availability": rag_available,
        "conversation_support": True,
        "session_management": True,
        "examples": {
            "real_time_query": "What are the latest AI developments?",
            "document_query": "What does the PDF say about AI?",
            "complex_query": "Compare AI approaches and their real-world applications based on current trends"
        }
    }


# POST endpoint - Session Management (Create/Get)
@app.post("/agent-session")
def agent_session(session_id: str = None):
    """
    Create or retrieve a conversation session.
    
    Args:
        session_id (str, optional): Existing session ID
    
    Returns:
        JSONResponse: Session information
    """
    if not session_id:
        session_id = str(uuid.uuid4())
    
    context = agent_brain.get_or_create_session(session_id)
    
    return {
        "session_id": session_id,
        "created_at": context.created_at.isoformat(),
        "message_count": len(context.messages),
        "status": "active"
    }


# GET endpoint - Session Conversation History
@app.get("/agent-session/{session_id}")
def get_session_history(session_id: str):
    """
    Get conversation history for a session.
    
    Args:
        session_id (str): Session ID
    
    Returns:
        JSONResponse: Conversation history and metadata
    """
    export = agent_brain.export_conversation(session_id)
    
    if not export:
        return JSONResponse(
            status_code=404,
            content={"error": f"Session {session_id} not found"}
        )
    
    return export


# ==============================
# PHASE 5: UNIFIED SYSTEM INTEGRATION
# ==============================

# POST endpoint - Unified AURA Endpoint (Main Integration Point)
@app.post("/aura")
async def unified_aura_query(request: UnifiedAuraRequest):
    """
    PHASE 5: UNIFIED AI ASSISTANT
    
    The main integration point combining all AURA components into one intelligent assistant.
    
    This endpoint:
    1. Accepts a user query
    2. Uses the intelligent agent to analyze the query
    3. Agent selects the optimal tool (RAG, Web Search, or Direct LLM)
    4. Executes the selected tool with full context
    5. Generates comprehensive response using Groq LLM
    6. Returns structured response with analysis and context
    
    The system can work in two modes:
    - "intelligent" (default): Agent decides which tool to use
    - Specific tool mode: "rag", "web", "llm" to force a specific tool
    
    Args:
        request (UnifiedAuraRequest): JSON body containing:
            - query (str): User's question
            - session_id (str, optional): For multi-turn conversations
            - mode (str, optional): "intelligent" or specific tool name
            - include_analysis (bool, optional): Include decision reasoning
    
    Returns:
        JSONResponse: Comprehensive response with:
            - response: The AI's answer
            - tool_used: Which tool was used
            - confidence: Agent's confidence in the decision (0.0-1.0)
            - analysis: Query analysis if requested
            - context: Source context used
            - session_id: For tracking conversations
    """
    try:
        if not request.query or request.query.strip() == "":
            return JSONResponse(
                status_code=400,
                content={"error": "Query field is required and cannot be empty"}
            )
        
        # Generate or use existing session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        print(f"\n{'='*80}")
        print(f"[LAUNCH] PHASE 5 - UNIFIED AURA QUERY")
        print(f"Query: {request.query}")
        print(f"Session ID: {session_id}")
        print(f"Mode: {request.mode}")
        print(f"{'='*80}")
        
        # Update agent brain with current RAG status
        rag_available = check_database_exists(DB_PATH)
        agent_brain.set_rag_availability(rag_available)
        
        # Determine tool to use
        tool_to_use = None
        analysis = None
        
        if request.mode == "intelligent":
            # Let agent decide
            print("\n[BRAIN] AGENT DECISION MODE - Analyzing query...")
            analysis = agent_brain.analyze_query(request.query)
            tool_to_use = analysis.suggested_tool
            confidence = analysis.confidence
            intent = analysis.intent
            reasoning = analysis.reasoning
        else:
            # Use specified tool
            print(f"\n FORCED MODE - Using {request.mode}")
            mode_to_tool = {
                "rag": ToolType.RAG,
                "web": ToolType.WEB_SEARCH,
                "llm": ToolType.DIRECT_LLM
            }
            tool_to_use = mode_to_tool.get(request.mode, ToolType.DIRECT_LLM)
            
            # Create analysis object for forced mode
            if analysis is None:
                analysis = agent_brain.analyze_query(request.query)
                confidence = 0.95  # High confidence when forced
                intent = "user_specified"
                reasoning = f"User explicitly requested {request.mode} tool"
        
        # Build response structure
        response_data = {
            "query": request.query,
            "session_id": session_id,
            "tool_used": tool_to_use.value,
            "confidence": confidence if request.mode == "intelligent" else 0.95,
            "response": None,
            "context": None,
            "source": None,
            "status": "success"
        }
        
        # Add analysis if requested
        if request.include_analysis:
            response_data["analysis"] = {
                "intent": intent if request.mode == "intelligent" else "user_specified",
                "reasoning": reasoning if request.mode == "intelligent" else f"User explicitly requested {request.mode}",
                "requires_realtime": analysis.requires_realtime,
                "requires_document": analysis.requires_document,
                "complexity": analysis.complexity
            }
        
        # Execute selected tool
        print(f"\n[AURA] EXECUTING: {tool_to_use.value.upper()}")
        
        if tool_to_use == ToolType.WEB_SEARCH:
            # Execute Web Search
            print("[WEB] Web Search Execution")
            web_results = web_searcher.search(request.query)
            
            if web_results:
                print(f"[OK] Found {len(web_results)} web results")
                web_context = web_searcher.generate_web_context(request.query, web_results)
                llm_response = query_llm_with_web(request.query, web_context)
                response_data["response"] = llm_response
                response_data["context"] = web_context
                response_data["source"] = "web_search"
            else:
                # Fallback when no web results
                print("[WARNING] No web results - using fallback response")
                llm_response = query_llm_with_web_fallback(request.query, "No live results available")
                response_data["response"] = llm_response
                response_data["context"] = "Fallback: Using training data"
                response_data["source"] = "web_search_fallback"
            
            # Add to conversation history
            agent_brain.add_to_history(session_id, "assistant", llm_response, tool_used="web_search")
        
        elif tool_to_use == ToolType.RAG:
            # Execute RAG (Document-based)
            print(" RAG Execution")
            
            if not rag_available:
                response_data["response"] = "No PDF documents are currently loaded. Please upload a PDF using /upload-pdf endpoint to enable document-based queries."
                response_data["source"] = "none"
                response_data["status"] = "rag_unavailable"
            else:
                vector_store = load_vector_database(DB_PATH)
                if vector_store is None:
                    response_data["response"] = "Failed to load document database."
                    response_data["source"] = "none"
                    response_data["status"] = "error"
                else:
                    # Generate RAG context
                    rag_context, _ = generate_rag_context(request.query, vector_store, k=4)
                    rag_response = query_llm_with_rag(request.query, rag_context)
                    response_data["response"] = rag_response
                    response_data["context"] = rag_context
                    response_data["source"] = "rag"
                    
                    # Add to conversation history
                    agent_brain.add_to_history(session_id, "assistant", rag_response, tool_used="rag")
        
        # ===== PHASE 6: REAL-TIME APIs =====
        elif tool_to_use == ToolType.WEATHER:
            # Execute Weather API
            print("[WEATHER] Weather API Execution")
            city = extract_city_from_query(request.query) or "Chennai"  # Default city
            
            if not WEATHER_API_KEY:
                response_data["response"] = "[ERROR] Weather API key not configured. Please add WEATHER_API_KEY to .env file."
                response_data["status"] = "error"
            else:
                weather_data = get_weather(city, WEATHER_API_KEY)
                response_data["response"] = weather_data
                response_data["context"] = f"Weather data for {city}"
                response_data["source"] = "weather_api"
                agent_brain.add_to_history(session_id, "assistant", weather_data, tool_used="weather_api")
        
        elif tool_to_use == ToolType.SPORTS:
            # Execute Cricket/Sports API
            print("[SPORTS] Cricket API Execution")
            
            if not CRIC_API_KEY:
                response_data["response"] = "[ERROR] Cricket API key not configured. Please add CRIC_API_KEY to .env file."
                response_data["status"] = "error"
            else:
                cricket_data = get_cricket_score(CRIC_API_KEY)
                response_data["response"] = cricket_data
                response_data["context"] = "Live cricket/IPL scores"
                response_data["source"] = "sports_api"
                agent_brain.add_to_history(session_id, "assistant", cricket_data, tool_used="sports_api")
        
        elif tool_to_use == ToolType.STOCKS:
            # Execute Stock Price API
            print("[STOCKS] Stock API Execution")
            symbol = extract_stock_symbol_from_query(request.query)
            
            stock_data = get_stock(symbol)
            response_data["response"] = stock_data
            response_data["context"] = f"Stock price for {symbol}"
            response_data["source"] = "stocks_api"
            agent_brain.add_to_history(session_id, "assistant", stock_data, tool_used="stocks_api")
        
        elif tool_to_use == ToolType.CRYPTO:
            # Execute Crypto API
            print(" Crypto API Execution")
            crypto_id = extract_crypto_from_query(request.query)
            
            crypto_data = get_crypto(crypto_id)
            response_data["response"] = crypto_data
            response_data["context"] = f"Cryptocurrency price for {crypto_id}"
            response_data["source"] = "crypto_api"
            agent_brain.add_to_history(session_id, "assistant", crypto_data, tool_used="crypto_api")
        
        elif tool_to_use == ToolType.NEWS:
            # Execute News API
            print("[NEWS] News API Execution")
            
            if not NEWS_API_KEY:
                response_data["response"] = "[ERROR] News API key not configured. Please add NEWS_API_KEY to .env file."
                response_data["status"] = "error"
            else:
                # Extract news query from user's query using helper function
                news_query = extract_news_query_from_query(request.query)
                
                print(f"[NEWS] News Query: '{news_query}'")
                
                news_data = get_news(NEWS_API_KEY, query=news_query)
                response_data["response"] = news_data
                response_data["context"] = f"Real-time news articles about '{news_query}'"
                response_data["source"] = "news_api"
                agent_brain.add_to_history(session_id, "assistant", news_data, tool_used="news_api")
        
        elif tool_to_use == ToolType.TIME:
            # Execute Time API (built-in)
            print("[TIME] Time API Execution")
            time_data = get_datetime()
            response_data["response"] = time_data
            response_data["context"] = "Current date and time"
            response_data["source"] = "time_api"
            agent_brain.add_to_history(session_id, "assistant", time_data, tool_used="time_api")
        
        elif tool_to_use == ToolType.MAPS:
            # Execute Maps API (placeholder)
            print(" Maps API Execution")
            maps_data = " **Maps API** - Location-based services coming soon!\n\nFor now, please use web search for nearby places."
            response_data["response"] = maps_data
            response_data["context"] = "Maps/location services"
            response_data["source"] = "maps_api"
            agent_brain.add_to_history(session_id, "assistant", maps_data, tool_used="maps_api")
        
        elif tool_to_use == ToolType.DIRECT_LLM:
            # Execute Direct LLM
            print(" Direct LLM Execution")
            
            # Get conversation context if multi-turn
            conversation_context = agent_brain.get_session_context(session_id)
            if conversation_context and len(conversation_context.strip()) > 0:
                prompt = f"Previous conversation context:\n{conversation_context}\n\nCurrent query: {request.query}"
            else:
                prompt = request.query
            
            llm_response = query_llm(prompt)
            response_data["response"] = llm_response
            response_data["context"] = "Direct LLM - General knowledge from training data"
            response_data["source"] = "direct_llm"
            
            # Add to conversation history
            agent_brain.add_to_history(session_id, "assistant", llm_response, tool_used="direct_llm")
        
        print(f"\n[OK] PHASE 5 EXECUTION COMPLETE")
        print(f"Tool Used: {response_data['tool_used']}")
        print(f"Response Length: {len(response_data['response']) if response_data['response'] else 0} characters")
        print(f"{'='*80}\n")
        
        return response_data
    
    except Exception as e:
        print(f"[ERROR] Error in unified AURA query: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error processing AURA query: {str(e)}",
                "status": "error"
            }
        )


# GET endpoint - AURA System Information
@app.get("/aura-info")
def aura_system_info():
    """
    Get comprehensive information about the unified AURA system.
    
    Returns:
        JSONResponse: System architecture, capabilities, and status
    """
    rag_available = check_database_exists(DB_PATH)
    
    return {
        "system": "AURA Phase 5 - Unified AI Assistant",
        "description": "Complete integration of RAG, Web Search, Agent Brain, and LLM",
        "status": "active",
        "version": "5.0",
        "architecture": {
            "components": [
                "Phase 1: Chat System (LLM)",
                "Phase 2: Document Intelligence (RAG)",
                "Phase 3: Real-Time Data (Web Search)",
                "Phase 4: Intelligent Decision-Making (Agent Brain)",
                "Phase 5: Unified Integration"
            ],
            "workflow": [
                "User sends query to /aura endpoint",
                "Agent analyzes query intent and requirements",
                "Agent selects optimal tool (RAG, Web, or LLM)",
                "Selected tool retrieves/processes data",
                "Groq LLM generates comprehensive response",
                "Response returned with full analysis"
            ]
        },
        "capabilities": {
            "intelligent_routing": "Automatic tool selection via agent brain",
            "multi_turn_conversations": "Multi-turn support with context management",
            "document_intelligence": f"RAG available: {rag_available}",
            "real_time_information": "Web search with fallback to training data",
            "general_knowledge": "Direct LLM for general queries",
            "transparency": "Full reasoning and analysis included in responses"
        },
        "endpoints": {
            "main": "/aura (POST) - Main unified endpoint",
            "info": "/aura-info (GET) - This endpoint",
            "status": "/aura-status (GET) - System status"
        },
        "query_modes": {
            "intelligent": "Let agent decide (recommended)",
            "rag": "Force document-based search",
            "web": "Force web search",
            "llm": "Force direct LLM"
        },
        "example_request": {
            "query": "What are the latest AI trends?",
            "session_id": "user_123",
            "mode": "intelligent",
            "include_analysis": True
        },
        "supported_features": [
            "Multi-turn conversation with context awareness",
            "Session management",
            "Confidence scoring (0.0-1.0)",
            "Intent detection",
            "Real-time information detection",
            "Document reference detection",
            "Query complexity assessment",
            "Comprehensive decision reasoning",
            "Multiple knowledge sources",
            "Fallback mechanisms",
            "Error recovery"
        ],
        "rag_status": {
            "available": rag_available,
            "database_path": DB_PATH
        },
        "agent_capabilities": {
            "decision_factors": [
                "Real-time information needs",
                "Document references",
                "Query complexity",
                "User intent"
            ],
            "tool_options": ["RAG", "Web Search", "Direct LLM"],
            "session_management": True,
            "context_awareness": True
        }
    }


# GET endpoint - AURA System Status
@app.get("/aura-status")
def aura_status():
    """
    Get current status of all AURA components.
    
    Returns:
        JSONResponse: Real-time status of each component
    """
    rag_available = check_database_exists(DB_PATH)
    
    return {
        "system": "AURA Phase 5",
        "overall_status": "operational",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "phase1_llm": {
                "status": "active",
                "description": "Groq LLaMA 3.1-8b"
            },
            "phase2_rag": {
                "status": "active" if rag_available else "inactive",
                "description": "FAISS Vector Database",
                "database_initialized": rag_available
            },
            "phase3_web_search": {
                "status": "active",
                "description": "DuckDuckGo Web Search with fallback scraping"
            },
            "phase4_agent": {
                "status": "active",
                "description": "Intelligent decision-making brain"
            },
            "phase5_integration": {
                "status": "active",
                "description": "Unified assistant endpoint"
            }
        },
        "main_endpoint": "/aura (POST)",
        "ready_for_queries": True
    }


# ==============================
# PHASE 6+: WEB AUTOMATION ENDPOINTS
# ==============================

# Define request model for Browser Action
class BrowserActionRequest(BaseModel):
    command: str
    session_id: str = None
    
    class Config:
        json_schema_extra = {
            "example": {
                "command": "Open Google",
                "session_id": "user_123"
            }
        }


# ==============================
# CODING MODE INFO ENDPOINT
# ==============================

@app.get("/coding-mode-info")
def coding_mode_info():
    """
    Get information about the Coding Mode detection system.
    Useful for debugging and understanding which keywords trigger Coding Mode.
    """
    from coding_mode import get_coding_mode_info
    return get_coding_mode_info()


# POST endpoint - Browser Automation / Web Action
@app.post("/browser-action")
def browser_action(request: BrowserActionRequest):
    """
    Execute a browser automation command (Open websites or perform searches).
    Phase 6+ feature that extends AURA into an action-oriented AI agent.
    
    Supported Commands:
    - Open {website}: "Open Google", "Open YouTube", "Open ChatGPT", etc.
    - Search {query}: "Search AI Engineering Roadmap", "Search Java Interview Questions"
    - Visit {url}: Alternative syntax for opening websites
    - Browse {website}: Alternative syntax for opening websites
    
    Args:
        request (BrowserActionRequest): JSON body containing:
            - command (str): The browser action command
            - session_id (str, optional): For conversation context
    
    Returns:
        JSONResponse: Action result with status and message
    """
    try:
        if not request.command or request.command.strip() == "":
            return JSONResponse(
                status_code=400,
                content={
                    "error": "Command field is required and cannot be empty",
                    "examples": [
                        "Open Google",
                        "Search AI Engineering Roadmap",
                        "Visit LinkedIn",
                        "Browse GitHub"
                    ]
                }
            )
        
        # Generate or use existing session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        print(f"\n{'='*70}")
        print(f"[WEB] BROWSER ACTION REQUEST")
        print(f"Command: {request.command}")
        print(f"Session ID: {session_id}")
        print(f"{'='*70}")
        
        # Get web automation manager
        web_automation = get_web_automation_manager()
        
        # Execute the command
        success, message = web_automation.execute_command(request.command)
        
        # Store in conversation history
        agent_brain.add_to_history(
            session_id, 
            "user", 
            request.command,
            metadata={"type": "browser_action"}
        )
        
        agent_brain.add_to_history(
            session_id,
            "assistant",
            message,
            tool_used="browser_action",
            metadata={"success": success}
        )
        
        response_data = {
            "command": request.command,
            "session_id": session_id,
            "success": success,
            "message": message,
            "status": "success" if success else "failed",
            "tool_used": "browser_action"
        }
        
        print(f"\n[OK] BROWSER ACTION EXECUTED")
        print(f"Success: {success}")
        print(f"Message: {message}")
        print(f"{'='*70}\n")
        
        return response_data
    
    except Exception as e:
        print(f"[ERROR] Error in browser action: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error executing browser action: {str(e)}",
                "status": "error"
            }
        )


# POST endpoint - Unified AURA Query with Browser Automation Support
@app.post("/aura-v2")
async def unified_aura_query_v2(request: UnifiedAuraRequest):
    """
    PHASE 6+: UNIFIED AI AGENT WITH BROWSER AUTOMATION
    
    Enhanced version of the unified endpoint that includes browser automation capabilities.
    Automatically detects and routes browser action commands.
    
    Supported Command Types:
    1. Browser Actions (Open/Search): "Open Google", "Search AI trends"
    2. Real-time Queries: "What's the latest news?"
    3. Document Queries: "What does the PDF say about AI?"
    4. Complex Queries: Multi-step reasoning questions
    
    Args:
        request (UnifiedAuraRequest): JSON body containing:
            - query (str): User's command or question
            - session_id (str, optional): For multi-turn conversations
            - mode (str, optional): "intelligent" or specific tool
            - include_analysis (bool, optional): Include decision reasoning
    
    Returns:
        JSONResponse: Response with action result or AI answer
    """
    try:
        if not request.query or request.query.strip() == "":
            return JSONResponse(
                status_code=400,
                content={"error": "Query field is required and cannot be empty"}
            )
        
        # Generate or use existing session ID
        session_id = request.session_id or str(uuid.uuid4())
        
        print(f"\n{'='*80}")
        print(f"[LAUNCH] PHASE 6+ - UNIFIED AURA V2 (WITH BROWSER AUTOMATION)")
        print(f"Query: {request.query}")
        print(f"Session ID: {session_id}")
        print(f"Mode: {request.mode}")
        print(f"{'='*80}")
        
        # Update agent brain with current RAG status
        rag_available = check_database_exists(DB_PATH)
        agent_brain.set_rag_availability(rag_available)
        
        # Determine tool to use
        tool_to_use = None
        analysis = None
        
        if request.mode == "intelligent":
            # Auto-detect coding queries BEFORE agent brain runs
            detected_mode = detect_mode(request.query)
            if detected_mode == "coding":
                print(" CODING MODE (auto-detected) - Bypassing agent brain")
                coding_intent = detect_coding_intent(request.query)
                coding_response = query_llm_coding(request.query, coding_intent)
                agent_brain.add_to_history(session_id, "assistant", coding_response, tool_used="coding_mode")
                return {
                    "query": request.query,
                    "session_id": session_id,
                    "tool_used": "coding_mode",
                    "coding_intent": coding_intent,
                    "confidence": 0.99,
                    "response": coding_response,
                    "source": "coding_mode",
                    "mode": "coding",
                    "status": "success",
                }

            # Let agent decide (non-coding query)
            print("[BRAIN] AGENT DECISION MODE - Analyzing query...")
            analysis = agent_brain.analyze_query(request.query)
            tool_to_use = analysis.suggested_tool
            confidence = analysis.confidence
            intent = analysis.intent
            reasoning = analysis.reasoning
        else:
            # Use specified tool
            print(f"\n FORCED MODE - Using {request.mode}")

            #  Coding Mode (forced via frontend mode selector) 
            if request.mode == "coding":
                print(" CODING MODE (forced) - Routing to Coding Assistant")
                coding_intent = detect_coding_intent(request.query)
                coding_response = query_llm_coding(request.query, coding_intent)
                session_id = request.session_id or str(uuid.uuid4())
                agent_brain.add_to_history(session_id, "assistant", coding_response, tool_used="coding_mode")
                return {
                    "query": request.query,
                    "session_id": session_id,
                    "tool_used": "coding_mode",
                    "coding_intent": coding_intent,
                    "confidence": 0.99,
                    "response": coding_response,
                    "source": "coding_mode",
                    "mode": "coding",
                    "status": "success",
                }

            mode_to_tool = {
                "rag": ToolType.RAG,
                "web": ToolType.WEB_SEARCH,
                "llm": ToolType.DIRECT_LLM,
                "browser": ToolType.BROWSER_ACTION
            }
            tool_to_use = mode_to_tool.get(request.mode, ToolType.DIRECT_LLM)

            if analysis is None:
                analysis = agent_brain.analyze_query(request.query)
                confidence = 0.95
                intent = "user_specified"
                reasoning = f"User explicitly requested {request.mode} tool"
        
        # Build response structure
        response_data = {
            "query": request.query,
            "session_id": session_id,
            "tool_used": tool_to_use.value,
            "confidence": confidence if request.mode == "intelligent" else 0.95,
            "response": None,
            "context": None,
            "source": None,
            "status": "success"
        }
        
        # Add analysis if requested
        if request.include_analysis:
            response_data["analysis"] = {
                "intent": intent if request.mode == "intelligent" else "user_specified",
                "reasoning": reasoning if request.mode == "intelligent" else f"User explicitly requested {request.mode}",
                "requires_realtime": analysis.requires_realtime if analysis else False,
                "requires_document": analysis.requires_document if analysis else False,
                "complexity": analysis.complexity if analysis else "unknown"
            }
        
        # Execute selected tool
        print(f"\n[AURA] EXECUTING: {tool_to_use.value.upper()}")
        
        # ===== NEW: BROWSER ACTION HANDLING =====
        if tool_to_use == ToolType.BROWSER_ACTION:
            print("[WEB] Browser Automation Execution")
            web_automation = get_web_automation_manager()
            success, message = web_automation.execute_command(request.query)
            
            response_data["response"] = message
            response_data["context"] = "Browser automation action"
            response_data["source"] = "browser_action"
            response_data["action_status"] = "success" if success else "failed"
            
            agent_brain.add_to_history(
                session_id,
                "assistant",
                message,
                tool_used="browser_action",
                metadata={"success": success}
            )
        
        # ===== ORIGINAL TOOLS (WEB, RAG, LLM, APIs) =====
        elif tool_to_use == ToolType.WEB_SEARCH:
            # Execute Web Search
            print("[WEB] Web Search Execution")
            web_results = web_searcher.search(request.query)
            
            if web_results:
                print(f"[OK] Found {len(web_results)} web results")
                web_context = web_searcher.generate_web_context(request.query, web_results)
                llm_response = query_llm_with_web(request.query, web_context)
                response_data["response"] = llm_response
                response_data["context"] = web_context
                response_data["source"] = "web_search"
            else:
                print("[WARNING] No web results - using fallback response")
                llm_response = query_llm_with_web_fallback(request.query, "No live results available")
                response_data["response"] = llm_response
                response_data["context"] = "Fallback: Using training data"
                response_data["source"] = "web_search_fallback"
            
            agent_brain.add_to_history(session_id, "assistant", llm_response, tool_used="web_search")
        
        elif tool_to_use == ToolType.RAG:
            # Execute RAG
            print(" RAG Execution")
            
            if not rag_available:
                response_data["response"] = "No PDF documents are currently loaded. Please upload a PDF using /upload-pdf endpoint."
                response_data["source"] = "none"
                response_data["status"] = "rag_unavailable"
            else:
                vector_store = load_vector_database(DB_PATH)
                if vector_store is None:
                    response_data["response"] = "Failed to load document database."
                    response_data["source"] = "none"
                    response_data["status"] = "error"
                else:
                    rag_context, _ = generate_rag_context(request.query, vector_store, k=4)
                    rag_response = query_llm_with_rag(request.query, rag_context)
                    response_data["response"] = rag_response
                    response_data["context"] = rag_context
                    response_data["source"] = "rag"
                    
                    agent_brain.add_to_history(session_id, "assistant", rag_response, tool_used="rag")
        
        elif tool_to_use == ToolType.DIRECT_LLM:
            # Execute Direct LLM
            print(" Direct LLM Execution")
            
            conversation_context = agent_brain.get_session_context(session_id)
            if conversation_context and len(conversation_context.strip()) > 0:
                prompt = f"Previous conversation context:\n{conversation_context}\n\nCurrent query: {request.query}"
            else:
                prompt = request.query
            
            llm_response = query_llm(prompt)
            response_data["response"] = llm_response
            response_data["context"] = "Direct LLM - General knowledge from training data"
            response_data["source"] = "direct_llm"
            
            agent_brain.add_to_history(session_id, "assistant", llm_response, tool_used="direct_llm")
        
        # ===== APIs (WEATHER, SPORTS, STOCKS, CRYPTO, NEWS, TIME, MAPS) =====
        elif tool_to_use == ToolType.WEATHER:
            print("[WEATHER] Weather API Execution")
            city = extract_city_from_query(request.query) or "Chennai"
            
            if not WEATHER_API_KEY:
                response_data["response"] = "[ERROR] Weather API key not configured."
                response_data["status"] = "error"
            else:
                weather_data = get_weather(city, WEATHER_API_KEY)
                response_data["response"] = weather_data
                response_data["context"] = f"Weather data for {city}"
                response_data["source"] = "weather_api"
                agent_brain.add_to_history(session_id, "assistant", weather_data, tool_used="weather_api")
        
        elif tool_to_use == ToolType.SPORTS:
            print("[SPORTS] Cricket API Execution")
            
            if not CRIC_API_KEY:
                response_data["response"] = "[ERROR] Cricket API key not configured."
                response_data["status"] = "error"
            else:
                cricket_data = get_cricket_score(CRIC_API_KEY)
                response_data["response"] = cricket_data
                response_data["context"] = "Live cricket/IPL scores"
                response_data["source"] = "sports_api"
                agent_brain.add_to_history(session_id, "assistant", cricket_data, tool_used="sports_api")
        
        elif tool_to_use == ToolType.STOCKS:
            print("[STOCKS] Stock API Execution")
            symbol = extract_stock_symbol_from_query(request.query)
            
            stock_data = get_stock(symbol)
            response_data["response"] = stock_data
            response_data["context"] = f"Stock price for {symbol}"
            response_data["source"] = "stocks_api"
            agent_brain.add_to_history(session_id, "assistant", stock_data, tool_used="stocks_api")
        
        elif tool_to_use == ToolType.CRYPTO:
            print(" Crypto API Execution")
            crypto_id = extract_crypto_from_query(request.query)
            
            crypto_data = get_crypto(crypto_id)
            response_data["response"] = crypto_data
            response_data["context"] = f"Cryptocurrency price for {crypto_id}"
            response_data["source"] = "crypto_api"
            agent_brain.add_to_history(session_id, "assistant", crypto_data, tool_used="crypto_api")
        
        elif tool_to_use == ToolType.NEWS:
            print("[NEWS] News API Execution")
            
            if not NEWS_API_KEY:
                response_data["response"] = "[ERROR] News API key not configured."
                response_data["status"] = "error"
            else:
                news_query = extract_news_query_from_query(request.query)
                news_data = get_news(NEWS_API_KEY, query=news_query)
                response_data["response"] = news_data
                response_data["context"] = f"Real-time news articles about '{news_query}'"
                response_data["source"] = "news_api"
                agent_brain.add_to_history(session_id, "assistant", news_data, tool_used="news_api")
        
        elif tool_to_use == ToolType.TIME:
            print("[TIME] Time API Execution")
            time_data = get_datetime()
            response_data["response"] = time_data
            response_data["context"] = "Current date and time"
            response_data["source"] = "time_api"
            agent_brain.add_to_history(session_id, "assistant", time_data, tool_used="time_api")
        
        elif tool_to_use == ToolType.MAPS:
            print(" Maps API Execution")
            maps_data = " **Maps API** - Location-based services coming soon!"
            response_data["response"] = maps_data
            response_data["context"] = "Maps/location services"
            response_data["source"] = "maps_api"
            agent_brain.add_to_history(session_id, "assistant", maps_data, tool_used="maps_api")
        
        print(f"\n[OK] PHASE 6+ EXECUTION COMPLETE")
        print(f"Tool Used: {response_data['tool_used']}")
        print(f"Response Length: {len(response_data['response']) if response_data['response'] else 0} characters")
        print(f"{'='*80}\n")
        
        return response_data
    
    except Exception as e:
        print(f"[ERROR] Error in AURA V2 query: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return JSONResponse(
            status_code=500,
            content={
                "error": f"Error processing AURA V2 query: {str(e)}",
                "status": "error"
            }
        )


# GET endpoint - Browser Automation Status and Supported Websites
@app.get("/browser-automation-status")
def browser_automation_status():
    """
    Get status and information about browser automation capabilities.
    
    Returns:
        JSONResponse: Supported websites, commands, and system status
    """
    web_automation = get_web_automation_manager()
    status = web_automation.get_status()
    
    return {
        "system": "Browser Automation Module",
        "status": "active",
        "version": "1.0",
        "capabilities": {
            "open_websites": True,
            "perform_searches": True,
            "custom_websites": True
        },
        "supported_websites": status["websites_list"],
        "total_websites": status["supported_websites"],
        "actions_performed": status["actions_performed"],
        "commands": {
            "open_website": [
                "Open Google",
                "Open YouTube",
                "Open ChatGPT",
                "Open LinkedIn",
                "Open GitHub",
                "Open Gmail",
                "Open Amazon",
                "Open BookMyShow",
                "Open SRM Portal",
                "Visit {any_url}",
                "Browse {website}"
            ],
            "search": [
                "Search AI Engineering Roadmap",
                "Search Java Interview Questions",
                "Search Best Laptops Under 50000",
                "Google {query}",
                "Find {topic}"
            ]
        },
        "endpoints": {
            "browser_action": "/browser-action (POST) - Execute browser actions",
            "aura_v2": "/aura-v2 (POST) - Unified endpoint with browser automation support",
            "status": "/browser-automation-status (GET) - This endpoint",
            "history": "/browser-action-history (GET) - View action history"
        },
        "example_requests": {
            "open_website": {
                "command": "Open Google",
                "session_id": "user_123"
            },
            "search": {
                "command": "Search AI Engineering Roadmap",
                "session_id": "user_123"
            },
            "aura_v2": {
                "query": "Open YouTube",
                "session_id": "user_123",
                "mode": "intelligent",
                "include_analysis": True
            }
        }
    }


# GET endpoint - Browser Action History
@app.get("/browser-action-history")
def browser_action_history(session_id: str = None):
    """
    Get history of browser automation actions performed.
    
    Args:
        session_id (str, optional): Filter by session ID
    
    Returns:
        JSONResponse: List of browser automation actions
    """
    web_automation = get_web_automation_manager()
    history = web_automation.get_action_history()
    
    # Filter by session if provided
    if session_id and session_id in agent_brain.conversation_history:
        context = agent_brain.conversation_history[session_id]
        filtered_history = [
            h for h in history if h.get("session_id") == session_id
        ]
        return {
            "session_id": session_id,
            "action_count": len(filtered_history),
            "actions": filtered_history
        }
    
    return {
        "total_actions": len(history),
        "actions": history
    }


# ==============================
# DO NOT TOUCH BELOW
# ==============================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
