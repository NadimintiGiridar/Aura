"""
AURA Coding Mode — Intent Detection Module

Detects whether a user query is coding-related and determines
the specific coding sub-intent (code generation, explanation, dry run, debug).

Used by main.py to short-circuit the agent brain and route directly
to a coding-specialized LLM prompt.
"""

# 
# 1. KEYWORD SETS
# 

CODING_KEYWORDS = {
    # Languages
    "python", "java", "c++", "cpp", "javascript", "typescript", "js", "ts",
    "golang", "go", "rust", "kotlin", "swift", "ruby", "php", "scala", "dart",
    "html", "css", "sql", "bash", "shell", "r", "matlab",
    # Programming concepts
    "code", "program", "function", "class", "method", "variable", "loop",
    "recursion", "recursive", "iteration", "implement", "implementation",
    # DSA / Algorithms
    "algorithm", "dsa", "data structure", "leetcode", "hackerrank", "codeforces",
    "linked list", "linkedlist", "binary tree", "binary search", "bst",
    "stack", "queue", "heap", "graph", "trie", "hash map", "hashmap",
    "array", "string", "pointer", "matrix", "dp", "dynamic programming",
    "backtracking", "greedy", "divide and conquer", "memoization",
    "sorting", "searching", "merge sort", "quick sort", "bubble sort",
    "insertion sort", "selection sort", "bfs", "dfs", "dijkstra",
    "two sum", "fibonacci", "factorial", "palindrome", "anagram",
    "longest common subsequence", "lcs", "knapsack", "sliding window",
    "two pointer", "binary search", "inorder", "preorder", "postorder",
    # Action verbs that signal coding intent
    "write", "build", "create", "solve", "fix", "debug", "optimize",
    "reverse", "sort", "find", "detect", "check", "count", "calculate",
    # Complexity
    "time complexity", "space complexity", "big o", "o(n)", "o(log n)",
    # Interview
    "interview", "leetcode", "coding challenge", "coding problem",
}

EXPLAIN_KEYWORDS = {
    "explain", "how does", "what is", "how do", "describe", "tell me about",
    "what are", "define", "clarify", "break down", "how it works",
}

DRY_RUN_KEYWORDS = {
    "dry run", "trace", "step by step", "walkthrough", "walk through",
    "simulate", "example run", "show steps", "show me the steps",
    "trace through", "step-by-step",
}

DEBUG_KEYWORDS = {
    "debug", "fix", "error", "bug", "issue", "not working", "why isn't",
    "wrong output", "failing", "broken", "crash", "exception", "traceback",
    "syntax error", "runtime error", "logic error",
}

BROWSER_KEYWORDS = {
    "open", "visit", "go to", "launch", "browse", "navigate to",
    "search on google", "search on youtube",
}

RAG_KEYWORDS = {
    "pdf", "document", "uploaded", "chapter", "summarize this",
    "in the file", "from the document", "according to the pdf",
    "what does the pdf", "what does the document",
}


# 
# 2. DETECTION FUNCTIONS
# 

def is_coding_query(query: str) -> bool:
    """
    Returns True if the query is coding/DSA/algorithm related.

    Args:
        query: User's input string

    Returns:
        bool: True if coding-related
    """
    q = query.lower().strip()

    # Check for explicit coding keywords
    for keyword in CODING_KEYWORDS:
        if keyword in q:
            return True

    # Check if starts with action verbs paired with programming nouns
    action_words = {"write", "build", "create", "implement", "solve", "code", "make"}
    words = set(q.split())
    if action_words & words:
        # Has action verb — likely a code request
        return True

    return False


def detect_coding_intent(query: str) -> str:
    """
    Detects the specific coding sub-intent from the query.

    Returns one of:
        "code"     — Default: generate code only
        "explain"  — Explain an algorithm/concept
        "dry_run"  — Step-by-step trace of an algorithm
        "debug"    — Help fix/debug existing code
    """
    q = query.lower().strip()

    # Debug check first (most specific)
    for kw in DEBUG_KEYWORDS:
        if kw in q:
            return "debug"

    # Dry run check
    for kw in DRY_RUN_KEYWORDS:
        if kw in q:
            return "dry_run"

    # Explain check
    for kw in EXPLAIN_KEYWORDS:
        if kw in q:
            return "explain"

    # Default: generate code
    return "code"


def detect_mode(query: str) -> str:
    """
    Top-level mode detector. Determines which AURA mode to activate.

    Returns one of:
        "coding"  — Coding Mode (DSA, algorithms, code generation)
        "browser" — Browser Mode (open websites, perform searches)
        "rag"     — RAG Mode (PDF/document questions)
        "normal"  — Normal Mode (general conversation)
    """
    q = query.lower().strip()

    # Browser mode — check action keywords that start the query
    for kw in BROWSER_KEYWORDS:
        if q.startswith(kw):
            return "browser"

    # RAG mode — references to uploaded documents
    for kw in RAG_KEYWORDS:
        if kw in q:
            return "rag"

    # Coding mode
    if is_coding_query(q):
        return "coding"

    # Default: normal mode
    return "normal"


def get_coding_mode_info() -> dict:
    """Returns metadata about the coding mode detection system."""
    return {
        "module": "AURA Coding Mode",
        "version": "1.0",
        "keyword_count": {
            "coding": len(CODING_KEYWORDS),
            "explain": len(EXPLAIN_KEYWORDS),
            "dry_run": len(DRY_RUN_KEYWORDS),
            "debug": len(DEBUG_KEYWORDS),
            "browser": len(BROWSER_KEYWORDS),
            "rag": len(RAG_KEYWORDS),
        },
        "supported_intents": ["code", "explain", "dry_run", "debug"],
        "supported_modes": ["coding", "browser", "rag", "normal"],
        "description": (
            "Intent-based routing module that detects coding queries and "
            "routes them to a specialized coding LLM prompt, bypassing the "
            "general agent brain for faster and cleaner code responses."
        ),
    }
