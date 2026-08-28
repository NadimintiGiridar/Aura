import os
import re
import sys
import time
from dotenv import load_dotenv
from groq import Groq

# Configure UTF-8 encoding for Windows standard output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

# Load .env from backend directory and current directory
_backend_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_backend_dir, ".env"))
load_dotenv()

def _get_client() -> Groq:
    """Get or create Groq client with latest API key."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        load_dotenv(os.path.join(_backend_dir, ".env"))
        api_key = os.getenv("GROQ_API_KEY")
    return Groq(api_key=api_key)

# ── Model selection ──────────────────────────────────────────────────────────
# Verified live models on Groq in priority order (high capacity + fast fallbacks)
MODELS = [
    "openai/gpt-oss-120b",  # Primary   — flagship 120B reasoning, high quality
    "openai/gpt-oss-20b",   # Fast      — low latency, high throughput
    "qwen/qwen3.8-27b",     # Fallback 1 — 27B high context reasoning
    "qwen/qwen3.6-27b",     # Fallback 2 — lightweight Qwen
    "groq/compound",        # Fallback 3 — Groq native
    "groq/compound-mini",   # Fallback 4 — smallest, ultra-available
]

# Fast model specifically for quick prompt enhancement
FAST_PROMPT_MODELS = [
    "openai/gpt-oss-20b",
    "groq/compound-mini",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]

# ── Token budget constants ───────────────────────────────────────────────────
MAX_USER_INPUT_CHARS   = 3000    # ~750 tokens  — user query cap
MAX_CONTEXT_CHARS      = 12000   # ~3000 tokens — RAG context
MAX_WEB_CONTEXT_CHARS  = 4000    # ~1000 tokens — web search context cap
MAX_CONV_CONTEXT_CHARS = 1500    # ~375 tokens  — conversation history cap


def _trim(text: str, max_chars: int) -> str:
    """Trim text to max_chars, appending '...' if truncated."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."


def _call_groq(messages: list, max_tokens: int = 800, model_list: list = None) -> str:
    """
    Call Groq with instant failover across model families.
    On 429 / rate-limit / errors: immediately tries the next model family
    without blocking sleep, ensuring sub-second response times.
    """
    models_to_try = model_list or MODELS
    client = _get_client()

    for attempt_round in range(2):  # up to 2 full passes
        for model in models_to_try:
            msgs = list(messages)
            try:
                print(f"[GROQ] Calling {model} (pass {attempt_round+1})...")
                response = client.chat.completions.create(
                    model=model,
                    messages=msgs,
                    temperature=0.7,
                    max_tokens=max_tokens,
                )
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()

            except Exception as e:
                err = str(e)

                # Rate limit (429) -> immediately failover to next model family
                if "429" in err or "rate_limit" in err.lower():
                    print(f"[GROQ] Rate limit on {model}. Fast failover to next model...")
                    continue

                # Request too large (413) -> trim largest non-system message and retry
                if "413" in err or "request_too_large" in err.lower():
                    print(f"[GROQ] 413 payload on {model}. Trimming...")
                    for m in msgs:
                        if m["role"] != "system" and len(m.get("content", "")) > 200:
                            m["content"] = m["content"][:len(m["content"]) // 2] + "..."
                    try:
                        resp = client.chat.completions.create(
                            model=model,
                            messages=msgs,
                            temperature=0.7,
                            max_tokens=max_tokens,
                        )
                        return resp.choices[0].message.content.strip()
                    except Exception:
                        pass
                    continue

                # Model not found or terms required -> skip
                if any(k in err.lower() for k in ["404", "model_not_found", "terms", "does not exist"]):
                    print(f"[GROQ] Model {model} unavailable ({err[:60]}). Trying next...")
                    continue

                # Any other error -> try next model immediately
                print(f"[GROQ] Error on {model}: {err[:80]}. Trying next...")
                continue

        # If first pass exhausted, sleep 1s and try second pass
        if attempt_round == 0:
            time.sleep(1.0)

    return "⚠️ AURA is temporarily busy. Please try again in a few seconds."


def improve_prompt(user_input: str) -> str:
    """
    Intelligently refines and enhances the user's raw prompt into a clear,
    well-structured query for the AI assistant.

    - Ultra-fast execution (< 300ms)
    - Strictly preserves user intent without answering the query
    - Never outputs conversational refusals or 'Please provide' statements
    """
    stripped = user_input.strip()
    if len(stripped) < 6 or stripped.lower() in {"hi", "hello", "hey", "ok", "okay", "thanks", "bye", "help"}:
        return user_input

    system_prompt = (
        "You are an expert search and prompt optimizer.\n"
        "Your ONLY task is to rewrite the user query into a clear, detailed, and well-structured prompt for an AI assistant.\n\n"
        "STRICT RULES:\n"
        "1. DO NOT answer or reply to the user prompt.\n"
        "2. DO NOT say 'I cannot', 'Please provide', 'As an AI', 'Sure', or ask questions back.\n"
        "3. If the user mentions a document/PDF/resume/file, refer to it as 'the uploaded document'.\n"
        "4. Fix broken grammar and expand vague requests into specific requirements.\n"
        "5. Keep the improved prompt concise (1-2 sentences max).\n"
        "6. Output ONLY the refined prompt text with no quotes, preamble, or explanations."
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"User: {stripped}\nRefined prompt:"},
    ]

    try:
        improved = _call_groq(messages, max_tokens=120, model_list=FAST_PROMPT_MODELS)
        
        # Validation safeguards: Reject responses that try to answer or refuse
        if not improved or len(improved.strip()) < 5:
            return user_input

        lower_imp = improved.lower().strip()
        invalid_prefixes = (
            "i cannot", "i can't", "please provide", "please upload", 
            "as an ai", "sure,", "here is", "⚠️", "temporarily busy"
        )
        if any(lower_imp.startswith(p) for p in invalid_prefixes) or "i do not have access" in lower_imp:
            print(f"[PROMPT] Rejected conversational output from improver: {improved[:60]!r}")
            return user_input

        # Strip any accidental wrapping quotes
        cleaned = improved.strip().strip('"').strip("'")
        return cleaned if cleaned else user_input

    except Exception as e:
        print(f"[PROMPT] Improvement fallback: {e}")
        return user_input


def query_llm(user_input: str) -> str:
    """Query the LLM with user input (basic mode, no external context)."""
    user_input = _trim(user_input, MAX_USER_INPUT_CHARS)
    messages = [
        {"role": "system", "content": "You are AURA, an intelligent, helpful AI assistant. Provide clear, accurate, and well-formatted answers."},
        {"role": "user",   "content": user_input},
    ]
    return _call_groq(messages, max_tokens=600)

def query_llm_with_rag(user_input: str, context: str) -> str:
    """Query the LLM with RAG document context."""
    user_input = _trim(user_input, MAX_USER_INPUT_CHARS)
    context    = _trim(context,    MAX_CONTEXT_CHARS)
    messages = [
        {
            "role": "system",
            "content": (
                "You are AURA, an intelligent document-analysis assistant. "
                "The user has uploaded a document and you have been given relevant excerpts from it. "
                "Your job is to answer the user's question as completely and helpfully as possible.\n\n"

                "UNDERSTANDING THE USER'S INTENT:\n"
                "- If they ask 'tell me about this PDF / document / file' → give a full overview: what it is, who it belongs to, key sections.\n"
                "- If they ask about PROJECTS → list every project with: name, description, technologies used, outcomes.\n"
                "- If they ask about SKILLS / TECHNOLOGIES → list all skills grouped by category.\n"
                "- If they ask for KEY POINTS / HIGHLIGHTS → extract the most important facts as a bulleted list.\n"
                "- If they ask about EXPERIENCE / WORK HISTORY → list each role: company, title, duration, responsibilities.\n"
                "- If they ask about EDUCATION → list degrees, institutions, years.\n"
                "- If they ask a SPECIFIC QUESTION → answer it precisely using document evidence.\n\n"

                "FORMATTING RULES:\n"
                "- Use ## headings to separate major sections.\n"
                "- Use bullet points (•) for lists. Use bold (**text**) for names/titles.\n"
                "- Be SPECIFIC: include real names, dates, tools, numbers found in the document.\n"
                "- Do NOT say 'the document mentions' repeatedly — just present the information directly.\n"
                "- If something is genuinely not in the provided excerpts, say: 'This information was not found in the uploaded document.'\n"
                "- Do NOT invent or assume any information not present in the context."
            ),
        },
        {
            "role": "user",
            "content": (
                f"=== DOCUMENT EXCERPTS ===\n{context}\n\n"
                f"=== USER QUESTION ===\n{user_input}\n\n"
                "Please provide a thorough, well-structured answer based on the document excerpts above."
            ),
        },
    ]
    return _call_groq(messages, max_tokens=1500)

def query_llm_with_web(user_input: str, web_context: str) -> str:
    """Query the LLM with real-time web search results."""
    user_input  = _trim(user_input,  MAX_USER_INPUT_CHARS)
    web_context = _trim(web_context, MAX_WEB_CONTEXT_CHARS)
    messages = [
        {
            "role": "system",
            "content": (
                "You are AURA, an AI assistant with real-time web knowledge. "
                "Use the provided web results to give an accurate, concise answer. "
                "Cite sources when helpful."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Web Results:\n{web_context}\n\n"
                f"Question: {user_input}\n\n"
                "Give a clear, concise answer using the web results above."
            ),
        },
    ]
    return _call_groq(messages, max_tokens=400)

def query_llm_with_web_fallback(user_input: str, fallback_context: str) -> str:
    """Fallback when web search fails — answer from training data."""
    user_input = _trim(user_input, MAX_USER_INPUT_CHARS)
    messages = [
        {
            "role": "system",
            "content": (
                "You are AURA, a knowledgeable AI assistant. "
                "Live web search is unavailable, so answer from your training data. "
                "Be clear about your knowledge cutoff when relevant."
            ),
        },
        {"role": "user", "content": user_input},
    ]
    return _call_groq(messages, max_tokens=400)


# 
# CODING MODE — Specialized LLM functions for code generation & DSA
# 

# System prompts for each coding sub-intent
_CODING_SYSTEM_PROMPTS = {
    "code": """You are AURA, an AI Coding Assistant — similar to GitHub Copilot or Cursor.

STRICT RULES:
1. Return ONLY the code. Nothing else.
2. Use Python by default unless another language is explicitly requested.
3. Do NOT include a problem statement or description.
4. Do NOT include time complexity or space complexity analysis unless the user asks.
5. Do NOT add lengthy explanations before or after the code.
6. You may add very brief inline comments (max 1 per block) only if they genuinely help.
7. Always wrap code in triple backticks with the language tag (e.g. ```python).
8. The code must be clean, correct, and immediately executable.
9. Do not add greetings, sign-offs, or meta-commentary.""",

    "explain": """You are AURA, an AI Coding Assistant in Explanation Mode.

RULES:
1. Give a clear, concise explanation of the algorithm or concept.
2. Keep it under 6 sentences / bullet points unless more depth is essential.
3. If a short code snippet helps illustrate the concept, include it.
4. No textbook-length content. Be developer-friendly.
5. No greetings or sign-offs.""",

    "dry_run": """You are AURA, an AI Coding Assistant in Dry Run Mode.

RULES:
1. Choose a small, concrete example input.
2. Trace through the algorithm step-by-step using that input.
3. Format each step as a numbered list showing variable states.
4. Keep it concise — under 15 steps unless absolutely necessary.
5. End with the final output/result.
6. No lengthy preamble — go straight into the trace.""",

    "debug": """You are AURA, an AI Coding Assistant in Debug Mode.

RULES:
1. Identify the bug or error directly.
2. Show the corrected code.
3. Add a brief (1-2 line) explanation of what was wrong.
4. Do not pad the response with unnecessary text.""",
}


def query_llm_coding(user_input: str, intent: str = "code") -> str:
    """
    Query the LLM in Coding Mode with a specialized, developer-friendly prompt.

    Bypasses general-purpose prompts to produce clean, concise code output.

    Args:
        user_input (str): The user's coding query
        intent (str): One of "code", "explain", "dry_run", "debug"

    Returns:
        str: Code or explanation, formatted for developers
    """
    user_input    = _trim(user_input, MAX_USER_INPUT_CHARS)
    system_prompt = _CODING_SYSTEM_PROMPTS.get(intent, _CODING_SYSTEM_PROMPTS["code"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user",   "content": user_input},
    ]
    return _call_groq(messages, max_tokens=500)


def query_llm_normal(user_input: str, conversation_context: str = "") -> str:
    """
    Query the LLM in Normal Mode — general conversational AI with AURA persona.

    Args:
        user_input (str): User's query
        conversation_context (str): Optional prior conversation summary

    Returns:
        str: Conversational AI response
    """
    user_input           = _trim(user_input,           MAX_USER_INPUT_CHARS)
    conversation_context = _trim(conversation_context, 1000)

    messages = [
        {
            "role": "system",
            "content": (
                "You are AURA (AI-powered User Response Assistant), "
                "an intelligent, helpful, and friendly AI assistant. "
                "Answer questions clearly and concisely."
            ),
        }
    ]

    if conversation_context.strip():
        messages.append({"role": "user",      "content": f"Previous context:\n{conversation_context}"})
        messages.append({"role": "assistant", "content": "Understood. How can I help?"})

    messages.append({"role": "user", "content": user_input})
    return _call_groq(messages, max_tokens=600)
