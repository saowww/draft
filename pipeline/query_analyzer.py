import json
import os
import enum
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from enum import Enum
import uuid

from backend.utils.logging import get_logger
from backend.llm.providers.gemini.gemini_client import GeminiClient


logger = get_logger(__name__)


class QueryIntent(Enum):
    """Enum representing different types of query intents for AUTOSAR domain."""
    GREETING = "greeting"
    AUTOSAR_RELATED = "autosar_related"
    TECHNICAL_QUERY = "technical_query"
    GENERAL = "general"


os.makedirs("user_data", exist_ok=True)


async def analyze_query(
    query: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    client: Any = None,
    user_id: Optional[str] = None
) -> QueryIntent:
    """
    Analyze the user query to determine its intent.

    Args:
        query: The query string from the user
        conversation_history: Optional history of conversation messages
        client: Client for LLM operations (GeminiClient or OllamaClient)
        user_id: Optional user identifier for personal info storage

    Returns:
        The classified query intent
    """
    logger.info(f"Analyzing query intent: {query}")

    if not query:
        logger.warning("Empty query received")
        return QueryIntent.GENERAL

    # Check if there's valid conversation history
    if not conversation_history:
        conversation_history = []

    # LLM-based intent classification
    if client:
        try:
            system_prompt = """
            You are an expert system that classifies user queries into specific intents for AUTOSAR domain.
            Classify the user's query into ONE of these categories:
            1. GREETING - General greetings, casual conversation, politeness
            2. AUTOSAR_RELATED - Any query about AUTOSAR standards, components, architecture, configuration, modules, layers, platforms, etc.
            3. TECHNICAL_QUERY - Technical questions about implementation, specifications, APIs, interfaces, protocols
            4. GENERAL - Any other topic not covered by the above

            Respond with ONLY the category name, nothing else. For example: 'AUTOSAR_RELATED'
            
            For AUTOSAR_RELATED, look for:
            - AUTOSAR components, modules, layers
            - AUTOSAR architecture and standards
            - Configuration, parameters, settings
            - Platform-specific queries
            - Software components, services, interfaces
            """

            # Prepare context from conversation history
            context = ""
            if conversation_history:
                recent_history = conversation_history[-3:] if len(
                    conversation_history) > 3 else conversation_history
                context = "\n".join(
                    [f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in recent_history])
                context = f"Previous conversation:\n{context}\n\n"

            # Build full prompt
            prompt = f"## {system_prompt} \n\n {context}Current query: {query}\n\nClassify this query's intent into one of the categories."

            # Call LLM to classify intent - work with either GeminiClient or OllamaClient
            if isinstance(client, GeminiClient):
                response = client.generate(
                    user_prompt=prompt,
                    system_prompt=system_prompt
                )
            else:
                # For other clients (Ollama), keep async if needed
                response = await client.generate(
                    prompt=prompt,
                    system_prompt=system_prompt
                )

            # Extract intent from response - handle either client's response format
            if isinstance(client, GeminiClient):
                # GeminiClient returns LLMResponse object with message attribute
                if hasattr(response, 'message'):
                    intent_text = str(response.message).strip().upper()
                elif isinstance(response, dict):
                    intent_text = response.get("message", "").strip().upper() if isinstance(response.get(
                        "message"), str) else response.get("message", {}).get("content", "").strip().upper()
                else:
                    intent_text = str(response).strip().upper()
            else:
                # Other clients (Ollama) return dict
                intent_text = response.get("message", {}).get(
                    "content", "").strip().upper()

            logger.debug(f"LLM classified intent as: {intent_text}")

            # Map LLM response to QueryIntent enum
            if "GREETING" in intent_text:
                intent = QueryIntent.GREETING
            elif "AUTOSAR" in intent_text:
                intent = QueryIntent.AUTOSAR_RELATED
            elif "TECHNICAL" in intent_text:
                intent = QueryIntent.TECHNICAL_QUERY
            else:
                intent = QueryIntent.GENERAL

            logger.info(f"Query intent determined: {intent.name}")
            return intent

        except Exception as e:
            logger.error(f"Error using LLM for intent classification: {e}")
            # Fall back to rule-based approach

    # Rule-based intent classification as fallback
    query_lower = query.lower()

    # Check for greeting intent
    greeting_phrases = ["hello", "hi", "hey", "good morning", "good afternoon", "good evening",
                        "how are you", "what's up", "greetings"]
    if any(phrase in query_lower for phrase in greeting_phrases):
        return QueryIntent.GREETING

    # Check for AUTOSAR-related intent
    autosar_keywords = [
        "autosar", "ecu", "swc", "software component", "rte", "runtime environment",
        "bsw", "basic software", "application layer", "platform", "module",
        "configuration", "parameter", "interface", "port", "service",
        "com", "can", "lin", "flexray", "ethernet", "diagnostic",
        "dem", "dcm", "nvm", "os", "scheduler", "task", "interrupt"
    ]
    if any(keyword in query_lower for keyword in autosar_keywords):
        return QueryIntent.AUTOSAR_RELATED

    # Check for technical query intent
    technical_keywords = [
        "how to", "implement", "specification", "api", "protocol",
        "architecture", "design", "function", "method", "algorithm"
    ]
    if any(keyword in query_lower for keyword in technical_keywords):
        return QueryIntent.TECHNICAL_QUERY

    # Default to general intent
    return QueryIntent.GENERAL


async def save_personal_info(user_id: str, query: str, llm_client: Any) -> Dict[str, Any]:
    """
    DEPRECATED: This function is kept for backward compatibility but not used in AUTOSAR domain.
    """
    """
    Extract and save personal information from a user query.

    Args:
        user_id: User identifier
        query: The user query containing personal information
        llm_client: Client for LLM operations (GeminiClient or OllamaClient)

    Returns:
        Dict with status of the operation
    """
    logger.info(f"Extracting personal info for user {user_id}")

    if not user_id:
        logger.error("No user_id provided for personal info extraction")
        return {"status": "error", "message": "No user ID provided"}

    try:
        # Ensure user_data directory exists
        os.makedirs("user_data", exist_ok=True)

        # File path for user's personal info
        file_path = f"user_data/{user_id}_personal_info.json"

        # Load existing data if available
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, "r") as f:
                    existing_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning(
                    f"Could not parse existing data for user {user_id}, starting fresh")
                existing_data = {}

        # Use LLM to extract structured info from the query
        if llm_client:
            system_prompt = """
            You are an expert system that extracts personal information from user messages.
            Extract any personal information such as:
            - Name
            - Age
            - Medical conditions
            - Medications
            - Allergies
            - Contact information
            - Any other personal health data

            Return the extracted information as a JSON object with the appropriate fields.
            Only include fields that were explicitly mentioned in the message.
            If you can't extract any personal information, return an empty JSON object {}.
            
            Example output format:
            {
                "name": "John Doe",
                "age": 45,
                "conditions": ["Type 2 diabetes", "hypertension"],
                "medications": ["metformin 500mg twice daily", "lisinopril"],
                "contact": {"email": "john@example.com"}
            }
            """

            # Format to be passed for JSON structured output
            format_schema = {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "number"},
                    "conditions": {"type": "array", "items": {"type": "string"}},
                    "medications": {"type": "array", "items": {"type": "string"}},
                    "allergies": {"type": "array", "items": {"type": "string"}},
                    "contact": {"type": "object"}
                }
            }

            # Handle different client types
            is_gemini = isinstance(llm_client, GeminiClient)

            if is_gemini:
                # Gemini prefers structured schema
                response = llm_client.generate(
                    user_prompt=query,
                    system_prompt=system_prompt,
                    response_schema=format_schema
                )
            else:
                # Ollama format (keep async if Ollama supports it)
                response = await llm_client.generate(
                    prompt=query,
                    system_prompt=system_prompt,
                    format={"type": "json"}
                )

            extracted_data = {}
            try:
                # Handle different response formats based on client type
                if is_gemini:
                    # GeminiClient returns LLMResponse object with message attribute
                    if hasattr(response, 'message'):
                        content = response.message
                    elif isinstance(response, dict):
                        content = response.get("message", "{}")
                    else:
                        content = str(response)

                    # Parse JSON from content
                    if isinstance(content, str):
                        extracted_data = json.loads(content)
                    else:
                        extracted_data = content
                else:
                    # Ollama format
                    content = response.get("message", {}).get("content", "{}")
                    if isinstance(content, str):
                        extracted_data = json.loads(content)
                    else:
                        extracted_data = content

                logger.info(
                    f"Successfully extracted personal info: {list(extracted_data.keys())}")
            except json.JSONDecodeError:
                logger.error(f"Failed to parse LLM response as JSON")
                extracted_data = {}
        else:
            # Simple fallback extraction if LLM is not available
            extracted_data = {"raw_query": query}

        # Add metadata
        extracted_data["_metadata"] = {
            "timestamp": datetime.now().isoformat(),
            "raw_query": query
        }

        # Merge with existing data
        for key, value in extracted_data.items():
            if key != "_metadata":
                if key in existing_data:
                    # If the field already exists and is a list, append new values
                    if isinstance(existing_data[key], list) and isinstance(value, list):
                        existing_data[key].extend(value)
                    # If field exists but types don't match, prefer the new value
                    else:
                        existing_data[key] = value
                else:
                    # Add new field
                    existing_data[key] = value

        # Update metadata
        if "_metadata" not in existing_data:
            existing_data["_metadata"] = {"queries": []}

        if "queries" not in existing_data["_metadata"]:
            existing_data["_metadata"]["queries"] = []

        # Add this query to the history
        existing_data["_metadata"]["queries"].append({
            "timestamp": datetime.now().isoformat(),
            "query": query
        })

        # Save updated data
        with open(file_path, "w") as f:
            json.dump(existing_data, f, indent=2)

        logger.info(f"Personal info saved for user {user_id}")
        return {
            "status": "success",
            "message": "Personal information saved successfully",
            "file_path": file_path
        }

    except Exception as e:
        logger.error(f"Error saving personal info: {e}", exc_info=True)
        return {"status": "error", "message": f"Error saving personal information: {str(e)}"}
