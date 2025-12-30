import json
import logging
import time
import asyncio
import os
from enum import Enum
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
from datetime import datetime
from pydantic import BaseModel, Field

from backend.pipeline.query_analyzer import QueryIntent, analyze_query
from backend.pipeline.keyword_extractor import extract_keywords
from backend.retrieval.triple_level_retrieval import (
    retrieve_from_knowledge_graph,
    format_retrieval_results,
    create_rag_prompt,
    save_retrieval_result,
    sanitize_filename,
    self_refine,
)
# Nebula Graph client will be passed as Any type
from backend.utils.logging import get_logger
from backend.pipeline_prompts import (
    PROMPTS,
    DEFAULT_SYSTEM_PROMPT,
    ERROR_RESPONSE_MESSAGE,
    EMPTY_RESPONSE_MESSAGE
)

# Configure logger
logger = get_logger(__name__)


async def generate_response_from_rag_prompt(
    rag_prompt: str,
    gemini_client: Any,
    empty_response_message: str = EMPTY_RESPONSE_MESSAGE,
    error_response_message: str = ERROR_RESPONSE_MESSAGE,
    return_none_on_error: bool = False
) -> Optional[str]:
    """
    Generate response from LLM using RAG prompt.

    Args:
        rag_prompt: Pre-formatted RAG prompt containing query, knowledge base, and conversation history
        gemini_client: Gemini client to call LLM
        empty_response_message: Message to return if LLM returns empty response
        error_response_message: Message to return if error occurs
        return_none_on_error: If True, return None on error instead of error message

    Returns:
        Generated response text or None if return_none_on_error is True and error occurs
    """
    try:
        # Generate response using Gemini with RAG prompt
        # Since generate() is a sync method, run it in executor to avoid blocking event loop
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: gemini_client.generate(
                user_prompt=rag_prompt,
                system_prompt=DEFAULT_SYSTEM_PROMPT
            )
        )

        # Extract message from LLMResponse object
        answer = extract_llm_response(response)

        if not answer:
            logger.warning("LLM returned empty response")
            return None if return_none_on_error else empty_response_message

        return answer

    except Exception as e:
        logger.error(f"Error generating response from RAG prompt: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None if return_none_on_error else error_response_message


def extract_llm_response(response: Any) -> str:
    """
    Extract message text from LLM response object.

    Args:
        response: LLM response object (can be LLMResponse, dict, or string)

    Returns:
        Extracted message text as string
    """
    if hasattr(response, 'message'):
        return str(response.message).strip()
    elif hasattr(response, 'text'):
        return str(response.text).strip()
    elif isinstance(response, dict):
        answer = response.get("message", "").strip()
        if isinstance(answer, dict):
            answer = answer.get("content", "").strip()
        return answer
    else:
        return str(response).strip()


def format_conversation_history(
    conversation_history: Optional[Union[str, List[Dict[str, str]]]]
) -> Optional[str]:
    """
    Format conversation history for RAG prompt.

    Args:
        conversation_history: Conversation history as string or list of dicts

    Returns:
        Formatted conversation history string or None
    """
    if not conversation_history:
        return None

    if isinstance(conversation_history, str):
        return conversation_history

    if isinstance(conversation_history, list):
        return "\n".join([
            f"{msg.get('role', 'unknown').capitalize()}: {msg.get('content', '')}"
            for msg in conversation_history[-3:]
        ])

    return None


def convert_conversation_history_to_list(
    conversation_history: Optional[Union[str, List[Dict[str, str]]]]
) -> Optional[List[Dict[str, str]]]:
    """
    Convert conversation history to list format for process_kg_query.

    Args:
        conversation_history: Conversation history as string or list of dicts

    Returns:
        List of conversation messages or None
    """
    if not conversation_history:
        return None

    if isinstance(conversation_history, str):
        return [{"role": "user", "content": conversation_history}]

    if isinstance(conversation_history, list):
        return conversation_history

    return None


async def retrieve_and_format_kg_results(
    query: str,
    high_level_keywords: List[str],
    low_level_keywords: List[str],
    nebula_client: Any,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> Tuple[Dict[str, Any], str]:
    """
    Retrieve and format knowledge graph results from Nebula Graph.

    Args:
        query: User query for context
        high_level_keywords: High-level keywords for retrieval
        low_level_keywords: Low-level keywords for retrieval
        nebula_client: Nebula Graph database client
        top_k: Number of top results to retrieve
        similarity_threshold: Similarity threshold for filtering nodes (for future use)

    Returns:
        Tuple of (retrieval_result dict, formatted_text string)
    """
    nodes, _, relations = await retrieve_from_knowledge_graph(
        high_level_keywords=high_level_keywords,
        low_level_keywords=low_level_keywords,
        nebula_client=nebula_client,
        top_k=top_k,
        similarity_threshold=similarity_threshold
    )

    # Self-refine if needed (simplified for Nebula Graph)
    nodes_refined, relations_refined = await self_refine(
        nodes=nodes,
        relationships=relations,
        nebula_client=nebula_client,
        gemini_client=None,  # Can be passed if needed
        query=query,
        max_iterations=2
    )

    retrieval_result = {
        "level1_nodes": nodes_refined,
        "level2_nodes": [],  # Not used in simplified Nebula Graph version
        "relationships": relations_refined
    }

    formatted_text = ""
    if retrieval_result.get('level1_nodes') or retrieval_result.get('level2_nodes'):
        formatted_text = format_retrieval_results(
            level1_nodes=retrieval_result.get('level1_nodes', []),
            level2_nodes=retrieval_result.get('level2_nodes', []),
            relationships=retrieval_result.get('relationships', [])
        )
        logger.info(
            f"Retrieved {len(retrieval_result.get('level1_nodes', []))} Level 1 nodes, "
            f"{len(retrieval_result.get('level2_nodes', []))} Level 2 nodes, "
            f"{len(retrieval_result.get('relationships', []))} relationships"
        )
    else:
        logger.warning("No nodes retrieved from knowledge graph")
        formatted_text = "No relevant information found in the knowledge base for this query."

    return retrieval_result, formatted_text


class ResponseFormat(BaseModel):
    """Pydantic model for structured response output."""
    response: str = Field(...,
                          description="The response text answering the user's query")
    sources: List[str] = Field(
        default_factory=list, description="Source references used in the response")
    confidence: float = Field(
        default=0.0, description="Confidence score for the response")


class KnowledgeGraphQueryProcessor:
    """
    Processes queries against the two-level knowledge graph.

    Implements a comprehensive query processing pipeline that:
    1. Analyzes query intent
    2. Retrieves relevant information from knowledge graph
    3. Generates appropriate responses based on intent and context
    4. Handles different query types with specialized prompts
    """

    def __init__(
        self,
        nebula_client: Any,
        gemini_client: Any,
        max_tokens: int = 10000,
        top_k: int = 5
    ):
        """
        Initialize the query processor for AUTOSAR knowledge graph.

        Args:
            nebula_client: Nebula Graph database client
            gemini_client: Gemini LLM client for response generation
            max_tokens: Maximum tokens for context
            top_k: Number of top results to retrieve
        """
        self.nebula_client = nebula_client
        self.gemini_client = gemini_client
        self.max_tokens = max_tokens
        self.top_k = top_k

        # Define base data directory for personal information
        self.user_data_dir = os.path.join(os.getcwd(), "user_data")
        os.makedirs(self.user_data_dir, exist_ok=True)

    async def process_query(
        self,
        query: str,
        intent: QueryIntent,
        high_level_keywords: List[str],
        low_level_keywords: List[str],
        conversation_history: Optional[List[Dict[str, str]]] = None,
        user_id: Optional[str] = None,
        response_type: str = "concise"
    ) -> Dict[str, Any]:
        """
        Main method to process a user query against the knowledge graph.

        Args:
            query: User's natural language query
            intent: Analyzed query intent
            high_level_keywords: High-level keywords extracted from query
            low_level_keywords: Low-level keywords extracted from query
            conversation_history: Optional list of previous conversation messages
            user_id: Optional user identifier for personal info management
            response_type: Type of response ('concise' or 'detailed')

        Returns:
            Dictionary containing query processing results
        """
        try:
            start_time = datetime.now()

            # Log the incoming query and intent
            logger.info(
                f"Processing query: '{query}' with intent: {intent.name}")
            if high_level_keywords or low_level_keywords:
                logger.info(f"High-level keywords: {high_level_keywords}")
                logger.info(f"Low-level keywords: {low_level_keywords}")

            # Create initial result structure
            result = {
                "query": query,
                "intent": intent.value,
                "keywords": {
                    "high_level": high_level_keywords,
                    "low_level": low_level_keywords
                },
                "response": "",
                "processing_time_seconds": 0,
                "sources": []
            }

            # Process based on intent type
            if intent == QueryIntent.GREETING:
                # For greetings, generate a friendly response without KG retrieval
                result["response"] = await self._generate_greeting_response(
                    query,
                    conversation_history
                )

            elif intent == QueryIntent.TECHNICAL_QUERY:
                # For technical queries, retrieve from knowledge graph and generate response
                if high_level_keywords or low_level_keywords:
                    try:
                        logger.info(
                            f"Retrieving knowledge graph for technical query with "
                            f"high-level keywords: {high_level_keywords}, "
                            f"low-level keywords: {low_level_keywords}"
                        )

                        # Retrieve and format knowledge graph results
                        retrieval_result, formatted_text = await retrieve_and_format_kg_results(
                            query=query,
                            high_level_keywords=high_level_keywords,
                            low_level_keywords=low_level_keywords,
                            nebula_client=self.nebula_client,
                            top_k=self.top_k,
                            similarity_threshold=0.7
                        )

                        # Format conversation history for RAG prompt
                        history_text = format_conversation_history(
                            conversation_history)

                        # Create RAG prompt
                        rag_prompt = create_rag_prompt(
                            query=query,
                            formatted_text=formatted_text,
                            conversation_history=history_text
                        )

                        # Generate response using RAG prompt
                        result["response"] = await self._generate_general_response_with_rag(
                            rag_prompt=rag_prompt,
                            response_type=response_type
                        )

                        # Store retrieval result
                        result["retrieval_result"] = retrieval_result
                        result["formatted_text"] = formatted_text
                        result["sources"] = retrieval_result.get("sources", [])
                        result["context"] = formatted_text

                    except Exception as e:
                        logger.error(
                            f"Error in knowledge graph retrieval for technical query: {str(e)}")
                        result["response"] = (
                            "I encountered an error while searching the knowledge base. "
                            "Please try rephrasing your question."
                        )
                else:
                    # No keywords, generate general response without KG context
                    result["response"] = await self._generate_general_response(
                        query=query,
                        context="",
                        conversation_history=conversation_history,
                        response_type=response_type
                    )

            elif intent == QueryIntent.AUTOSAR_RELATED:
                # For AUTOSAR queries, retrieve from knowledge graph and generate response
                # Only attempt retrieval if we have keywords
                if high_level_keywords or low_level_keywords:
                    try:
                        logger.info(
                            f"Retrieving knowledge graph with high-level keywords: {high_level_keywords}, "
                            f"low-level keywords: {low_level_keywords}"
                        )

                        # Retrieve and format knowledge graph results
                        retrieval_result, formatted_text = await retrieve_and_format_kg_results(
                            query=query,
                            high_level_keywords=high_level_keywords,
                            low_level_keywords=low_level_keywords,
                            nebula_client=self.nebula_client,
                            top_k=self.top_k,
                            similarity_threshold=0.7
                        )

                        # Format conversation history for RAG prompt
                        history_text = format_conversation_history(
                            conversation_history)

                        # Create RAG prompt using the template
                        rag_prompt = create_rag_prompt(
                            query=query,
                            formatted_text=formatted_text,
                            conversation_history=history_text
                        )

                        # Generate response using RAG prompt
                        result["response"] = await self._generate_autosar_response_with_rag(
                            rag_prompt=rag_prompt,
                            response_type=response_type
                        )

                        # Store retrieval result in response
                        result["retrieval_result"] = retrieval_result
                        result["formatted_text"] = formatted_text

                        # Include sources if available
                        result["sources"] = retrieval_result.get("sources", [])

                    except Exception as e:
                        logger.error(
                            f"Error in knowledge graph retrieval: {str(e)}")
                        result["response"] = (
                            "I encountered an error while searching the knowledge base. "
                            "Please try rephrasing your question."
                        )
                else:
                    logger.warning("No keywords provided for healthcare query")
                    result["response"] = (
                        "I couldn't extract specific keywords from your query. "
                        "Could you please rephrase your question with more specific medical terms?"
                    )

            else:  # QueryIntent.GENERAL
                # For general queries, retrieve from KG and generate general response
                if high_level_keywords or low_level_keywords:
                    try:
                        logger.info(
                            f"Retrieving knowledge graph for general query with "
                            f"high-level keywords: {high_level_keywords}, "
                            f"low-level keywords: {low_level_keywords}"
                        )

                        # Retrieve and format knowledge graph results
                        retrieval_result, formatted_text = await retrieve_and_format_kg_results(
                            query=query,
                            high_level_keywords=high_level_keywords,
                            low_level_keywords=low_level_keywords,
                            nebula_client=self.nebula_client,
                            top_k=self.top_k,
                            similarity_threshold=0.7
                        )

                        # Format conversation history for RAG prompt
                        history_text = format_conversation_history(
                            conversation_history)

                        # Create RAG prompt
                        rag_prompt = create_rag_prompt(
                            query=query,
                            formatted_text=formatted_text,
                            conversation_history=history_text
                        )

                        # Generate response using RAG prompt
                        result["response"] = await self._generate_general_response_with_rag(
                            rag_prompt=rag_prompt,
                            response_type=response_type
                        )

                        # Store retrieval result
                        result["retrieval_result"] = retrieval_result
                        result["formatted_text"] = formatted_text
                        result["sources"] = retrieval_result.get("sources", [])
                        result["context"] = formatted_text

                    except Exception as e:
                        logger.error(
                            f"Error in knowledge graph retrieval for general query: {str(e)}")
                        result["response"] = (
                            "I encountered an error while searching the knowledge base. "
                            "Please try rephrasing your question."
                        )
                else:
                    # No keywords, generate general response without KG context
                    result["response"] = await self._generate_general_response(
                        query=query,
                        context="",
                        conversation_history=conversation_history,
                        response_type=response_type
                    )

            # Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            result["processing_time_seconds"] = processing_time

            logger.info(f"Query processed in {processing_time:.2f} seconds")
            return result

        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "query": query,
                "error": str(e),
                "response": "I'm sorry, but I couldn't process your query successfully."
            }

    async def _generate_greeting_response(
        self,
        query: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a friendly greeting response.

        Args:
            query: User's greeting query
            conversation_history: Optional conversation history

        Returns:
            Greeting response text
        """
        # Format conversation history for prompt
        history_text = conversation_history[-3:
                                            ] if conversation_history else ''

        # Get prompt template from pipeline_prompts
        prompt_template = PROMPTS.get("greeting_response", "")
        prompt = prompt_template.format(
            query=query,
            conversation_history=history_text
        )

        try:
            # Generate greeting using Gemini
            response = await self.gemini_client.generate(
                prompt=prompt
            )

            # Extract content from response
            answer = extract_llm_response(response)
            if not answer:
                return "Hello! How can I help you with your health questions today?"
            return answer

        except Exception as e:
            logger.error(f"Error generating greeting response: {str(e)}")
            return "Hello! How can I help you with your health questions today?"

    # The rest of the methods remain the same...

    async def _save_personal_info(
        self,
        query: str,
        user_id: str
    ) -> Dict[str, Any]:
        """
        Extract and save personal information from user query.

        Args:
            query: User's query containing personal information
            user_id: User identifier

        Returns:
            Dict with operation status and message
        """
        # Implementation remains the same as before...
        # Create filename for user data
        file_path = os.path.join(
            self.user_data_dir, f"{user_id}_personal_info.json")

        # Load existing data if available
        existing_data = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    existing_data = json.load(f)
            except Exception as e:
                logger.error(f"Error loading existing user data: {str(e)}")

        # Get prompt template from pipeline_prompts
        prompt_template = PROMPTS.get("personal_info_extraction_kg", "")
        prompt = prompt_template.format(query=query)

        try:
            # Use Gemini to extract structured information
            response = await self.gemini_client.generate(
                prompt=prompt
            )

            # Parse the response
            content = response.get("message", {}).get("content", "{}")
            if isinstance(content, str):
                try:
                    extracted_data = json.loads(content)
                except json.JSONDecodeError:
                    logger.error(
                        f"Failed to parse response as JSON: {content}")
                    extracted_data = {
                        "extracted_fields": {},
                        "acknowledgment": "I've noted your information, but there was an issue processing it. Could you try again?"
                    }
            else:
                extracted_data = content

            # Get extracted fields and acknowledgment
            extracted_fields = extracted_data.get("extracted_fields", {})
            acknowledgment = extracted_data.get(
                "acknowledgment", "I've saved your information to your profile.")

            # Update existing data with new data
            for key, value in extracted_fields.items():
                existing_data[key] = value

            # Add timestamp
            existing_data["last_updated"] = datetime.now().isoformat()

            # Save to file
            with open(file_path, 'w') as f:
                json.dump(existing_data, f, indent=2)

            logger.info(f"Saved personal information for user {user_id}")

            return {
                "status": "success",
                "message": acknowledgment,
                "file_path": file_path
            }

        except Exception as e:
            logger.error(f"Error saving personal information: {str(e)}")
            return {
                "status": "error",
                "message": "I had trouble saving your information. Could you try again?"
            }

    async def _generate_autosar_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        response_type: str = "concise",
        grounding_context: str = ""
    ) -> str:
        """
        Generate a response for AUTOSAR-related queries.

        Args:
            query: User's query
            context: Knowledge graph context
            conversation_history: Optional conversation history
            response_type: Type of response ('concise' or 'detailed')
            grounding_context: Additional context from web search

        Returns:
            Generated response
        """
        # Format conversation history for prompt
        history_text = conversation_history[-3:
                                            ] if conversation_history else ''

        # Get prompt template from pipeline_prompts (use general_response or create autosar_response)
        prompt_template = PROMPTS.get(
            "autosar_response", PROMPTS.get("general_response", ""))
        prompt = prompt_template.format(
            query=query,
            context=context,
            grounding_context=grounding_context,
            conversation_history=history_text,
            response_type=response_type
        )

        try:
            # Generate response using Gemini
            response = await self.gemini_client.generate(
                prompt=prompt
            )

            # Extract content from response
            answer = extract_llm_response(response)
            if not answer:
                return "I don't have enough information about that specific AUTOSAR topic in my knowledge sources. Would you like to ask about something else related to AUTOSAR?"
            return answer

        except Exception as e:
            logger.error(f"Error generating AUTOSAR response: {str(e)}")
            return ERROR_RESPONSE_MESSAGE

    async def _generate_general_response(
        self,
        query: str,
        context: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        response_type: str = "concise"
    ) -> str:
        """
        Generate a response for general queries.

        Args:
            query: User's query
            context: Knowledge graph context
            conversation_history: Optional conversation history
            response_type: Type of response ('concise' or 'detailed')

        Returns:
            Generated response
        """
        # Format conversation history for prompt
        history_text = conversation_history[-3:
                                            ] if conversation_history else ''

        # Get prompt template from pipeline_prompts
        prompt_template = PROMPTS.get("general_response", "")
        prompt = prompt_template.format(
            query=query,
            context=context,
            conversation_history=history_text,
            response_type=response_type
        )

        try:
            # Generate response using Gemini
            response = await self.gemini_client.generate(
                prompt=prompt
            )

            # Extract content from response
            answer = extract_llm_response(response)
            if not answer:
                return "I don't have enough information to answer that question properly. Is there something else I can help you with?"
            return answer

        except Exception as e:
            logger.error(f"Error generating general response: {str(e)}")
            return ERROR_RESPONSE_MESSAGE

    async def _generate_autosar_response_with_rag(
        self,
        rag_prompt: str,
        response_type: str = "concise"
    ) -> str:
        """
        Generate an AUTOSAR response using RAG prompt.

        Args:
            rag_prompt: Pre-formatted RAG prompt containing query, knowledge base, and conversation history
            response_type: Type of response ('concise' or 'detailed')

        Returns:
            Generated response text
        """
        answer = await generate_response_from_rag_prompt(
            rag_prompt=rag_prompt,
            gemini_client=self.gemini_client,
            empty_response_message=EMPTY_RESPONSE_MESSAGE,
            error_response_message=ERROR_RESPONSE_MESSAGE,
            return_none_on_error=False
        )
        return answer or ERROR_RESPONSE_MESSAGE

    async def _generate_general_response_with_rag(
        self,
        rag_prompt: str,
        response_type: str = "concise"
    ) -> str:
        """
        Generate a general response using RAG prompt.

        Args:
            rag_prompt: Pre-formatted RAG prompt containing query, knowledge base, and conversation history
            response_type: Type of response ('concise' or 'detailed')

        Returns:
            Generated response text
        """
        answer = await generate_response_from_rag_prompt(
            rag_prompt=rag_prompt,
            gemini_client=self.gemini_client,
            empty_response_message=EMPTY_RESPONSE_MESSAGE,
            error_response_message=ERROR_RESPONSE_MESSAGE,
            return_none_on_error=False
        )
        return answer or ERROR_RESPONSE_MESSAGE
