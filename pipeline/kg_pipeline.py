"""
Knowledge Graph Pipeline Module

This module contains the high-level pipeline functions for processing queries
through the complete knowledge graph pipeline.
"""
import json
import logging
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from backend.pipeline.query_analyzer import QueryIntent, analyze_query
from backend.pipeline.keyword_extractor import extract_keywords
from backend.pipeline.kg_query_processor import (
    KnowledgeGraphQueryProcessor,
    convert_conversation_history_to_list
)
from backend.retrieval.triple_level_retrieval import (
    save_retrieval_result,
    sanitize_filename
)
# Nebula Graph client will be passed as Any type
from backend.utils.logging import get_logger

# Configure logger
logger = get_logger(__name__)


def save_llm_answer(
    query: str,
    answer: str,
    result: dict,
    prompt_file: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Save LLM answer to file.

    Args:
        query: Original query
        answer: Answer from LLM
        result: Dictionary containing all results
        prompt_file: Path to saved prompt file (optional)
        output_dir: Output directory (optional, defaults to output_pipeline_retrie at root)

    Returns:
        Path to saved file or None if error
    """
    try:
        # Create output directory if not provided
        if output_dir is None:
            # Get root path from current file (backend/pipeline/kg_pipeline.py)
            # Go up 2 levels to reach root: backend/pipeline -> backend -> root
            current_file = Path(__file__)
            root_path = current_file.parent.parent.parent
            output_dir = root_path / "output_pipeline_retrie"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Create filename based on query and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_sanitized = sanitize_filename(query, max_length=50)
        filename = f"answer_{timestamp}_{query_sanitized}.txt"
        file_path = output_dir / filename

        # Create file content
        content_parts = []

        content_parts.append("=" * 80)
        content_parts.append("LLM ANSWER - KNOWLEDGE GRAPH QUERY")
        content_parts.append("=" * 80)
        content_parts.append("")
        content_parts.append(f"Query: {query}")
        content_parts.append(
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Intent (handle both QueryIntent enum and string)
        intent = result.get('intent')
        if intent:
            if hasattr(intent, 'name'):
                content_parts.append(f"Intent: {intent.name}")
            elif hasattr(intent, 'value'):
                content_parts.append(f"Intent: {intent.value}")
            else:
                content_parts.append(f"Intent: {intent}")
        else:
            content_parts.append("Intent: N/A")

        if prompt_file:
            content_parts.append(f"Prompt File: {prompt_file}")

        content_parts.append("")
        content_parts.append("=" * 80)
        content_parts.append("ANSWER")
        content_parts.append("=" * 80)
        content_parts.append("")
        content_parts.append(answer)
        content_parts.append("")

        # Metadata
        content_parts.append("=" * 80)
        content_parts.append("METADATA")
        content_parts.append("=" * 80)

        # Keywords (if AUTOSAR related)
        if result.get('intent') == QueryIntent.AUTOSAR_RELATED or (
            isinstance(result.get('intent'), str) and
            result.get('intent') == QueryIntent.AUTOSAR_RELATED.value
        ):
            high_level_keywords = result.get('high_level_keywords', [])
            low_level_keywords = result.get('low_level_keywords', [])
            if high_level_keywords:
                content_parts.append(
                    f"High-level keywords: {', '.join(high_level_keywords)}")
            if low_level_keywords:
                content_parts.append(
                    f"Low-level keywords: {', '.join(low_level_keywords)}")

        # Retrieval statistics
        if result.get('retrieval_result'):
            retrieval = result['retrieval_result']
            content_parts.append(f"Retrieval Statistics:")
            content_parts.append(
                f"  - Level 1 nodes: {len(retrieval.get('level1_nodes', []))}")
            content_parts.append(
                f"  - Level 2 nodes: {len(retrieval.get('level2_nodes', []))}")
            content_parts.append(
                f"  - Relationships: {len(retrieval.get('relationships', []))}")

        content_parts.append(f"Answer length: {len(answer)} characters")

        # Write file
        full_content = "\n".join(content_parts)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        logger.info(f"Saved LLM answer to: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Error saving LLM answer: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None


async def process_kg_query(
    query: str,
    intent: QueryIntent,
    high_level_keywords: List[str],
    low_level_keywords: List[str],
    conversation_history: Optional[List[Dict[str, str]]] = None,
    nebula_client: Any = None,
    gemini_client: Any = None,
    user_id: Optional[str] = None,
    response_type: str = "concise"
) -> Dict[str, Any]:
    """
    Process a complete query against the AUTOSAR knowledge graph in Nebula Graph.

    Args:
        query: User's natural language query
        intent: Analyzed query intent (GREETING, AUTOSAR_RELATED, etc.)
        high_level_keywords: High-level conceptual keywords extracted from query
        low_level_keywords: Low-level specific keywords extracted from query
        conversation_history: Optional conversation history for context
        nebula_client: Nebula Graph database client
        gemini_client: Gemini client for text generation
        user_id: Optional user identifier (for future use)
        response_type: Type of response ('concise' or 'detailed')

    Returns:
        Dictionary with processed query results
    """
    start_time = time.time()
    logger.info(f"Processing query: '{query}' with intent: {intent}")
    logger.info(f"High-level keywords: {high_level_keywords}")
    logger.info(f"Low-level keywords: {low_level_keywords}")

    # Create a processor instance
    processor = KnowledgeGraphQueryProcessor(
        nebula_client=nebula_client,
        gemini_client=gemini_client
    )

    # Process the query using the processor
    result = await processor.process_query(
        query=query,
        intent=intent,
        high_level_keywords=high_level_keywords,
        low_level_keywords=low_level_keywords,
        conversation_history=conversation_history,
        user_id=user_id,
        response_type=response_type
    )

    # Add processing time
    processing_time = time.time() - start_time
    logger.info(f"Query processed in {processing_time:.2f} seconds")

    # If result doesn't already include processing time, add it
    if "processing_time_seconds" not in result:
        result["processing_time_seconds"] = processing_time

    return result


async def process_query_full_pipeline(
    query: str,
    conversation_history=None,
    clients=None
) -> Optional[Dict[str, Any]]:
    """
    Process query through the full pipeline for AUTOSAR knowledge graph:
    1. Intent classification
    2. Keyword extraction (if AUTOSAR_RELATED)
    3. Knowledge graph retrieval from Nebula Graph
    4. Generate answer from LLM Gemini

    Args:
        query: User's query string
        conversation_history: Optional conversation history (string or list of dicts)
        clients: Dictionary containing all necessary clients:
            - gemini_client: Gemini LLM client
            - nebula_client: Nebula Graph database client

    Returns:
        Dictionary containing processing results or None if error
    """
    logger.info("=" * 80)
    logger.info("🚀 PROCESSING QUERY - FULL PIPELINE")
    logger.info("=" * 80)
    logger.info(f"📝 Query: {query}")
    logger.info("=" * 80)

    if not clients:
        logger.error("No clients provided, cannot process")
        return None

    gemini_client = clients.get("gemini_client")
    if not gemini_client:
        logger.error("No Gemini client provided")
        return None

    result = {
        "query": query,
        "intent": None,
        "high_level_keywords": [],
        "low_level_keywords": [],
        "retrieval_result": None,
        "formatted_text": "",
        "saved_file": None,
        "llm_answer": None,
        "answer_file": None
    }

    # STEP 1: Intent classification
    logger.info("-" * 80)
    logger.info("📋 STEP 1: INTENT CLASSIFICATION")
    logger.info("-" * 80)
    try:
        intent = await analyze_query(
            query=query,
            conversation_history=conversation_history,
            client=gemini_client
        )
        result["intent"] = intent
        logger.info(f"✅ Intent: {intent.name} ({intent.value})")
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return result

    # STEP 2: Keyword extraction (only for AUTOSAR_RELATED)
    if intent == QueryIntent.AUTOSAR_RELATED:
        logger.info("-" * 80)
        logger.info("🔑 STEP 2: KEYWORD EXTRACTION")
        logger.info("-" * 80)
        try:
            high_level_keywords, low_level_keywords = await extract_keywords(
                query=query,
                conversation_history=conversation_history,
                llm_client=gemini_client
            )
            result["high_level_keywords"] = high_level_keywords
            result["low_level_keywords"] = low_level_keywords

            logger.info(f"✅ High-level keywords ({len(high_level_keywords)}):")
            for i, keyword in enumerate(high_level_keywords, 1):
                logger.info(f"   {i}. {keyword}")

            logger.info(f"\n✅ Low-level keywords ({len(low_level_keywords)}):")
            for i, keyword in enumerate(low_level_keywords, 1):
                logger.info(f"   {i}. {keyword}")
        except Exception as e:
            logger.error(f"Error extracting keywords: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return result

        # STEP 3 & 4: Process query using kg_query_processor
        if high_level_keywords or low_level_keywords:
            logger.info("-" * 80)
            logger.info(
                "🔍 STEP 3 & 4: PROCESSING QUERY WITH KG_QUERY_PROCESSOR")
            logger.info("-" * 80)

            nebula_client = clients.get("nebula_client")

            if not nebula_client:
                logger.error(
                    "Missing required clients for knowledge graph query")
                logger.error(f"   - Nebula Graph: {'✅' if nebula_client else '❌'}")
                return result

            try:
                # Format conversation history for process_kg_query
                history_list = convert_conversation_history_to_list(
                    conversation_history)

                # Process query using kg_query_processor
                logger.info(f"\n📌 Keywords for query:")
                logger.info(f"   High-level: {high_level_keywords}")
                logger.info(f"   Low-level: {low_level_keywords}")

                processed_result = await process_kg_query(
                    query=query,
                    intent=intent,
                    high_level_keywords=high_level_keywords,
                    low_level_keywords=low_level_keywords,
                    conversation_history=history_list,
                    nebula_client=nebula_client,
                    gemini_client=gemini_client,
                    response_type="detailed"
                )

                # Update result with processed_result data
                result["retrieval_result"] = processed_result.get(
                    "retrieval_result")
                result["formatted_text"] = processed_result.get(
                    "formatted_text", "")
                result["llm_answer"] = processed_result.get("response", "")

                # Display retrieval results
                retrieval_result = result.get("retrieval_result")
                if retrieval_result:
                    l1_nodes = len(retrieval_result.get('level1_nodes', []))
                    l2_nodes = len(retrieval_result.get('level2_nodes', []))
                    relationships = len(
                        retrieval_result.get('relationships', []))

                    logger.info(f"\n✅ Retrieval results:")
                    logger.info(f"   - Level 1 nodes: {l1_nodes}")
                    logger.info(f"   - Level 2 nodes: {l2_nodes}")
                    logger.info(f"   - Relationships: {relationships}")

                # Save retrieval result to file
                formatted_text = result.get("formatted_text", "")
                if formatted_text:
                    saved_file = save_retrieval_result(
                        query, formatted_text, result, conversation_history)
                    if saved_file:
                        result["saved_file"] = saved_file
                        logger.info(f"💾 Saved prompt to: {saved_file}")

                # Save LLM answer to file
                llm_answer = result.get("llm_answer", "")
                if llm_answer:
                    logger.info("\n✅ Answer from LLM:")
                    logger.info("=" * 80)
                    if len(llm_answer) > 1000:
                        logger.info(llm_answer[:1000])
                        logger.info(
                            f"\n... (remaining {len(llm_answer) - 1000} characters)")
                    else:
                        logger.info(llm_answer)
                    logger.info("=" * 80)

                    answer_file = save_llm_answer(
                        query, llm_answer, result, result.get("saved_file"))
                    if answer_file:
                        result["answer_file"] = answer_file
                        logger.info(f"\n💾 Saved answer to: {answer_file}")

            except Exception as e:
                logger.error(f"Error processing query: {e}")
                import traceback
                logger.error(traceback.format_exc())
        else:
            logger.warning("\n⚠️  No keywords to query")
    else:
        logger.info(
            f"\n⏭️  Intent is {intent.name} - Skipping keyword extraction and graph query")
        logger.info("   (Only processing for AUTOSAR_RELATED)")

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("📊 SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Query: {query}")
        logger.info(
            f"Intent: {result['intent'].name if result['intent'] else 'N/A'}")
    if result['intent'] == QueryIntent.AUTOSAR_RELATED:
        logger.info(
            f"High-level keywords: {len(result['high_level_keywords'])}")
        logger.info(f"Low-level keywords: {len(result['low_level_keywords'])}")
        if result['retrieval_result']:
            logger.info(
                f"Level 1 nodes: {len(result['retrieval_result'].get('level1_nodes', []))}")
            logger.info(
                f"Level 2 nodes: {len(result['retrieval_result'].get('level2_nodes', []))}")
            logger.info(
                f"Relationships: {len(result['retrieval_result'].get('relationships', []))}")
    logger.info("=" * 80)

    return result
