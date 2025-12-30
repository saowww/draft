"""
AUTOSAR Knowledge Graph Retriever for Nebula Graph

This module retrieves relevant information from the AUTOSAR knowledge graph
stored in Nebula Graph based on high-level and low-level keywords extracted from user queries.
It queries nodes and relationships directly from Nebula Graph.
"""
import logging
import asyncio
import re
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Union, Tuple

from backend.utils.logging import get_logger
from backend.pipeline_prompts import PROMPTS

# Configure logger
logger = get_logger(__name__)


async def retrieve_from_knowledge_graph(
    high_level_keywords: List[str],
    low_level_keywords: List[str],
    nebula_client: Any,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve relevant information from the AUTOSAR knowledge graph in Nebula Graph.

    Args:
        high_level_keywords: High-level keywords from the query
        low_level_keywords: Low-level keywords from the query
        nebula_client: Nebula Graph database client
        top_k: Number of top results to retrieve for each keyword
        similarity_threshold: Minimum similarity score threshold (for future use)

    Returns:
        Tuple of (nodes, relationships) lists
    """
    # retrieval_context = {
    #     "level1_nodes": [],
    #     "level2_nodes": [],
    #     "relationships": [],
    #     "sources": [],
    #     "combined_text": ""
    # }

    if not high_level_keywords and not low_level_keywords:
        logger.warning("No keywords provided for knowledge graph retrieval")
        return [], [], []

    logger.info(
        f"Retrieving knowledge with high-level keywords: {high_level_keywords}")
    logger.info(
        f"Retrieving knowledge with low-level keywords: {low_level_keywords}")

    all_keywords = high_level_keywords + low_level_keywords

    try:
        # STEP 1: Retrieve nodes by text matching from Nebula Graph
        nodes = await retrieve_nodes_by_text(
            all_keywords,
            nebula_client,
            top_k=top_k
        )

        logger.info(
            f"Found {len(nodes)} nodes by text matching")

        # STEP 2: Retrieve relationships for the found nodes
        relationships = await retrieve_relationships(
            nodes,
            nebula_client,
            max_references=top_k
        )

        logger.info(
            f"Found {len(relationships)} relationships")

        return nodes, [], relationships

    except Exception as e:
        logger.error(f"Error during knowledge graph retrieval: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return [], [], []


async def retrieve_nodes_by_text(
    keywords: List[str],
    nebula_client: Any,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve Entity nodes from Nebula Graph using text matching on node names.
    According to schema: Entity tag with properties: name, entity_type, description, aliases, etc.

    Args:
        keywords: List of keywords to search for
        nebula_client: Nebula Graph database client
        top_k: Maximum number of results per keyword

    Returns:
        List of retrieved Entity node dictionaries
    """
    retrieved_nodes = []
    unique_node_ids = set()

    try:
        for keyword in keywords:
            # Escape quotes in keyword to prevent injection
            safe_keyword = keyword.replace('"', '\\"')

            # nGQL query using MATCH and WHERE with properties
            query = f'''
            MATCH (e:Entity)
            WHERE properties(e).name == "{safe_keyword}"
            RETURN id(e) AS id,
                properties(e).name AS name,
                properties(e).entity_type AS entity_type,
                properties(e).description AS description,
                properties(e).aliases AS aliases,
                properties(e).identifiers AS identifiers,
                properties(e).evidence_pages AS evidence_pages,
                properties(e).evidence_items AS evidence_items,
                properties(e).source AS source
            LIMIT {top_k}
            '''

            try:
                logger.info(
                    f"Searching for Entity nodes with name: '{keyword}'")

                # Execute query using nebula client
                if hasattr(nebula_client, 'execute_query'):
                    results = await nebula_client.execute_query(query)
                elif hasattr(nebula_client, 'execute'):
                    results = await nebula_client.execute(query)
                else:
                    # Fallback: assume it's a sync method
                    loop = asyncio.get_event_loop()
                    results = await loop.run_in_executor(
                        None,
                        lambda q=query: nebula_client.execute(q)
                    )

                logger.info(
                    f"Found {len(results) if results else 0} Entity nodes for keyword '{keyword}'")

                if results:
                    for record in results:
                        # Handle different response formats from Nebula
                        if isinstance(record, dict):
                            node_id = record.get("id")
                            name = record.get("name", "")
                            entity_type = record.get("entity_type", "")
                            description = record.get("description", "")
                            aliases = record.get("aliases", "")
                            identifiers = record.get("identifiers", "")
                            evidence_pages = record.get("evidence_pages", "")
                            evidence_items = record.get("evidence_items", "")
                            source = record.get("source", "")
                        else:
                            # If record is a row object
                            node_id = getattr(record, 'id', None)
                            name = getattr(record, 'name', "")
                            entity_type = getattr(record, 'entity_type', "")
                            description = getattr(record, 'description', "")
                            aliases = getattr(record, 'aliases', "")
                            identifiers = getattr(record, 'identifiers', "")
                            evidence_pages = getattr(
                                record, 'evidence_pages', "")
                            evidence_items = getattr(
                                record, 'evidence_items', "")
                            source = getattr(record, 'source', "")

                        if not node_id or str(node_id) in unique_node_ids:
                            continue

                        node_data = {
                            "id": str(node_id),
                            "entity_id": str(node_id),
                            "name": name,
                            "entity_type": entity_type,
                            "semantic_type": entity_type,  # For backward compatibility
                            "description": description,
                            "aliases": aliases,
                            "identifiers": identifiers,
                            "evidence_pages": evidence_pages,
                            "evidence_items": evidence_items,
                            "source": source,
                            "similarity_score": 1.0,
                            "match_type": "text"
                        }

                        retrieved_nodes.append(node_data)
                        unique_node_ids.add(str(node_id))
                        logger.debug(
                            f"Found Entity node: {node_data.get('name')} (id: {node_id})")

            except Exception as e:
                logger.error(
                    f"Error searching for keyword '{keyword}': {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info(
            f"Retrieved {len(retrieved_nodes)} unique Entity nodes by text matching")
        return retrieved_nodes

    except Exception as e:
        logger.error(f"Error in text-based node retrieval: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []


async def retrieve_relationships(
    nodes: List[Dict[str, Any]],
    nebula_client: Any,
    max_references: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve relationships from Nebula Graph for given Entity nodes.
    According to schema:
    - REL: Entity -> Entity (with properties: keywords, description, evidence_pages, evidence_items, source_chunk)
    - HAVE_INFORMATION_IN: Entity -> Chunk (with properties: description, evidence_pages, evidence_items)
    - Also retrieve target Entity names for REL edges

    Args:
        nodes: List of Entity node dictionaries
        nebula_client: Nebula Graph database client
        max_references: Maximum number of relationships to retrieve per node

    Returns:
        List of relationship dictionaries
    """
    relationships = []
    unique_relationship_ids = set()
    target_entity_cache = {}  # Cache for target entity names

    try:
        for node in nodes:
            node_id = node.get("id") or node.get("entity_id")
            if not node_id:
                continue

            # Query 1: Get REL edges (Entity -> Entity)
            rel_query = f"""
            GO FROM "{node_id}" OVER REL
            YIELD src(edge) AS source_id, dst(edge) AS target_id,
                   properties(edge).keywords AS keywords,
                   properties(edge).description AS description,
                   properties(edge).evidence_pages AS evidence_pages,
                   properties(edge).evidence_items AS evidence_items,
                   properties(edge).source_chunk AS source_chunk
            LIMIT {max_references}
            """

            # Query 2: Get HAVE_INFORMATION_IN edges (Entity -> Chunk)
            chunk_query = f"""
            GO FROM "{node_id}" OVER HAVE_INFORMATION_IN
            YIELD src(edge) AS source_id, dst(edge) AS target_id,
                   properties(edge).description AS description,
                   properties(edge).evidence_pages AS evidence_pages,
                   properties(edge).evidence_items AS evidence_items
            LIMIT {max_references}
            """

            try:
                # Execute REL query
                if hasattr(nebula_client, 'execute_query'):
                    rel_results = await nebula_client.execute_query(rel_query)
                elif hasattr(nebula_client, 'execute'):
                    rel_results = await nebula_client.execute(rel_query)
                else:
                    loop = asyncio.get_event_loop()
                    rel_results = await loop.run_in_executor(
                        None,
                        lambda: nebula_client.execute(rel_query)
                    )

                # Execute HAVE_INFORMATION_IN query
                if hasattr(nebula_client, 'execute_query'):
                    chunk_results = await nebula_client.execute_query(chunk_query)
                elif hasattr(nebula_client, 'execute'):
                    chunk_results = await nebula_client.execute(chunk_query)
                else:
                    loop = asyncio.get_event_loop()
                    chunk_results = await loop.run_in_executor(
                        None,
                        lambda: nebula_client.execute(chunk_query)
                    )

                # Process REL edges (Entity -> Entity)
                if rel_results:
                    for record in rel_results:
                        if isinstance(record, dict):
                            source_id = record.get("source_id")
                            target_id = record.get("target_id")
                            keywords = record.get("keywords", "")
                            rel_description = record.get("description", "")
                            evidence_pages = record.get("evidence_pages", "")
                            evidence_items = record.get("evidence_items", "")
                            source_chunk = record.get("source_chunk", "")
                        else:
                            source_id = getattr(record, 'source_id', None)
                            target_id = getattr(record, 'target_id', None)
                            keywords = getattr(record, 'keywords', "")
                            rel_description = getattr(
                                record, 'description', "")
                            evidence_pages = getattr(
                                record, 'evidence_pages', "")
                            evidence_items = getattr(
                                record, 'evidence_items', "")
                            source_chunk = getattr(record, 'source_chunk', "")

                        rel_id = f"{source_id}_REL_{target_id}"
                        if rel_id and rel_id not in unique_relationship_ids:
                            # Get target entity name (cache if not already fetched)
                            target_name = target_entity_cache.get(
                                str(target_id))
                            if not target_name:
                                target_name = await _get_entity_name(target_id, nebula_client)
                                if target_name:
                                    target_entity_cache[str(
                                        target_id)] = target_name

                            rel_data = {
                                "source_id": str(source_id),
                                "target_id": str(target_id),
                                "source_name": node.get("name", "Unknown"),
                                "target_name": target_name or "Unknown",
                                "type": "REL",
                                "keywords": keywords,
                                "description": rel_description or f"{node.get('name', 'Unknown')} relates to {target_name or target_id}",
                                "evidence_pages": evidence_pages,
                                "evidence_items": evidence_items,
                                "source_chunk": source_chunk
                            }
                            relationships.append(rel_data)
                            unique_relationship_ids.add(rel_id)

                # Process HAVE_INFORMATION_IN edges (Entity -> Chunk)
                if chunk_results:
                    for record in chunk_results:
                        if isinstance(record, dict):
                            source_id = record.get("source_id")
                            target_id = record.get("target_id")
                            chunk_description = record.get("description", "")
                            evidence_pages = record.get("evidence_pages", "")
                            evidence_items = record.get("evidence_items", "")
                        else:
                            source_id = getattr(record, 'source_id', None)
                            target_id = getattr(record, 'target_id', None)
                            chunk_description = getattr(
                                record, 'description', "")
                            evidence_pages = getattr(
                                record, 'evidence_pages', "")
                            evidence_items = getattr(
                                record, 'evidence_items', "")

                        rel_id = f"{source_id}_HAVE_INFORMATION_IN_{target_id}"
                        if rel_id and rel_id not in unique_relationship_ids:
                            # Get chunk name
                            chunk_name = await _get_chunk_name(target_id, nebula_client)

                            rel_data = {
                                "source_id": str(source_id),
                                "target_id": str(target_id),
                                "source_name": node.get("name", "Unknown"),
                                "target_name": chunk_name or f"Chunk_{target_id}",
                                "type": "HAVE_INFORMATION_IN",
                                "description": chunk_description or f"{node.get('name', 'Unknown')} has information in chunk",
                                "evidence_pages": evidence_pages,
                                "evidence_items": evidence_items
                            }
                            relationships.append(rel_data)
                            unique_relationship_ids.add(rel_id)

            except Exception as e:
                logger.error(
                    f"Error retrieving relationships for node {node_id}: {str(e)}")
                import traceback
                logger.error(traceback.format_exc())

        logger.info(f"Retrieved {len(relationships)} relationships")
        return relationships

    except Exception as e:
        logger.error(f"Error retrieving relationships: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return []


async def _get_entity_name(entity_id: str, nebula_client: Any) -> Optional[str]:
    """Helper function to get Entity name by ID."""
    try:
        query = f"""
        FETCH PROP ON Entity "{entity_id}"
        YIELD properties(vertex).name AS name
        """

        if hasattr(nebula_client, 'execute_query'):
            results = await nebula_client.execute_query(query)
        elif hasattr(nebula_client, 'execute'):
            results = await nebula_client.execute(query)
        else:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: nebula_client.execute(query)
            )

        if results and len(results) > 0:
            record = results[0]
            if isinstance(record, dict):
                return record.get("name", "")
            else:
                return getattr(record, 'name', "")
    except Exception as e:
        logger.debug(f"Error getting entity name for {entity_id}: {str(e)}")
    return None


async def _get_chunk_name(chunk_id: str, nebula_client: Any) -> Optional[str]:
    """Helper function to get Chunk name by ID."""
    try:
        query = f"""
        FETCH PROP ON Chunk "{chunk_id}"
        YIELD properties(vertex).name AS name
        """

        if hasattr(nebula_client, 'execute_query'):
            results = await nebula_client.execute_query(query)
        elif hasattr(nebula_client, 'execute'):
            results = await nebula_client.execute(query)
        else:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: nebula_client.execute(query)
            )

        if results and len(results) > 0:
            record = results[0]
            if isinstance(record, dict):
                return record.get("name", "")
            else:
                return getattr(record, 'name', "")
    except Exception as e:
        logger.debug(f"Error getting chunk name for {chunk_id}: {str(e)}")
    return None


async def self_refine(
    nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]],
    nebula_client: Any,
    gemini_client: Any,
    query: str,
    max_iterations: int = 2
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Refine the retrieval results using self-refinement with LLM.
    Simplified version for Nebula Graph - returns nodes and relationships.
    """
    # For now, return the original results without refinement
    # This can be enhanced later with LLM-based refinement if needed
    logger.info(
        "Self-refinement skipped for Nebula Graph (simplified implementation)")
    return nodes, relationships


async def retrieve_level1_nodes_by_text(
    keywords: List[str],
    neo4j_client: Neo4jClient,
    top_k: int = 5
) -> List[Dict[str, Any]]:
    """
    Retrieve Level 1 nodes using exact text matching on node names.

    Args:
        keywords: List of keywords to search for
        neo4j_client: Neo4j database client
        top_k: Maximum number of results per keyword

    Returns:
        List of retrieved Level 1 node dictionaries
    """
    retrieved_entities = []
    unique_entity_ids = set()

    try:
        for keyword in keywords:
            # Search for nodes where name contains the keyword (case-insensitive)
            # Using CONTAINS for partial matching, or use "= $keyword" for exact match
            query = """
            MATCH (n:Level1)
            WHERE toLower(n.name) CONTAINS toLower($keyword)
            RETURN n.id as id, n.name as name, n.semantic_type as semantic_type,
                   n.cui as cui, n.definition as definition, n.icd as icd, n.level as level
            LIMIT $limit
            """

            try:
                logger.info(
                    f"Searching for Level1 nodes containing: '{keyword}'")
                results = await neo4j_client.execute_query(
                    query,
                    {"keyword": keyword, "limit": top_k}
                )

                logger.info(
                    f"Found {len(results) if results else 0} nodes for keyword '{keyword}'")

                for record in results:
                    entity_id = record.get("id")

                    if not entity_id or entity_id in unique_entity_ids:
                        continue

                    node_data = {
                        "id": entity_id,
                        "entity_id": entity_id,
                        "name": record.get("name", ""),
                        "semantic_type": record.get("semantic_type", ""),
                        "cui": record.get("cui", ""),
                        "definition": record.get("definition", ""),
                        "icd": record.get("icd", ""),
                        "level": record.get("level", "Level 1"),
                        "entity_type": record.get("semantic_type", "CONCEPT"),
                        "description": record.get("definition", ""),
                        "similarity_score": 1.0,  # Perfect match for exact text search
                        "match_type": "text"
                    }

                    retrieved_entities.append(node_data)
                    unique_entity_ids.add(entity_id)
                    logger.debug(
                        f"Found node: {node_data.get('name')} (id: {entity_id})")

            except Exception as e:
                logger.error(
                    f"Error searching for keyword '{keyword}': {str(e)}")

        logger.info(
            f"Retrieved {len(retrieved_entities)} unique Level 1 nodes by text matching")
        return retrieved_entities

    except Exception as e:
        logger.error(f"Error in text-based Level 1 node retrieval: {str(e)}")
        return []


async def retrieve_level1_nodes(
    embeddings: List[List[float]],
    qdrant_client: Any,
    neo4j_client: Neo4jClient,
    top_k: int = 5,
    similarity_threshold: float = 0.7
) -> List[Dict[str, Any]]:
    """
    Retrieve Level 1 nodes using vector similarity search.

    Args:
        embeddings: List of embedding vectors for keywords
        qdrant_client: Vector database client for similarity search
        neo4j_client: Neo4j database client for retrieving node data
        top_k: Number of top results to retrieve for each keyword
        similarity_threshold: Minimum similarity score threshold

    Returns:
        List of retrieved Level 1 node dictionaries
    """
    retrieved_entities = []
    unique_entity_ids = set()

    try:
        for embedding in embeddings:
            try:
                # Query Qdrant for similar vectors
                similar_nodes = qdrant_client.query_points(
                    collection_name="kg_lv1_nodes",
                    query=embedding,
                    limit=top_k
                )

                logger.info(
                    f"Found {len(similar_nodes.points) if hasattr(similar_nodes, 'points') else 0} similar Level 1 nodes")

                for node in similar_nodes.points:
                    # Extracting node ID directly from Qdrant point id, which corresponds to the 'id' field in Neo4j
                    # Using node.id directly as the identifier
                    entity_id = str(node.id) if hasattr(
                        node, 'id') and node.id is not None else None
                    similarity_score = getattr(node, 'score', None)

                    if not entity_id:
                        logger.warning(
                            f"Node missing id. "
                            f"Payload keys: {list(node.payload.keys()) if node.payload else 'None'}, "
                            f"Score: {similarity_score}")
                        continue

                    # Apply similarity threshold filter
                    if similarity_score is not None and similarity_score < similarity_threshold:
                        logger.debug(
                            f"Skipping node {entity_id} with similarity score {similarity_score:.4f} "
                            f"below threshold {similarity_threshold}")
                        continue

                    if entity_id in unique_entity_ids:
                        logger.debug(
                            f"Skipping duplicate node_id: {entity_id} (already in unique_entity_ids)")
                        continue

                    # Querying Neo4j to retrieve full node data matching the structure: id, name, semantic_type, cui, definition, icd, level
                    # Using parameterized query first, with fallback to direct string interpolation if needed
                    query = """
                    MATCH (n:Level1 {id: $entity_id})
                    RETURN n.id as id, n.name as name, n.semantic_type as semantic_type,
                           n.cui as cui, n.definition as definition, n.icd as icd, n.level as level
                    """

                    try:
                        logger.info(
                            f"Querying Neo4j with entity_id: {entity_id} (type: {type(entity_id).__name__})")
                        results = await neo4j_client.execute_query(query, {"entity_id": entity_id})
                        logger.info(
                            f"Neo4j query result: {type(results).__name__}, length: {len(results) if results else 0}, "
                            f"result content: {results[:1] if results else 'empty'}")

                        # Fallback to direct string interpolation if parameterized query returns no results
                        if not results or len(results) == 0:
                            # Checking total number of Level1 nodes in database for diagnostic purposes
                            count_query = "MATCH (n:Level1) RETURN count(n) as total"
                            count_result = await neo4j_client.execute_query(count_query)
                            total_nodes = count_result[0].get(
                                'total', 0) if count_result and len(count_result) > 0 else 0
                            logger.warning(
                                f"Node not found with id: {entity_id}. "
                                f"Total Level1 nodes in database: {total_nodes}")

                            # Attempting direct query with string interpolation as fallback
                            logger.warning(
                                f"Attempting direct query with string interpolation for entity_id: {entity_id}")
                            query_direct = f"""
                            MATCH (n:Level1 {{id: '{entity_id}'}})
                            RETURN n.id as id, n.name as name, n.semantic_type as semantic_type,
                                   n.cui as cui, n.definition as definition, n.icd as icd, n.level as level
                            """
                            results_direct = await neo4j_client.execute_query(query_direct)
                            logger.info(
                                f"Direct query result: {type(results_direct).__name__}, length: {len(results_direct) if results_direct else 0}")
                            if results_direct and len(results_direct) > 0:
                                logger.warning(
                                    f"Direct query found node but parameterized query did not. "
                                    f"This may indicate a parameter binding issue.")
                                results = results_direct

                        if results and len(results) > 0:
                            # Creating node data dictionary matching the actual node structure
                            # Neo4j stores: id, name, semantic_type, cui, definition, icd, level
                            # Qdrant payload stores: name, semantic_type
                            node_data = {
                                "id": results[0].get("id", entity_id),
                                # Keeping entity_id for backward compatibility
                                "entity_id": results[0].get("id", entity_id),
                                "name": results[0].get("name", ""),
                                "semantic_type": results[0].get("semantic_type", ""),
                                # CUI code (may be empty)
                                "cui": results[0].get("cui", ""),
                                # Definition (may be empty)
                                "definition": results[0].get("definition", ""),
                                # ICD code (may be empty)
                                "icd": results[0].get("icd", ""),
                                # Level (should be "Level 1")
                                "level": results[0].get("level", "Level 1"),
                                # Keeping entity_type for backward compatibility (using semantic_type)
                                "entity_type": results[0].get("semantic_type", "CONCEPT"),
                                # Using definition as description if available, otherwise empty
                                "description": results[0].get("definition", ""),
                                "similarity_score": similarity_score
                            }

                            retrieved_entities.append(node_data)
                            unique_entity_ids.add(entity_id)
                            logger.debug(
                                f"Retrieved node from Neo4j: {node_data.get('name')} "
                                f"(id: {entity_id}, score: {similarity_score})")
                        else:
                            logger.warning(
                                f"Node not found in Neo4j with id: {entity_id} "
                                f"(score: {similarity_score})")
                    except Exception as query_error:
                        logger.error(
                            f"Error querying Neo4j for entity_id {entity_id}: {str(query_error)}")
                        import traceback
                        logger.error(traceback.format_exc())

            except Exception as e:
                logger.error(f"Error retrieving similar nodes: {str(e)}")

        # Sort by similarity score
        retrieved_entities.sort(key=lambda x: x.get(
            "similarity_score", 0), reverse=True)

        # Limit to top_k * 2 most relevant nodes overall
        max_nodes = top_k * 2
        if len(retrieved_entities) > max_nodes:
            retrieved_entities = retrieved_entities[:max_nodes]

        logger.info(
            f"Retrieved {len(retrieved_entities)} unique Level 1 nodes")
        return retrieved_entities

    except Exception as e:
        logger.error(f"Error in Level 1 node retrieval: {str(e)}")
        return []


async def retrieve_level2_references(
    level1_nodes: List[Dict[str, Any]],
    neo4j_client: Neo4jClient,
    max_references: int = 5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve Level 2 nodes referenced by Level 1 nodes.

    Args:
        level1_nodes: List of Level 1 node dictionaries
        neo4j_client: Neo4j database client
        max_references: Maximum number of references to retrieve per Level 1 node

    Returns:
        Tuple of (level2_nodes, relationships) lists
    """
    level2_nodes = []
    relationships = []
    unique_level2_ids = set()
    unique_relationship_ids = set()

    try:
        for level1_node in level1_nodes:
            # Using 'id' as the primary identifier matching the actual node structure
            entity_id = level1_node.get("id") or level1_node.get("entity_id")
            if not entity_id:
                continue

            # Retrieving Level 2 nodes connected to this Level 1 node
            # Query matches any relationship type between Level1 and Level2 nodes
            # Returns node properties: id, name, cui, definition, semantic_types, semantic_type, icd, and relationship type
            # Relationship types in database may include IS_A, REFERENCES, or other types
            query = """
            MATCH (l1:Level1 {id: $entity_id})-[r]->(l2:Level2)
            RETURN l2.id AS id, l2.name AS name, l2.cui AS cui, 
                   l2.definition AS definition, l2.semantic_types AS semantic_types,
                   l2.semantic_type AS semantic_type, l2.icd AS icd,
                   type(r) AS relationship_type
            LIMIT $limit
            """

            results = await neo4j_client.execute_query(
                query,
                {"entity_id": entity_id, "limit": max_references}
            )

            for record in results:
                # Creating Level 2 node data dictionary matching the actual node structure
                level2_id = record.get("id", "")
                semantic_types = record.get("semantic_types", [])
                # Single semantic_type field
                semantic_type = record.get("semantic_type", "")
                # Extracting first semantic type if available, defaulting to semantic_type or "CONCEPT"
                entity_type = semantic_types[0] if semantic_types else (
                    semantic_type if semantic_type else "CONCEPT")
                # Extracting actual relationship type from query result
                relationship_type = record.get(
                    "relationship_type", "RELATED_TO")

                level2_data = {
                    "id": level2_id,
                    "entity_id": level2_id,  # Keeping entity_id for backward compatibility
                    "name": record.get("name", "Unknown"),
                    "cui": record.get("cui", ""),
                    "definition": record.get("definition", ""),
                    "semantic_types": semantic_types,  # Array of semantic types
                    "semantic_type": semantic_type,  # Single semantic type field
                    "icd": record.get("icd", ""),  # ICD code
                    "entity_type": entity_type,  # Keeping entity_type for backward compatibility
                    # Using definition as description for backward compatibility
                    "description": record.get("definition", "")
                }

                # Adding Level 2 node to list if not already present
                if level2_id and level2_id not in unique_level2_ids:
                    level2_nodes.append(level2_data)
                    unique_level2_ids.add(level2_id)

                # Creating relationship data dictionary with actual relationship type from database
                rel_id = f"{entity_id}_to_{level2_id}"
                if rel_id and rel_id not in unique_relationship_ids:
                    rel_data = {
                        "source_id": entity_id,
                        "target_id": level2_id,
                        "target_name": record.get("name", "Unknown"),
                        "source_name": level1_node.get("name", "Unknown"),
                        "type": relationship_type,  # Using actual relationship type from database
                        "description": f"{level1_node.get('name', 'Unknown')} {relationship_type.lower()} {record.get('name', 'Unknown')}"
                    }

                    relationships.append(rel_data)
                    unique_relationship_ids.add(rel_id)

        logger.info(
            f"Retrieved {len(level2_nodes)} Level 2 nodes and {len(relationships)} relationships")
        return level2_nodes, relationships

    except Exception as e:
        logger.error(f"Error retrieving Level 2 references: {str(e)}")
        return [], []


async def retrieve_level3_references(
    level2_nodes: List[Dict[str, Any]],
    neo4j_client: Neo4jClient,
    max_references: int = 5
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Retrieve Level 3 nodes referenced by Level 2 nodes.
    Matches Level2 nodes by their CUI to Level3 nodes where level2_node_id equals the CUI.

    Args:
        level2_nodes: List of Level 2 node dictionaries containing CUI
        neo4j_client: Neo4j database client
        max_references: Maximum number of references to retrieve per Level 2 node

    Returns:
        Tuple of (level3_nodes, relationships) lists
    """
    level3_nodes = []
    relationships = []
    unique_level3_ids = set()
    unique_relationship_ids = set()

    try:
        for level2_node in level2_nodes:
            # Extracting CUI from Level2 node, which should match level2_node_id in Level3
            level2_cui = level2_node.get("cui")
            if not level2_cui:
                logger.debug(
                    f"Level2 node {level2_node.get('name', 'Unknown')} has no CUI, skipping Level3 retrieval")
                continue

            # Retrieving Level 3 nodes where level2_node_id exactly matches the Level2 CUI
            # Query matches Level3 nodes based on exact property match: level2_node_id = CUI
            # Returns all Level3 node properties including patient information
            query = """
            MATCH (l3:Level3)
            WHERE l3.level2_node_id = $level2_cui
            RETURN l3.id AS id, l3.level2_node_id AS level2_node_id,
                   l3.admission_info_json AS admission_info_json,
                   l3.anchor_age AS anchor_age,
                   l3.anchor_year AS anchor_year,
                   l3.anchor_year_group AS anchor_year_group,
                   l3.gender AS gender,
                   l3.medications_json AS medications_json,
                   l3.procedures_json AS procedures_json,
                   l3.relevant_diagnoses_json AS relevant_diagnoses_json,
                   l3.services_json AS services_json,
                   l3.source AS source,
                   l3.subject_id AS subject_id,
                   l3.total_diagnoses_count AS total_diagnoses_count,
                   l3.total_procedures_count AS total_procedures_count,
                   l3.total_services_count AS total_services_count
            LIMIT $limit
            """

            results = []
            relationship_type = "RELATED_TO"

            try:
                results = await neo4j_client.execute_query(
                    query,
                    {"level2_cui": level2_cui, "limit": max_references}
                )
            except Exception as query_error:
                logger.warning(
                    f"Error querying Level3 nodes for CUI {level2_cui}: {query_error}")
                continue

            for record in results:
                # Creating Level 3 node data dictionary with all properties
                level3_id = record.get("id", "")

                level3_data = {
                    "id": level3_id,
                    "entity_id": level3_id,  # Keeping entity_id for backward compatibility
                    "level2_node_id": record.get("level2_node_id", ""),
                    "admission_info_json": record.get("admission_info_json"),
                    "anchor_age": record.get("anchor_age"),
                    "anchor_year": record.get("anchor_year"),
                    "anchor_year_group": record.get("anchor_year_group"),
                    "gender": record.get("gender"),
                    "medications_json": record.get("medications_json"),
                    "procedures_json": record.get("procedures_json"),
                    "relevant_diagnoses_json": record.get("relevant_diagnoses_json"),
                    "services_json": record.get("services_json"),
                    "source": record.get("source"),
                    "subject_id": record.get("subject_id"),
                    "total_diagnoses_count": record.get("total_diagnoses_count"),
                    "total_procedures_count": record.get("total_procedures_count"),
                    "total_services_count": record.get("total_services_count")
                }

                # Adding Level 3 node to list if not already present
                if level3_id and level3_id not in unique_level3_ids:
                    level3_nodes.append(level3_data)
                    unique_level3_ids.add(level3_id)

                # Creating relationship data dictionary
                rel_id = f"{level2_cui}_to_{level3_id}"
                if rel_id and rel_id not in unique_relationship_ids:
                    # Using subject_id as name identifier for Level3
                    level3_name = f"Patient {record.get('subject_id', 'Unknown')}"
                    rel_data = {
                        "source_id": level2_node.get("id", ""),
                        "source_cui": level2_cui,
                        "target_id": level3_id,
                        "target_name": level3_name,
                        "source_name": level2_node.get("name", "Unknown"),
                        "type": relationship_type,
                        "description": f"{level2_node.get('name', 'Unknown')} {relationship_type.lower()} {level3_name}"
                    }

                    relationships.append(rel_data)
                    unique_relationship_ids.add(rel_id)

        logger.info(
            f"Retrieved {len(level3_nodes)} Level 3 nodes and {len(relationships)} relationships")
        return level3_nodes, relationships

    except Exception as e:
        logger.error(f"Error retrieving Level 3 references: {str(e)}")
        return [], []


def format_retrieval_results(
    level1_nodes: List[Dict[str, Any]],
    level2_nodes: List[Dict[str, Any]],
    relationships: List[Dict[str, Any]]
) -> str:
    """
    Format the retrieval results into a more informative structured text representation.
    Adapted for Nebula Graph schema: Entity nodes with REL and HAVE_INFORMATION_IN relationships.

    Args:
        level1_nodes: List of Entity node dictionaries (from Nebula Graph)
        level2_nodes: List of Chunk node dictionaries (not used in current implementation, kept for compatibility)
        relationships: List of relationship dictionaries (REL and HAVE_INFORMATION_IN edges)

    Returns:
        Formatted text representation of the retrieved information
    """
    # Group relationships by type and source entity ID
    entity_to_entity = {}  # REL edges: Entity -> Entity
    entity_to_chunk = {}  # HAVE_INFORMATION_IN edges: Entity -> Chunk

    # Create dictionaries to quickly look up nodes
    entity_by_id = {node.get('id') or node.get('entity_id'): node
                    for node in level1_nodes if node.get('id') or node.get('entity_id')}

    # Group relationships by source entity ID and type
    for rel in relationships:
        source_id = rel.get('source_id')
        target_id = rel.get('target_id')
        target_name = rel.get('target_name')
        rel_type = rel.get('type', '')

        if source_id:
            # REL edge: Entity -> Entity
            if rel_type == 'REL':
                if source_id not in entity_to_entity:
                    entity_to_entity[source_id] = []
                entity_to_entity[source_id].append({
                    'target_name': target_name,
                    'target_id': target_id,
                    'relationship': rel
                })
            # HAVE_INFORMATION_IN edge: Entity -> Chunk
            elif rel_type == 'HAVE_INFORMATION_IN':
                if source_id not in entity_to_chunk:
                    entity_to_chunk[source_id] = []
                entity_to_chunk[source_id].append({
                    'target_name': target_name,
                    'target_id': target_id,
                    'relationship': rel
                })

    # Format the text with each Entity node and its relationships
    sections = []

    # Add main content section with detailed information
    main_content = []

    for entity_node in level1_nodes:
        # Using 'id' as the primary identifier
        entity_id = entity_node.get('id') or entity_node.get('entity_id')
        entity_name = entity_node.get('name', 'Unknown')
        entity_type = entity_node.get('entity_type') or entity_node.get(
            'semantic_type', 'Unknown')
        entity_desc = entity_node.get('description', '')
        aliases = entity_node.get('aliases', '')
        identifiers = entity_node.get('identifiers', '')
        source = entity_node.get('source', '')

        # Add Entity node info
        node_section = [
            f"## {entity_name} ({entity_type})",
        ]
        if entity_desc:
            node_section.append(f"**Description:** {entity_desc}")
        if aliases:
            node_section.append(f"**Aliases:** {aliases}")
        if identifiers:
            node_section.append(f"**Identifiers:** {identifiers}")
        if source:
            node_section.append(f"**Source:** {source}")
        node_section.append("")

        # Add related Entity nodes (REL edges) if any
        related_entities = entity_to_entity.get(entity_id, [])
        if related_entities:
            node_section.append(f"### Related Entities (REL):")
            for item in related_entities:
                target_name = item.get('target_name', 'Unknown')
                rel = item.get('relationship', {})
                rel_description = rel.get('description', '')
                keywords = rel.get('keywords', '')
                evidence_pages = rel.get('evidence_pages', '')
                evidence_items = rel.get('evidence_items', '')

                rel_info = []
                if keywords:
                    rel_info.append(f"Keywords: {keywords}")
                if rel_description:
                    rel_info.append(f"Description: {rel_description}")
                if evidence_pages:
                    rel_info.append(f"Evidence pages: {evidence_pages}")
                if evidence_items:
                    rel_info.append(f"Evidence items: {evidence_items}")

                rel_text = " | ".join(
                    rel_info) if rel_info else "Related entity"
                node_section.append(f"* **{target_name}**: {rel_text}")

            node_section.append("")

        # Add related Chunks (HAVE_INFORMATION_IN edges) if any
        related_chunks = entity_to_chunk.get(entity_id, [])
        if related_chunks:
            node_section.append(f"### Related Information Chunks:")
            for item in related_chunks:
                chunk_name = item.get('target_name', 'Unknown')
                rel = item.get('relationship', {})
                rel_description = rel.get('description', '')
                evidence_pages = rel.get('evidence_pages', '')
                evidence_items = rel.get('evidence_items', '')

                chunk_info = []
                if rel_description:
                    chunk_info.append(f"Description: {rel_description}")
                if evidence_pages:
                    chunk_info.append(f"Evidence pages: {evidence_pages}")
                if evidence_items:
                    chunk_info.append(f"Evidence items: {evidence_items}")

                chunk_text = " | ".join(
                    chunk_info) if chunk_info else "Information chunk"
                node_section.append(f"* **{chunk_name}**: {chunk_text}")

            node_section.append("")

        main_content.extend(node_section)

    # Create a key concepts summary section
    concept_summary = ["# KEY ENTITIES", ""]

    # Group Entity nodes by entity type
    entity_types = {}
    for node in level1_nodes:
        entity_type = node.get('entity_type') or node.get(
            'semantic_type', 'Unknown')
        if entity_type not in entity_types:
            entity_types[entity_type] = []
        entity_types[entity_type].append(node)

    # Add a summary for each entity type
    for entity_type, nodes in entity_types.items():
        concept_summary.append(f"## {entity_type.upper()}S")
        for node in nodes:
            name = node.get('name', 'Unknown')
            desc = node.get('description', '')
            # Creating short description from first sentence or truncating
            if desc:
                short_desc = desc.split('.')[0] if '.' in desc else desc[:100]
                concept_summary.append(f"* **{name}**: {short_desc}")
            else:
                concept_summary.append(f"* **{name}**")
        concept_summary.append("")

    # Add a relationships summary
    relationship_summary = ["# RELATIONSHIPS", ""]

    # Group relationships by type (REL, HAVE_INFORMATION_IN)
    rel_types = {}
    for rel in relationships:
        rel_type = rel.get('type', 'REL')
        if rel_type not in rel_types:
            rel_types[rel_type] = []
        rel_types[rel_type].append(rel)

    # Add a summary for each relationship type
    for rel_type, rels in rel_types.items():
        relationship_summary.append(f"## {rel_type}")
        # List only unique source-target pairs to avoid repetition
        unique_pairs = set()
        for rel in rels:
            source = rel.get('source_name', 'Unknown')
            target = rel.get('target_name', 'Unknown')
            pair = f"{source} → {target}"
            if pair not in unique_pairs:
                unique_pairs.add(pair)
                # Add additional info if available
                rel_desc = rel.get('description', '')
                if rel_desc and len(rel_desc) < 100:
                    relationship_summary.append(f"* {pair} ({rel_desc})")
                else:
                    relationship_summary.append(f"* {pair}")
        relationship_summary.append("")

    # Combine all sections
    sections.append("\n".join(concept_summary))
    sections.append("\n".join(relationship_summary))
    sections.append("# DETAILED INFORMATION\n")
    sections.append("\n".join(main_content))

    return "\n".join(sections)


def sanitize_filename(text: str, max_length: int = 100) -> str:
    """
    Convert text to a valid filename.

    Args:
        text: Text to convert
        max_length: Maximum length of the filename

    Returns:
        Sanitized filename
    """
    # Remove special symbols from text
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[-\s]+', '_', text)
    if len(text) > max_length:
        text = text[:max_length]
    return text.strip('_')


def create_rag_prompt(query: str, formatted_text: str, conversation_history: Optional[str] = None) -> str:
    """
    Create RAG prompt ready to be passed to LLM.

    Args:
        query: User's query
        formatted_text: Text formatted from format_retrieval_results
        conversation_history: Conversation history (optional)

    Returns:
        Prompt string ready to be passed to LLM
    """
    # Use prompt template from pipeline_prompts.py
    prompt_template = PROMPTS.get("test_full_rag_prompt", "")

    # Format conversation history
    history_text = conversation_history if conversation_history else "(No previous conversation)"

    # Format prompt with values
    prompt = prompt_template.format(
        query=query,
        formatted_text=formatted_text,
        conversation_history=history_text
    )

    return prompt


def save_retrieval_result(
    query: str,
    formatted_text: str,
    result: dict,
    conversation_history: Optional[str] = None,
    output_dir: Optional[Path] = None
) -> Optional[str]:
    """
    Save formatted retrieval result to output_pipeline_retrie folder.
    Format according to RAG prompt structure for direct use with LLM.

    Args:
        query: Original query
        formatted_text: Text formatted from format_retrieval_results
        result: Dictionary containing all results
        conversation_history: Conversation history (optional)
        output_dir: Output directory (optional, defaults to output_pipeline_retrie at root)

    Returns:
        Path to saved file or None if error
    """
    try:
        # Import QueryIntent if available (optional dependency)
        query_intent_enum = None
        has_query_intent = False
        try:
            from backend.pipeline.query_analyzer import QueryIntent
            query_intent_enum = QueryIntent
            has_query_intent = True
        except ImportError:
            pass

        # Create output_pipeline_retrie folder if it doesn't exist
        if output_dir is None:
            # Get root path from current file (backend/retrieval/triple_level_retrieval.py)
            # Go up 2 levels to reach root: backend/retrieval -> backend -> root
            current_file = Path(__file__)
            root_path = current_file.parent.parent.parent
            output_dir = root_path / "output_pipeline_retrie"
        else:
            output_dir = Path(output_dir)

        output_dir.mkdir(exist_ok=True)

        # Create filename based on query and timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        query_sanitized = sanitize_filename(query, max_length=50)
        filename = f"prompt_{timestamp}_{query_sanitized}.txt"
        file_path = output_dir / filename

        # Create RAG prompt ready to be passed to LLM
        rag_prompt = create_rag_prompt(
            query, formatted_text, conversation_history)

        # Create file content with prompt and metadata
        content_parts = []

        # Main prompt section (ready to copy to LLM)
        content_parts.append("=" * 80)
        content_parts.append(
            "PROMPT READY FOR LLM (Copy the section below to use)")
        content_parts.append("=" * 80)
        content_parts.append("")
        content_parts.append(rag_prompt)
        content_parts.append("")

        # Metadata section (for reference)
        content_parts.append("=" * 80)
        content_parts.append("METADATA (For reference only)")
        content_parts.append("=" * 80)
        content_parts.append(
            f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Intent (if available)
        intent = result.get('intent')
        if intent:
            if has_query_intent and hasattr(intent, 'name'):
                content_parts.append(f"Intent: {intent.name}")
            else:
                content_parts.append(f"Intent: {intent}")
        else:
            content_parts.append("Intent: N/A")

        # Keywords (if available)
        high_level_keywords = result.get('high_level_keywords', [])
        low_level_keywords = result.get('low_level_keywords', [])

        # Display keywords if available (for HEALTHCARE_RELATED or any keywords)
        if high_level_keywords or low_level_keywords:
            # Only display if intent is HEALTHCARE_RELATED (if QueryIntent exists) or if QueryIntent doesn't exist
            should_show_keywords = True
            if has_query_intent and query_intent_enum and intent:
                should_show_keywords = (
                    intent == query_intent_enum.HEALTHCARE_RELATED)

            if should_show_keywords:
                content_parts.append(
                    f"High-level keywords: {', '.join(high_level_keywords)}")
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

        # Write file
        full_content = "\n".join(content_parts)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(full_content)

        logger.info(f"Saved retrieval result to: {file_path}")
        return str(file_path)

    except Exception as e:
        logger.error(f"Error saving retrieval result: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None
