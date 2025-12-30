import logging
import asyncio
import json
from typing import List, Dict, Any, Optional, Union
from nebula3.gclient.net import ConnectionPool
from nebula3.Config import Config
from nebula3.common import ttypes
import os
from dotenv import load_dotenv
import uuid

logger = logging.getLogger(__name__)


class NebulaClient:
    """
    Nebula Graph client for connecting to and querying Nebula Graph database.
    Similar interface to Neo4jClient but uses Nebula Graph and nGQL queries.
    """

    def __init__(
        self,
        address: str = None,
        port: int = None,
        user: str = None,
        password: str = None,
        space: str = None,
        min_connection_pool_size: int = 2,
        max_connection_pool_size: int = 10
    ):
        """
        Initialize Nebula Graph client.

        Args:
            address: Nebula Graph address (default from env: NEBULA_ADDRESS)
            port: Nebula Graph port (default from env: NEBULA_PORT or 9669)
            user: Username (default from env: NEBULA_USER)
            password: Password (default from env: NEBULA_PASSWORD)
            space: Space name (default from env: NEBULA_SPACE)
            min_connection_pool_size: Minimum connection pool size
            max_connection_pool_size: Maximum connection pool size
        """
        if address is None or user is None or password is None:
            load_dotenv()
            address = address or os.getenv("NEBULA_ADDRESS", "127.0.0.1")
            port = port or int(os.getenv("NEBULA_PORT", "9669"))
            user = user or os.getenv("NEBULA_USER", "root")
            password = password or os.getenv("NEBULA_PASSWORD", "password")
            space = space or os.getenv("NEBULA_SPACE", "autosar")

        self.address = address
        self.port = port
        self.user = user
        self.password = password
        self.space = space
        self._connection_pool = None
        self._session = None
        self._initialize_connection_pool()

    def _initialize_connection_pool(self):
        """Initialize Nebula Graph connection pool."""
        try:
            config = Config()
            config.max_connection_pool_size = 10
            self._connection_pool = ConnectionPool()

            # Initialize connection pool
            if not self._connection_pool.init(
                [(self.address, self.port)],
                Config()
            ):
                raise Exception("Failed to initialize connection pool")

            logger.info(
                f"Successfully initialized Nebula Graph connection pool for {self.address}:{self.port}")
        except Exception as e:
            logger.error(
                f"Failed to initialize Nebula Graph connection pool: {str(e)}")
            raise

    async def close(self):
        """Close Nebula Graph connection pool."""
        if self._session:
            self._session.release()
            self._session = None

        if self._connection_pool:
            self._connection_pool.close()
            self._connection_pool = None
            logger.info("Nebula Graph connection pool closed")

    def close_sync(self):
        """Synchronous version of close method."""
        if self._session:
            self._session.release()
            self._session = None

        if self._connection_pool:
            self._connection_pool.close()
            self._connection_pool = None
            logger.info("Nebula Graph connection pool closed")

    def _get_session(self):
        """Get or create a session for executing queries."""
        if not self._session:
            self._session = self._connection_pool.get_session(
                self.user, self.password)
            # Use the space
            result = self._session.execute(f"USE {self.space}")
            if not result.is_succeeded():
                error_msg = result.error_msg()
                logger.error(f"Failed to use space {self.space}: {error_msg}")
                raise Exception(f"Failed to use space: {error_msg}")
        return self._session

    async def verify_connectivity(self) -> bool:
        """
        Verify that the Nebula Graph connection is working.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            session = self._get_session()
            result = session.execute("SHOW SPACES")
            if result.is_succeeded():
                logger.info("Nebula Graph connectivity verified")
                return True
            else:
                logger.error(
                    f"Nebula Graph connectivity check failed: {result.error_msg()}")
                return False
        except Exception as e:
            logger.error(f"Nebula Graph connectivity check failed: {str(e)}")
            return False

    async def execute_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Execute a custom nGQL query against the Nebula Graph database.

        Args:
            query: nGQL query to execute
            params: Parameters for the query (not used in current implementation)

        Returns:
            List of query result records as dictionaries
        """
        results = []

        try:
            session = self._get_session()

            # Execute query
            result = session.execute(query)

            if not result.is_succeeded():
                error_msg = result.error_msg()
                logger.error(f"Query execution failed: {error_msg}")
                logger.error(f"Query: {query}")
                return []

            # Convert result to list of dictionaries
            column_names = result.keys()
            for row in result:
                record = {}
                for i, col_name in enumerate(column_names):
                    value = row.values[i]
                    # Handle different value types from Nebula
                    if isinstance(value, ttypes.Value):
                        if value.getSetVal() is not None:
                            record[col_name] = list(value.getSetVal().values)
                        elif value.getListVal() is not None:
                            record[col_name] = [
                                v for v in value.getListVal().values]
                        elif value.getMapVal() is not None:
                            record[col_name] = {
                                k: v for k, v in value.getMapVal().kvs.items()}
                        elif value.getNVal() is not None:
                            # Node value
                            node = value.getNVal()
                            record[col_name] = {
                                "vid": node.getVid().getSVal() if node.getVid() else None,
                                "tags": [tag.name for tag in node.getTags()] if node.getTags() else []
                            }
                        elif value.getEVAL() is not None:
                            # Edge value
                            edge = value.getEVAL()
                            record[col_name] = {
                                "src": edge.getSrc().getSVal() if edge.getSrc() else None,
                                "dst": edge.getDst().getSVal() if edge.getDst() else None,
                                "name": edge.name if edge.name else None
                            }
                        elif value.sVal is not None:
                            record[col_name] = value.sVal.decode(
                                'utf-8') if isinstance(value.sVal, bytes) else value.sVal
                        elif value.iVal is not None:
                            record[col_name] = value.iVal
                        elif value.fVal is not None:
                            record[col_name] = value.fVal
                        elif value.bVal is not None:
                            record[col_name] = value.bVal
                        elif value.tVal is not None:
                            record[col_name] = str(value.tVal)
                        elif value.dVal is not None:
                            record[col_name] = str(value.dVal)
                        elif value.dtVal is not None:
                            record[col_name] = str(value.dtVal)
                        elif value.tmVal is not None:
                            record[col_name] = str(value.tmVal)
                        elif value.isNull():
                            record[col_name] = None
                        else:
                            record[col_name] = str(value)
                    else:
                        record[col_name] = value

                results.append(record)

            return results
        except Exception as e:
            logger.error(f"Error executing query: {str(e)}")
            logger.error(f"Query: {query}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def execute(self, query: str) -> List[Dict[str, Any]]:
        """
        Synchronous version of execute_query.

        Args:
            query: nGQL query to execute

        Returns:
            List of query result records as dictionaries
        """
        return asyncio.run(self.execute_query(query))

    async def setup_schema(self) -> bool:
        """
        Set up Nebula Graph schema with tags and edges for the knowledge graph.
        According to schema: Entity, Chunk, Document tags and REL, HAVE_INFORMATION_IN, PARAGRAPH_OF edges.

        Returns:
            True if schema setup is successful, False otherwise
        """
        schema_queries = [
            # Create tags
            "CREATE TAG IF NOT EXISTS Entity(name string, entity_type string, description string, aliases string, identifiers string, evidence_pages string, evidence_items string, source string)",
            "CREATE TAG IF NOT EXISTS Chunk(name string, toc_type string, description string, doc_filename string, docling_chunk_id string, heading_path string, evidence_pages string, evidence_items string, cited_images string)",
            "CREATE TAG IF NOT EXISTS Document(name string, filename string)",

            # Create edges
            "CREATE EDGE IF NOT EXISTS REL(keywords string, description string, evidence_pages string, evidence_items string, source_chunk string)",
            "CREATE EDGE IF NOT EXISTS HAVE_INFORMATION_IN(description string, evidence_pages string, evidence_items string)",
            "CREATE EDGE IF NOT EXISTS PARAGRAPH_OF(description string)",

            # Create indexes (optional, for better query performance)
            "CREATE TAG INDEX IF NOT EXISTS entity_name_index ON Entity(name(255))",
        ]

        try:
            session = self._get_session()

            for query in schema_queries:
                try:
                    result = session.execute(query)
                    if result.is_succeeded():
                        logger.info(
                            f"Successfully executed schema query: {query}")
                    else:
                        error_msg = result.error_msg()
                        # Some queries might fail if already exist, which is okay
                        if "existed" in error_msg.lower() or "already" in error_msg.lower():
                            logger.info(
                                f"Schema element already exists: {query}")
                        else:
                            logger.warning(
                                f"Schema query warning: {error_msg}")
                except Exception as e:
                    logger.warning(
                        f"Error executing schema query (may already exist): {str(e)}")

            logger.info("Nebula Graph schema setup completed")
            return True
        except Exception as e:
            logger.error(f"Error setting up Nebula Graph schema: {str(e)}")
            return False

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the knowledge graph in Nebula Graph.

        Returns:
            Dictionary with graph statistics
        """
        try:
            session = self._get_session()
            stats = {}

            queries = {
                "total_entities": "MATCH (e:Entity) RETURN count(e) AS count",
                "total_chunks": "MATCH (c:Chunk) RETURN count(c) AS count",
                "total_documents": "MATCH (d:Document) RETURN count(d) AS count",
                "total_rel_edges": "MATCH ()-[r:REL]->() RETURN count(r) AS count",
                "total_have_info_edges": "MATCH ()-[r:HAVE_INFORMATION_IN]->() RETURN count(r) AS count",
                "total_paragraph_edges": "MATCH ()-[r:PARAGRAPH_OF]->() RETURN count(r) AS count"
            }

            for key, query in queries.items():
                result = session.execute(query)
                if result.is_succeeded():
                    for row in result:
                        if row.values:
                            stats[key] = row.values[0].getIVal() if hasattr(
                                row.values[0], 'getIVal') else 0
                else:
                    stats[key] = 0
                    logger.warning(
                        f"Failed to get statistic for {key}: {result.error_msg()}")

            logger.info(f"Graph statistics retrieved: {json.dumps(stats)}")
            return stats
        except Exception as e:
            logger.error(f"Error getting graph statistics: {str(e)}")
            return {}

    async def clear_database(self) -> bool:
        """
        Clear all data from the Nebula Graph space.

        WARNING: This deletes all nodes and relationships!

        Returns:
            True if clearing is successful, False otherwise
        """
        try:
            session = self._get_session()

            # Delete all edges first
            queries = [
                "MATCH ()-[r:REL]->() DELETE r",
                "MATCH ()-[r:HAVE_INFORMATION_IN]->() DELETE r",
                "MATCH ()-[r:PARAGRAPH_OF]->() DELETE r",
                # Delete all vertices
                "MATCH (v) DELETE v"
            ]

            for query in queries:
                result = session.execute(query)
                if not result.is_succeeded():
                    logger.warning(
                        f"Error executing clear query: {result.error_msg()}")

            logger.info(
                "Successfully cleared all data from Nebula Graph space")
            return True
        except Exception as e:
            logger.error(f"Error clearing Nebula Graph space: {str(e)}")
            return False


async def main():
    """Example main function demonstrating the NebulaClient usage."""
    client = NebulaClient()

    try:
        connected = await client.verify_connectivity()
        if not connected:
            logger.error("Failed to connect to Nebula Graph, exiting")
            return

        schema_success = await client.setup_schema()
        if not schema_success:
            logger.warning("Schema setup had issues")

        stats = await client.get_graph_statistics()
        logger.info(f"Graph statistics: {json.dumps(stats, indent=2)}")

        # Example query
        query_result = await client.execute_query(
            "MATCH (e:Entity) RETURN e.name AS name LIMIT 5"
        )
        logger.info(f"Sample entities: {query_result}")

    finally:
        await client.close()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    asyncio.run(main())
