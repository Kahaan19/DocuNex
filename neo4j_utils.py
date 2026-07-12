"""Neo4j helpers for persisting the GraphRAG knowledge graph.

Neo4j is an optional dependency: it is imported lazily inside
``create_neo4j_driver`` so the pure helpers (label/relationship
sanitizers) can be imported and unit-tested without the driver installed.
"""
import re
import traceback

import config


def create_neo4j_driver():
    """Create and verify a Neo4j driver, or return None if unavailable."""
    try:
        from neo4j import GraphDatabase

        driver = GraphDatabase.driver(
            config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
        )
        with driver.session() as session:
            session.run("RETURN 1")
        print("✅ Connected to Neo4j successfully")
        return driver
    except Exception as e:  # noqa: BLE001
        print(f"⚠️ Could not connect to Neo4j: {e}")
        return None


def sanitize_rel_type(rel_type) -> str:
    """Normalize an arbitrary string into a valid Neo4j relationship type."""
    if not rel_type:
        return "RELATED_TO"

    rel_type = re.sub(r"[^a-zA-Z0-9_]", "_", str(rel_type)).upper()
    rel_type = re.sub(r"_+", "_", rel_type)
    rel_type = rel_type.strip("_")

    # Relationship types cannot be empty or start with a digit.
    if not rel_type or rel_type[0].isdigit():
        rel_type = "RELATED_TO"
    if len(rel_type) > 50:
        rel_type = rel_type[:50]
    return rel_type


def sanitize_node_label(label) -> str:
    """Normalize an arbitrary string into a valid Neo4j node label."""
    if not label:
        return "Entity"

    label = re.sub(r"[^a-zA-Z0-9_]", "_", str(label))
    label = re.sub(r"_+", "_", label)
    label = label.strip("_")

    # Labels must start with a letter.
    if not label or label[0].isdigit():
        label = "Entity_" + label if label else "Entity"
    if len(label) > 50:
        label = label[:50]
    return label


def clear_neo4j_database(driver) -> bool:
    """Detach-delete all nodes/relationships from the connected database."""
    if not driver:
        return False
    try:
        with driver.session() as session:
            session.run("MATCH (n) WITH n LIMIT 10000 DETACH DELETE n")
            result = session.run("MATCH (n) RETURN count(n) as count")
            count = result.single()["count"]
            if count > 0:
                print(f"⚠️ Still {count} nodes remaining - may need multiple clears")
        print("✅ Cleared Neo4j database")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error clearing Neo4j: {e}")
        return False


def insert_graph_to_neo4j(driver, nodes, relationships) -> bool:
    """Insert knowledge-graph nodes and relationships into Neo4j."""
    if not driver:
        print("❌ Neo4j not connected.")
        return False

    import uuid

    try:
        with driver.session() as session:
            nodes_created = 0
            for node in nodes:
                try:
                    node_id = str(node.get("id", f"node_{uuid.uuid4().hex[:8]}"))
                    node_label = sanitize_node_label(node.get("label", "Entity"))
                    clean_props = _clean_props(node.get("properties", {}))
                    session.run(
                        f"MERGE (n:{node_label} {{id: $id}}) SET n += $props",
                        id=node_id, props=clean_props,
                    )
                    nodes_created += 1
                except Exception as e:  # noqa: BLE001
                    print(f"⚠️ Error creating node {node.get('id', 'unknown')}: {e}")
                    continue

            relationships_created = 0
            for rel in relationships:
                try:
                    source_id = str(rel.get("source", ""))
                    target_id = str(rel.get("target", ""))
                    rel_type = sanitize_rel_type(rel.get("type", "RELATED_TO"))
                    if not source_id or not target_id:
                        continue
                    clean_props = _clean_props(rel.get("properties", {}))
                    session.run(
                        f"""
                        MATCH (a {{id: $source}}), (b {{id: $target}})
                        MERGE (a)-[r:{rel_type}]->(b)
                        SET r += $props
                        """,
                        source=source_id, target=target_id, props=clean_props,
                    )
                    relationships_created += 1
                except Exception as e:  # noqa: BLE001
                    print(
                        f"⚠️ Error creating relationship "
                        f"{rel.get('source', 'unknown')} -> {rel.get('target', 'unknown')}: {e}"
                    )
                    continue

        print(
            f"✅ Successfully inserted {nodes_created} nodes and "
            f"{relationships_created} relationships into Neo4j"
        )
        return True
    except Exception as e:  # noqa: BLE001
        print(f"❌ Error inserting graph into Neo4j: {e}")
        traceback.print_exc()
        return False


def _clean_props(props: dict) -> dict:
    """Coerce property values into Neo4j-serializable primitives."""
    clean = {}
    for key, value in props.items():
        clean[key] = value if isinstance(value, (str, int, float, bool)) else str(value)
    return clean
