"""
Task 3: RAG Database for PICO-8 Code Generation
=================================================
Builds a Retrieval-Augmented Generation (RAG) system using:
  - ChromaDB as the vector store
  - sentence-transformers for embeddings
  - The scraped pico8_games.csv dataset

Usage:
  python task3_rag.py                    # Build the DB and run a test query
  python task3_rag.py "your query here"  # Query the RAG system
"""

import os
import sys
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer


# --- Configuration ---
CSV_FILE = "pico8_games.csv"
CHROMA_DB_DIR = "./chroma_db"
COLLECTION_NAME = "pico8_games"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5


def load_dataset(csv_path):
    """Load the scraped PICO-8 games dataset."""
    if not os.path.exists(csv_path):
        print(f"[✗] Dataset not found: {csv_path}")
        print("    Please run task2_scraper.py first to generate the CSV.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[✓] Loaded {len(df)} games from {csv_path}")
    return df


def build_documents(df):
    """
    Construct documents for embedding.
    Each document combines the game's name, description, and code
    to create a rich searchable text.
    """
    documents = []
    metadatas = []
    ids = []

    for idx, row in df.iterrows():
        name = str(row.get("name", ""))
        description = str(row.get("description", ""))
        code = str(row.get("code", ""))
        author = str(row.get("author", ""))
        license_info = str(row.get("license", ""))

        # Construct a combined document text for embedding
        doc_text = f"""Game: {name}
Author: {author}
Description: {description}
Code: {code[:3000]}"""  # Limit code length for embedding

        documents.append(doc_text)
        metadatas.append({
            "name": name,
            "author": author,
            "license": license_info,
            "like_count": str(row.get("like_count", 0)),
            "artwork_url": str(row.get("artwork_url", "")),
        })
        ids.append(f"game_{idx}")

    return documents, metadatas, ids


def create_vector_store(documents, metadatas, ids):
    """Create and populate the ChromaDB vector store."""
    print("\n--- Building Vector Store ---")

    # Initialize embedding model
    print(f"  Loading embedding model: {EMBEDDING_MODEL}...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # Generate embeddings
    print(f"  Generating embeddings for {len(documents)} documents...")
    embeddings = embedder.encode(documents, show_progress_bar=True)
    embeddings_list = [emb.tolist() for emb in embeddings]

    # Initialize ChromaDB
    print(f"  Initializing ChromaDB at {CHROMA_DB_DIR}...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # Delete existing collection if it exists (for clean rebuild)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "PICO-8 games from Lexaloffle BBS"},
    )

    # Add documents to the collection
    print(f"  Adding {len(documents)} documents to collection...")
    collection.add(
        documents=documents,
        embeddings=embeddings_list,
        metadatas=metadatas,
        ids=ids,
    )

    print(f"[✓] Vector store created with {collection.count()} entries.")
    return client, collection, embedder


def query_rag(collection, embedder, query, top_k=TOP_K):
    """
    Query the RAG system.
    Returns the top-K most relevant documents.
    """
    # Embed the query
    query_embedding = embedder.encode([query])[0].tolist()

    # Retrieve from ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    return results


def generate_rag_prompt(query, results):
    """
    Construct a prompt for an LLM (e.g., Claude) using retrieved context.
    This function builds the prompt; the actual LLM call would be external.
    """
    context_blocks = []
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        context_blocks.append(
            f"--- Retrieved Game {i+1} (similarity: {1-dist:.3f}) ---\n"
            f"Name: {meta.get('name', 'N/A')}\n"
            f"Author: {meta.get('author', 'N/A')}\n"
            f"Likes: {meta.get('like_count', 'N/A')}\n"
            f"{doc}\n"
        )

    context = "\n".join(context_blocks)

    prompt = f"""You are a PICO-8 game development assistant. Based on the following reference games
and their code from the PICO-8 community, help the user with their request.

== REFERENCE GAMES FROM DATABASE ==
{context}

== USER REQUEST ==
{query}

== INSTRUCTIONS ==
Using the patterns, techniques, and code examples from the reference games above,
generate PICO-8 Lua code that addresses the user's request. Follow PICO-8 conventions:
- Use _init(), _update(), and _draw() functions
- Use PICO-8 API functions (spr, map, btn, sfx, etc.)
- Keep code within PICO-8 token limits
- Include comments explaining the code

Please provide the complete PICO-8 code:
"""

    return prompt


def main():
    print("=" * 60)
    print("  PICO-8 RAG Database - Task 3")
    print("=" * 60)

    # Step 1: Load dataset
    print("\n[1/3] Loading dataset...")
    df = load_dataset(CSV_FILE)

    # Step 2: Build documents and create vector store
    print("\n[2/3] Building vector store...")
    documents, metadatas, ids = build_documents(df)
    client, collection, embedder = create_vector_store(documents, metadatas, ids)

    # Step 3: Query the RAG system
    print("\n[3/3] Testing RAG system...")

    # Default test query or user-provided query
    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
    else:
        query = "Write PICO-8 code for a simple platformer with player jumping and gravity"

    print(f"\n  Query: \"{query}\"")
    print("-" * 50)

    results = query_rag(collection, embedder, query)

    # Display results
    print(f"\n  Top {TOP_K} Retrieved Games:")
    for i, (meta, dist) in enumerate(zip(
        results["metadatas"][0],
        results["distances"][0]
    )):
        similarity = 1 - dist
        print(f"    {i+1}. {meta.get('name', 'N/A')} "
              f"(by {meta.get('author', 'N/A')}, "
              f"likes: {meta.get('like_count', '?')}, "
              f"sim: {similarity:.3f})")

    # Generate the RAG prompt
    prompt = generate_rag_prompt(query, results)

    # Save the prompt to a file for inspection
    prompt_file = "rag_prompt_output.txt"
    with open(prompt_file, "w", encoding="utf-8") as f:
        f.write(prompt)

    print(f"\n[✓] RAG prompt saved to '{prompt_file}'")
    print(f"    The prompt ({len(prompt)} chars) contains retrieved context")
    print(f"    and can be sent to an LLM (e.g., Claude) for code generation.")

    # Print a snippet of the prompt
    print("\n--- Prompt Preview (first 500 chars) ---")
    print(prompt[:500])
    print("...")

    print("\n=== Task 3 Complete ===")


if __name__ == "__main__":
    main()
