#!/usr/bin/env python3

import argparse
import json
import sys
import os
import meilisearch
import textwrap
import requests
import tempfile
import webbrowser
import subprocess
import re
from urllib.parse import urljoin
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich.progress import Progress
from rich import box

console = Console()

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Search documents in Meilisearch with advanced options and LLM integration',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic keyword search
  python search.py "neural networks"
  
  # Semantic search with LLM key points extraction
  python search.py "machine learning optimization methods" --semantic --llm-function key_points
  
  # Hybrid search with file type filter
  python search.py "blockchain decentralized finance" --hybrid --filter "fileType=pdf"
  
  # Generate a report on a specific topic with LLM summarization
  python search.py "climate change impacts" --hybrid --llm-function summarize
  
  # Ask a specific question about search results
  python search.py "quantum computing applications" --semantic --llm-function ask --question "How is quantum computing used in cryptography?"
  
  # Create a chat interface with document context
  python search.py "artificial intelligence ethics" --llm-function chat
'''
    )
    parser.add_argument('query', help='Search query string')
    
    # Meilisearch connection
    connection_group = parser.add_argument_group('Connection Options')
    connection_group.add_argument('-u', '--url', default='http://localhost:7700', 
                      help='Meilisearch URL (default: http://localhost:7700)')
    connection_group.add_argument('-i', '--index', default='documents', 
                      help='Meilisearch index name (default: documents)')
    connection_group.add_argument('-k', '--key', 
                      help='Meilisearch API key for authentication (if required)')
    
    # Search options
    search_group = parser.add_argument_group('Search Options')
    search_group.add_argument('-l', '--limit', type=int, default=10, 
                   help='Maximum number of results to return (default: 10)')
    search_group.add_argument('-f', '--filter', 
                   help='Filter query (e.g., "fileType=pdf" or "fileSize > 1000000")')
    search_group.add_argument('-s', '--sort', 
                   help='Sort results (e.g., "createdAt:desc" for newest first)')
    search_group.add_argument('-a', '--attributes', 
                   help='Comma-separated list of attributes to retrieve (e.g., "title,path,content")')
    search_group.add_argument('--facets', 
                   help='Comma-separated list of facets to retrieve for aggregations')
    search_group.add_argument('--matching-strategy', choices=['last', 'all', 'any'], default='all',
                   help='Matching strategy for keyword search (default: all)')
    
    # Vector search options
    vector_group = parser.add_argument_group('Vector Search Options')
    vector_group.add_argument('--semantic', action='store_true',
                    help='Enable primarily semantic (vector) search with minimal keyword matching')
    vector_group.add_argument('--hybrid', action='store_true',
                    help='Enable balanced hybrid search (keywords + vectors)')
    vector_group.add_argument('--embedding-model', default='nomic-embed-text',
                    help='Embedding model to use (default: nomic-embed-text)')
    vector_group.add_argument('--semantic-weight', type=float, default=0.5,
                    help='Weight for semantic search vs keyword search (0.0-1.0, default: 0.5)')
    vector_group.add_argument('--exact-matches', action='store_true',
                    help='Prioritize exact matches even in semantic/hybrid search')
    
    # LLM options
    llm_group = parser.add_argument_group('LLM Integration Options')
    llm_group.add_argument('--llm-function', choices=['key_points', 'summarize', 'compare', 'ask', 'chat', 'analyze'],
                  help='LLM function to apply to search results')
    llm_group.add_argument('--llm-model', default='cas/llama-3.2-3b-instruct',
                  help='LLM model to use for functions (default: cas/llama-3.2-3b-instruct)')
    llm_group.add_argument('--question',
                  help='Question to ask the LLM about documents (required for --llm-function=ask)')
    llm_group.add_argument('--ollama-url', default='http://localhost:11434',
                  help='URL for Ollama API (default: http://localhost:11434)')
    llm_group.add_argument('--max-context-docs', type=int, default=3,
                  help='Maximum number of documents to include in LLM context (default: 3)')
    
    # Display options
    display_group = parser.add_argument_group('Display Options')
    display_group.add_argument('-o', '--output', choices=['text', 'json', 'table', 'rich'], default='rich', 
                    help='Output format (default: rich console output)')
    display_group.add_argument('-c', '--context-words', type=int, default=30, 
                    help='Number of WORDS of context around each match (default: 30)')
    display_group.add_argument('-n', '--content-lines', type=int, default=3, 
                    help='Maximum NUMBER OF CONTENT SNIPPETS to show per result (default: 3)')
    display_group.add_argument('--content-size', type=int, default=500, 
                    help='Maximum NUMBER OF CHARACTERS per content snippet (default: 500)')
    display_group.add_argument('--open', action='store_true', 
                    help='Enable interactive mode to open matching files')
    display_group.add_argument('--show-path', action='store_true', 
                    help='Show full path to files instead of relative paths')
    display_group.add_argument('--highlight-style', default='bold red on white', 
                    help='Rich text style for highlights (default: "bold red on white")')
    display_group.add_argument('--no-highlight', action='store_true', 
                    help='Disable highlighting of search terms')
    display_group.add_argument('--preview', action='store_true', 
                    help='Generate HTML preview of results and open in browser')
    display_group.add_argument('--handle-chunks', choices=['group', 'flat', 'parent-only'], default='group',
                    help='How to handle chunked documents: group by parent, show all chunks, or parent only (default: group)')
    display_group.add_argument('-d', '--debug', action='store_true', 
                    help='Show debug information and raw API responses')
    display_group.add_argument('--explain-scores', action='store_true',
                    help='Show relevance scores and explain ranking where available')
    
    return parser.parse_args()


def create_meilisearch_client(url, api_key=None):
    client_args = {'url': url}
    if api_key:
        client_args['api_key'] = api_key
    
    try:
        client = meilisearch.Client(**client_args)
        # Quick health check
        health = client.health()
        if health["status"] == "available":
            return client
    except Exception as e:
        console.print(f"[bold red]Error connecting to Meilisearch:[/] {str(e)}")
        sys.exit(1)
        
    return client

def generate_embedding(text, model="nomic-embed-text", ollama_url="http://localhost:11434"):
    """Generate embeddings for text using Ollama"""
    if not text or len(text.strip()) == 0:
        return None
        
    # Truncate very long text (adjust based on model limits)
    if len(text) > 8192:
        text = text[:8192]
    
    try:
        response = requests.post(
            urljoin(ollama_url, "/api/embeddings"),
            json={"model": model, "prompt": text},
            timeout=30
        )
        
        if response.status_code == 200:
            embedding_data = response.json()
            if "embedding" in embedding_data:
                return embedding_data["embedding"]
        
        console.print(f"[yellow]Warning:[/] Failed to generate embedding: {response.text}")
        return None
    except Exception as e:
        console.print(f"[yellow]Warning:[/] Error generating embedding: {str(e)}")
        return None

def query_llm(prompt, model="cas/llama-3.2-3b-instruct", ollama_url="http://localhost:11434", system_prompt=None):
    """Send a query to the Ollama LLM API"""
    request_data = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }
    
    if system_prompt:
        request_data["system"] = system_prompt
    
    try:
        response = requests.post(
            urljoin(ollama_url, "/api/generate"),
            json=request_data,
            timeout=120  # Longer timeout for LLM responses
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get("response", "Error: No response generated")
        
        console.print(f"[bold red]Error from LLM API:[/] {response.text}")
        return f"Error: {response.status_code} - {response.text}"
    except Exception as e:
        console.print(f"[bold red]Error querying LLM:[/] {str(e)}")
        return f"Error: {str(e)}"

def build_search_params(args):
    params = {
        'limit': args.limit,
    }
    
    if args.filter:
        params['filter'] = args.filter
    
    if args.sort:
        params['sort'] = [args.sort]
    
    if args.attributes:
        params['attributesToRetrieve'] = args.attributes.split(',')
    else:
        # Default attributes to retrieve
        params['attributesToRetrieve'] = [
            'id', 'title', 'path', 'content', 'fileType', 'fileSize', 'modifiedAt',
            'is_chunk', 'is_parent', 'parent_id', 'chunk_index', 'chunks_count'
        ]
    
    if args.facets:
        params['facets'] = args.facets.split(',')
    
    # Always add highlighting for content
    if not args.no_highlight:
        # Set highlighting parameters
        params['attributesToHighlight'] = ['content']
        params['highlightPreTag'] = '<<<HIGHLIGHT>>>'
        params['highlightPostTag'] = '<<<END_HIGHLIGHT>>>'
        # Make sure we get matches in the content
        params['showMatchesPosition'] = True
    
    # Add vector search if requested
    if args.semantic or args.hybrid:
        query_embedding = generate_embedding(
            args.query,
            model=args.embedding_model,
            ollama_url=args.ollama_url
        )
        
        if query_embedding:
            # Set the embedding vector
            params['vector'] = query_embedding
            
            # IMPROVED: Better configuration of hybrid search parameters
            if args.semantic:
                # For semantic search, we'll use hybrid but with a high semantic ratio
                # This ensures we'll get some keyword matches when they exist
                params['hybrid'] = {
                    'embedder': args.embedding_model,
                    'semanticRatio': 0.85  # Mostly semantic with some keyword matching
                }
                
                # Add query text for matching as well (helps with exact term matches)
                # This will highlight the query term if it exists in the content
                params['q'] = args.query
            
            elif args.hybrid:
                # For hybrid search, use the specified semantic weight but with better defaults
                params['hybrid'] = {
                    'embedder': args.embedding_model,
                    'semanticRatio': args.semantic_weight
                }
                
                # Add optional parameters for better hybrid search
                if 'matchingStrategy' not in params:
                    params['matchingStrategy'] = 'all'  # Require all query terms to match for keyword part
            
            # Add additional parameters for both search types
            params['attributesToSearchOn'] = ['content', 'title']  # Prioritize these fields
        else:
            console.print("[yellow]Warning:[/] Could not generate embedding. Falling back to keyword search.")
    
    return params

def manually_highlight(content, query_terms, context_size=30):
    """Manually highlight occurrences of query terms in content"""
    if not content or not query_terms:
        return content
    
    # Split query into individual terms and clean them
    terms = [term.lower() for term in re.findall(r'\w+', query_terms)]
    
    # Find all occurrences of each term
    results = []
    content_lower = content.lower()
    
    for term in terms:
        if len(term) < 3:  # Skip very short terms
            continue
        
        # Find all occurrences of this term
        start_pos = 0
        while True:
            pos = content_lower.find(term, start_pos)
            if pos == -1:
                break
                
            # Get the surrounding context
            context_start = max(0, pos - context_size * 5)  # Approximate 5 chars per word
            context_end = min(len(content), pos + len(term) + context_size * 5)
            
            # Extract the context with the term
            context = content[context_start:context_end]
            term_in_context_pos = pos - context_start
            
            # Add highlight markers
            highlighted_context = (
                context[:term_in_context_pos] +
                "<<<HIGHLIGHT>>>" + context[term_in_context_pos:term_in_context_pos + len(term)] + "<<<END_HIGHLIGHT>>>" +
                context[term_in_context_pos + len(term):]
            )
            
            # Add ellipsis if needed
            if context_start > 0:
                highlighted_context = "... " + highlighted_context
            if context_end < len(content):
                highlighted_context = highlighted_context + " ..."
                
            results.append(highlighted_context)
            start_pos = pos + len(term)
            
            # Limit number of matches per term
            if len(results) >= 10:
                break
    
    # Return original content if no matches found
    if not results:
        return content[:200] + "..." if len(content) > 200 else content
        
    return results

def extract_context(content, highlight_tag_start, highlight_tag_end, context_size):
    """Extract content around highlighted terms"""
    if not content:
        return []
        
    if highlight_tag_start not in content:
        # If no highlighting, return first part of content
        return [content[:min(len(content), 200)] + "..." if len(content) > 200 else content]
    
    contexts = []
    current_pos = 0
    
    while highlight_tag_start in content[current_pos:]:
        # Find the next highlight
        highlight_start = content.find(highlight_tag_start, current_pos)
        if highlight_start == -1:
            break
            
        highlight_end = content.find(highlight_tag_end, highlight_start)
        if highlight_end == -1:
            break
            
        # Find context boundaries (word-based)
        context_start = max(0, highlight_start)
        for _ in range(context_size):
            space_pos = content.rfind(' ', 0, context_start)
            if space_pos == -1:
                context_start = 0
                break
            context_start = space_pos
        
        context_end = min(len(content), highlight_end + len(highlight_tag_end))
        for _ in range(context_size):
            space_pos = content.find(' ', context_end)
            if space_pos == -1:
                context_end = len(content)
                break
            context_end = space_pos
        
        # Extract context with highlight
        context = content[context_start:context_end].strip()
        
        # Add ellipsis if needed
        if context_start > 0:
            context = "... " + context
        if context_end < len(content):
            context = context + " ..."
            
        contexts.append(context)
        current_pos = highlight_end + len(highlight_tag_end)
    
    return contexts if contexts else [content[:min(len(content), 200)] + "..." if len(content) > 200 else content]

def extract_highlights(content, highlight_start_tag, highlight_end_tag, context_size):
    """Extract content around highlighted terms"""
    if not content or highlight_start_tag not in content:
        return []
        
    highlights = []
    pos = 0
    
    while True:
        # Find next highlight
        start_pos = content.find(highlight_start_tag, pos)
        if start_pos == -1:
            break
            
        end_pos = content.find(highlight_end_tag, start_pos)
        if end_pos == -1:
            break
            
        # Get context before and after the highlight
        context_start = max(0, start_pos - context_size * 5)  # Approximate 5 chars per word
        context_end = min(len(content), end_pos + len(highlight_end_tag) + context_size * 5)
        
        # Extract the context with highlight
        context = content[context_start:context_end]
        
        # Add ellipsis if needed
        if context_start > 0:
            context = "... " + context
        if context_end < len(content):
            context = context + " ..."
            
        highlights.append(context)
        pos = end_pos + len(highlight_end_tag)
    
    return highlights

def group_chunks_by_parent(hits):
    """Group document chunks by their parent document"""
    # First, separate parents and chunks
    parents = {}
    chunks = []
    regular_docs = []
    
    for hit in hits:
        if hit.get('is_parent', False):
            parents[hit['id']] = hit
        elif hit.get('is_chunk', False) and 'parent_id' in hit:
            chunks.append(hit)
        else:
            regular_docs.append(hit)
    
    # Group chunks by parent_id
    grouped_chunks = {}
    for chunk in chunks:
        parent_id = chunk['parent_id']
        if parent_id not in grouped_chunks:
            grouped_chunks[parent_id] = []
        grouped_chunks[parent_id].append(chunk)
    
    # Sort chunks by index
    for parent_id, chunk_list in grouped_chunks.items():
        grouped_chunks[parent_id] = sorted(chunk_list, key=lambda x: x.get('chunk_index', 0))
    
    # Create result structure
    results = []
    
    # Add regular documents
    results.extend(regular_docs)
    
    # Add parents with their chunks
    for parent_id, chunk_list in grouped_chunks.items():
        if parent_id in parents:
            parent = parents[parent_id]
            parent['_chunks'] = chunk_list
            results.append(parent)
        else:
            # If parent is missing, add chunks as individual results
            results.extend(chunk_list)
    
    # Add parents without chunks
    for parent_id, parent in parents.items():
        if parent_id not in grouped_chunks:
            results.append(parent)
    
    return results

def format_hit_for_display(hit, args):
    """Format a search hit for display, handling both regular documents and chunks"""
    is_parent = hit.get('is_parent', False)
    is_chunk = hit.get('is_chunk', False)
    
    # Basic document info
    doc_info = {
        'title': hit.get('title', 'Untitled'),
        'path': hit.get('path', 'Unknown location'),
        'type': hit.get('fileType', '').upper(),
        'size': hit.get('fileSize', 0),
        'modified': hit.get('modifiedAt', None),
    }
    
    # For chunked documents
    if is_parent:
        doc_info['chunks_count'] = hit.get('chunks_count', 0)
        doc_info['has_chunks'] = '_chunks' in hit and len(hit['_chunks']) > 0
    
    if is_chunk:
        doc_info['parent_id'] = hit.get('parent_id', None)
        doc_info['chunk_index'] = hit.get('chunk_index', 0)
    
    # Content extraction
    contexts = []
    formatted = None
    
    if '_formatted' in hit:
        formatted = hit['_formatted']
    elif hasattr(hit, '_formatted'):
        formatted = hit._formatted
        
    if formatted and 'content' in formatted:
        formatted_content = formatted['content']
        contexts = extract_context(
            formatted_content, 
            '<<<HIGHLIGHT>>>', 
            '<<<END_HIGHLIGHT>>>', 
            args.context_words
        )
    elif 'content' in hit and hit['content']:
        # Try manual highlighting
        manual_contexts = manually_highlight(hit['content'], args.query, args.context_words)
        if isinstance(manual_contexts, list):
            contexts = manual_contexts
        else:
            # Get the max characters based on content_size
            max_chars = min(len(hit['content']), args.content_size)
            contexts = [hit['content'][:max_chars] + ("..." if max_chars < len(hit['content']) else "")]
    
    # IMPORTANT: If no contexts were found but this is a valid result,
    # add a content preview so we don't show empty results
    if not contexts and 'content' in hit and hit['content']:
        # Extract a preview from the beginning of the content
        content_preview = hit['content'][:args.content_size]
        if len(hit['content']) > args.content_size:
            content_preview += "..."
        contexts = [content_preview]
    
    # Limit contexts based on content_lines
    contexts = contexts[:args.content_lines]
    
    # If this is a parent with chunks, also extract contexts from chunks
    chunk_contexts = []
    if is_parent and '_chunks' in hit and len(hit['_chunks']) > 0:
        for chunk in hit['_chunks'][:3]:  # Limit to first 3 chunks
            chunk_formatted = None
            if '_formatted' in chunk:
                chunk_formatted = chunk['_formatted']
            
            if chunk_formatted and 'content' in chunk_formatted:
                chunk_contexts.extend(extract_context(
                    chunk_formatted['content'],
                    '<<<HIGHLIGHT>>>',
                    '<<<END_HIGHLIGHT>>>',
                    args.context_words
                )[:2])  # Limit to 2 contexts per chunk
            elif 'content' in chunk:
                manual_chunk_contexts = manually_highlight(chunk['content'], args.query, args.context_words)
                if isinstance(manual_chunk_contexts, list):
                    chunk_contexts.extend(manual_chunk_contexts[:2])
                else:
                    # Add a preview of the chunk content if no highlights
                    content_preview = chunk['content'][:min(len(chunk['content']), args.content_size)]
                    if len(chunk['content']) > args.content_size:
                        content_preview += "..."
                    chunk_contexts.append(content_preview)
    
    # Add formatted result
    doc_info['contexts'] = contexts
    doc_info['chunk_contexts'] = chunk_contexts
    
    return doc_info

def extract_document_content(hit, max_chars=10000):
    """Extract clean document content for LLM processing"""
    content = ""
    
    # Handle regular documents
    if 'content' in hit and hit['content']:
        content = hit['content']
        
    # Handle parent documents with chunks
    if hit.get('is_parent', False) and '_chunks' in hit:
        chunks_content = []
        for chunk in hit['_chunks']:
            if 'content' in chunk and chunk['content']:
                chunks_content.append(chunk['content'])
        
        if chunks_content:
            content = "\n\n".join(chunks_content)
    
    # Truncate if too long
    if len(content) > max_chars:
        content = content[:max_chars] + "..."
    
    return content

def extract_key_points(document, model="cas/llama-3.2-3b-instruct", ollama_url="http://localhost:11434"):
    """Extract key points from a document using an LLM"""
    
    content = extract_document_content(document)
    
    prompt = f"""Analyze the following document and extract the 5 most important key points:
Title: {document.get('title', 'Untitled')}

Content:
{content[:8000]}  # Limit for LLM context

Format your response as a bullet point list with a brief explanation for each point.
"""

    return query_llm(prompt, model=model, ollama_url=ollama_url)

def summarize_document(document, model="cas/llama-3.2-3b-instruct", ollama_url="http://localhost:11434"):
    """Generate a comprehensive summary of a document"""
    
    content = extract_document_content(document)
    
    prompt = f"""Please provide a detailed summary of the following document:
Title: {document.get('title', 'Untitled')}

Content:
{content[:6000]}  # Limit for LLM context

Your summary should:
1. Capture the main ideas and arguments
2. Highlight key findings or conclusions
3. Preserve the original structure where relevant
4. Be around 300-500 words

Format your response in well-structured paragraphs.
"""

    return query_llm(prompt, model=model, ollama_url=ollama_url)

def compare_documents(documents, query, model="cas/llama-3.2-3b-instruct", ollama_url="http://localhost:11434"):
    """Compare multiple documents and highlight similarities and differences"""
    
    if len(documents) < 2:
        return "Need at least 2 documents to compare."
    
    # Prepare document summaries for comparison
    doc_summaries = []
    for i, doc in enumerate(documents[:3]):  # Limit to 3 documents max
        content = extract_document_content(doc, max_chars=3000)
        doc_summaries.append(f"Document {i+1}: {doc.get('title', f'Document {i+1}')}\n{content[:3000]}")
    
    doc_text = "\n\n".join(doc_summaries)  # Create the document text separately

    prompt = f"""Compare and contrast the following documents in relation to the query: "{query}"

{doc_text}

Please provide:
1. A brief summary of each document
2. Key similarities between the documents
3. Important differences or unique perspectives
4. An evaluation of which document(s) best address the query and why

Format your response in clearly labeled sections.
"""

    return query_llm(prompt, model=model, ollama_url=ollama_url)

def ask_about_documents(documents, question, model="cas/llama-3.2-3b-instruct", ollama_url="http://localhost:11434"):
    """Ask a specific question about the documents"""
    
    # Prepare document contexts
    doc_contexts = []
    for i, doc in enumerate(documents):
        content = extract_document_content(doc, max_chars=4000)
        doc_contexts.append(f"Document {i+1}: {doc.get('title', f'Document {i+1}')}\n{content[:4000]}")
    
    doc_text = "\n\n".join(doc_contexts)  # Create the document text separately

    prompt = f"""Based on the following documents, please answer this question:

Question: {question}

Documents:
{doc_text}

Your answer should:
1. Directly address the question
2. Cite specific information from the documents
3. Include document references (e.g., "According to Document 1...")
4. State if the documents don't contain sufficient information to answer

Please provide a comprehensive answer based solely on the provided documents.
"""

    system_prompt = "You are a research assistant that helps answer questions based on provided documents. Only use information from the provided documents to answer questions."

    return query_llm(prompt, model=model, ollama_url=ollama_url, system_prompt=system_prompt)

def analyze_documents(documents, query, model="cas/llama-3.2-3b-instruct", ollama_url="http://localhost:11434"):
    """Perform detailed analysis of documents in relation to a query"""
    
    # Prepare document contexts
    doc_contexts = []
    for i, doc in enumerate(documents):
        content = extract_document_content(doc, max_chars=3000)
        doc_contexts.append(f"Document {i+1}: {doc.get('title', f'Document {i+1}')}\n{content[:3000]}")
    
    doc_text = "\n\n".join(doc_contexts)  # Create the document text separately
    prompt = f"""Perform a detailed analysis of the following documents in relation to the query: "{query}"

{doc_text}

Your analysis should include:
1. Overview of how each document relates to the query
2. Critical evaluation of the arguments, evidence, or information presented
3. Identification of potential biases, limitations, or gaps
4. Assessment of the credibility and relevance of each source
5. Synthesis of key insights across all documents

Format your response as a scholarly analysis with clearly labeled sections.
"""

    return query_llm(prompt, model=model, ollama_url=ollama_url)

def chat_with_documents(documents, query, model="cas/llama-3.2-3b-instruct", ollama_url="http://localhost:11434"):
    """Start an interactive chat session with document context"""
    console.print("\n[bold cyan]Starting chat session with document context[/]")
    console.print("[italic]Type 'exit' or 'quit' to end the chat session[/]")
    
    # Prepare document contexts
    doc_contexts = []
    for i, doc in enumerate(documents):
        title = doc.get('title', f'Document {i+1}')
        content = extract_document_content(doc, max_chars=3000)
        doc_contexts.append(f"Document {i+1}: {title}\n{content[:3000]}")
    
    # Initial system context
    doc_text = "\n\n".join(doc_contexts)  # Create the document text separately

    system_context = f"""You are an assistant that helps with document analysis. You have access to the following documents that were retrieved based on the user's query: "{query}"

{doc_text}

When answering questions:
1. Use information from these documents when relevant
2. Cite specific documents when referencing their content (e.g., "According to Document 1...")
3. If the documents don't contain relevant information, say so and provide your best answer
4. Keep responses concise but informative
"""
    
    # Chat history for context
    chat_history = []
    
    # Start with an initial greeting that mentions the documents
    initial_response = query_llm(
        f"The user is researching: {query}. Introduce yourself and briefly mention what documents you have available to discuss.",
        model=model,
        ollama_url=ollama_url,
        system_prompt=system_context
    )
    
    console.print(f"[bold green]Assistant:[/] {initial_response}\n")
    chat_history.append({"role": "assistant", "content": initial_response})
    
    # Chat loop
    while True:
        user_input = console.input("[bold blue]You:[/] ")
        
        if user_input.lower() in ['exit', 'quit', 'bye', 'goodbye']:
            console.print("[italic]Ending chat session...[/]")
            break
        
        chat_history.append({"role": "user", "content": user_input})
        
        # Construct prompt with chat history
        history_text = ""
        for msg in chat_history[-6:]:  # Use last 6 messages for context
            if msg["role"] == "user":
                history_text += f"User: {msg['content']}\n"
            else:
                history_text += f"Assistant: {msg['content']}\n"
        
        # Get response
        response = query_llm(
            f"{history_text}\nUser: {user_input}\nAssistant:",
            model=model,
            ollama_url=ollama_url,
            system_prompt=system_context
        )
        
        console.print(f"[bold green]Assistant:[/] {response}\n")
        chat_history.append({"role": "assistant", "content": response})

def output_rich(results, args):
    """Output results with rich formatting"""
    hits = results.get('hits', [])
    total_hits = results.get('estimatedTotalHits', results.get('nbHits', len(hits)))
    processing_time = results.get('processingTimeMs', None)
    
    # Handle document chunks
    if args.handle_chunks != 'flat':
        hits = group_chunks_by_parent(hits)
    
    # Create header
    search_type = "Semantic Search" if args.semantic else "Hybrid Search" if args.hybrid else "Keyword Search"
    console.print(Panel(
        f"[bold blue]Search Results for:[/] [yellow]'{args.query}'[/] ([cyan]{search_type}[/])", 
        subtitle=f"Found {total_hits} results" + (f" in {processing_time}ms" if processing_time else "")
    ))
    
    # Display results
    for i, hit in enumerate(hits):
        # Skip based on chunk handling preference
        is_parent = hit.get('is_parent', False)
        is_chunk = hit.get('is_chunk', False)
        
        if is_parent and args.handle_chunks == 'flat':
            continue
        if is_chunk and args.handle_chunks == 'parent-only':
            continue
        
        # Determine document type and styling
        if is_parent:
            doc_type = f"[blue]PARENT DOCUMENT ({hit.get('chunks_count', 0)} chunks)[/]"
            border_style = "blue"
        elif is_chunk:
            doc_type = f"[cyan]DOCUMENT CHUNK {hit.get('chunk_index', 0)+1}[/]"
            border_style = "cyan"
        else:
            doc_type = f"[{hit.get('fileType', '').upper()}]" if 'fileType' in hit else ""
            border_style = "green"
        
        # Build result content
        result_text = Text()
        result_text.append(f"[{i+1}] ", style="cyan bold")
        result_text.append(f"{hit.get('title', 'Untitled')} ", style="green bold")
        result_text.append(f"{doc_type}\n", style="blue")
        
        # Path display
        if 'path' in hit:
            if args.show_path:
                result_text.append(f"Path: {hit['path']}\n", style="yellow")
            else:
                path_display = hit['path'][-40:] if len(hit['path']) > 40 else hit['path']
                result_text.append(f"Path: ...{path_display}\n", style="yellow")
        
        # File metadata
        metadata = []
        if 'fileSize' in hit:
            size_kb = hit['fileSize'] / 1024
            if size_kb < 1024:
                metadata.append(f"Size: {size_kb:.1f} KB")
            else:
                metadata.append(f"Size: {size_kb/1024:.1f} MB")
        
        if 'modifiedAt' in hit:
            try:
                mod_date = datetime.fromtimestamp(hit['modifiedAt']).strftime('%Y-%m-%d')
                metadata.append(f"Modified: {mod_date}")
            except:
                pass
                
        if metadata:
            result_text.append(" | ".join(metadata) + "\n", style="dim")
        
        # Content highlights
        content_added = False
        if '_formatted' in hit and 'content' in hit['_formatted']:
            # Extract highlights with context
            highlights = extract_highlights(
                hit['_formatted']['content'], 
                '<<<HIGHLIGHT>>>', 
                '<<<END_HIGHLIGHT>>>', 
                args.context_words
            )
            
            if highlights:
                result_text.append("\nHighlights:\n", style="magenta")
                for j, highlight in enumerate(highlights[:args.content_lines]):
                    result_text.append(f"  [{j+1}] ", style="dim")
                    
                    # Format highlighted text
                    parts = highlight.split('<<<HIGHLIGHT>>>')
                    for k, part in enumerate(parts):
                        if k > 0:  # Not the first part
                            mark_parts = part.split('<<<END_HIGHLIGHT>>>', 1)
                            if len(mark_parts) > 1:
                                result_text.append(mark_parts[0], style=args.highlight_style)
                                result_text.append(mark_parts[1])
                            else:
                                result_text.append(part)
                        else:
                            result_text.append(part)
                    
                    result_text.append("\n")
                content_added = True
        
        # If no highlights were found but content exists, show content preview
        if not content_added and 'content' in hit:
            result_text.append("\nContent Preview:\n", style="magenta")
            content_preview = hit['content'][:args.content_size]
            if len(hit['content']) > args.content_size:
                content_preview += "..."
            result_text.append(f"  {content_preview}\n")
            content_added = True
        
        # If still no content displayed (which shouldn't happen now), add a note
        if not content_added:
            result_text.append("\n[Semantic match - no direct keyword match in content]\n", style="yellow italic")
        
        # For parent documents, show chunk highlights
        chunk_content_added = False
        if is_parent and '_chunks' in hit and len(hit['_chunks']) > 0:
            result_text.append("\nChunk Highlights:\n", style="cyan")
            for j, chunk in enumerate(hit['_chunks'][:3]):  # Show only first 3 chunks
                result_text.append(f"  Chunk {chunk.get('chunk_index', j)+1}: ", style="cyan")
                
                if '_formatted' in chunk and 'content' in chunk['_formatted']:
                    # Get first highlight
                    chunk_highlights = extract_highlights(
                        chunk['_formatted']['content'], 
                        '<<<HIGHLIGHT>>>', 
                        '<<<END_HIGHLIGHT>>>', 
                        args.context_words
                    )
                    
                    if chunk_highlights:
                        # Format highlighted text
                        highlight = chunk_highlights[0]
                        parts = highlight.split('<<<HIGHLIGHT>>>')
                        for k, part in enumerate(parts):
                            if k > 0:  # Not the first part
                                mark_parts = part.split('<<<END_HIGHLIGHT>>>', 1)
                                if len(mark_parts) > 1:
                                    result_text.append(mark_parts[0], style=args.highlight_style)
                                    result_text.append(mark_parts[1])
                                else:
                                    result_text.append(part)
                            else:
                                result_text.append(part)
                        chunk_content_added = True
                    else:
                        # No highlights, show content preview
                        content_preview = chunk['content'][:args.content_size] if 'content' in chunk else ""
                        result_text.append(content_preview)
                        chunk_content_added = True
                elif 'content' in chunk:
                    # No formatting, plain snippet
                    content_preview = chunk['content'][:args.content_size]
                    if len(chunk['content']) > args.content_size:
                        content_preview += "..."
                    result_text.append(content_preview)
                    chunk_content_added = True
                else:
                    result_text.append("[No content available]", style="italic dim")
                
                result_text.append("\n")
            
            # If no chunk content was displayed, say so
            if not chunk_content_added and is_parent:
                result_text.append("  [No matching content in chunks]\n", style="yellow italic")
        
        # Display the panel with all content
        console.print(Panel(
            result_text,
            title=f"Result {i+1}",
            border_style=border_style,
            expand=False,
            padding=(1, 2)
        ))
        
        # Interactive file opening
        if args.open and 'path' in hit and os.path.exists(hit['path']):
            open_option = console.input(f"Open this file? ([green]y[/green]/[red]n[/red]/[yellow]q[/yellow]): ").lower()
            if open_option == 'y':
                try:
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', hit['path']])
                    elif sys.platform == 'win32':  # Windows
                        os.startfile(hit['path'])
                    else:  # Linux
                        subprocess.run(['xdg-open', hit['path']])
                except Exception as e:
                    console.print(f"[red]Error opening file: {e}[/]")
            elif open_option == 'q':
                console.print("[yellow]Exiting...[/]")
                break
    
    # Apply LLM function if requested
    if args.llm_function and hits:
        console.print(Panel("[bold purple]LLM Processing[/]", border_style="purple"))
        
        # Process top documents with LLM
        top_docs = hits[:args.max_context_docs]
        
        if args.llm_function == 'summarize':
            for i, doc in enumerate(top_docs):
                with console.status(f"[bold green]Summarizing document {i+1}...[/]"):
                    summary = summarize_document(doc, model=args.llm_model, ollama_url=args.ollama_url)
                
                console.print(Panel(
                    Markdown(summary),
                    title=f"[purple]Summary:[/] {doc.get('title', 'Document ' + str(i+1))}",
                    border_style="purple",
                    padding=(1, 2)
                ))
        
        elif args.llm_function == 'key_points':
            for i, doc in enumerate(top_docs):
                with console.status(f"[bold green]Extracting key points from document {i+1}...[/]"):
                    key_points = extract_key_points(doc, model=args.llm_model, ollama_url=args.ollama_url)
                
                console.print(Panel(
                    Markdown(key_points),
                    title=f"[purple]Key Points:[/] {doc.get('title', 'Document ' + str(i+1))}",
                    border_style="purple",
                    padding=(1, 2)
                ))
        
        elif args.llm_function == 'compare':
            with console.status("[bold green]Comparing documents...[/]"):
                comparison = compare_documents(top_docs, args.query, model=args.llm_model, ollama_url=args.ollama_url)
            
            console.print(Panel(
                Markdown(comparison),
                title="[purple]Document Comparison[/]",
                border_style="purple",
                padding=(1, 2)
            ))
        
        elif args.llm_function == 'ask':
            if args.question:
                with console.status(f"[bold green]Answering: {args.question}[/]"):
                    answer = ask_about_documents(top_docs, args.question, model=args.llm_model, ollama_url=args.ollama_url)
                
                console.print(Panel(
                    Markdown(answer),
                    title=f"[purple]Answer:[/] {args.question}",
                    border_style="purple",
                    padding=(1, 2)
                ))
            else:
                console.print("[bold red]Error:[/] --question is required for the 'ask' function")
        
        elif args.llm_function == 'chat':
            chat_with_documents(top_docs, args.query, model=args.llm_model, ollama_url=args.ollama_url)

def output_text_v1(results, args):
    """Output results as plain text"""
    hits = results.get('hits', [])
    total_hits = results.get('estimatedTotalHits', results.get('nbHits', len(hits)))
    processing_time = results.get('processingTimeMs', None)
    
    # Handle document chunks based on preference
    if args.handle_chunks != 'flat':
        hits = group_chunks_by_parent(hits)
    
    for i, hit in enumerate(hits):
        is_parent = hit.get('is_parent', False)
        is_chunk = hit.get('is_chunk', False)
        
        # Skip parent documents if we only want flat view
        if is_parent and args.handle_chunks == 'flat':
            continue
            
        # Skip chunks if we only want parent documents
        if is_chunk and args.handle_chunks == 'parent-only':
            continue
        
        print(f"\n[{i+1}] {hit.get('title', 'Untitled')}")
        print("-" * 70)
        
        # Show document type
        if is_parent:
            print(f"Type: PARENT DOCUMENT ({hit.get('chunks_count', 0)} chunks)")
        elif is_chunk:
            print(f"Type: DOCUMENT CHUNK {hit.get('chunk_index', 0)+1}")
        elif 'fileType' in hit:
            print(f"Type: {hit['fileType'].upper()}")
        
        # Show path
        if args.show_path and 'path' in hit:
            print(f"Path: {hit['path']}")
        elif 'path' in hit:
            print(f"Path: ...{hit['path'][-40:]}" if len(hit['path']) > 40 else f"Path: {hit['path']}")
        
        # Show size if available
        if 'fileSize' in hit:
            print(f"Size: {hit['fileSize']/1024:.1f} KB")
        
        # Extract and display context
        formatted = None
        if '_formatted' in hit:
            formatted = hit['_formatted']
        elif hasattr(hit, '_formatted'):
            formatted = hit._formatted
            
        if formatted and 'content' in formatted:
            formatted_content = formatted['content']
            contexts = extract_context(
                formatted_content, 
                '<<<HIGHLIGHT>>>', 
                '<<<END_HIGHLIGHT>>>', 
                args.context_words
            )
            
            print("\nContent:")
            for j, context in enumerate(contexts[:args.content_lines]):
                # Add ANSI color codes for highlighting in terminal
                highlighted_text = context.replace('<<<HIGHLIGHT>>>', '\033[1;31m').replace('<<<END_HIGHLIGHT>>>', '\033[0m')
                print(f"  [{j+1}] {highlighted_text}")
        else:
            # Try manual highlighting
            if 'content' in hit and hit['content']:
                contexts = manually_highlight(hit['content'], args.query, args.context_words)
                if isinstance(contexts, list):
                    print("\nContent:")
                    for j, context in enumerate(contexts[:args.content_lines]):
                        highlighted_text = context.replace('<<<HIGHLIGHT>>>', '\033[1;31m').replace('<<<END_HIGHLIGHT>>>', '\033[0m')
                        print(f"  [{j+1}] {highlighted_text}")
                else:
                    print("\nContent Preview:")
                    print(f"  {textwrap.shorten(hit['content'], width=args.content_size, placeholder='...')}")
            elif 'content' in hit:
                print("\nContent Preview:")
                print(f"  {textwrap.shorten(hit['content'], width=args.content_size, placeholder='...')}")
        
        # Display chunk content if this is a parent document
        if is_parent and '_chunks' in hit and len(hit['_chunks']) > 0:
            print("\nChunk Previews:")
            for j, chunk in enumerate(hit['_chunks'][:3]):  # Show only first 3 chunks
                chunk_formatted = None
                if '_formatted' in chunk:
                    chunk_formatted = chunk['_formatted']
                
                print(f"  Chunk {chunk.get('chunk_index', j)+1}:")
                if chunk_formatted and 'content' in chunk_formatted:
                    chunk_contexts = extract_context(
                        chunk_formatted['content'],
                        '<<<HIGHLIGHT>>>',
                        '<<<END_HIGHLIGHT>>>',
                        args.context_words
                    )
                    if chunk_contexts:
                        highlighted_text = chunk_contexts[0].replace('<<<HIGHLIGHT>>>', '\033[1;31m').replace('<<<END_HIGHLIGHT>>>', '\033[0m')
                        print(f"    {highlighted_text}")
                elif 'content' in chunk:
                    print(f"    {textwrap.shorten(chunk['content'], width=args.content_size, placeholder='...')}")
        
        if args.open and 'path' in hit and os.path.exists(hit['path']):
            open_option = input(f"\nOpen this file? (y/n/q): ").lower()
            if open_option == 'y':
                try:
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', hit['path']])
                    elif sys.platform == 'win32':  # Windows
                        os.startfile(hit['path'])
                    else:  # Linux
                        subprocess.run(['xdg-open', hit['path']])
                except Exception as e:
                    print(f"Error opening file: {e}")
            elif open_option == 'q':
                print("Exiting...")
                break

    # Display search stats
    print(f"\nFound {total_hits} results (showing {len(hits)})")
    if processing_time is not None:
        print(f"Search completed in {processing_time}ms")
    
    # Apply LLM functions if requested
    if args.llm_function and hits:
        print("\n" + "="*70)
        print(f"APPLYING LLM FUNCTION: {args.llm_function.upper()}")
        print("="*70)
        
        # Prepare documents for LLM processing
        if args.handle_chunks != 'flat':
            processed_hits = group_chunks_by_parent(hits)
        else:
            processed_hits = hits
            
        # Take only the top documents for LLM processing
        top_docs = processed_hits[:args.max_context_docs]
        
        if args.llm_function == 'key_points':
            for i, doc in enumerate(top_docs):
                print(f"\n[{i+1}] KEY POINTS FROM: {doc.get('title', 'Document ' + str(i+1))}")
                print("-" * 70)
                key_points = extract_key_points(doc, model=args.llm_model, ollama_url=args.ollama_url)
                print(key_points)
                
        elif args.llm_function == 'summarize':
            for i, doc in enumerate(top_docs):
                print(f"\n[{i+1}] SUMMARY OF: {doc.get('title', 'Document ' + str(i+1))}")
                print("-" * 70)
                summary = summarize_document(doc, model=args.llm_model, ollama_url=args.ollama_url)
                print(summary)
        
        elif args.llm_function == 'compare':
            print("\nDOCUMENT COMPARISON:")
            print("-" * 70)
            comparison = compare_documents(top_docs, args.query, model=args.llm_model, ollama_url=args.ollama_url)
            print(comparison)
        
        elif args.llm_function == 'ask':
            if not args.question:
                print("Error: --question is required when using --llm-function=ask")
            else:
                print(f"\nQUESTION: {args.question}")
                print("-" * 70)
                answer = ask_about_documents(top_docs, args.question, model=args.llm_model, ollama_url=args.ollama_url)
                print(answer)
        
        elif args.llm_function == 'analyze':
            print("\nDOCUMENT ANALYSIS:")
            print("-" * 70)
            analysis = analyze_documents(top_docs, args.query, model=args.llm_model, ollama_url=args.ollama_url)
            print(analysis)
        
        elif args.llm_function == 'chat':
            chat_with_documents(top_docs, args.query, model=args.llm_model, ollama_url=args.ollama_url)

def output_text(results, args):
    """Output results as plain text"""
    hits = results.get('hits', [])
    total_hits = results.get('estimatedTotalHits', results.get('nbHits', len(hits)))
    processing_time = results.get('processingTimeMs', None)
    
    if args.handle_chunks != 'flat':
        hits = group_chunks_by_parent(hits)
    
    print(f"Search results for: '{args.query}'")
    print(f"Found {total_hits} results (showing {len(hits)})")
    if processing_time is not None:
        print(f"Search completed in {processing_time}ms")
    print("-" * 70)
    
    for i, hit in enumerate(hits):
        # Skip based on chunk handling preference
        is_parent = hit.get('is_parent', False)
        is_chunk = hit.get('is_chunk', False)
        
        if is_parent and args.handle_chunks == 'flat':
            continue
        if is_chunk and args.handle_chunks == 'parent-only':
            continue
        
        # Print title and type
        print(f"\n[{i+1}] {hit.get('title', 'Untitled')}")
        
        if is_parent:
            print(f"Type: PARENT DOCUMENT ({hit.get('chunks_count', 0)} chunks)")
        elif is_chunk:
            print(f"Type: DOCUMENT CHUNK {hit.get('chunk_index', 0)+1}")
        elif 'fileType' in hit:
            print(f"Type: {hit['fileType'].upper()}")
        
        # Print path
        if 'path' in hit:
            if args.show_path:
                print(f"Path: {hit['path']}")
            else:
                print(f"Path: ...{hit['path'][-40:]}" if len(hit['path']) > 40 else f"Path: {hit['path']}")
        
        # Print file metadata
        if 'fileSize' in hit:
            print(f"Size: {hit['fileSize']/1024:.1f} KB")
        if 'modifiedAt' in hit:
            mod_date = datetime.fromtimestamp(hit['modifiedAt']).strftime('%Y-%m-%d %H:%M')
            print(f"Modified: {mod_date}")
        
        # Print content highlights
        print("\nContent:")
        if '_formatted' in hit and 'content' in hit['_formatted']:
            # Extract highlights with context
            highlights = extract_highlights(
                hit['_formatted']['content'], 
                '<<<HIGHLIGHT>>>', 
                '<<<END_HIGHLIGHT>>>', 
                args.context_words
            )
            
            for j, highlight in enumerate(highlights[:args.content_lines]):
                # Replace HTML tags with terminal formatting
                highlight_text = highlight.replace('<<<HIGHLIGHT>>>', '\033[1;31m').replace('<<<END_HIGHLIGHT>>>', '\033[0m')
                print(f"  [{j+1}] {highlight_text}")
        elif 'content' in hit:
            # No highlights, show plain snippet
            content_preview = hit['content'][:args.content_size]
            if len(hit['content']) > args.content_size:
                content_preview += "..."
            print(f"  {content_preview}")
        
        # For parent documents, show chunk previews
        if is_parent and '_chunks' in hit and len(hit['_chunks']) > 0:
            print("\nChunk Previews:")
            for j, chunk in enumerate(hit['_chunks'][:3]):  # Show only first 3 chunks
                print(f"  Chunk {chunk.get('chunk_index', j)+1}:")
                
                if '_formatted' in chunk and 'content' in chunk['_formatted']:
                    # Get first highlight
                    chunk_highlights = extract_highlights(
                        chunk['_formatted']['content'], 
                        '<<<HIGHLIGHT>>>', 
                        '<<<END_HIGHLIGHT>>>', 
                        args.context_words
                    )
                    
                    if chunk_highlights:
                        highlight_text = chunk_highlights[0].replace('<<<HIGHLIGHT>>>', '\033[1;31m').replace('<<<END_HIGHLIGHT>>>', '\033[0m')
                        print(f"    {highlight_text}")
                elif 'content' in chunk:
                    # No highlights, show plain snippet
                    content_preview = chunk['content'][:args.content_size]
                    if len(chunk['content']) > args.content_size:
                        content_preview += "..."
                    print(f"    {content_preview}")
        
        # Interactive file opening
        if args.open and 'path' in hit and os.path.exists(hit['path']):
            open_option = input(f"\nOpen this file? (y/n/q): ").lower()
            if open_option == 'y':
                try:
                    if sys.platform == 'darwin':  # macOS
                        subprocess.run(['open', hit['path']])
                    elif sys.platform == 'win32':  # Windows
                        os.startfile(hit['path'])
                    else:  # Linux
                        subprocess.run(['xdg-open', hit['path']])
                except Exception as e:
                    print(f"Error opening file: {e}")
            elif open_option == 'q':
                print("Exiting...")
                break
    
    # Apply LLM function if requested
    if args.llm_function and hits:
        print("\n" + "="*70)
        print(f"LLM FUNCTION: {args.llm_function.upper()}")
        print("="*70)
        
        # Process top documents with LLM
        top_docs = hits[:args.max_context_docs]
        
        if args.llm_function == 'summarize':
            for i, doc in enumerate(top_docs):
                print(f"\n[{i+1}] SUMMARY: {doc.get('title', 'Document ' + str(i+1))}")
                print("-" * 70)
                summary = summarize_document(doc, model=args.llm_model, ollama_url=args.ollama_url)
                print(summary)
        
        elif args.llm_function == 'key_points':
            for i, doc in enumerate(top_docs):
                print(f"\n[{i+1}] KEY POINTS: {doc.get('title', 'Document ' + str(i+1))}")
                print("-" * 70)
                key_points = extract_key_points(doc, model=args.llm_model, ollama_url=args.ollama_url)
                print(key_points)
        
        elif args.llm_function == 'compare':
            print("\nDOCUMENT COMPARISON:")
            print("-" * 70)
            comparison = compare_documents(top_docs, args.query, model=args.llm_model, ollama_url=args.ollama_url)
            print(comparison)
        
        elif args.llm_function == 'ask':
            if args.question:
                print(f"\nQUESTION: {args.question}")
                print("-" * 70)
                answer = ask_about_documents(top_docs, args.question, model=args.llm_model, ollama_url=args.ollama_url)
                print(answer)
            else:
                print("Error: --question is required for the 'ask' function")
        
        elif args.llm_function == 'chat':
            chat_with_documents(top_docs, args.query, model=args.llm_model, ollama_url=args.ollama_url)

def output_json(results, args):
    """Output results as JSON"""
    # Handle document chunks if needed
    if args.handle_chunks != 'flat':
        results['hits'] = group_chunks_by_parent(results.get('hits', []))
    
    # Format timestamps for better readability
    if 'hits' in results:
        for hit in results['hits']:
            if 'modifiedAt' in hit and hit['modifiedAt']:
                hit['modifiedAtFormatted'] = datetime.fromtimestamp(hit['modifiedAt']).strftime('%Y-%m-%d %H:%M')
    
    # Include the query and settings in the output
    output = {
        'query': args.query,
        'settings': {
            'index': args.index,
            'limit': args.limit,
            'filter': args.filter,
            'sort': args.sort,
            'semantic': args.semantic,
            'hybrid': args.hybrid,
            'semantic_weight': args.semantic_weight
        },
        'results': results
    }
    
    print(json.dumps(output, indent=2, default=str))

def generate_html_preview(results, args):
    """Generate an HTML preview of the search results"""
    hits = results.get('hits', [])
    total_hits = results.get('estimatedTotalHits', results.get('nbHits', len(hits)))
    processing_time = results.get('processingTimeMs', None)
    
    # Handle document chunks based on preference
    if args.handle_chunks != 'flat':
        hits = group_chunks_by_parent(hits)
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Search Results: {args.query}</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .result {{
            margin-bottom: 25px;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            background-color: #fff;
            transition: transform 0.2s;
        }}
        .result:hover {{
            transform: translateY(-3px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        }}
        .result-num {{
            display: inline-block;
            width: 24px;
            height: 24px;
            line-height: 24px;
            text-align: center;
            background-color: #3498db;
            color: white;
            border-radius: 50%;
            margin-right: 8px;
            font-weight: bold;
            font-size: 14px;
        }}
        .title {{
            font-size: 20px;
            font-weight: bold;
            margin-bottom: 8px;
            color: #3498db;
            display: flex;
            align-items: center;
        }}
        .file-type {{
            display: inline-block;
            margin-left: 10px;
            padding: 2px 8px;
            background-color: #e0e0e0;
            color: #555;
            border-radius: 4px;
            font-size: 14px;
            font-weight: normal;
        }}
        .meta {{
            font-size: 14px;
            color: #7f8c8d;
            margin-bottom: 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 15px;
        }}
        .meta-item {{
            display: flex;
            align-items: center;
            gap: 5px;
        }}
        .path {{
            font-family: monospace;
            padding: 8px 12px;
            background-color: #f8f9fa;
            border-radius: 4px;
            font-size: 14px;
            overflow-wrap: break-word;
            margin-bottom: 15px;
            border-left: 3px solid #3498db;
        }}
        .context-container {{
            margin-top: 15px;
        }}
        .context {{
            margin-bottom: 10px;
            padding: 15px;
            background-color: #f8f9fa;
            border-radius: 4px;
            line-height: 1.7;
            border-left: 3px solid #ddd;
        }}
        .chunk-container {{
            margin-top: 15px;
            border-top: 1px dashed #ccc;
            padding-top: 15px;
        }}
        .chunk-heading {{
            font-weight: bold;
            color: #2980b9;
            margin-bottom: 10px;
        }}
        .context-num {{
            font-weight: bold;
            color: #7f8c8d;
            margin-right: 5px;
        }}
        .highlight {{
            background-color: #ffff72;
            padding: 2px 4px;
            font-weight: bold;
            border-radius: 3px;
            box-shadow: 0 0 0 1px rgba(255, 213, 0, 0.4);
        }}
        .stats {{
            margin-top: 30px;
            padding: 15px;
            background-color: #eee;
            border-radius: 8px;
            font-size: 16px;
            text-align: center;
        }}
        .open-link {{
            display: inline-block;
            margin-top: 15px;
            padding: 8px 16px;
            background-color: #3498db;
            color: white;
            text-decoration: none;
            border-radius: 4px;
            font-size: 14px;
            font-weight: bold;
            transition: background-color 0.2s;
        }}
        .open-link:hover {{
            background-color: #2980b9;
        }}
        .query-term {{
            background-color: #e6f7ff;
            border-radius: 3px;
            padding: 2px 6px;
            font-family: monospace;
            margin: 0 2px;
        }}
        .parent-doc {{
            border-left: 4px solid #3498db;
        }}
        .chunk-doc {{
            border-left: 4px solid #2ecc71;
        }}
        .chunk-badge {{
            background-color: #2ecc71;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 8px;
        }}
        .parent-badge {{
            background-color: #3498db;
            color: white;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 12px;
            margin-left: 8px;
        }}
        @media (max-width: 600px) {{
            .meta {{
                flex-direction: column;
                gap: 5px;
            }}
        }}
    </style>
</head>
<body>
    <h1>Search Results: "<span class="query-term">{args.query}</span>"</h1>
    
    <div class="results">
"""

    for i, hit in enumerate(hits):
        # Skip based on chunk handling preference
        is_parent = hit.get('is_parent', False) 
        is_chunk = hit.get('is_chunk', False)
        
        if is_parent and args.handle_chunks == 'flat':
            continue
        if is_chunk and args.handle_chunks == 'parent-only':
            continue
        
        # Format hit for display
        formatted_hit = format_hit_for_display(hit, args)
        
        title = formatted_hit['title']
        
        # Document type badge
        doc_type_html = ""
        result_class = ""
        
        if is_parent:
            doc_type_html = f'<span class="parent-badge">PARENT</span>'
            result_class = "parent-doc"
        elif is_chunk:
            doc_type_html = f'<span class="chunk-badge">CHUNK {hit.get("chunk_index", 0)+1}</span>'
            result_class = "chunk-doc"
        elif 'fileType' in hit:
            doc_type_html = f'<span class="file-type">{hit["fileType"].upper()}</span>'
        
        # Path
        path = formatted_hit['path']
        
        # Metadata
        metadata_items = []
        if 'fileType' in hit and not is_parent and not is_chunk:
            metadata_items.append(f'<div class="meta-item"><span>Type:</span> {hit["fileType"].upper()}</div>')
        if 'fileSize' in hit:
            metadata_items.append(f'<div class="meta-item"><span>Size:</span> {hit["fileSize"]/1024:.1f} KB</div>')
        if 'modifiedAt' in hit:
            from datetime import datetime
            modified = datetime.fromtimestamp(hit['modifiedAt'])
            metadata_items.append(f'<div class="meta-item"><span>Modified:</span> {modified.strftime("%Y-%m-%d")}</div>')
        
        # Context with highlights
        contexts_html = '<div class="context-container">'
        
        # Add formatted contexts
        for j, context in enumerate(formatted_hit['contexts'][:args.content_lines]):
            # Replace highlight tags with HTML
            snippet_html = context.replace('<<<HIGHLIGHT>>>', '<span class="highlight">').replace('<<<END_HIGHLIGHT>>>', '</span>')
            contexts_html += f'<div class="context"><span class="context-num">{j+1}.</span> {snippet_html}</div>'
            
        contexts_html += '</div>'
        
        # Add chunk content if this is a parent document
        chunks_html = ""
        if is_parent and formatted_hit['chunk_contexts']:
            chunks_html = '<div class="chunk-container">'
            chunks_html += '<div class="chunk-heading">Highlights from chunks:</div>'
            
            for j, context in enumerate(formatted_hit['chunk_contexts'][:5]):  # Limit to 5 chunk contexts
                # Replace highlight tags with HTML
                snippet_html = context.replace('<<<HIGHLIGHT>>>', '<span class="highlight">').replace('<<<END_HIGHLIGHT>>>', '</span>')
                chunks_html += f'<div class="context"><span class="context-num">Chunk {j//2 + 1} [{j%2 + 1}]</span> {snippet_html}</div>'
                
            chunks_html += '</div>'
        
        # File URI for opening locally
        file_uri = ''
        if 'path' in hit and os.path.exists(hit['path']):
            file_uri = f"file://{os.path.abspath(hit['path'])}"
        
        html += f"""
    <div class="result {result_class}">
        <div class="title">
            <span class="result-num">{i+1}</span>
            {title} 
            {doc_type_html}
        </div>
        <div class="meta">
            {''.join(metadata_items)}
        </div>
        <div class="path">{path}</div>
        {contexts_html}
        {chunks_html}
        {f'<a href="{file_uri}" class="open-link">Open File</a>' if file_uri else ''}
    </div>
"""

    html += f"""
    </div>
    
    <div class="stats">
        Found {total_hits} results (showing {len(hits)})
        {f"<br>Search completed in {processing_time}ms" if processing_time is not None else ''}
    </div>
"""

    # Add LLM processing results if requested
    if args.llm_function and hits and not args.llm_function == 'chat':
        html += f"""
    <div style="margin-top: 30px; border-top: 2px solid #ddd; padding-top: 20px;">
        <h2>LLM Processing: {args.llm_function.upper()}</h2>
"""
        
        # Take only the top documents for LLM processing
        top_docs = hits[:args.max_context_docs]
        
        if args.llm_function == 'key_points':
            for i, doc in enumerate(top_docs):
                html += f"""
        <div class="result" style="border-left: 4px solid #9b59b6;">
            <h3>Key Points from: {doc.get('title', 'Document ' + str(i+1))}</h3>
            <div style="white-space: pre-line;">
                {extract_key_points(doc, model=args.llm_model, ollama_url=args.ollama_url)}
            </div>
        </div>
"""
                
        elif args.llm_function == 'summarize':
            for i, doc in enumerate(top_docs):
                html += f"""
        <div class="result" style="border-left: 4px solid #9b59b6;">
            <h3>Summary of: {doc.get('title', 'Document ' + str(i+1))}</h3>
            <div style="white-space: pre-line;">
                {summarize_document(doc, model=args.llm_model, ollama_url=args.ollama_url)}
            </div>
        </div>
"""
        
        elif args.llm_function == 'compare':
            html += f"""
        <div class="result" style="border-left: 4px solid #9b59b6;">
            <h3>Document Comparison</h3>
            <div style="white-space: pre-line;">
                {compare_documents(top_docs, args.query, model=args.llm_model, ollama_url=args.ollama_url)}
            </div>
        </div>
"""
        
        elif args.llm_function == 'ask':
            if args.question:
                html += f"""
        <div class="result" style="border-left: 4px solid #9b59b6;">
            <h3>Question: {args.question}</h3>
            <div style="white-space: pre-line;">
                {ask_about_documents(top_docs, args.question, model=args.llm_model, ollama_url=args.ollama_url)}
            </div>
        </div>
"""
        
        elif args.llm_function == 'analyze':
            html += f"""
        <div class="result" style="border-left: 4px solid #9b59b6;">
            <h3>Document Analysis</h3>
            <div style="white-space: pre-line;">
                {analyze_documents(top_docs, args.query, model=args.llm_model, ollama_url=args.ollama_url)}
            </div>
        </div>
"""
        
        html += """
    </div>
"""

    html += """
</body>
</html>
"""

    # Save HTML to a temporary file and open in browser
    with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html') as f:
        f.write(html)
        temp_file = f.name
    
    webbrowser.open('file://' + temp_file)
    return temp_file


def main():
    args = parse_arguments()
    
    try:
        # Create client and get index
        client = create_meilisearch_client(args.url, args.key)
        index = client.index(args.index)
        
        # Build search parameters
        search_params = build_search_params(args)
        
        # Execute search
        results = index.search(args.query, search_params)
        
        if args.debug:
            console.print("[bold cyan]DEBUG: Search parameters:[/]")
            console.print_json(json.dumps(search_params, indent=2))
            console.print("\n[bold cyan]DEBUG: Raw search results:[/]")
            console.print_json(json.dumps(results, indent=2, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)))
            console.print("\n" + "-"*50 + "\n")
        
        # Output results in the requested format
        if args.output == 'json':
            output_json(results, args)
        elif args.output == 'table':
            # If table output is needed, implement output_table function
            console.print("[bold red]Table output format not implemented yet.[/]")
        elif args.output == 'rich':
            output_rich(results, args)
        elif args.output == 'text':
            output_text(results, args)
        
        # Generate HTML preview if requested
        if args.preview:
            temp_file = generate_html_preview(results, args)
            console.print(f"[italic]HTML preview saved to: {temp_file}[/]")
        
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        console.print(f"[bold red]Error: {str(e)}[/]")
        sys.exit(1)


if __name__ == "__main__":
    main()
