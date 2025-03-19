#!/usr/bin/env python3

import os
import json
import subprocess
import argparse
import glob
import time
import hashlib
import tempfile
import logging
import sys
import re
import requests
import numpy as np
from urllib.parse import urljoin
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import meilisearch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('meilisearch_indexer.log')
    ]
)
logger = logging.getLogger('meilisearch_indexer')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Index documents to Meilisearch with vector search support')
    parser.add_argument('directory', help='Directory to scan for documents')
    parser.add_argument('search_mask', nargs='+', help='One or more file patterns (e.g., "*.pdf" "*.epub")')
    
    # Connection options
    connection_group = parser.add_argument_group('Connection Options')
    connection_group.add_argument('-u', '--url', default='http://localhost:7700', 
                       help='Meilisearch URL (default: http://localhost:7700)')
    connection_group.add_argument('-i', '--index', default='documents', 
                       help='Meilisearch index name (default: documents)')
    connection_group.add_argument('-k', '--key', help='Meilisearch API key (if required)')
    connection_group.add_argument('--primaryKey', default='id', 
                       help='Primary key field for documents (default: id)')
    
    # Processing options
    processing_group = parser.add_argument_group('Processing Options')
    processing_group.add_argument('-b', '--batch-size', type=int, default=100, 
                       help='Maximum number of documents per batch (default: 100)')
    processing_group.add_argument('-m', '--max-batch-mb', type=int, default=90, 
                       help='Maximum batch size in MB (default: 90, Meilisearch limit is ~95MB)')
    processing_group.add_argument('-t', '--threads', type=int, default=4,
                       help='Number of threads for parallel processing (default: 4)')
    processing_group.add_argument('-c', '--content-size-mb', type=int, default=10,
                       help='Maximum content size per document in MB (default: 10)')
    processing_group.add_argument('-r', '--recursive', action='store_true', 
                       help='Recursively scan subdirectories')
    
    # Rest of the function remains the same
    # Vector/Embedding options
    embedding_group = parser.add_argument_group('Embedding Options')
    embedding_group.add_argument('--embed', action='store_true',
                       help='Generate and store vector embeddings for documents')
    embedding_group.add_argument('--chunk', action='store_true',
                       help='Chunk large documents for better vector search')
    embedding_group.add_argument('--max-chunk-size', type=int, default=8000,
                       help='Maximum size (chars) for document chunks (default: 8000)')
    embedding_group.add_argument('--chunk-overlap', type=int, default=200,
                       help='Overlap size (chars) between chunks (default: 200)')
    embedding_group.add_argument('--embedding-model', default='nomic-embed-text',
                       help='Embedding model to use (default: nomic-embed-text)')
    embedding_group.add_argument('--embedding-dimensions', type=int, default=768,
                       help='Dimensions of embeddings from model (default: 768 for nomic-embed-text)')
    embedding_group.add_argument('--ollama-url', default='http://localhost:11434',
                       help='URL for Ollama API (default: http://localhost:11434)')
    
    # Cache and control options
    cache_group = parser.add_argument_group('Cache & Control Options')
    cache_group.add_argument('--cache-file', 
                    help='Cache file to store indexed file information (default: .meilisearch_<index>_cache.json)')
    cache_group.add_argument('-f', '--force', action='store_true', 
                    help='Force reindexing of all files regardless of modification time')
    cache_group.add_argument('--continue-on-error', action='store_true',
                    help='Continue processing if a batch fails to index')
    cache_group.add_argument('--dry-run', action='store_true',
                    help='Scan files but do not index them (test mode)')
    
    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument('-v', '--verbose', action='store_true',
                    help='Enable verbose output')
    output_group.add_argument('-q', '--quiet', action='store_true',
                    help='Suppress most output messages')
    
    args = parser.parse_args()
    
    # Set default cache file based on index name if not specified
    if not args.cache_file:
        args.cache_file = f'.meilisearch_{args.index}_cache.json'
    
    return args

def configure_logging(verbose, quiet):
    """Configure logging based on verbosity options"""
    if quiet:
        logger.setLevel(logging.WARNING)
    elif verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

def create_meilisearch_client(url, api_key=None):
    """Create and test a connection to Meilisearch"""
    client_args = {'url': url}
    if api_key:
        client_args['api_key'] = api_key
    
    client = meilisearch.Client(**client_args)
    
    # Test the connection immediately
    try:
        # A simple health check to ensure server is reachable
        health = client.health()
        logger.info(f"Connected to Meilisearch at {url} (version: {health.get('version', 'unknown')})")
        return client
    except Exception as e:
        logger.error(f"Cannot connect to Meilisearch at {url}")
        logger.error(f"Error details: {str(e)}")
        sys.exit(1)  # Exit immediately

def setup_index(client, index_name, embedding_dimensions=768, use_embeddings=False, args=None):
    """Set up and configure the Meilisearch index with vector search support if requested"""
    # Check if a primaryKey was specified in args
    primary_key = getattr(args, 'primaryKey', 'id')  # Default to 'id' if not specified
    
    # First check if the index already exists
    try:
        # Get all indexes
        indexes = client.get_indexes()
        index_exists = any(index['uid'] == index_name for index in indexes['results']) if 'results' in indexes else False
        
        if not index_exists:
            # Create the index - this is async in newer Meilisearch versions
            task = client.create_index(index_name, {'primaryKey': primary_key})
            logger.info(f"Initiated creation of index '{index_name}' with primary key '{primary_key}'")
            
            # For newer clients, we need to wait for the task to finish
            try:
                client.wait_for_task(task['taskUid'])
                logger.info(f"Index creation complete")
            except (AttributeError, TypeError):
                # Older clients or different response format
                logger.debug("Could not wait for task - assuming index was created synchronously")
        else:
            logger.info(f"Index '{index_name}' already exists")
    except Exception as e:
        logger.warning(f"Error during index check/creation: {str(e)}")
    
    # Get the index object (this should work regardless of whether we just created it or not)
    index = client.index(index_name)
    
    # Update the primary key if needed
    try:
        # Try to get the current primary key
        current_pk = None
        try:
            index_info = client.get_index(index_name)
            current_pk = index_info.get('primaryKey', None)
        except:
            pass
            
        # Set or update primary key if needed
        if current_pk is None or current_pk != primary_key:
            try:
                task = index.update_primary_key(primary_key)
                logger.info(f"Updated primary key for '{index_name}' to '{primary_key}'")
                
                # Wait for the task to complete if possible
                try:
                    client.wait_for_task(task['taskUid'])
                except:
                    pass
            except Exception as e:
                logger.warning(f"Could not update primary key: {str(e)}")
    except Exception as e:
        logger.debug(f"Error checking/updating primary key: {str(e)}")
    
    try:
        # Basic settings that work with all Meilisearch versions
        settings = {
            'searchableAttributes': ['title', 'content', 'path', 'filename'],
            'filterableAttributes': [
                'fileType', 'extension', 'path', 'directory', 'lastIndexed',
                'is_chunk', 'is_parent', 'parent_id', 'chunk_index'
            ],
            'sortableAttributes': ['createdAt', 'fileSize', 'lastIndexed', 'modifiedAt', 'chunk_index']
        }
        
        # Apply the basic settings - this might return a task
        task = index.update_settings(settings)
        
        # If it's a task, wait for it to complete
        try:
            if isinstance(task, dict) and 'taskUid' in task:
                client.wait_for_task(task['taskUid'])
        except:
            pass
            
        logger.info(f"Basic index settings for '{index_name}' configured successfully")
        
        # If vector search is requested, configure embedders
        if use_embeddings and args:
            try:
                # Fix the URL format for Ollama
                ollama_url = args.ollama_url
                if ollama_url and not (ollama_url.endswith('/api/embed') or ollama_url.endswith('/api/embeddings')):
                    # Make sure the URL ends with the right endpoint
                    ollama_url = ollama_url.rstrip('/') + '/api/embeddings'
                
                # Configure Ollama embedder
                embedder_settings = {
                    args.embedding_model: {
                        'source': 'ollama',
                        'url': ollama_url,  # Use the corrected URL
                        'model': args.embedding_model,
                        'documentTemplate': '{{doc.content}}'
                    }
                }
                
                # Try the dedicated method first
                try:
                    task = index.update_embedders(embedder_settings)
                    logger.info(f"Vector search enabled with {embedding_dimensions} dimensions using ollama embedder")
                    
                    # Wait for the task if possible
                    try:
                        if isinstance(task, dict) and 'taskUid' in task:
                            client.wait_for_task(task['taskUid'])
                    except:
                        pass
                except AttributeError:
                    # Fall back to patch request if method doesn't exist in older client versions
                    response = requests.patch(
                        f"{client.config.url}/indexes/{index_name}/settings/embedders",
                        headers={
                            "Content-Type": "application/json",
                            "Authorization": f"Bearer {client.config.api_key}" if client.config.api_key else None
                        },
                        json=embedder_settings
                    )
                    
                    if response.status_code == 202:
                        logger.info(f"Vector search enabled with {embedding_dimensions} dimensions using REST API call")
                    else:
                        logger.warning(f"Failed to enable vector search via REST API: {response.text}")
                        
            except Exception as e:
                logger.warning(f"Could not configure embedder: {str(e)}")
                logger.warning("Documents will be indexed without vector search capabilities.")
                # Continue without embedders - don't exit the program
        
    except Exception as e:
        logger.error(f"Could not configure index settings: {str(e)}")
        logger.error("Check that the Meilisearch server is running and accessible.")
        sys.exit(1)
    
    return index

def extract_text(file_path):
    """Extract text from various document formats with ebook-converter as primary method"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Direct handling for text files - fastest
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        # Try ebook-converter first for all non-text files
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            try:
                result = subprocess.run(
                    ['ebook-converter', file_path, temp_file.name, '-q'],
                    capture_output=True, 
                    text=True,
                    check=True,
                    timeout=90
                )
                
                with open(temp_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                logger.debug(f"ebook-converter failed for {file_path}, trying format-specific tools: {e}")
                
                # Fall back to format-specific extractors
                if ext == '.pdf':
                    try:
                        result = subprocess.run(['pdftotext', file_path, '-'], 
                                              capture_output=True, text=True, timeout=60)
                        return result.stdout
                    except:
                        pass
                
                elif ext in ['.doc', '.docx']:
                    try:
                        result = subprocess.run(['textutil', '-convert', 'txt', '-stdout', file_path], 
                                              capture_output=True, text=True, timeout=60)
                        return result.stdout
                    except:
                        pass
                
                elif ext == '.epub':
                    try:
                        result = subprocess.run(['epub2txt', file_path], 
                                              capture_output=True, text=True, timeout=60)
                        return result.stdout
                    except:
                        pass
                
                logger.warning(f"All extraction methods failed for {file_path}")
                return f"[Could not extract text from {ext} file]"
    
    except Exception as e:
        logger.warning(f"Error extracting text from {file_path}: {str(e)}")
        return f"[Error extracting text: {str(e)}]"
    
def get_file_details(file_path):
    """Get detailed information about a file"""
    stats = os.stat(file_path)
    return {
        'fileSize': stats.st_size,
        'createdAt': stats.st_ctime,
        'modifiedAt': stats.st_mtime,
        'accessedAt': stats.st_atime,
        'directory': os.path.dirname(file_path),
        'extension': os.path.splitext(file_path)[1].lower().lstrip('.')
    }

def compute_file_hash(file_path):
    """Compute an MD5 hash of the file contents"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def load_cache(cache_file):
    """Load the cache of previously indexed files"""
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load cache file: {e}")
    return {}

def save_cache(cache_file, cache):
    """Save the cache of indexed files"""
    try:
        # Create directory if it doesn't exist
        cache_dir = os.path.dirname(cache_file)
        if cache_dir and not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
            
        with open(cache_file, 'w') as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Could not save cache file: {e}")

def process_content(content, max_size_mb=10):
    """Process content to ensure it's within size limits and properly formatted"""
    if not content:
        return ""
        
    max_size = max_size_mb * 1024 * 1024  # Convert MB to bytes
    
    # Check if content is too large
    if len(content) > max_size:
        # Try to find a good truncation point
        truncated = content[:max_size]
        last_para = truncated.rfind('\n\n')
        last_sentence = truncated.rfind('. ')
        
        if last_para != -1 and last_para > max_size * 0.9:
            return truncated[:last_para] + "\n\n[Content truncated due to size limitations...]"
        elif last_sentence != -1 and last_sentence > max_size * 0.9:
            return truncated[:last_sentence+1] + " [Content truncated due to size limitations...]"
        else:
            return truncated + " [Content truncated due to size limitations...]"
    
    return content

def chunk_document(text, max_chunk_size=8000, overlap=200):
    """
    Chunk document text at natural boundaries (paragraphs, then sentences)
    with optional overlap between chunks.
    """
    if not text or len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    
    # First try to split by paragraphs (double newlines)
    paragraphs = text.split('\n\n')
    current_chunk = ""
    
    for para in paragraphs:
        # If adding this paragraph would exceed max size, store current chunk and start new one
        if len(current_chunk) + len(para) + 2 > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk)
                # Add overlap from end of previous chunk if possible
                current_chunk = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
            
            # If paragraph itself is too long, we need to split by sentences
            if len(para) > max_chunk_size:
                # Split paragraph into sentences
                sentences = re.split(r'(?<=[.!?])\s+', para)
                para_current = ""
                
                for sentence in sentences:
                    if len(para_current) + len(sentence) + 1 > max_chunk_size:
                        if para_current:
                            chunks.append(para_current)
                            # Add overlap from end of previous chunk
                            para_current = para_current[-overlap:] if len(para_current) > overlap else para_current
                        
                        # If sentence itself is too long, we split it by character count
                        if len(sentence) > max_chunk_size:
                            sent_chunks = [sentence[i:i+max_chunk_size] for i in range(0, len(sentence), max_chunk_size-overlap)]
                            chunks.extend(sent_chunks[:-1])
                            para_current = sent_chunks[-1]
                        else:
                            para_current = sentence
                    else:
                        para_current += " " + sentence if para_current else sentence
                
                if para_current:
                    current_chunk = para_current
            else:
                current_chunk = para
        else:
            current_chunk += "\n\n" + para if current_chunk else para
    
    # Add the last chunk if it's not empty
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def generate_embedding(text, model="nomic-embed-text", ollama_url="http://localhost:11434"):
    """Generate embedding for text using Ollama"""
    if not text or len(text.strip()) == 0:
        return None
        
    # Truncate very long text (adjust based on model requirements)
    if len(text) > 8192:
        text = text[:8192]
    
    try:
        response = requests.post(
            urljoin(ollama_url, "/api/embeddings"),
            json={"model": model, "prompt": text},
            timeout=30  # Add timeout to prevent hanging
        )
        
        if response.status_code == 200:
            embedding_data = response.json()
            if "embedding" in embedding_data:
                return embedding_data["embedding"]
        
        logger.warning(f"Failed to generate embedding: {response.text}")
        return None
    except Exception as e:
        logger.warning(f"Error generating embedding: {str(e)}")
        return None

def generate_embeddings_for_document(content, model="nomic-embed-text", ollama_url="http://localhost:11434", 
                                    max_chunk_size=8000, overlap=200, use_chunking=True):
    """Generate embeddings for document content with intelligent chunking"""
    if not content or len(content.strip()) == 0:
        return None, []
    
    # If chunking is enabled and document is large enough, chunk it
    if use_chunking and len(content) > max_chunk_size:
        chunks = chunk_document(content, max_chunk_size=max_chunk_size, overlap=overlap)
        logger.debug(f"Document chunked into {len(chunks)} parts")
        
        # Generate embeddings for each chunk
        chunk_embeddings = []
        for chunk in chunks:
            embedding = generate_embedding(chunk, model=model, ollama_url=ollama_url)
            if embedding:
                chunk_embeddings.append(embedding)
        
        # Also generate an embedding for the whole document if possible
        # (useful for document-level search)
        full_embedding = generate_embedding(content[:max_chunk_size], model=model, ollama_url=ollama_url)
        
        return full_embedding, (chunks, chunk_embeddings)
    else:
        # For small documents, just generate a single embedding
        embedding = generate_embedding(content, model=model, ollama_url=ollama_url)
        return embedding, []

def estimate_document_size(document):
    """Estimate the size of a document in bytes"""
    return len(json.dumps(document).encode('utf-8'))

def find_files(directory, patterns, recursive=False):
    """Find all files matching the patterns"""
    all_files = []
    for pattern in patterns:
        if recursive:
            for root, _, _ in os.walk(directory):
                matches = glob.glob(os.path.join(root, pattern))
                all_files.extend(matches)
        else:
            matches = glob.glob(os.path.join(directory, pattern))
            all_files.extend(matches)
    
    # Remove duplicates and filter out directories
    unique_files = list(set(all_files))
    return [f for f in unique_files if os.path.isfile(f)]

def sanitize_id(path):
    """Convert a file path to a valid Meilisearch document ID"""
    # Hash the path to create a fixed-length, valid ID
    # This ensures uniqueness while meeting Meilisearch's requirements
    path_hash = hashlib.md5(path.encode('utf-8')).hexdigest()
    return path_hash

def process_file(file_path, cache, args):
    """Process a single file and prepare it for indexing"""
    try:
        abs_path = os.path.abspath(file_path)
        
        # Get file details
        file_details = get_file_details(file_path)
        
        # Check if file is new/modified
        file_hash = compute_file_hash(file_path)
        status = "unchanged"
        
        # Create a sanitized document ID
        doc_id = sanitize_id(abs_path)
        
        if not args.force and abs_path in cache:
            cached_info = cache[abs_path]
            if cached_info.get('hash') == file_hash:
                # File hasn't changed, return cached info
                return {
                    'path': abs_path,
                    'status': "unchanged",
                    'cached_info': cached_info
                }
            status = "updated"
        else:
            status = "new"
            
        # Extract and process content
        content = extract_text(file_path)
        processed_content = process_content(content, args.content_size_mb)
        
        filename = os.path.basename(file_path)
        
        # Use the new sanitized document ID
        # Note: We'll still store the original path for reference
        
        # Generate embeddings if requested
        embedding = None
        chunks_data = []
        
        if args.embed:
            embedding, chunks_data = generate_embeddings_for_document(
                processed_content,
                model=args.embedding_model,
                ollama_url=args.ollama_url,
                max_chunk_size=args.max_chunk_size,
                overlap=args.chunk_overlap,
                use_chunking=args.chunk
            )
        
        # Create documents
        if args.chunk and chunks_data and len(chunks_data) > 0:
            # We have chunks, create parent-child structure
            chunks, chunk_embeddings = chunks_data
            
            # Create parent document (without content to save space)
            parent_document = {
                'id': doc_id,
                'original_path': file_path,  # Store original path for reference
                'title': filename,
                'filename': filename,
                'path': file_path,
                'content_preview': processed_content[:1000] + "..." if len(processed_content) > 1000 else processed_content,
                'is_parent': True,
                'chunks_count': len(chunks),
                'fileType': file_details['extension'],
                'lastIndexed': datetime.now().timestamp(),
                **file_details
            }
            
            # Add parent embedding if available
            if embedding and args.embed:
                parent_document['_vectors'] = {
                    args.embedding_model: embedding
                }
            
            # Create chunk documents
            chunk_documents = []
            for i, (chunk, chunk_embedding) in enumerate(zip(chunks, chunk_embeddings)):
                # Create a unique chunk ID based on parent ID and chunk index
                chunk_id = f"{doc_id}_chunk_{i}"
                
                chunk_document = {
                    'id': chunk_id,
                    'parent_id': doc_id,
                    'original_path': file_path,  # Store original path for reference
                    'chunk_index': i,
                    'title': f"{filename} (part {i+1}/{len(chunks)})",
                    'filename': filename,
                    'path': file_path,
                    'content': chunk,
                    'is_chunk': True,
                    'fileType': file_details['extension'],
                    'lastIndexed': datetime.now().timestamp()
                }
                
                # Add embedding if available
                if chunk_embedding and args.embed:
                    chunk_document['_vectors'] = {
                        args.embedding_model: chunk_embedding
                    }
                
                chunk_documents.append(chunk_document)
            
            # Create a list of all documents (parent + chunks)
            all_documents = [parent_document] + chunk_documents
            
            # Create cache entry (just for the parent)
            cache_entry = {
                'id': doc_id,
                'modifiedAt': file_details['modifiedAt'],
                'hash': file_hash,
                'lastIndexed': datetime.now().timestamp(),
                'fileSize': file_details['fileSize'],
                'chunks_count': len(chunks)
            }
            
            return {
                'path': abs_path,
                'documents': all_documents,
                'cache_entry': cache_entry,
                'status': status
            }
        else:
            # No chunks, create a single document
            single_document = {
                'id': doc_id,
                'original_path': file_path,  # Store original path for reference
                'title': filename,
                'filename': filename,
                'path': file_path,
                'content': processed_content,
                'fileType': file_details['extension'],
                'lastIndexed': datetime.now().timestamp(),
                **file_details
            }
            
            # Add embedding if available
            if embedding and args.embed:
                single_document['_vectors'] = {
                    args.embedding_model: embedding
                }
            
            # Create cache entry
            cache_entry = {
                'id': doc_id,
                'modifiedAt': file_details['modifiedAt'],
                'hash': file_hash,
                'lastIndexed': single_document['lastIndexed'],
                'fileSize': file_details['fileSize']
            }
            
            return {
                'path': abs_path,
                'documents': [single_document],
                'cache_entry': cache_entry,
                'status': status
            }
    except Exception as e:
        logger.warning(f"Error processing {file_path}: {str(e)}")
        return {
            'path': file_path,
            'status': "error",
            'error': str(e)
        }

def add_documents_with_retry(index, documents, batch_num, continue_on_error=False, max_retries=3):
    """Add documents to the index with retry logic"""
    for attempt in range(max_retries):
        try:
            batch_size_mb = sum(len(json.dumps(doc).encode('utf-8')) for doc in documents) / (1024 * 1024)
            
            start_time = time.time()
            task = index.add_documents(documents)
            duration = time.time() - start_time
            
            logger.info(f"Indexed batch {batch_num} ({len(documents)} docs, {batch_size_mb:.2f} MB) in {duration:.2f}s")
            return True
        except Exception as e:
            if attempt < max_retries - 1:
                # If it's a connection error or timeout, wait and retry
                logger.warning(f"Batch {batch_num} failed: {str(e)}. Retrying ({attempt+1}/{max_retries})...")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Failed to index batch {batch_num} after {max_retries} attempts: {str(e)}")
                if not continue_on_error:
                    raise
                return False
    return False

def index_documents(processed_files, index, args):
    """Index processed documents in batches"""
    documents = []
    current_batch_size = 0
    batch_count = 0
    indexed_count = 0
    max_batch_size_bytes = args.max_batch_mb * 1024 * 1024
    
    # Prepare current cache
    current_cache = {}
    
    # Flatten all documents
    all_documents = []
    for result in processed_files:
        # Skip unchanged and error files
        if result['status'] == 'unchanged':
            if 'cached_info' in result:
                current_cache[result['path']] = result['cached_info']
            continue
        elif result['status'] == 'error':
            continue
        
        # Add all documents from this file to the list
        if 'documents' in result:
            all_documents.extend(result['documents'])
            
        # Update cache
        if 'cache_entry' in result:
            current_cache[result['path']] = result['cache_entry']
    
    # Process all documents in batches
    for i, document in enumerate(all_documents):
        # Check document size
        doc_size = estimate_document_size(document)
        
        # If this document would make the batch too large, send the current batch first
        if documents and current_batch_size + doc_size > max_batch_size_bytes:
            if not args.dry_run:
                success = add_documents_with_retry(
                    index, documents, batch_count + 1, 
                    continue_on_error=args.continue_on_error
                )
                if success:
                    indexed_count += len(documents)
            else:
                logger.info(f"[DRY RUN] Would index batch {batch_count + 1} "
                          f"({len(documents)} docs, {current_batch_size / (1024*1024):.2f} MB)")
            
            batch_count += 1
            documents = []
            current_batch_size = 0
        
        # Add document to batch
        documents.append(document)
        current_batch_size += doc_size
        
        # If we've reached the max batch size, send the batch
        if len(documents) >= args.batch_size:
            if not args.dry_run:
                success = add_documents_with_retry(
                    index, documents, batch_count + 1,
                    continue_on_error=args.continue_on_error
                )
                if success:
                    indexed_count += len(documents)
            else:
                logger.info(f"[DRY RUN] Would index batch {batch_count + 1} "
                          f"({len(documents)} docs, {current_batch_size / (1024*1024):.2f} MB)")
            
            batch_count += 1
            documents = []
            current_batch_size = 0
    
    # Index any remaining documents
    if documents:
        if not args.dry_run:
            success = add_documents_with_retry(
                index, documents, batch_count + 1,
                continue_on_error=args.continue_on_error
            )
            if success:
                indexed_count += len(documents)
        else:
            logger.info(f"[DRY RUN] Would index batch {batch_count + 1} "
                      f"({len(documents)} docs, {current_batch_size / (1024*1024):.2f} MB)")
        
        batch_count += 1
    
    return current_cache, batch_count, indexed_count

def count_status(processed_files):
    """Count files by status"""
    counts = {
        'total': len(processed_files),
        'new': sum(1 for r in processed_files if r['status'] == 'new'),
        'updated': sum(1 for r in processed_files if r['status'] == 'updated'),
        'unchanged': sum(1 for r in processed_files if r['status'] == 'unchanged'),
        'error': sum(1 for r in processed_files if r['status'] == 'error')
    }
    
    # Count total documents (including chunks)
    counts['documents'] = sum(
        len(r.get('documents', [])) 
        for r in processed_files 
        if r['status'] in ('new', 'updated')
    )
    
    # Count parent documents and chunks separately
    counts['parents'] = sum(
        1 for r in processed_files 
        if r['status'] in ('new', 'updated') and 'documents' in r and len(r['documents']) > 0
    )
    
    counts['chunks'] = counts['documents'] - counts['parents']
    
    return counts

def main():
    args = parse_arguments()
    
    # Configure logging
    configure_logging(args.verbose, args.quiet)
    
    # Validate directory
    if not os.path.isdir(args.directory):
        logger.error(f"Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    try:
        # Find all files matching the patterns
        logger.info(f"Searching for files in {args.directory}...")
        files = find_files(args.directory, args.search_mask, args.recursive)
        logger.info(f"Found {len(files)} files matching patterns: {args.search_mask}")
        
        if not files:
            logger.warning("No files found. Check your patterns and directory.")
            return
        
        # Load cache
        cache = load_cache(args.cache_file)
        logger.info(f"Loaded cache with {len(cache)} entries")
        
        # Create client and set up index (skip in dry-run mode)
        if not args.dry_run:
            client = create_meilisearch_client(args.url, args.key)
            index = setup_index(client, args.index, args.embedding_dimensions, args.embed, args)
        else:
            logger.info("[DRY RUN] Skipping Meilisearch connection check")
            index = None
        
        # Process files (potentially in parallel)
        logger.info("Processing files...")
        processed_files = []
        
        with tqdm(total=len(files), desc="Processing documents", disable=args.quiet) as pbar:
            if args.threads > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=args.threads) as executor:
                    futures = {
                        executor.submit(
                            process_file, file_path, cache, args
                        ): file_path for file_path in files
                    }
                    
                    for future in as_completed(futures):
                        result = future.result()
                        processed_files.append(result)
                        pbar.update(1)
            else:
                # Sequential processing
                for file_path in files:
                    result = process_file(file_path, cache, args, args.embedding_model)
                    processed_files.append(result)
                    pbar.update(1)
        
        # Count files by status
        counts = count_status(processed_files)
        
        # Display stats
        logger.info(f"Processing summary:")
        logger.info(f"  - Total files processed: {counts['total']}")
        logger.info(f"  - New files: {counts['new']}")
        logger.info(f"  - Updated files: {counts['updated']}")
        logger.info(f"  - Unchanged files (skipped): {counts['unchanged']}")
        logger.info(f"  - Files with errors: {counts['error']}")
        
        if counts['new'] + counts['updated'] > 0:
            if counts['chunks'] > 0:
                logger.info(f"  - Documents created: {counts['documents']} "
                          f"({counts['parents']} parents + {counts['chunks']} chunks)")
            else:
                logger.info(f"  - Documents created: {counts['documents']}")
                
            # Index documents
            if not args.dry_run:
                logger.info("Indexing documents...")
                current_cache, batch_count, indexed_count = index_documents(
                    processed_files, index, args
                )
                
                # Save updated cache
                save_cache(args.cache_file, current_cache)
                logger.info(f"Cache saved to {args.cache_file}")
                
                logger.info(f"Indexing complete! {indexed_count} documents indexed in {batch_count} batches.")
            else:
                logger.info("[DRY RUN] Indexing skipped.")
        else:
            logger.info("No new or updated files to index.")
        
    except KeyboardInterrupt:
        logger.warning("\nOperation cancelled by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        if args.verbose:
            import traceback
            logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
