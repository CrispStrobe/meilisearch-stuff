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
    parser = argparse.ArgumentParser(description='Index documents to Meilisearch')
    parser.add_argument('directory', help='Directory to scan for documents')
    parser.add_argument('search_mask', nargs='+', help='One or more file patterns (e.g., "*.pdf" "*.epub")')
    parser.add_argument('-u', '--url', default='http://localhost:7700', help='Meilisearch URL')
    parser.add_argument('-i', '--index', default='documents', help='Meilisearch index name')
    parser.add_argument('-k', '--key', help='Meilisearch API key (if required)')
    parser.add_argument('-b', '--batch-size', type=int, default=100, 
                      help='Maximum number of documents per batch')
    parser.add_argument('-m', '--max-batch-mb', type=int, default=90, 
                      help='Maximum batch size in MB (default: 90, Meilisearch limit is ~95MB)')
    parser.add_argument('-t', '--threads', type=int, default=4,
                      help='Number of threads for parallel processing (default: 4)')
    parser.add_argument('-c', '--content-size-mb', type=int, default=10,
                      help='Maximum content size per document in MB (default: 10)')
    parser.add_argument('-r', '--recursive', action='store_true', 
                      help='Recursively scan subdirectories')
    parser.add_argument('--cache-file', default='.meilisearch_cache.json', 
                      help='Cache file to store indexed file information')
    parser.add_argument('-f', '--force', action='store_true', 
                      help='Force reindexing of all files regardless of modification time')
    parser.add_argument('--continue-on-error', action='store_true',
                      help='Continue processing if a batch fails to index')
    parser.add_argument('--dry-run', action='store_true',
                      help='Scan files but do not index them (test mode)')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Enable verbose output')
    parser.add_argument('-q', '--quiet', action='store_true',
                      help='Suppress most output messages')
    return parser.parse_args()

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

def setup_index(client, index_name):
    """Set up and configure the Meilisearch index"""
    index = client.index(index_name)
    try:
        index.update_settings({
            'searchableAttributes': ['title', 'content', 'path', 'filename'],
            'filterableAttributes': ['fileType', 'extension', 'path', 'directory', 'lastIndexed'],
            'sortableAttributes': ['createdAt', 'fileSize', 'lastIndexed', 'modifiedAt']
        })
        logger.info(f"Index '{index_name}' configured successfully")
    except Exception as e:
        logger.error(f"Could not configure index settings: {e}")
        logger.error("Check that the Meilisearch server is running and accessible.")
        sys.exit(1)
    return index

def extract_text(file_path):
    """Extract text from various document formats with optimized approach"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        # Direct handling for text files - fastest
        if ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        
        # Handle specific formats with best tools
        if ext == '.pdf':
            try:
                result = subprocess.run(['pdftotext', file_path, '-'], 
                                      capture_output=True, text=True, timeout=60)
                return result.stdout
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug(f"pdftotext failed for {file_path}, trying ebook-converter")
        
        elif ext in ['.doc', '.docx']:
            try:
                result = subprocess.run(['textutil', '-convert', 'txt', '-stdout', file_path], 
                                      capture_output=True, text=True, timeout=60)
                return result.stdout
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug(f"textutil failed for {file_path}, trying ebook-converter")
        
        elif ext == '.epub':
            try:
                result = subprocess.run(['epub2txt', file_path], 
                                      capture_output=True, text=True, timeout=60)
                return result.stdout
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug(f"epub2txt failed for {file_path}, trying ebook-converter")
        
        # Fallback to ebook-converter for any format it can handle
        with tempfile.NamedTemporaryFile(suffix='.txt') as temp_file:
            try:
                result = subprocess.run(
                    ['ebook-converter', file_path, temp_file.name, '-q'],
                    capture_output=True, 
                    text=True,
                    check=True,
                    timeout=90  # Longer timeout for complex documents
                )
                
                with open(temp_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            except (subprocess.SubprocessError, FileNotFoundError) as e:
                logger.warning(f"All extraction methods failed for {file_path}: {str(e)}")
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

def process_file(file_path, cache, force=False, max_content_size_mb=10):
    """Process a single file and prepare it for indexing"""
    try:
        abs_path = os.path.abspath(file_path)
        
        # Get file details
        file_details = get_file_details(file_path)
        
        # Check if file is new/modified
        file_hash = compute_file_hash(file_path)
        status = "unchanged"
        
        if not force and abs_path in cache:
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
        processed_content = process_content(content, max_content_size_mb)
        
        filename = os.path.basename(file_path)
        
        # Reuse document ID if it exists
        doc_id = cache.get(abs_path, {}).get('id', None)
        
        # Create document
        document = {
            'id': doc_id if doc_id else abs_path,  # Use path as ID if none exists
            'title': filename,
            'filename': filename,
            'path': file_path,
            'content': processed_content,
            'fileType': file_details['extension'],
            'lastIndexed': datetime.now().timestamp(),
            **file_details
        }
        
        # Create cache entry
        cache_entry = {
            'id': document['id'],
            'modifiedAt': file_details['modifiedAt'],
            'hash': file_hash,
            'lastIndexed': document['lastIndexed'],
            'fileSize': file_details['fileSize']
        }
        
        return {
            'path': abs_path,
            'document': document,
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

def index_documents(processed_files, index, batch_size=100, max_batch_mb=90, 
                   dry_run=False, continue_on_error=False):
    """Index processed documents in batches"""
    documents = []
    current_batch_size = 0
    batch_count = 0
    indexed_count = 0
    max_batch_size_bytes = max_batch_mb * 1024 * 1024
    
    # Prepare current cache
    current_cache = {}
    for result in processed_files:
        # Skip unchanged and error files
        if result['status'] == 'unchanged':
            current_cache[result['path']] = result['cached_info']
            continue
        elif result['status'] == 'error':
            continue
            
        document = result['document']
        cache_entry = result['cache_entry']
        
        # Check document size
        doc_size = estimate_document_size(document)
        
        # If this document would make the batch too large, send the current batch first
        if documents and current_batch_size + doc_size > max_batch_size_bytes:
            if not dry_run:
                success = add_documents_with_retry(
                    index, documents, batch_count + 1, 
                    continue_on_error=continue_on_error
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
        
        # Add to cache
        current_cache[result['path']] = cache_entry
        
        # If we've reached the max batch size, send the batch
        if len(documents) >= batch_size:
            if not dry_run:
                success = add_documents_with_retry(
                    index, documents, batch_count + 1,
                    continue_on_error=continue_on_error
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
        if not dry_run:
            success = add_documents_with_retry(
                index, documents, batch_count + 1,
                continue_on_error=continue_on_error
            )
            if success:
                indexed_count += len(documents)
        else:
            logger.info(f"[DRY RUN] Would index batch {batch_count + 1} "
                      f"({len(documents)} docs, {current_batch_size / (1024*1024):.2f} MB)")
        
        batch_count += 1
    
    return current_cache, batch_count, indexed_count

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
            index = setup_index(client, args.index)
        else:
            logger.info("[DRY RUN] Skipping Meilisearch connection check")
            index = None
        
        # Process files (potentially in parallel)
        logger.info("Processing files...")
        processed_files = []
        
        with tqdm(total=len(files), desc="Processing", disable=args.quiet) as pbar:
            if args.threads > 1:
                # Parallel processing
                with ThreadPoolExecutor(max_workers=args.threads) as executor:
                    futures = {
                        executor.submit(
                            process_file, file_path, cache, args.force, args.content_size_mb
                        ): file_path for file_path in files
                    }
                    
                    for future in as_completed(futures):
                        result = future.result()
                        processed_files.append(result)
                        pbar.update(1)
            else:
                # Sequential processing
                for file_path in files:
                    result = process_file(file_path, cache, args.force, args.content_size_mb)
                    processed_files.append(result)
                    pbar.update(1)
        
        # Count results by status
        status_counts = {
            'new': sum(1 for r in processed_files if r['status'] == 'new'),
            'updated': sum(1 for r in processed_files if r['status'] == 'updated'),
            'unchanged': sum(1 for r in processed_files if r['status'] == 'unchanged'),
            'error': sum(1 for r in processed_files if r['status'] == 'error')
        }
        
        logger.info(f"File processing summary:")
        logger.info(f"  - New files: {status_counts['new']}")
        logger.info(f"  - Updated files: {status_counts['updated']}")
        logger.info(f"  - Unchanged files (skipped): {status_counts['unchanged']}")
        logger.info(f"  - Files with errors: {status_counts['error']}")
        
        # Skip indexing if all files are unchanged or there were errors
        if status_counts['new'] + status_counts['updated'] == 0:
            logger.info("No new or updated files to index.")
            return
        
        # Index documents
        logger.info("Indexing documents...")
        current_cache, batch_count, indexed_count = index_documents(
            processed_files,
            index,
            batch_size=args.batch_size,
            max_batch_mb=args.max_batch_mb,
            dry_run=args.dry_run,
            continue_on_error=args.continue_on_error
        )
        
        # Save cache
        if not args.dry_run:
            save_cache(args.cache_file, current_cache)
            logger.info(f"Cache saved to {args.cache_file}")
        
        # Final summary
        prefix = "[DRY RUN] " if args.dry_run else ""
        logger.info(f"{prefix}Indexing complete!")
        logger.info(f"Total files processed: {len(files)}")
        if not args.dry_run:
            logger.info(f"Documents indexed: {indexed_count} in {batch_count} batches")
        else:
            logger.info(f"Documents that would be indexed: "
                      f"{status_counts['new'] + status_counts['updated']} in {batch_count} batches")
        
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
