#!/usr/bin/env python3

import os
import json
import subprocess
import argparse
import glob
from tqdm import tqdm
import meilisearch
import sys
from datetime import datetime

def parse_arguments():
    parser = argparse.ArgumentParser(description='Index documents to Meilisearch')
    parser.add_argument('directory', help='Directory to scan for documents')
    parser.add_argument('search_mask', help='Space-separated file patterns (e.g., "*.pdf *.epub")')
    parser.add_argument('-u', '--url', default='http://localhost:7700', help='Meilisearch URL')
    parser.add_argument('-i', '--index', default='documents', help='Meilisearch index name')
    parser.add_argument('-k', '--key', help='Meilisearch API key (if required)')
    parser.add_argument('-b', '--batch-size', type=int, default=100, help='Batch size for indexing')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recursively scan subdirectories')
    return parser.parse_args()

def create_meilisearch_client(url, api_key=None):
    client_args = {'url': url}
    if api_key:
        client_args['api_key'] = api_key
    return meilisearch.Client(**client_args)

def setup_index(client, index_name):
    index = client.index(index_name)
    try:
        index.update_settings({
            'searchableAttributes': ['title', 'content', 'path', 'filename'],
            'filterableAttributes': ['fileType', 'extension', 'path', 'directory'],
            'sortableAttributes': ['createdAt', 'fileSize']
        })
        print(f"Index '{index_name}' configured successfully")
    except Exception as e:
        print(f"Warning: Could not configure index settings: {e}")
    return index

def extract_text(file_path):
    """Extract text from various document formats"""
    ext = os.path.splitext(file_path)[1].lower()
    
    try:
        if ext == '.pdf':
            result = subprocess.run(['pdftotext', file_path, '-'], capture_output=True, text=True)
            return result.stdout
        elif ext in ['.doc', '.docx']:
            result = subprocess.run(['textutil', '-convert', 'txt', '-stdout', file_path], capture_output=True, text=True)
            return result.stdout
        elif ext == '.txt':
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        elif ext == '.epub':
            # You may need to install additional tools for EPUB extraction
            try:
                result = subprocess.run(['epub2txt', file_path], capture_output=True, text=True)
                return result.stdout
            except FileNotFoundError:
                return f"[EPUB conversion requires epub2txt tool]"
        else:
            return f"[Unsupported file format: {ext}]"
    except Exception as e:
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

def process_files(directory, patterns, index, batch_size=100, recursive=False):
    """Process files matching the given patterns"""
    documents = []
    batch_count = 0
    total_count = 0
    
    # Expand patterns to full paths
    all_files = []
    for pattern in patterns.split():
        if recursive:
            for root, _, _ in os.walk(directory):
                matches = glob.glob(os.path.join(root, pattern))
                all_files.extend(matches)
        else:
            matches = glob.glob(os.path.join(directory, pattern))
            all_files.extend(matches)
    
    unique_files = list(set(all_files))  # Remove duplicates
    
    print(f"Found {len(unique_files)} files matching pattern(s): {patterns}")
    
    for file_path in tqdm(unique_files, desc="Processing documents"):
        try:
            # Skip directories
            if os.path.isdir(file_path):
                continue
                
            # Extract content and file details
            content = extract_text(file_path)
            file_details = get_file_details(file_path)
            filename = os.path.basename(file_path)
            
            # Create document
            document = {
                'id': str(total_count + 1),
                'title': filename,
                'filename': filename,
                'path': file_path,
                'content': content,
                'fileType': file_details['extension'],
                **file_details
            }
            
            documents.append(document)
            total_count += 1
            
            # Index in batches
            if len(documents) >= batch_size:
                try:
                    index.add_documents(documents)
                    batch_count += 1
                    print(f"Indexed batch {batch_count} ({len(documents)} documents)")
                except Exception as e:
                    print(f"Error indexing batch {batch_count}: {str(e)}")
                documents = []
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
    
    # Index remaining documents
    if documents:
        try:
            index.add_documents(documents)
            print(f"Indexed final batch ({len(documents)} documents)")
        except Exception as e:
            print(f"Error indexing final batch: {str(e)}")
    
    print(f"Indexing complete! Processed {total_count} documents in {batch_count + 1} batches")

def main():
    args = parse_arguments()
    
    # Validate directory
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    try:
        # Create client and set up index
        client = create_meilisearch_client(args.url, args.key)
        index = setup_index(client, args.index)
        
        # Process files
        process_files(
            args.directory, 
            args.search_mask, 
            index, 
            batch_size=args.batch_size,
            recursive=args.recursive
        )
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
