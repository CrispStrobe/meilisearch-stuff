#!/usr/bin/env python3

import argparse
import json
import sys
import os
import meilisearch
import textwrap
from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich.panel import Panel
from rich.markdown import Markdown
from rich import box
import tempfile
import subprocess
import webbrowser
import re

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Search documents in Meilisearch with advanced options',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Basic search
  python search.py "neural networks"
  
  # Search with filters and more context
  python search.py "machine learning" --filter "fileType=pdf" --context-words 50

  # Show more content and highlight matches
  python search.py "blockchain" --content-lines 20 --highlight-style "bold red"
  
  # Generate HTML preview with full paths
  python search.py "artificial intelligence" --preview --show-path
  
  # Output as JSON for programmatic use
  python search.py "quantum computing" --output json
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
    display_group.add_argument('-d', '--debug', action='store_true', 
                    help='Show debug information and raw API responses')
    
    return parser.parse_args()

def create_meilisearch_client(url, api_key=None):
    client_args = {'url': url}
    if api_key:
        client_args['api_key'] = api_key
    return meilisearch.Client(**client_args)

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
        params['attributesToRetrieve'] = ['id', 'title', 'path', 'content', 'fileType', 'fileSize', 'modifiedAt']
    
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

def output_text(results, args):
    """Output results as plain text"""
    hits = results.get('hits', [])
    total_hits = results.get('estimatedTotalHits', results.get('nbHits', len(hits)))
    processing_time = results.get('processingTimeMs', None)
    
    for i, hit in enumerate(hits):
        print(f"\n[{i+1}] {hit.get('title', 'Untitled')}")
        print("-" * 70)
        
        # Show path
        if args.show_path and 'path' in hit:
            print(f"Path: {hit['path']}")
        elif 'path' in hit:
            print(f"Path: ...{hit['path'][-40:]}" if len(hit['path']) > 40 else f"Path: {hit['path']}")
        
        # Show document type and size
        if 'fileType' in hit:
            print(f"Type: {hit['fileType'].upper()}")
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

    print(f"\nFound {total_hits} results (showing {len(hits)})")
    if processing_time is not None:
        print(f"Search completed in {processing_time}ms")

def output_json(results, args):
    """Output results as JSON"""
    print(json.dumps(results, indent=2, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)))

def output_table(results, args):
    """Output results as a formatted table"""
    console = Console()
    
    hits = results.get('hits', [])
    total_hits = results.get('estimatedTotalHits', results.get('nbHits', len(hits)))
    processing_time = results.get('processingTimeMs', None)
    
    table = Table(title=f"Search Results: {args.query}", box=box.ROUNDED)
    table.add_column("#", style="cyan", no_wrap=True)
    table.add_column("Title", style="green")
    table.add_column("Type", style="blue")
    
    if args.show_path:
        table.add_column("Path", style="yellow")
    else:
        table.add_column("Location", style="yellow")
        
    table.add_column("Preview", style="magenta")
    
    for i, hit in enumerate(hits):
        # Title
        title = hit.get('title', 'Untitled')
        
        # Type
        doc_type = hit.get('fileType', '').upper()
        
        # Path
        if args.show_path and 'path' in hit:
            path = hit['path']
        elif 'path' in hit:
            path_parts = hit['path'].split(os.sep)
            path = os.sep.join(path_parts[-2:]) if len(path_parts) > 1 else hit['path']
        else:
            path = "Unknown"
        
        # Context
        preview = ""
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
            if contexts:
                # Get the first context with highlighting
                preview_text = contexts[0]
                # Create rich Text object with highlighting
                final_preview = Text()
                parts = preview_text.split('<<<HIGHLIGHT>>>')
                for j, part in enumerate(parts):
                    if j > 0:
                        highlight_parts = part.split('<<<END_HIGHLIGHT>>>', 1)
                        if len(highlight_parts) > 1:
                            final_preview.append(highlight_parts[0], style="bold red on yellow")
                            final_preview.append(highlight_parts[1])
                        else:
                            final_preview.append(part)
                    else:
                        final_preview.append(part)
                preview = final_preview
        elif 'content' in hit and hit['content']:
            # Try manual highlighting
            contexts = manually_highlight(hit['content'], args.query, args.context_words)
            if isinstance(contexts, list) and contexts:
                preview_text = contexts[0]
                # Create rich Text object with highlighting
                final_preview = Text()
                parts = preview_text.split('<<<HIGHLIGHT>>>')
                for j, part in enumerate(parts):
                    if j > 0:
                        highlight_parts = part.split('<<<END_HIGHLIGHT>>>', 1)
                        if len(highlight_parts) > 1:
                            final_preview.append(highlight_parts[0], style="bold red on yellow")
                            final_preview.append(highlight_parts[1])
                        else:
                            final_preview.append(part)
                    else:
                        final_preview.append(part)
                preview = final_preview
            else:
                preview = textwrap.shorten(hit['content'], width=60, placeholder="...")
        
        # Convert rich Text to string for table if needed
        if isinstance(preview, Text):
            # We can directly use Text objects in Rich tables
            table.add_row(str(i+1), title, doc_type, path, preview)
        else:
            table.add_row(str(i+1), title, doc_type, path, str(preview))
    
    console.print(table)
    console.print(f"Found {total_hits} results (showing {len(hits)})")
    if processing_time is not None:
        console.print(f"Search completed in {processing_time}ms")

def output_rich(results, args):
    """Output results with rich formatting"""
    console = Console()
    
    hits = results.get('hits', [])
    total_hits = results.get('estimatedTotalHits', results.get('nbHits', len(hits)))
    processing_time = results.get('processingTimeMs', None)
    
    console.print(Panel(f"[bold blue]Search Results for:[/] [yellow]'{args.query}'[/]", 
                        subtitle=f"Found {total_hits} results"))
    
    for i, hit in enumerate(hits):
        # Title with document type
        title = hit.get('title', 'Untitled')
        doc_type = f"[{hit.get('fileType', '').upper()}]" if 'fileType' in hit else ""
        
        # Format path
        if 'path' in hit:
            if args.show_path:
                path_display = hit['path']
            else:
                path_parts = hit['path'].split(os.sep)
                path_display = os.sep.join(path_parts[-2:]) if len(path_parts) > 1 else hit['path']
        else:
            path_display = "Unknown location"
        
        # Extract file metadata
        metadata = []
        if 'fileSize' in hit:
            metadata.append(f"Size: {hit['fileSize']/1024:.1f} KB")
        if 'modifiedAt' in hit:
            from datetime import datetime
            modified = datetime.fromtimestamp(hit['modifiedAt'])
            metadata.append(f"Modified: {modified.strftime('%Y-%m-%d')}")
        
        # Context with highlights
        contexts = []
        formatted = None
        if '_formatted' in hit:
            formatted = hit['_formatted']
        elif hasattr(hit, '_formatted'):
            formatted = hit._formatted
            
        if formatted and 'content' in formatted:
            formatted_content = formatted['content']
            context_snippets = extract_context(
                formatted_content, 
                '<<<HIGHLIGHT>>>', 
                '<<<END_HIGHLIGHT>>>', 
                args.context_words
            )
            
            for snippet in context_snippets[:args.content_lines]:
                # Replace highlight tags with rich formatting
                text = Text()
                parts = snippet.split('<<<HIGHLIGHT>>>')
                for j, part in enumerate(parts):
                    if j > 0:
                        highlight_parts = part.split('<<<END_HIGHLIGHT>>>', 1)
                        if len(highlight_parts) > 1:
                            text.append(highlight_parts[0], style=args.highlight_style)
                            text.append(highlight_parts[1])
                        else:
                            text.append(part)
                    else:
                        text.append(part)
                contexts.append(text)
        elif 'content' in hit and hit['content']:
            # Try manual highlighting
            manual_contexts = manually_highlight(hit['content'], args.query, args.context_words)
            if isinstance(manual_contexts, list):
                for snippet in manual_contexts[:args.content_lines]:
                    # Replace highlight tags with rich formatting
                    text = Text()
                    parts = snippet.split('<<<HIGHLIGHT>>>')
                    for j, part in enumerate(parts):
                        if j > 0:
                            highlight_parts = part.split('<<<END_HIGHLIGHT>>>', 1)
                            if len(highlight_parts) > 1:
                                text.append(highlight_parts[0], style=args.highlight_style)
                                text.append(highlight_parts[1])
                            else:
                                text.append(part)
                        else:
                            text.append(part)
                    contexts.append(text)
            else:
                # Get the max characters based on content_size
                max_chars = min(len(hit['content']), args.content_size)
                text = Text(hit['content'][:max_chars])
                if max_chars < len(hit['content']):
                    text.append("...")
                contexts.append(text)
        elif 'content' in hit:
            # Get the max characters based on content_size
            max_chars = min(len(hit['content']), args.content_size)
            text = Text(hit['content'][:max_chars])
            if max_chars < len(hit['content']):
                text.append("...")
            contexts.append(text)
        
        # Create panel with result information
        result_text = Text()
        result_text.append(f"[{i+1}] ", style="cyan bold")
        result_text.append(f"{title} ", style="green bold")
        result_text.append(f"{doc_type}\n", style="blue")
        result_text.append(f"Path: {path_display}\n", style="yellow")
        
        if metadata:
            result_text.append(" | ".join(metadata) + "\n", style="dim")
        
        if contexts:
            result_text.append("\nContent:\n", style="magenta")
            for j, context in enumerate(contexts):
                result_text.append(f"  [{j+1}] ", style="dim")
                result_text.append(context)
                result_text.append("\n")
        
        console.print(Panel(result_text, box=box.ROUNDED))
        
        # Option to open file
        if args.open and 'path' in hit and os.path.exists(hit['path']):
            open_option = input(f"Open this file? (y/n/q): ").lower()
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
                console.print("Exiting...")
                break
    
    console.print(f"Found {total_hits} results (showing {len(hits)})")
    if processing_time is not None:
        console.print(f"Search completed in {processing_time}ms")

def generate_html_preview(results, args):
    """Generate an HTML preview of the search results"""
    hits = results.get('hits', [])
    total_hits = results.get('estimatedTotalHits', results.get('nbHits', len(hits)))
    processing_time = results.get('processingTimeMs', None)
    
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
            max-width: 900px;
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
        title = hit.get('title', 'Untitled')
        doc_type = hit.get('fileType', '').upper()
        
        # Path
        path = hit.get('path', 'Unknown location')
        
        # Metadata
        metadata_items = []
        if 'fileType' in hit:
            metadata_items.append(f'<div class="meta-item"><span>Type:</span> {doc_type}</div>')
        if 'fileSize' in hit:
            metadata_items.append(f'<div class="meta-item"><span>Size:</span> {hit["fileSize"]/1024:.1f} KB</div>')
        if 'modifiedAt' in hit:
            from datetime import datetime
            modified = datetime.fromtimestamp(hit['modifiedAt'])
            metadata_items.append(f'<div class="meta-item"><span>Modified:</span> {modified.strftime("%Y-%m-%d")}</div>')
        
        # Context with highlights
        contexts_html = '<div class="context-container">'
        formatted = None
        if '_formatted' in hit:
            formatted = hit['_formatted']
        elif hasattr(hit, '_formatted'):
            formatted = hit._formatted
            
        if formatted and 'content' in formatted:
            formatted_content = formatted['content']
            context_snippets = extract_context(
                formatted_content, 
                '<<<HIGHLIGHT>>>', 
                '<<<END_HIGHLIGHT>>>', 
                args.context_words
            )
            
            for j, snippet in enumerate(context_snippets[:args.content_lines]):
                # Replace highlight tags with HTML
                snippet_html = snippet.replace('<<<HIGHLIGHT>>>', '<span class="highlight">').replace('<<<END_HIGHLIGHT>>>', '</span>')
                contexts_html += f'<div class="context"><span class="context-num">{j+1}.</span> {snippet_html}</div>'
        elif 'content' in hit and hit['content']:
            # Try manual highlighting
            manual_contexts = manually_highlight(hit['content'], args.query, args.context_words)
            if isinstance(manual_contexts, list):
                for j, snippet in enumerate(manual_contexts[:args.content_lines]):
                    # Replace highlight tags with HTML
                    snippet_html = snippet.replace('<<<HIGHLIGHT>>>', '<span class="highlight">').replace('<<<END_HIGHLIGHT>>>', '</span>')
                    contexts_html += f'<div class="context"><span class="context-num">{j+1}.</span> {snippet_html}</div>'
            else:
                content_preview = hit["content"][:args.content_size]
                if len(hit["content"]) > args.content_size:
                    content_preview += "..."
                contexts_html += f'<div class="context">{content_preview}</div>'
        elif 'content' in hit:
            content_preview = hit["content"][:args.content_size]
            if len(hit["content"]) > args.content_size:
                content_preview += "..."
            contexts_html += f'<div class="context">{content_preview}</div>'
            
        contexts_html += '</div>'
        
        # File URI for opening locally
        file_uri = ''
        if 'path' in hit and os.path.exists(hit['path']):
            file_uri = f"file://{os.path.abspath(hit['path'])}"
        
        html += f"""
    <div class="result">
        <div class="title">
            <span class="result-num">{i+1}</span>
            {title} 
            {f'<span class="file-type">{doc_type}</span>' if doc_type else ''}
        </div>
        <div class="meta">
            {''.join(metadata_items)}
        </div>
        <div class="path">{path}</div>
        {contexts_html}
        {f'<a href="{file_uri}" class="open-link">Open File</a>' if file_uri else ''}
    </div>
"""

    html += f"""
    </div>
    
    <div class="stats">
        Found {total_hits} results (showing {len(hits)})
        {f"<br>Search completed in {processing_time}ms" if processing_time is not None else ''}
    </div>
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
            print("DEBUG: Search parameters:")
            print(json.dumps(search_params, indent=2))
            print("\nDEBUG: Raw search results:")
            print(json.dumps(results, indent=2, default=lambda o: o.__dict__ if hasattr(o, '__dict__') else str(o)))
            print("\n" + "-"*50 + "\n")
        
        # Output results in the requested format
        if args.output == 'json':
            output_json(results, args)
        elif args.output == 'table':
            output_table(results, args)
        elif args.output == 'rich':
            output_rich(results, args)
        elif args.output == 'text':
            output_text(results, args)
        
        # Generate HTML preview if requested
        if args.preview:
            temp_file = generate_html_preview(results, args)
            print(f"HTML preview saved to: {temp_file}")
        
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()