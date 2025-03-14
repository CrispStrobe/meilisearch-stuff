#!/usr/bin/env python3

import argparse
import json
import sys
import os
import humanize
import meilisearch
import requests
from rich.console import Console
from rich.table import Table
from rich import box
from rich.panel import Panel
from rich.progress import Progress

def parse_arguments():
    parser = argparse.ArgumentParser(description='Check Meilisearch index stats and size')
    parser.add_argument('-u', '--url', default='http://localhost:7700', help='Meilisearch URL')
    parser.add_argument('-i', '--index', help='Specific index name to check (optional)')
    parser.add_argument('-k', '--key', help='Meilisearch API key (if required)')
    parser.add_argument('-r', '--raw', action='store_true', help='Output raw JSON')
    parser.add_argument('-s', '--sample', type=int, default=0, 
                        help='Show sample documents (specify number of documents to sample)')
    parser.add_argument('--show-attributes', action='store_true', 
                        help='Show detailed attribute distribution')
    return parser.parse_args()

def create_meilisearch_client(url, api_key=None):
    client_args = {'url': url}
    if api_key:
        client_args['api_key'] = api_key
    return meilisearch.Client(**client_args)

def get_auth_headers(api_key=None):
    """Get authorization headers if API key is provided"""
    if api_key:
        return {'Authorization': f'Bearer {api_key}'}
    return {}

def get_version(url, api_key=None):
    """Get Meilisearch version"""
    try:
        headers = get_auth_headers(api_key)
        response = requests.get(f"{url}/version", headers=headers)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_stats(url, api_key=None):
    """Get Meilisearch stats"""
    try:
        headers = get_auth_headers(api_key)
        response = requests.get(f"{url}/stats", headers=headers)
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def get_indexes(url, api_key=None):
    """Get list of indexes"""
    try:
        headers = get_auth_headers(api_key)
        response = requests.get(f"{url}/indexes", headers=headers)
        data = response.json()
        # Return just the results array containing index objects
        return data.get('results', [])
    except Exception as e:
        return [{"error": str(e)}]

def get_index_settings(client, index_name):
    """Get index settings"""
    try:
        index = client.index(index_name)
        return index.get_settings()
    except Exception as e:
        return {"error": str(e)}

def get_sample_documents(client, index_name, limit=3):
    """Get sample documents from an index"""
    try:
        index = client.index(index_name)
        result = index.search('', {'limit': limit})
        
        # Handle different response formats depending on client version
        if hasattr(result, 'hits'):
            return result.hits
        elif isinstance(result, dict) and 'hits' in result:
            return result['hits']
        return []
    except Exception as e:
        return [{"error": str(e)}]

def display_stats(stats, indexes, settings, samples, args):
    """Display statistics in a readable format"""
    # If raw JSON output is requested
    if args.raw:
        output = {
            "meiliStats": stats,
            "indexes": indexes,
            "settings": settings
        }
        if args.sample > 0:
            output["samples"] = samples
        print(json.dumps(output, indent=2, default=str))
        return
    
    console = Console()
    
    # Check for errors in stats
    if "error" in stats:
        console.print(f"[bold red]Error getting stats:[/] {stats['error']}")
        return
    
    # Create a table for indexes
    table = Table(title="Meilisearch Index Statistics", box=box.ROUNDED)
    table.add_column("Index", style="cyan")
    table.add_column("Documents", style="green", justify="right")
    table.add_column("Size", style="blue", justify="right")
    table.add_column("Avg Doc Size", style="magenta", justify="right")
    table.add_column("Fields", style="yellow", justify="right")
    table.add_column("Status", style="cyan", justify="center")
    
    # Track totals
    total_docs = 0
    total_size = 0
    
    # Get index information
    index_stats = stats.get('indexes', {})
    
    # Filter to specific index if requested
    if args.index:
        if args.index in index_stats:
            index_stats = {args.index: index_stats[args.index]}
        else:
            console.print(f"[bold red]Index '{args.index}' not found![/]")
            return
    
    # Add rows for each index
    for idx_name, idx_stats in index_stats.items():
        # Skip pagination metadata if it somehow got included
        if idx_name in ['offset', 'limit', 'total']:
            continue
            
        # Get document count
        doc_count = idx_stats.get('numberOfDocuments', 0)
        total_docs += doc_count
        
        # Get size information
        raw_size = idx_stats.get('rawDocumentDbSize', 0)
        total_size += raw_size
        
        # Get average document size
        avg_size = idx_stats.get('avgDocumentSize', 0)
        
        # Get field count
        field_dist = idx_stats.get('fieldDistribution', {})
        field_count = len(field_dist)
        
        # Get indexing status
        is_indexing = idx_stats.get('isIndexing', False)
        status = "[yellow]Indexing[/]" if is_indexing else "[green]Ready[/]"
        
        # Add the row
        table.add_row(
            idx_name,
            f"{doc_count:,}",
            humanize.naturalsize(raw_size),
            humanize.naturalsize(avg_size),
            str(field_count),
            status
        )
    
    # Add total row if multiple indexes
    if len(index_stats) > 1:
        table.add_row(
            "[bold]Total[/]",
            f"[bold]{total_docs:,}[/]",
            f"[bold]{humanize.naturalsize(total_size)}[/]",
            "",
            "",
            ""
        )
    
    console.print(table)
    
    # Display database size information
    db_size = stats.get('databaseSize', 0)
    used_size = stats.get('usedDatabaseSize', 0)
    last_update = stats.get('lastUpdate', 'Unknown')
    
    db_table = Table(title="Database Information", box=box.ROUNDED)
    db_table.add_column("Metric", style="cyan")
    db_table.add_column("Value", style="green")
    
    db_table.add_row("Total Database Size", humanize.naturalsize(db_size))
    db_table.add_row("Used Database Size", humanize.naturalsize(used_size))
    
    if db_size > 0:
        usage_percent = (used_size / db_size) * 100
        db_table.add_row("Space Utilization", f"{usage_percent:.1f}%")
        
    db_table.add_row("Last Update", last_update)
    
    console.print(db_table)
    
    # Show field distribution if requested
    if args.show_attributes:
        for idx_name, idx_stats in index_stats.items():
            field_dist = idx_stats.get('fieldDistribution', {})
            if field_dist:
                field_table = Table(title=f"Field Distribution for '{idx_name}'", box=box.SIMPLE)
                field_table.add_column("Field", style="cyan")
                field_table.add_column("Count", style="green", justify="right")
                field_table.add_column("% Coverage", style="blue", justify="right")
                
                doc_count = idx_stats.get('numberOfDocuments', 0)
                
                for field, count in field_dist.items():
                    coverage = (count / doc_count * 100) if doc_count > 0 else 0
                    field_table.add_row(
                        field,
                        f"{count:,}",
                        f"{coverage:.1f}%"
                    )
                
                console.print(field_table)
    
    # Display version information if available
    version_info = get_version(args.url, args.key)
    if version_info and not "error" in version_info:
        console.print(f"\nMeilisearch Version: [bold]{version_info.get('pkgVersion', 'Unknown')}[/]")
    
    # Show index settings summary
    for idx_name, idx_settings in settings.items():
        if isinstance(idx_settings, dict) and not "error" in idx_settings:
            console.print(f"\n[bold cyan]Index '{idx_name}' Settings Summary:[/]")
            
            # Create a table for important settings
            settings_table = Table(box=box.SIMPLE, show_header=False)
            settings_table.add_column("Setting", style="yellow")
            settings_table.add_column("Value", style="green")
            
            # Add searchable attributes
            searchable = idx_settings.get('searchableAttributes', ['*'])
            settings_table.add_row(
                "Searchable Attributes",
                ", ".join(searchable) if len(searchable) < 10 else 
                f"{', '.join(searchable[:10])}... ({len(searchable) - 10} more)"
            )
            
            # Add filterable attributes
            filterable = idx_settings.get('filterableAttributes', [])
            settings_table.add_row(
                "Filterable Attributes",
                ", ".join(filterable) if len(filterable) < 10 else 
                f"{', '.join(filterable[:10])}... ({len(filterable) - 10} more)"
            )
            
            # Add sortable attributes
            sortable = idx_settings.get('sortableAttributes', [])
            settings_table.add_row(
                "Sortable Attributes",
                ", ".join(sortable) if len(sortable) < 10 else 
                f"{', '.join(sortable[:10])}... ({len(sortable) - 10} more)"
            )
            
            # Add displayed attributes
            displayed = idx_settings.get('displayedAttributes', ['*'])
            settings_table.add_row(
                "Displayed Attributes",
                ", ".join(displayed) if len(displayed) < 10 else 
                f"{', '.join(displayed[:10])}... ({len(displayed) - 10} more)"
            )
            
            console.print(settings_table)
    
    # Show sample documents if requested
    if args.sample > 0:
        for idx_name, sample_docs in samples.items():
            if sample_docs:
                console.print(f"\n[bold cyan]Sample Documents from '{idx_name}':[/]")
                
                for i, doc in enumerate(sample_docs[:args.sample]):
                    # Get key information to display
                    doc_id = doc.get('id', f"Doc #{i+1}")
                    title = doc.get('title', doc.get('name', 'Untitled'))
                    path = doc.get('path', '')
                    
                    # Create a summary panel
                    summary = f"ID: {doc_id}\nTitle: {title}"
                    if path:
                        summary += f"\nPath: {path}"
                    
                    # Add content preview if available
                    content = doc.get('content', '')
                    if content:
                        preview = content[:200] + "..." if len(content) > 200 else content
                        summary += f"\n\nContent Preview:\n{preview}"
                    
                    # Add other fields summary
                    other_fields = [k for k in doc.keys() if k not in ['id', 'title', 'name', 'path', 'content']]
                    if other_fields:
                        summary += f"\n\nOther Fields: {', '.join(other_fields)}"
                    
                    console.print(Panel(summary, title=f"Document {i+1}", border_style="blue"))
    
    # Display size optimization tips if database is large
    if db_size > 1_000_000_000:  # > 1GB
        console.print(Panel(
            "[yellow]Your Meilisearch database is relatively large. Consider these optimization strategies:[/]\n\n"
            "1. [bold]Limit document content[/]: Only index the most important parts of documents\n"
            "2. [bold]Use selective indexing[/]: Configure searchableAttributes to include only necessary fields\n"
            "3. [bold]Split into multiple indexes[/]: Create separate indexes for different document types\n"
            "4. [bold]Regular cleanup[/]: Remove outdated or unnecessary documents periodically",
            title="Optimization Recommendations",
            border_style="yellow"
        ))

def main():
    args = parse_arguments()
    
    try:
        console = Console()
        
        with Progress() as progress:
            task = progress.add_task("[cyan]Fetching Meilisearch information...", total=4)
            
            # Create client
            client = create_meilisearch_client(args.url, args.key)
            progress.update(task, advance=1)
            
            # Get stats
            stats = get_stats(args.url, args.key)
            progress.update(task, advance=1)
            
            # Get indexes
            indexes = get_indexes(args.url, args.key)
            progress.update(task, advance=1)
            
            # Get settings for each index
            settings = {}
            samples = {}
            
            # Filter to specific index if requested
            index_list = []
            if args.index:
                index_list = [idx for idx in indexes if idx.get('uid') == args.index]
            else:
                index_list = indexes
            
            for idx in index_list:
                idx_name = idx.get('uid')
                if idx_name:
                    settings[idx_name] = get_index_settings(client, idx_name)
                    if args.sample > 0:
                        samples[idx_name] = get_sample_documents(client, idx_name, args.sample)
            
            progress.update(task, advance=1)
        
        # Display stats
        display_stats(stats, indexes, settings, samples, args)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
