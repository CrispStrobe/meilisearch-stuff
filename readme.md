# Meilisearch Document Indexer and Search

This repository contains mainly two Python scripts for efficient document management with [Meilisearch](https://www.meilisearch.com/):

- **indexer.py**: Index documents from your filesystem to Meilisearch
- **search.py**: Search through indexed documents with advanced filtering and display options

## Requirements

- Python 3.6+
- Meilisearch server (running locally or remote)
- Python dependencies:
  ```
  pip install meilisearch tqdm rich
  ```
- For document text extraction:
  - `pdftotext` (for PDF files)
  - `textutil` (for DOC/DOCX files on macOS)
  - `ebook-converter` from Calibre (for various document formats)

## Setup

1. Install Meilisearch by following the [official installation guide](https://docs.meilisearch.com/learn/getting_started/installation.html)
2. Start the Meilisearch server (default: http://localhost:7700)
3. Place the scripts in your desired location
4. Make them executable: `chmod +x indexer.py search.py`

## Indexer Usage

The `indexer.py` script scans your filesystem for documents and indexes them in Meilisearch.

### Basic Usage

```bash
python indexer.py ~/Documents "*.pdf" "*.epub" "*.txt" -r
```

This will scan the ~/Documents directory recursively for PDF, EPUB, and TXT files, and index them.

### Advanced Options

```bash
python indexer.py DIRECTORY "PATTERN1" "PATTERN2" [OPTIONS]
```

#### Main Arguments

- `DIRECTORY`: Directory to scan for documents
- `PATTERN`: One or more file patterns (e.g., "*.pdf", "*.epub")

#### Common Options

- `-r, --recursive`: Scan subdirectories recursively
- `-u, --url URL`: Meilisearch URL (default: http://localhost:7700)
- `-i, --index NAME`: Meilisearch index name (default: documents)
- `-k, --key KEY`: Meilisearch API key (if required)
- `-b, --batch-size SIZE`: Maximum number of documents per batch (default: 100)
- `-m, --max-batch-mb SIZE`: Maximum batch size in MB (default: 90, limit is ~95MB)
- `-f, --force`: Force reindexing of all files regardless of modification time
- `-c, --cache-file FILE`: Cache file to store indexed file information
- `--dry-run`: Scan files but do not index them (test mode)
- `-t, --threads NUM`: Number of threads for parallel processing (default: 4)
- `--content-size-mb SIZE`: Maximum content size per document in MB (default: 10)

### Examples

```bash
# Index PDFs and EPUBs from the Books directory with an API key
python indexer.py ~/Books "*.pdf" "*.epub" -r -k your-api-key

# Test what would be indexed without actually indexing
python indexer.py ~/Documents "*.txt" --dry-run -r

# Force reindexing of all documents
python indexer.py ~/Papers "*.pdf" -r -f

# Use 8 threads for faster processing of many documents
python indexer.py ~/Archive "*.pdf" "*.epub" "*.txt" -r -t 8

# Set a maximum batch size to avoid timeouts
python indexer.py ~/BigArchive "*.pdf" -r -m 50
```

## Search Usage

The `search.py` script allows you to search through your indexed documents with rich output formatting.

### Basic Usage

```bash
python search.py "neural networks"
```

This will search for "neural networks" in all indexed documents and display results in a rich console format.

### Advanced Options

```bash
python search.py QUERY [OPTIONS]
```

#### Main Arguments

- `QUERY`: Search query string

#### Common Options

##### Connection Options
- `-u, --url URL`: Meilisearch URL (default: http://localhost:7700)
- `-i, --index NAME`: Meilisearch index name (default: documents)
- `-k, --key KEY`: Meilisearch API key (if required)

##### Search Options
- `-l, --limit NUM`: Maximum number of results to return (default: 10)
- `-f, --filter FILTER`: Filter query (e.g., "fileType=pdf")
- `-s, --sort SORT`: Sort results (e.g., "createdAt:desc")

##### Display Options
- `-o, --output FORMAT`: Output format: rich, text, json, or table (default: rich)
- `-c, --context-words NUM`: Number of words of context around each match (default: 30)
- `-n, --content-lines NUM`: Number of snippets to show per result (default: 3)
- `--content-size SIZE`: Maximum characters per content snippet (default: 500)
- `--open`: Enable interactive mode to open matching files
- `--show-path`: Show full path to files instead of relative paths
- `--highlight-style STYLE`: Style for highlights (default: "bold red on white")
- `--no-highlight`: Disable highlighting of search terms
- `--preview`: Generate HTML preview of results and open in browser

### Examples

```bash
# Basic search with filters and more context
python search.py "machine learning" --filter "fileType=pdf" --context-words 50

# Show more content and highlight matches
python search.py "blockchain" --content-lines 20 --highlight-style "bold red"

# Generate HTML preview with full paths
python search.py "artificial intelligence" --preview --show-path

# Output as JSON for programmatic use
python search.py "quantum computing" --output json

# Find recent documents about neural networks
python search.py "neural networks" --filter "fileType=pdf" --sort "modifiedAt:desc"

# Interactive mode to open matching files
python search.py "important document" --open
```

## Notes

1. **Incremental Indexing**: By default, the indexer only processes new or modified files, making subsequent runs much faster.

2. **Large Document Collections**: Use the `--threads` option for faster processing and the `--max-batch-mb` option to avoid payload size limitations.

3. **Content Size**: Very large documents are automatically truncated to prevent excessive memory usage. Adjust with `--content-size-mb`.

4. **HTML Preview**: Use `--preview` with search.py to generate an interactive HTML view of your search results.

5. **Debugging**: Add `--debug` to search.py to see raw API responses and detailed error information.

## Troubleshooting

### Common Issues

1. **Connection Refused**: Make sure your Meilisearch server is running

2. **Payload Too Large**: Try reducing batch size with `--max-batch-mb`

3. **Slow Indexing**: Increase thread count with `--threads`

4. **API Key Issues**: Check if your instance requires authentication and provide the correct key

### Logging

The indexer creates a log file `meilisearch_indexer.log` with detailed information about the indexing process.
