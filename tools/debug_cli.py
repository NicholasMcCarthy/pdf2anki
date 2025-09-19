# tools/debug_cli.py
"""
Debug/inspection CLI for pdf2anki development.

Features:
- Inspect PDF metadata & text extraction
- Try different chunking strategies (page, chars, overlap)
- Render and inspect prompt content BEFORE any LLM call
- Benchmark chunking and prompt sizes (chars/tokens) & timings
- Save artifacts (chunks, prompts) to a workspace folder

Usage examples:
  python tools/debug_cli.py inspect --pdf examples/sample.pdf
  python tools/debug_cli.py chunk --pdf examples/sample.pdf --method page --max-chars 800
  python tools/debug_cli.py render-prompts --pdf examples/sample.pdf --prompts-dir prompts --template qa_card.j2
  python tools/debug_cli.py bench --pdf examples/sample.pdf --methods page overlap chars --max-chars 900
"""

import json
import time
import re
from pathlib import Path
from typing import List, Optional, Dict, Any

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Optional deps: jinja2 for template rendering, tiktoken for token counts, fitz (PyMuPDF) for robust PDF text
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from jinja2 import Environment, FileSystemLoader, StrictUndefined
except Exception:
    Environment = None  # graceful fallback

try:
    import tiktoken
except Exception:
    tiktoken = None

app = typer.Typer(add_completion=False)
console = Console()


# ------------------------------
# Helpers
# ------------------------------

def _read_pdf_text(pdf_path: Path) -> Dict[str, Any]:
    """Read PDF text per page. Uses PyMuPDF if available; otherwise naive fallback."""
    start = time.perf_counter()
    pages: List[str] = []

    meta: Dict[str, Any] = {"title": "", "author": "", "pages": 0}
    if fitz is not None:
        with fitz.open(pdf_path.as_posix()) as doc:
            meta["pages"] = doc.page_count
            md = doc.metadata or {}
            meta["title"] = md.get("title") or pdf_path.stem
            meta["author"] = md.get("author") or ""
            for i in range(doc.page_count):
                text = doc.load_page(i).get_text("text")  # page text
                pages.append(text or "")
    else:
        # Minimal fallback: read bytes and hope for a text-like PDF (not reliable, but avoids hard dependency)
        # Recommend installing PyMuPDF for good results: `pip install pymupdf`
        console.print("[yellow]PyMuPDF not installed; using fallback extraction (may be poor).[/yellow]")
        try:
            data = pdf_path.read_bytes()
            pages = [data.decode(errors="ignore")]
            meta["pages"] = 1
            meta["title"] = pdf_path.stem
        except Exception as e:
            raise RuntimeError(f"Cannot read PDF: {e}")

    elapsed = time.perf_counter() - start
    return {"pages": pages, "meta": meta, "elapsed_s": elapsed}


def _token_len(text: str, model: str = "cl100k_base") -> int:
    """Token length using tiktoken if available; fall back to simple word split."""
    if tiktoken is None:
        # Fallback heuristic
        return max(1, len(re.findall(r"\w+|\S", text)))
    enc = tiktoken.get_encoding(model)
    return len(enc.encode(text))


def _mk_workspace(out_dir: Optional[Path]) -> Path:
    root = Path(out_dir or "workspace/debug")
    root.mkdir(parents=True, exist_ok=True)
    return root


# ------------------------------
# Chunking strategies
# ------------------------------

def chunk_by_page(pages: List[str]) -> List[Dict[str, Any]]:
    chunks = []
    for i, p in enumerate(pages):
        chunks.append({"index": i, "strategy": "page", "page": i + 1, "text": p.strip()})
    return chunks


def chunk_by_chars(pages: List[str], max_chars: int = 1200, overlap: int = 100) -> List[Dict[str, Any]]:
    """
    Greedy fixed-size char chunks across the whole document with overlap.
    - Good for controlling prompt size deterministically.
    """
    all_text = "\n\n".join(pages)
    chunks = []
    start = 0
    idx = 0
    n = len(all_text)
    while start < n:
        end = min(start + max_chars, n)
        text = all_text[start:end]
        chunks.append({"index": idx, "strategy": "chars", "start": start, "end": end, "text": text.strip()})
        idx += 1
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def chunk_by_overlap_paragraphs(pages: List[str], max_chars: int = 1200, overlap_paras: int = 1) -> List[Dict[str, Any]]:
    """
    Paragraph-based chunks with paragraph overlap.
    - More semantically coherent than raw chars.
    """
    paras = []
    for p in pages:
        # Split on blank lines; trim noisy whitespace
        parts = [x.strip() for x in re.split(r"\n\s*\n", p) if x.strip()]
        paras.extend(parts)

    chunks = []
    idx = 0
    i = 0
    while i < len(paras):
        buf = []
        buf_len = 0
        j = i
        while j < len(paras) and (buf_len + len(paras[j]) + 2) <= max_chars:
            buf.append(paras[j])
            buf_len += len(paras[j]) + 2
            j += 1
        if not buf:
            # Fallback: force at least one paragraph
            buf = [paras[i]]
            j = i + 1
        text = "\n\n".join(buf)
        chunks.append({"index": idx, "strategy": "overlap", "start_para": i, "end_para": j, "text": text})
        # Move window forward with overlap
        i = max(i + 1, j - overlap_paras)
        idx += 1
    return chunks


# ------------------------------
# Prompt rendering
# ------------------------------

def render_prompts_for_chunks(chunks: List[Dict[str, Any]],
                              prompts_dir: Path,
                              template_name: str,
                              pdf_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Render a Jinja2 template per chunk. Variables available: chunk, chunk_text, pdf_meta, index."""
    if Environment is None:
        raise RuntimeError("jinja2 is not installed. Install with `pip install jinja2`.")
    env = Environment(
        loader=FileSystemLoader(str(prompts_dir)),
        undefined=StrictUndefined,
        autoescape=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template(template_name)
    rendered = []
    for c in chunks:
        text = c["text"]
        payload = tmpl.render(chunk=c, chunk_text=text, pdf_meta=pdf_meta, index=c["index"])
        rendered.append({"index": c["index"], "strategy": c["strategy"], "prompt": payload})
    return rendered


# ------------------------------
# Commands
# ------------------------------

@app.command()
def inspect(pdf: Path = typer.Option(..., "--pdf", help="Path to sample PDF"),
            out_dir: Optional[Path] = typer.Option(None, "--out", help="Workspace output dir"),
            show_first_pages: int = typer.Option(2, help="Show first N pages' excerpts"),
            excerpt_chars: int = typer.Option(400, help="Chars per page excerpt")):
    """Open a PDF and show metadata + quick text excerpts."""
    info = _read_pdf_text(pdf)
    meta = info["meta"]
    pages = info["pages"]

    t = Table(title="PDF Metadata", show_header=True, header_style="bold")
    t.add_column("Key")
    t.add_column("Value")
    t.add_row("Title", meta.get("title", ""))
    t.add_row("Author", meta.get("author", ""))
    t.add_row("Pages", str(meta.get("pages", 0)))
    t.add_row("Load time (s)", f"{info['elapsed_s']:.3f}")
    console.print(t)

    for i in range(min(show_first_pages, len(pages))):
        snippet = (pages[i] or "")[:excerpt_chars].replace("\n", " ")
        console.print(Panel.fit(snippet, title=f"Page {i+1} excerpt", style="cyan"))

    root = _mk_workspace(out_dir)
    (root / "raw_pages.json").write_text(json.dumps(pages, ensure_ascii=False, indent=2))
    console.print(f"[green]Saved raw pages to[/green] {root/'raw_pages.json'}")


@app.command()
def chunk(pdf: Path = typer.Option(..., "--pdf"),
          method: str = typer.Option("page", help="page|chars|overlap"),
          max_chars: int = typer.Option(1200, help="Max chars per chunk (chars/overlap)"),
          overlap: int = typer.Option(100, help="Char overlap (chars) or paragraph overlap (overlap)"),
          out_dir: Optional[Path] = typer.Option(None, "--out")):
    """Chunk a PDF and preview the first few chunks with sizes."""
    info = _read_pdf_text(pdf)
    pages = info["pages"]
    meta = info["meta"]

    start = time.perf_counter()
    if method == "page":
        chunks = chunk_by_page(pages)
    elif method == "chars":
        chunks = chunk_by_chars(pages, max_chars=max_chars, overlap=overlap)
    elif method == "overlap":
        chunks = chunk_by_overlap_paragraphs(pages, max_chars=max_chars, overlap_paras=overlap)
    else:
        raise typer.BadParameter("method must be one of: page, chars, overlap")
    elapsed = time.perf_counter() - start

    # Summaries
    rows = Table(title=f"Chunk summary ({method})", show_header=True, header_style="bold")
    rows.add_column("#")
    rows.add_column("Chars")
    rows.add_column("Tokens")
    rows.add_column("Preview")
    total_chars = 0
    total_tokens = 0

    for c in chunks[:10]:  # show first 10
        text = c["text"]
        ch = len(text)
        tk = _token_len(text)
        total_chars += ch
        total_tokens += tk
        preview = text[:120].replace("\n", " ")
        rows.add_row(str(c["index"]), str(ch), str(tk), preview + ("…" if len(text) > 120 else ""))

    console.print(rows)
    console.print(Panel.fit(
        f"Chunks: {len(chunks)}   Total tokens (first 10 shown): {total_tokens}   Build time: {elapsed:.3f}s",
        title="Stats", style="green"
    ))

    root = _mk_workspace(out_dir)
    payload = {"meta": meta, "method": method, "params": {"max_chars": max_chars, "overlap": overlap}, "chunks": chunks}
    (root / f"chunks_{method}.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    console.print(f"[green]Saved chunks to[/green] {root / f'chunks_{method}.json'}")


@app.command("render-prompts")
def render_prompts(pdf: Path = typer.Option(..., "--pdf"),
                   prompts_dir: Path = typer.Option(..., "--prompts-dir"),
                   template: str = typer.Option(..., "--template", help="Template filename (e.g. qa_card.j2)"),
                   method: str = typer.Option("page", help="page|chars|overlap"),
                   max_chars: int = typer.Option(1200),
                   overlap: int = typer.Option(100),
                   out_dir: Optional[Path] = typer.Option(None, "--out")):
    """Render Jinja2 prompts for each chunk (no LLM call) and preview the exact text."""
    info = _read_pdf_text(pdf)
    pages, meta = info["pages"], info["meta"]

    if method == "page":
        chunks = chunk_by_page(pages)
    elif method == "chars":
        chunks = chunk_by_chars(pages, max_chars=max_chars, overlap=overlap)
    elif method == "overlap":
        chunks = chunk_by_overlap_paragraphs(pages, max_chars=max_chars, overlap_paras=overlap)
    else:
        raise typer.BadParameter("method must be one of: page, chars, overlap")

    prompts = render_prompts_for_chunks(chunks, prompts_dir, template, meta)

    # Show first two rendered prompts for sanity-check
    for p in prompts[:2]:
        console.rule(f"[bold]Prompt #{p['index']}[/bold]")
        console.print(p["prompt"])

    # Save artifacts
    root = _mk_workspace(out_dir)
    (root / f"prompts_{method}.jsonl").write_text(
        "\n".join(json.dumps(p, ensure_ascii=False) for p in prompts), encoding="utf-8"
    )
    console.print(f"[green]Saved prompts to[/green] {root / f'prompts_{method}.jsonl'}")

    # Small size table
    tbl = Table(title="Prompt sizes", show_header=True, header_style="bold")
    tbl.add_column("#"); tbl.add_column("Chars"); tbl.add_column("Tokens")
    for p in prompts[:10]:
        tbl.add_row(str(p["index"]), str(len(p["prompt"])), str(_token_len(p["prompt"])))
    console.print(tbl)


@app.command()
def bench(pdf: Path = typer.Option(..., "--pdf"),
          methods: List[str] = typer.Option(["page", "overlap", "chars"], "--methods"),
          max_chars: int = typer.Option(1200),
          overlap: int = typer.Option(100),
          prompts_dir: Optional[Path] = typer.Option(None, "--prompts-dir", help="If provided, also render prompts"),
          template: Optional[str] = typer.Option(None, "--template"),
          out_dir: Optional[Path] = typer.Option(None, "--out")):
    """
    Benchmark different chunkers (and optionally prompt rendering).
    Reports chunk counts, avg size, total tokens, and elapsed times.
    """
    info = _read_pdf_text(pdf)
    pages, meta = info["pages"], info["meta"]

    results = []
    for m in methods:
        t0 = time.perf_counter()
        if m == "page":
            chunks = chunk_by_page(pages)
        elif m == "chars":
            chunks = chunk_by_chars(pages, max_chars=max_chars, overlap=overlap)
        elif m == "overlap":
            chunks = chunk_by_overlap_paragraphs(pages, max_chars=max_chars, overlap_paras=overlap)
        else:
            raise typer.BadParameter("method must be one of: page, chars, overlap")
        t_chunk = time.perf_counter() - t0

        # size stats
        sizes = [len(c["text"]) for c in chunks]
        token_sizes = [_token_len(c["text"]) for c in chunks]
        row = {
            "method": m,
            "n_chunks": len(chunks),
            "avg_chars": (sum(sizes) / len(sizes)) if sizes else 0,
            "avg_tokens": (sum(token_sizes) / len(token_sizes)) if token_sizes else 0,
            "build_time_s": t_chunk,
        }

        # optionally render prompts to measure prompt size, without hitting an LLM
        if prompts_dir and template:
            t1 = time.perf_counter()
            prompts = render_prompts_for_chunks(chunks, prompts_dir, template, meta)
            t_render = time.perf_counter() - t1
            prompt_tokens = [_token_len(p["prompt"]) for p in prompts]
            row.update({
                "prompt_avg_tokens": (sum(prompt_tokens) / len(prompt_tokens)) if prompt_tokens else 0,
                "prompt_render_time_s": t_render
            })

            # Save per-method artifacts
            root = _mk_workspace(out_dir)
            (root / f"bench_{m}_chunks.json").write_text(json.dumps(chunks, ensure_ascii=False, indent=2))
            (root / f"bench_{m}_prompts.jsonl").write_text(
                "\n".join(json.dumps(p, ensure_ascii=False) for p in prompts),
                encoding="utf-8"
            )
        results.append(row)

    # Pretty table
    tbl = Table(title="Benchmark results", show_header=True, header_style="bold")
    tbl.add_column("Method")
    tbl.add_column("# Chunks")
    tbl.add_column("Avg chars")
    tbl.add_column("Avg tokens")
    tbl.add_column("Chunk time (s)")
    if prompts_dir and template:
        tbl.add_column("Prompt avg tokens")
        tbl.add_column("Prompt render (s)")

    for r in results:
        if prompts_dir and template:
            tbl.add_row(
                r["method"],
                str(r["n_chunks"]),
                f"{r['avg_chars']:.0f}",
                f"{r['avg_tokens']:.1f}",
                f"{r['build_time_s']:.3f}",
                f"{r.get('prompt_avg_tokens', 0):.1f}",
                f"{r.get('prompt_render_time_s', 0):.3f}",
            )
        else:
            tbl.add_row(
                r["method"],
                str(r["n_chunks"]),
                f"{r['avg_chars']:.0f}",
                f"{r['avg_tokens']:.1f}",
                f"{r['build_time_s']:.3f}",
            )
    console.print(tbl)

    # Persist summary
    root = _mk_workspace(out_dir)
    (root / "bench_summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
    console.print(f"[green]Saved summary to[/green] {root / 'bench_summary.json'}")


if __name__ == "__main__":
    app()
