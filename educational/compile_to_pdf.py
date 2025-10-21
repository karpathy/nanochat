#!/usr/bin/env python3
"""
Compile all educational markdown files into a single LaTeX PDF.

This script:
1. Combines all markdown files in order
2. Adds LaTeX preamble with proper formatting
3. Converts to PDF using pandoc

Requirements:
- pandoc
- LaTeX distribution (e.g., TeX Live, BasicTeX)

Usage:
    python compile_to_pdf.py
"""

import os
import subprocess
import sys
from pathlib import Path


# Configuration
MD_FILES = [
    "01_introduction.md",
    "02_mathematical_foundations.md",
    "03_tokenization.md",
    "04_transformer_architecture.md",
    "05_attention_mechanism.md",
    "06_training_process.md",
    "07_optimization.md",
    "08_putting_it_together.md",
]

OUTPUT_PDF = "nanochat_educational_guide.pdf"
COMBINED_MD = "combined.md"

# LaTeX preamble for better formatting
LATEX_PREAMBLE = r"""
---
title: "nanochat: Building a ChatGPT from Scratch"
subtitle: "A Comprehensive Educational Guide"
author: |
  | Based on nanochat by Andrej Karpathy
  |
  | Vibe Written by Matt Suiche (msuiche) with Claude Code
date: "October 21, 2025"
documentclass: book
geometry: margin=1in
fontsize: 11pt
linestretch: 1.2
toc: true
toc-depth: 3
numbersections: true
colorlinks: true
linkcolor: blue
urlcolor: blue
citecolor: blue
header-includes:
  - \usepackage{fancyhdr}
  - \pagestyle{fancy}
  - \fancyhead[L]{nanochat Educational Guide}
  - \fancyhead[R]{\thepage}
  - \usepackage{listings}
  - \usepackage{xcolor}
  - \usepackage{pmboxdraw}
  - \usepackage{newunicodechar}
  - \lstset{
      basicstyle=\ttfamily\small,
      breaklines=true,
      frame=single,
      backgroundcolor=\color{gray!10},
      literate={├}{|--}1 {└}{`--}1 {─}{-}1 {│}{|}1
    }
---

\newpage

"""


def check_dependencies():
    """Check if required tools are installed."""
    print("Checking dependencies...")

    # Check for pandoc
    try:
        result = subprocess.run(
            ["pandoc", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print(f"[OK] pandoc found: {result.stdout.split()[1]}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[FAIL] pandoc not found. Please install pandoc:")
        print("  macOS: brew install pandoc")
        print("  Ubuntu: sudo apt-get install pandoc")
        return False

    # Check for LaTeX
    try:
        result = subprocess.run(
            ["pdflatex", "--version"],
            capture_output=True,
            text=True,
            check=True
        )
        print("[OK] LaTeX found")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("[FAIL] LaTeX not found. Please install a LaTeX distribution:")
        print("  macOS: brew install basictex")
        print("  Ubuntu: sudo apt-get install texlive-full")
        return False

    return True


def combine_markdown_files():
    """Combine all markdown files into a single file."""
    import re

    print(f"\nCombining {len(MD_FILES)} markdown files...")

    with open(COMBINED_MD, "w", encoding="utf-8") as outfile:
        # Write preamble
        outfile.write(LATEX_PREAMBLE)

        # Combine all markdown files
        for i, md_file in enumerate(MD_FILES):
            print(f"  Adding {md_file}...")

            if not os.path.exists(md_file):
                print(f"  [WARNING] {md_file} not found, skipping...")
                continue

            with open(md_file, "r", encoding="utf-8") as infile:
                content = infile.read()

                # Remove problematic Unicode characters (emojis, special symbols)
                # Keep only ASCII and common Unicode characters
                content = re.sub(r'[^\x00-\x7F\u00A0-\u024F\u1E00-\u1EFF\u2000-\u206F\u2070-\u209F\u20A0-\u20CF\u2100-\u214F\u2190-\u21FF\u2200-\u22FF\u2460-\u24FF\u2500-\u257F]', '', content)

                # Add page break between sections (except first)
                if i > 0:
                    outfile.write("\n\\newpage\n\n")

                outfile.write(content)
                outfile.write("\n\n")

    print(f"[OK] Combined markdown saved to {COMBINED_MD}")


def convert_to_pdf():
    """Convert combined markdown to PDF using pandoc."""
    print(f"\nConverting to PDF...")

    # Pandoc command - use xelatex for better Unicode support
    cmd = [
        "pandoc",
        COMBINED_MD,
        "-o", OUTPUT_PDF,
        "--pdf-engine=xelatex",
        "--highlight-style=tango",
        "--standalone",
        "-V", "linkcolor:blue",
        "-V", "urlcolor:blue",
        "-V", "toccolor:blue",
    ]

    try:
        # Run pandoc
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )

        print(f"[OK] PDF created successfully: {OUTPUT_PDF}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"[FAIL] Error converting to PDF:")
        print(e.stderr)
        return False


def get_pdf_size():
    """Get size of the generated PDF."""
    if os.path.exists(OUTPUT_PDF):
        size_bytes = os.path.getsize(OUTPUT_PDF)
        size_mb = size_bytes / (1024 * 1024)
        return f"{size_mb:.2f} MB"
    return "N/A"


def cleanup():
    """Clean up temporary files."""
    print("\nCleaning up temporary files...")

    # Remove combined markdown
    if os.path.exists(COMBINED_MD):
        os.remove(COMBINED_MD)
        print(f"  Removed {COMBINED_MD}")

    # Remove LaTeX auxiliary files
    aux_extensions = [".aux", ".log", ".out", ".toc"]
    for ext in aux_extensions:
        aux_file = OUTPUT_PDF.replace(".pdf", ext)
        if os.path.exists(aux_file):
            os.remove(aux_file)
            print(f"  Removed {aux_file}")


def main():
    """Main compilation pipeline."""
    print("=" * 60)
    print("nanochat Educational Guide - PDF Compilation")
    print("=" * 60)

    # Check dependencies
    if not check_dependencies():
        print("\n[FAIL] Missing dependencies. Please install required tools.")
        sys.exit(1)

    # Combine markdown files
    try:
        combine_markdown_files()
    except Exception as e:
        print(f"\n[FAIL] Error combining markdown files: {e}")
        sys.exit(1)

    # Convert to PDF
    success = convert_to_pdf()

    # Cleanup
    cleanup()

    # Summary
    print("\n" + "=" * 60)
    if success:
        print("[SUCCESS] Compilation successful!")
        print(f"   Output: {OUTPUT_PDF}")
        print(f"   Size: {get_pdf_size()}")
        print(f"   Pages: ~{len(MD_FILES) * 5}-{len(MD_FILES) * 10} (estimated)")
        print("\n   You can now read the complete guide in PDF format!")
    else:
        print("[FAIL] Compilation failed. See errors above.")
        sys.exit(1)

    print("=" * 60)


if __name__ == "__main__":
    main()
