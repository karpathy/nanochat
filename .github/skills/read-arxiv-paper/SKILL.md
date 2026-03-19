---
name: read-arxiv-paper
description: Use this skill when asked to read, summarize, or extract implementation ideas from an arXiv paper given an arXiv URL or ID.
---

This skill is agent-neutral and is intended to work for GitHub Copilot, Claude, or any coding agent that can read files and run shell commands.

You may be given an arXiv ID or an arXiv URL, for example:

- `2601.07372`
- `https://arxiv.org/abs/2601.07372`
- `https://arxiv.org/pdf/2601.07372`
- `https://arxiv.org/src/2601.07372`

Follow this workflow.

## 1. Normalize the identifier

- Extract the canonical arXiv ID from the input.
- Accept `abs`, `pdf`, or `src` URLs.
- Preserve a version suffix such as `v2` if one is present.
- Convert the input to the source URL form: `https://arxiv.org/src/{arxiv_id}`.

The goal is to fetch the paper source, not the PDF.

## 2. Download the paper source

- Download the source archive to `~/.cache/nanochatter/knowledge/arxiv/{arxiv_id}.tar.gz`.
- Reuse an existing archive if it is already present.
- Do not download the PDF unless the user explicitly asks for it.

## 3. Unpack the archive

- Extract the contents into `~/.cache/nanochatter/knowledge/arxiv/{arxiv_id}/`.
- If the extraction directory already exists and looks complete, reuse it.

## 4. Locate the LaTeX entrypoint

- Look for likely root files such as `main.tex`, `paper.tex`, `ms.tex`, or `root.tex`.
- If there is no obvious root file, find the file that contains `\documentclass` and `\begin{document}`.
- Use the `\input{}` and `\include{}` graph to confirm the real entrypoint.
- Ignore generated artifacts when possible.

## 5. Read the paper source

- Start from the entrypoint and recursively read the relevant source files.
- Follow `\input{}`, `\include{}`, and bibliography references when they matter for understanding the paper.
- Ignore binary assets unless they are needed to understand a figure or table.
- Extract the important technical details:
  - problem statement
  - proposed method
  - experimental setup
  - results
  - limitations
  - implementation-relevant details

## 6. Connect the paper to the current repository

- Read only the relevant parts of the current repository needed to relate the paper to the codebase.
- In this repository, focus on concrete implications for `nanochatter`.
- Prefer actionable takeaways over generic commentary.
- Typical areas of interest include memory, architecture, prompting, evaluation, tooling, native acceleration, and developer workflow.

## 7. Write the report

- Create `./knowledge` if it does not already exist.
- Write the summary to `./knowledge/summary_{tag}.md`.
- Choose a short, descriptive, unique `{tag}` so you do not overwrite an existing summary.
- The summary should include:
  - paper title
  - arXiv ID
  - source URL
  - one-paragraph summary
  - key ideas
  - implementation notes
  - relevance to `nanochatter`
  - experiments or follow-up ideas to try
  - open questions or risks
  - repository files or modules consulted

## 8. Report back in chat

- Tell the user where the summary file was written.
- Provide a concise summary of the paper and the most relevant next steps for the repository.
- If the source archive cannot be fetched or unpacked, say exactly what failed and stop instead of guessing.
