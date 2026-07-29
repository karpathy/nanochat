---
name: read-arxiv-paper
description: Use this skill when asked to read an arxiv paper given an arxiv URL
---

You will be given a URL of an arxiv paper, for example:

https://www.arxiv.org/abs/2601.07372

### Part 1: Normalize the URL

The goal is to fetch the TeX Source of the paper (not the PDF!), which lives at:

https://www.arxiv.org/src/2601.07372

Notice the /src/ in the url. Don't assume the URL you were handed is an `/abs/` one — pull the bare
id out of it and rebuild. The forms that show up in practice are `/abs/{id}`, `/pdf/{id}`,
`/html/{id}`, any of those with a version suffix (`2601.07372v2`), with or without a trailing
slash, and old-style ids (`math/0211159`). So: strip the leading `arxiv.org/<section>/`, strip any
trailing slash or query string, keep the version suffix if one was given (it pins what you read),
and use `{arxiv_id}` for both the URL and the local paths below. Once you have the URL:

### Part 2: Download the paper source

Fetch the url to a local file. A good location is `~/.cache/nanochat/knowledge/{arxiv_id}.tar.gz`.

(If the file already exists, there is no need to re-download it).

If this 404s, there is no TeX source for that paper (some submissions are PDF-only). Don't quietly read
the PDF instead and write a summary that looks like all the others: say so, and if you do fall back
to the PDF or the abstract, put that in the summary header from Part 6 so the next reader knows the
source was not the TeX.

### Part 3: Unpack the file in that folder

Unpack the contents into `~/.cache/nanochat/knowledge/{arxiv_id}` directory.

Note the download is not always a tarball: single-file submissions come back as one gzipped `.tex`,
so if `tar` rejects it, `gunzip` it and treat the result as the entrypoint.

### Part 4: Locate the entrypoint

Every latex source usually has an entrypoint, such as `main.tex` or something like that.

### Part 5: Read the paper

Once you've found the entrypoint, Read the contents and then recurse through all other relevant source files to read the paper.

### Part 6: Report

Once you've read the paper, produce a summary of the paper into a markdown file at `./knowledge/summary_{tag}.md`. Notice that 1) use the local knowledge directory here (it's easier for me to open and reference here), not in `~/.cache`, and 2) generate some reasonable `tag` like e.g. `conditional_memory` or whatever seems appropriate given the paper. Check that `./knowledge/summary_{tag}.md` does not exist before writing; if it does, pick another tag rather than overwriting.

Start the file with a few lines saying where it came from, so a summary found six months later can be
traced back and re-read against the source:

```
arxiv: 2601.07372v2
title: <title as it appears in the source>
source: tex   # or: pdf / abstract-only, if Part 2 fell back
read: <date>
```

As for the summary itself, remember that you're processing this paper within the context of the nanochat repository, so most often we will be interested in how to apply the paper and its lessons to the nanochat project. Therefore, you should feel free to "remind yourself" of the related nanochat code by reading the relevant parts, and then explicitly make the connection of how this paper might relate to nanochat or what are things we might be inspired about or try.
