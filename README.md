PDF → JSON Outline Converter
----------------------------

This script scans every PDF placed in the `input` folder, discovers the document title and up-to-three heading levels, and writes a matching `<filename>.json` file to the `output` folder.
All decisions (font size cut-offs, whitespace limits, score thresholds, etc.) are calculated from the statistics of each individual PDF, so nothing is hand-tuned for a particular template.

Approach (summary)
------------------

1. Parse the PDF with PyMuPDF and turn raw glyphs into one logical line per baseline.
2. Collect typography and layout features for every line (font size, bold flag, centre position, surrounding whitespace, …).
3. Compute document-relative z-scores for those features so that rules work on *relative* values, not fixed numbers.
4. Identify the most prominent centred and isolated line on the first page as the title.
5. Score the remaining lines – centred alignment and vertical whitespace dominate, font size is secondary, minor bonuses for bold or numbering.
6. Keep only lines whose score is well above the document mean, group them by visual style, and label the largest three groups as H1, H2, H3.
7. Dump the result to JSON.

Dependencies
------------

```
PyMuPDF>=1.22
numpy>=1.21
scipy>=1.7
```

( `pandas` is imported for quick interactive inspection but is **not** required to run the batch script. )

Setup
-----

```
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Usage
-----

1. Create the folders and add PDFs:
```
mkdir -p input output
cp /path/to/*.pdf input/
```

2. Run the extractor:
```
python pdf_outline_extractor.py
```

3. For every `sample.pdf` in `input`, an `sample.json` outline appears in `output`.

Output format
-------------

```
{
  "title": "Document Title",
  "outline": [
    {"level": "H1", "text": "Introduction",         "page": 1},
    {"level": "H2", "text": "Background",           "page": 2},
    {"level": "H3", "text": "1.1 Related Work",     "page": 3}
  ]
}
```
