import json
import re

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def normalize_text(text):
    """Lowercase and strip whitespace for matching"""
    return re.sub(r'\s+', ' ', text.strip().lower())

def match_span_to_outline(span_text, outline_text):
    """
    Return True if span_text matches outline_text approximately.
    Approximate match can be:
       - Exact after normalization, OR
       - Outline text contained within span text, OR vice versa,
       - Or at least a fuzzy heuristic (e.g., first 80% of words appear)
    """
    span_norm = normalize_text(span_text)
    outline_norm = normalize_text(outline_text)
    
    if span_norm == outline_norm:
        return True
    if outline_norm in span_norm:
        return True
    if span_norm in outline_norm:
        return True
    
    # Partial word overlap heuristic:
    span_words = set(span_norm.split())
    outline_words = set(outline_norm.split())
    if not outline_words:
        return False
    common_words = span_words.intersection(outline_words)
    coverage = len(common_words) / len(outline_words)
    if coverage >= 0.8:  # 80% words match threshold
        return True
    
    return False

def assign_levels_top_down(heuristics, outline):
    """
    Assign levels to heuristic spans by going top-down through outline.
    Once an outline heading is matched and assigned, move forward in outline.
    Spans not matching current heading get level 0.
    """
    outline_items = outline.get("outline", [])
    outline_index = 0   # To track current heading
    n_outline = len(outline_items)
    
    # Pre-normalize outline texts once for efficiency
    for item in outline_items:
        item["_norm_text"] = normalize_text(item["text"])
    
    # Go through all spans in heuristics
    for span in heuristics:
        span_text = span.get("text", "")
        span_norm = normalize_text(span_text)
        
        if outline_index >= n_outline:
            # No more headings left in outline, assign 0
            span["level"] = 0
            continue
        
        current_heading = outline_items[outline_index]
        # Use approximate matching
        if match_span_to_outline(span_text, current_heading["text"]): # Parth look here
            # Assign the heading level to the span
            span["level"] = int(current_heading["level"].lstrip("H"))
            outline_index += 1  # Move to next heading in outline
        else:
            # No match, assign 0 level (non-heading)
            span["level"] = 0
    return heuristics

def pipeline(heuristics_path, outline_path, output_path):
    heuristics = load_json(heuristics_path)
    outline = load_json(outline_path)

    enriched = assign_levels_top_down(heuristics, outline)
    save_json(enriched, output_path)
    print(f"Saved enriched heuristics with top-down levels to {output_path}")

from pathlib import Path

if __name__ == "__main__":
    # Hardcoded paths — update as needed
    heuristics_path = "/home/sakamuri/Documents/Adobe-Hack/PYfiles/data/pre/1809.01477v1_preprocess.json"
    outline_path = "/home/sakamuri/Documents/Adobe-Hack/PYfiles/data/outlines/1809.01477v1.json"

    # Get current script directory
    current_dir = Path(__file__).resolve().parent

    # Create output directory: data/post/
    post_dir = current_dir / "data" / "post"
    post_dir.mkdir(parents=True, exist_ok=True)

    # Extract heuristics filename without extension
    heuristics_file_stem = Path(heuristics_path).stem
    output_path = post_dir / f"{heuristics_file_stem}_postprocess.json"

    # Call pipeline
    pipeline(heuristics_path, outline_path, str(output_path))

