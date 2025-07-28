import fitz  
import pandas as pd
import numpy as np
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any
import json
from scipy.spatial.distance import cosine
# from difflib import SequenceMatcher

class PDFHeuristicExtractor:
    """
    Comprehensive PDF text heuristic extractor using PyMuPDF.
    Extracts 50+ features for heading detection including contextual comparisons.
    """
    
    def __init__(self):
        self.base_font_size = None
        self.font_frequency = None
        self.common_patterns = [
            r'^(chapter|section|\d+\.?\d*)', 
            r'^[IVX]+\.', # Roman numerals
            r'^\d+(\.\d+)*\s+', # Numbered sections
            r'^[A-Z][A-Z\s]+$', # All caps
            r'^\d+[\.)]', # Numbered items
        ]

        
        
    def extract_document_features(self, pdf_path: str) -> List[Dict]:
        
        all_features = []

        doc = fitz.open(pdf_path)  # Open PDF here
        self._analyze_global_properties(doc)  # Analyze fonts, etc.

        for page_num, page in enumerate(doc):
            page_features = self._extract_page_features(page, page_num)
            all_features.extend(page_features)

        doc.close()

        self._compute_contextual_features(all_features)

        return all_features

    
    def _analyze_global_properties(self, doc):
        """Analyze global document properties for baseline comparison"""
        font_sizes = []
        font_names = Counter()
        
        for page in doc:
            text_dict = page.get_text("dict")
            for block in text_dict["blocks"]:
                if block.get("type") == 0:  # text block
                    for line in block["lines"]:
                        for span in line["spans"]:
                            font_sizes.append(span["size"])
                            font_names[span["font"]] += 1
        
        # Calculate base font size (most common)
        if font_sizes:
            size_counter = Counter(font_sizes)
            self.base_font_size = size_counter.most_common(1)[0][0]
        else:
            self.base_font_size = 12.0
            
        self.font_frequency = font_names

    def _merge_raw_spans(self, spans_data: List[Dict]) -> List[Dict]:
        merged = []
        EPS = 0.5
        i, n = 0, len(spans_data)

        while i < n:
            base     = spans_data[i]
            y1_base  = base["bbox"][3]
            group    = [base]
            j = i + 1
            while j < n and abs(spans_data[j]["bbox"][3] - y1_base) <= EPS:
                group.append(spans_data[j])
                j += 1
            group.sort(key=lambda s: s["bbox"][0])

            buf = ""
            for span_info in group:
                for token in re.split(r'(\s)', span_info["text"]):   # keep spaces as tokens
                    if not token:
                        continue

                    visible = token.rstrip()         # token without trailing blanks
                    buf += token                     # append ONCE

                    if not visible:                  # token was only spaces
                        continue

                    last_char = visible[-1]

                    # break if last char is not alnum / ! / ?        (dot counts as break)
                    if not re.match(r'[A-Za-z0-9!?]', last_char):
                        merged.append({
                            "span":      span_info["span"],
                            "block_idx": span_info["block_idx"],
                            "line_idx":  span_info["line_idx"],
                            "span_idx":  span_info["span_idx"],
                            "bbox": (
                                group[0]['bbox'][0],
                                y1_base - (group[0]['bbox'][3] - group[0]['bbox'][1]),
                                group[-1]['bbox'][2],
                                y1_base
                            ),
                            "text":     buf.strip(),
                            "page_num": span_info["page_num"]
                        })
                        buf = ""

            if buf.strip():
                merged.append({
                    "span":      group[0]["span"],
                    "block_idx": group[0]["block_idx"],
                    "line_idx":  group[0]["line_idx"],
                    "span_idx":  group[0]["span_idx"],
                    "bbox": (
                        group[0]['bbox'][0],
                        y1_base - (group[0]['bbox'][3] - group[0]['bbox'][1]),
                        group[-1]['bbox'][2],
                        y1_base
                    ),
                    "text":     buf.strip(),
                    "page_num": base["page_num"]
                })

            i = j

        return merged

    
    def _extract_page_features(self, page, page_num: int) -> List[Dict]:
        raw = []
        text_dict = page.get_text("dict")
        for b, block in enumerate(text_dict["blocks"]):
            if block.get("type") != 0: continue
            for l, line in enumerate(block["lines"]):
                for s, span in enumerate(line["spans"]):
                    raw.append({
                        'span': span,
                        'block_idx': b,
                        'line_idx': l,
                        'span_idx': s,
                        'bbox': span['bbox'],
                        'text': span['text'],
                        'page_num': page_num
                    })
        # Merge into logical spans
        merged = self._merge_raw_spans(raw)
        features = []
        for idx, span_data in enumerate(merged):
            feat = self._extract_span_features(span_data, merged, idx, page)
            if feat:
                features.append(feat)
        return features

    
    def _extract_span_features(self, span_data: Dict, all_spans: List[Dict], 
                              current_idx: int, page) -> Dict:
        """Extract comprehensive features for a single text span"""
        span = span_data['span']
        text = span_data['text'].strip()
        bbox = span_data['bbox']
        
        if not text or text.isspace() or len(re.sub(r'\W+', '', text)) <= 1:
            return None  # Skip junk spans

        
        features = {}
        
        # === BASIC TEXT FEATURES ===
        features['text'] = text
        features['text_length'] = len(text)
        features['word_count'] = len(text.split())
        features['char_count'] = len(text)
        features['page_num'] = span_data['page_num']
        
        # === TYPOGRAPHY FEATURES ===
        features['font_size'] = span['size']
        features['font_name'] = span['font']
        features['font_flags'] = span['flags']
        features['color'] = span['color']
        
        # Font analysis
        features['font_size_ratio'] = span['size'] / self.base_font_size if self.base_font_size > 0 else 1.0
        features['is_bold'] = bool(span['flags'] & 2**4)
        features['is_italic'] = bool(span['flags'] & 2**1)
        features['is_monospace'] = 'mono' in span['font'].lower()
        features['font_family'] = self._extract_font_family(span['font'])
        
        # === POSITIONAL FEATURES ===
        features['x0'] = bbox[0]
        features['y0'] = bbox[1] 
        features['x1'] = bbox[2]
        features['y1'] = bbox[3]
        features['width'] = bbox[2] - bbox[0]
        features['height'] = bbox[3] - bbox[1]
        features['center_x'] = (bbox[0] + bbox[2]) / 2
        features['center_y'] = (bbox[1] + bbox[3]) / 2
        
        # Page-relative positions
        page_rect = page.rect
        features['relative_x'] = bbox[0] / page_rect.width
        features['relative_y'] = bbox[1] / page_rect.height
        features['relative_width'] = (bbox[2] - bbox[0]) / page_rect.width
        features['relative_height'] = (bbox[3] - bbox[1]) / page_rect.height
        
        # === INDENTATION AND ALIGNMENT ===
        features['left_margin'] = bbox[0]
        features['right_margin'] = page_rect.width - bbox[2]
        features['is_left_aligned'] = bbox[0] < 100  # pixels from left
        features['is_right_aligned'] = bbox[2] > page_rect.width - 100
        features['is_centered'] = abs(features['center_x'] - page_rect.width/2) < 50
        
        # === TEXTUAL PATTERN FEATURES ===
        features['starts_with_number'] = bool(re.match(r'^\d', text))
        features['ends_with_period'] = text.endswith('.')
        features['ends_with_colon'] = text.endswith(':')
        features['is_all_caps'] = text.isupper() and len(text) > 1
        features['is_title_case'] = text.istitle()
        features['has_punctuation'] = bool(re.search(r'[^\w\s]', text))
        features['punctuation_count'] = len(re.findall(r'[^\w\s]', text))
        
        # Pattern matching for common heading styles
        for i, pattern in enumerate(self.common_patterns):
            features[f'pattern_match_{i}'] = bool(re.match(pattern, text.lower()))
        
        # === Y-AXIS WORD SIMILARITY FEATURES ===
        features.update(self._compute_y_axis_similarity(span_data, all_spans, current_idx))
        
        # === CONTEXTUAL COMPARISON FEATURES ===
        features.update(self._compute_surrounding_contrast(span_data, all_spans, current_idx))
        
        # === WHITESPACE FEATURES ===
        features.update(self._compute_whitespace_features(span_data, all_spans, current_idx))
        
        # === ADVANCED LINGUISTIC FEATURES ===
        features.update(self._compute_linguistic_features(text))
        
        # === DOCUMENT STRUCTURE FEATURES ===
        features['block_idx'] = span_data['block_idx']
        features['line_idx'] = span_data['line_idx']
        features['span_idx'] = span_data['span_idx']
        features['is_first_in_block'] = span_data['span_idx'] == 0
        features['is_first_in_line'] = span_data['line_idx'] == 0
        
        return features
    
    def _compute_y_axis_similarity(self, span_data: Dict, all_spans: List[Dict], 
                                  current_idx: int) -> Dict:
        """Compute similarity features for text at similar y-coordinates"""
        features = {}
        current_y = span_data['bbox'][1]
        current_text = span_data['text']
        current_words = current_text.split()
        
        # Find spans at similar y-coordinates (within 5 pixels)
        similar_y_spans = []
        for i, other_span in enumerate(all_spans):
            if i != current_idx:
                other_y = other_span['bbox'][1]
                if abs(current_y - other_y) <= 5:
                    similar_y_spans.append(other_span)
        
        if similar_y_spans:
            # Word-level similarity analysis
            all_similar_words = []
            for span in similar_y_spans:
                all_similar_words.extend(span['text'].split())
            
            # Cosine similarity based on word occurrence
            if current_words and all_similar_words:
                # Create word vectors
                all_words = set(current_words + all_similar_words)
                current_vector = [current_words.count(word) for word in all_words]
                similar_vector = [all_similar_words.count(word) for word in all_words]
                
                # Compute cosine similarity
                if sum(current_vector) > 0 and sum(similar_vector) > 0:
                    features['y_axis_word_similarity'] = 1 - cosine(current_vector, similar_vector)
                else:
                    features['y_axis_word_similarity'] = 0.0
            else:
                features['y_axis_word_similarity'] = 0.0
            
            # Other y-axis features
            features['y_axis_span_count'] = len(similar_y_spans)
            similar_font_sizes = [s['span']['size'] for s in similar_y_spans]
            features['y_axis_avg_font_size'] = np.mean(similar_font_sizes) if similar_font_sizes else 0
            features['y_axis_font_size_std'] = np.std(similar_font_sizes) if len(similar_font_sizes) > 1 else 0
            
            # Text pattern similarity on same line
            similar_texts = [s['text'] for s in similar_y_spans]
            features['y_axis_has_numbers'] = sum(1 for t in similar_texts if re.search(r'\d', t)) / len(similar_texts)
            features['y_axis_has_caps'] = sum(1 for t in similar_texts if t.isupper()) / len(similar_texts)
        else:
            features['y_axis_word_similarity'] = 0.0
            features['y_axis_span_count'] = 0
            features['y_axis_avg_font_size'] = 0
            features['y_axis_font_size_std'] = 0
            features['y_axis_has_numbers'] = 0
            features['y_axis_has_caps'] = 0
            
        return features
    
    def _compute_surrounding_contrast(self, span_data: Dict, all_spans: List[Dict], 
                                    current_idx: int) -> Dict:
        """Compute contrast features comparing with surrounding text"""
        features = {}
        current_span = span_data['span']
        current_bbox = span_data['bbox']
        
        # Find surrounding spans (within 100 pixels vertically)
        surrounding_spans = []
        for i, other_span in enumerate(all_spans):
            if i != current_idx:
                other_bbox = other_span['bbox']
                vertical_distance = abs(current_bbox[1] - other_bbox[1])
                if vertical_distance <= 100:
                    surrounding_spans.append(other_span)
        
        if surrounding_spans:
            # Font size contrast
            surrounding_sizes = [s['span']['size'] for s in surrounding_spans]
            avg_surrounding_size = np.mean(surrounding_sizes)
            features['font_size_contrast'] = current_span['size'] / avg_surrounding_size if avg_surrounding_size > 0 else 1.0
            features['font_size_difference'] = current_span['size'] - avg_surrounding_size
            features['is_largest_nearby'] = current_span['size'] > max(surrounding_sizes)
            
            # Color contrast
            surrounding_colors = [s['span']['color'] for s in surrounding_spans]
            features['color_uniqueness'] = 1.0 if current_span['color'] not in surrounding_colors else 0.0
            
            # Font family contrast
            surrounding_fonts = [s['span']['font'] for s in surrounding_spans]
            features['font_uniqueness'] = 1.0 if current_span['font'] not in surrounding_fonts else 0.0
            
            # Position contrast
            surrounding_x_positions = [s['bbox'][0] for s in surrounding_spans]
            avg_x = np.mean(surrounding_x_positions)
            features['x_position_contrast'] = abs(current_bbox[0] - avg_x)
            
            # Text length contrast
            surrounding_lengths = [len(s['text']) for s in surrounding_spans]
            avg_length = np.mean(surrounding_lengths)
            features['text_length_contrast'] = len(span_data['text']) / avg_length if avg_length > 0 else 1.0
            
            # Formatting contrast (bold, italic)
            surrounding_bold = [bool(s['span']['flags'] & 2**4) for s in surrounding_spans]
            surrounding_italic = [bool(s['span']['flags'] & 2**1) for s in surrounding_spans]
            
            bold_rate = sum(surrounding_bold) / len(surrounding_bold)
            italic_rate = sum(surrounding_italic) / len(surrounding_italic)
            
            current_bold = bool(current_span['flags'] & 2**4)
            current_italic = bool(current_span['flags'] & 2**1)
            
            features['bold_contrast'] = abs(float(current_bold) - bold_rate)
            features['italic_contrast'] = abs(float(current_italic) - italic_rate)
        else:
            # No surrounding spans
            features['font_size_contrast'] = 1.0
            features['font_size_difference'] = 0.0
            features['is_largest_nearby'] = False
            features['color_uniqueness'] = 0.0
            features['font_uniqueness'] = 0.0
            features['x_position_contrast'] = 0.0
            features['text_length_contrast'] = 1.0
            features['bold_contrast'] = 0.0
            features['italic_contrast'] = 0.0
            
        return features
    
    def _compute_whitespace_features(self, span_data: Dict, all_spans: List[Dict], 
                                   current_idx: int) -> Dict:
        """Compute whitespace and spacing features"""
        features = {}
        current_bbox = span_data['bbox']
        current_y = current_bbox[1]
        
        # Find vertically adjacent spans
        above_spans = [s for s in all_spans if s['bbox'][3] <= current_y and current_y - s['bbox'][3] <= 50]
        below_spans = [s for s in all_spans if s['bbox'][1] >= current_bbox[3] and s['bbox'][1] - current_bbox[3] <= 50]
        
        # Whitespace above
        if above_spans:
            min_distance_above = min(current_y - s['bbox'][3] for s in above_spans)
            features['whitespace_above'] = min_distance_above
        else:
            features['whitespace_above'] = 50.0  # Max distance considered
        
        # Whitespace below
        if below_spans:
            min_distance_below = min(s['bbox'][1] - current_bbox[3] for s in below_spans)
            features['whitespace_below'] = min_distance_below
        else:
            features['whitespace_below'] = 50.0
        
        # Total whitespace isolation
        features['total_whitespace'] = features['whitespace_above'] + features['whitespace_below']
        features['is_isolated'] = features['total_whitespace'] > 20
        
        # Horizontal whitespace (indentation analysis)
        # Find spans on the same approximate line
        same_line_spans = [s for s in all_spans if abs(s['bbox'][1] - current_y) <= 3]
        if len(same_line_spans) > 1:
            x_positions = [s['bbox'][0] for s in same_line_spans]
            features['horizontal_uniqueness'] = len(set(x_positions)) / len(same_line_spans)
        else:
            features['horizontal_uniqueness'] = 1.0
        
        # Leading whitespace (indentation)
        features['leading_whitespace'] = len(span_data['text']) - len(span_data['text'].lstrip())
        features['trailing_whitespace'] = len(span_data['text']) - len(span_data['text'].rstrip())
        
        return features
    
    def _compute_linguistic_features(self, text: str) -> Dict:
        """Compute advanced linguistic features"""
        features = {}
        words = text.split()
        
        # Word characteristics
        features['avg_word_length'] = np.mean([len(w) for w in words]) if words else 0
        features['max_word_length'] = max([len(w) for w in words]) if words else 0
        features['capital_word_ratio'] = sum(1 for w in words if w[0].isupper()) / len(words) if words else 0
        
        # Sentence characteristics
        features['sentence_count'] = len(re.split(r'[.!?]+', text))
        features['has_question_mark'] = '?' in text
        features['has_exclamation'] = '!' in text
        
        # Numeric content
        numbers = re.findall(r'\d+', text)
        features['number_count'] = len(numbers)
        features['has_decimal'] = '.' in text and any(c.isdigit() for c in text)
        features['has_percentage'] = '%' in text
        
        # Special characters
        features['has_parentheses'] = '(' in text or ')' in text
        features['has_quotes'] = '"' in text or "'" in text
        features['has_hyphen'] = '-' in text
        features['has_underscore'] = '_' in text
        
        # Text entropy (complexity measure)
        if text:
            char_counts = Counter(text.lower())
            total_chars = len(text)
            entropy = -sum((count/total_chars) * np.log2(count/total_chars) 
                          for count in char_counts.values())
            features['text_entropy'] = entropy
        else:
            features['text_entropy'] = 0.0
        
        return features
    
    def _compute_contextual_features(self, all_features: List[Dict]):
        """Compute features that require global document context"""
        if not all_features:
            return
            
        # Sort by page and position
        all_features.sort(key=lambda x: (x['page_num'], x['y0']))
        
        for i, feature_dict in enumerate(all_features):
            # Document position features
            feature_dict['document_position_ratio'] = i / len(all_features)
            feature_dict['is_document_start'] = i < len(all_features) * 0.1
            feature_dict['is_document_end'] = i > len(all_features) * 0.9
            
            # Sequence features
            feature_dict['sequence_number'] = i
            
            # Page-level features
            page_features = [f for f in all_features if f['page_num'] == feature_dict['page_num']]
            page_position = [f for f in page_features].index(feature_dict)
            feature_dict['page_position_ratio'] = page_position / len(page_features) if page_features else 0
            feature_dict['is_page_start'] = page_position < len(page_features) * 0.1
            feature_dict['is_page_end'] = page_position > len(page_features) * 0.9
            
            # Font frequency features
            feature_dict['font_rarity'] = 1.0 / (self.font_frequency.get(feature_dict['font_name'], 1) + 1)
            feature_dict['is_rare_font'] = feature_dict['font_rarity'] > 0.1
    
    def _extract_font_family(self, font_name: str) -> str:
        """Extract base font family name"""
        # Remove common suffixes
        base_name = re.sub(r'[-+](Bold|Italic|Light|Regular|Medium).*', '', font_name)
        return base_name.strip()
    
    def _get_empty_features(self) -> Dict:
        """Return feature dict for empty text spans"""
        return {
            'text': '',
            'text_length': 0,
            'word_count': 0,
            'char_count': 0,
            'font_size': 0,
            'font_size_ratio': 0,
            'is_heading': False  # Default label
        }

    def save_features(self, features: List[Dict], filepath: str):
        """Save extracted features to JSON file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(features, f, indent=2, ensure_ascii=False)
    
    def load_features(self, filepath: str) -> List[Dict]:
        """Load features from JSON file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def extract_outline(self, pdf_path: str,
                        max_heading_levels: int = 3) -> Dict:
        """
        1. run feature extraction
        2. detect title (dynamic)
        3. detect headings + assign H1/H2/H3 dynamically
        4. return the exact JSON structure you asked for
        """
        feats = self.extract_document_features(pdf_path)

        title_span   = self._find_title(feats)
        heading_spans = self._find_headings(feats,
                                            title_span,
                                            max_heading_levels=max_heading_levels)

        outline_json = [
            dict(level=sp['level'],
                text=sp['text'],
                page=sp['page_num'] + 1)          # human page numbers start at 1
            for sp in heading_spans
        ]

        return dict(title=title_span['text'] if title_span else "",
                    outline=outline_json)

    # --------------------------------------------------------------------------
    #  TITLE – purely data-driven, no constants
    # --------------------------------------------------------------------------
    def _find_title(self, feats: List[Dict]) -> Dict:
        first_page = [f for f in feats if f['page_num'] == 0]
        if not first_page:
            return {}

        # Build a score based on *relative* measures
        # - font_size_z           : bigger than typical page font
        # - vertical_position_z   : closer to top
        # - horizontal_centring   : centred
        # - whitespace_above_z    : isolated
        sizes = np.array([f['font_size'] for f in first_page])
        y0s   = np.array([f['y0']         for f in first_page])
        wsa   = np.array([f['whitespace_above'] for f in first_page])

        size_z = (sizes - sizes.mean()) / (sizes.std()  + 1e-6)
        y_inv  = (y0s.max() - y0s)                        # distance from top
        y_z    = (y_inv - y_inv.mean()) / (y_inv.std() + 1e-6)
        wsa_z  = (wsa  - wsa.mean()) / (wsa.std()  + 1e-6)

        for i, f in enumerate(first_page):
            f['_title_score'] = (
                3*size_z[i] +                       # heavy weight on size
                2*y_z[i] +
                1*wsa_z[i] +
                (1 if f['is_centered'] else 0)
            )

        return max(first_page, key=lambda f: f['_title_score'])

    # --------------------------------------------------------------------------
    #  HEADINGS – dynamic threshold + dynamic level mapping
    # --------------------------------------------------------------------------
    # ------------------------------------------------------------------
#  HEADINGS – dynamic threshold + better filtering
# ------------------------------------------------------------------
    def _find_headings(self,
                    feats: List[Dict],
                    title_span: Dict,
                    *,
                    max_heading_levels: int = 3) -> List[Dict]:

        # -----------------------------------------------------------
        # 0. pre-filter obvious body text
        # -----------------------------------------------------------
        prelim = [f for f in feats
                if f['word_count'] <= 20            # headings are short
                and not (f['ends_with_period'] and f['word_count'] > 5)]

        if title_span:
            prelim = [f for f in prelim if f is not title_span]
        if not prelim:
            return []

        # -----------------------------------------------------------
        # 1. build simple priority score
        #    – centred + whitespace dominate
        # -----------------------------------------------------------
        # normalise font sizes & whitespace once
        size      = np.array([f['font_size']        for f in prelim])
        size_z    = (size - size.mean()) / (size.std() + 1e-6)

        twhite    = np.array([f['whitespace_above'] + f['whitespace_below']
                            for f in prelim])
        twhite_z  = (twhite - twhite.mean()) / (twhite.std() + 1e-6)

        for i, f in enumerate(prelim):
            f['_heading_score'] = (
                5.0 * (1 if f['is_centered'] else 0)     # ← top priority
                + 4.0 * twhite_z[i]                        # ← isolation
                + 2.0 * size_z[i]                          # ← font size
                + 1.0 * (1 if f['is_bold'] else 0)         # ← extras
                + 0.5 * (1 if f['starts_with_number'] else 0)
            )

        # -----------------------------------------------------------
        # 2. keep only the top tail of scores  (µ + 0.8σ)
        # -----------------------------------------------------------
        scores  = np.array([f['_heading_score'] for f in prelim])
        cut_off = scores.mean() + 0.8 * scores.std()
        cands   = [f for f in prelim if f['_heading_score'] >= cut_off]
        if not cands:
            return []

        # -----------------------------------------------------------
        # 3. cluster by visual style  (font, bold, centred)
        # -----------------------------------------------------------
        def style_key(f):
            return (round(f['font_size'], 1), f['is_bold'], f['is_centered'])

        clusters = defaultdict(list)
        for f in cands:
            clusters[style_key(f)].append(f)

        # discard 1-off clusters – usually noise
        clusters = {k: v for k, v in clusters.items() if len(v) >= 2}
        if not clusters:
            return []

        # -----------------------------------------------------------
        # 4. map the largest three clusters → H1 / H2 / H3
        # -----------------------------------------------------------
        order = sorted(clusters.items(),
                    key=lambda kv: (-np.mean([x['font_size'] for x in kv[1]]),
                                    -len(kv[1])))[:max_heading_levels]

        level_by_key = {k: f"H{idx+1}" for idx, (k, _) in enumerate(order)}

        labelled = []
        for key, members in clusters.items():
            lvl = level_by_key.get(key)
            if not lvl:
                continue
            for m in members:
                m['level'] = lvl
                labelled.append(m)

        labelled.sort(key=lambda f: (f['page_num'], f['y0']))
        return labelled

if __name__ == "__main__":
    import time, json
    from pathlib import Path

    t0 = time.time()

    extractor   = PDFHeuristicExtractor()

    root_dir    = Path(__file__).resolve().parent
    in_dir      = root_dir / "app/input"
    out_dir     = root_dir / "app/output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # find all PDFs (recursively only if you want that)
    pdf_files = sorted(in_dir.glob("*.pdf"))

    if not pdf_files:
        print("No PDFs found in", in_dir)
        exit()

    for pdf_path in pdf_files:
        try:
            outline = extractor.extract_outline(str(pdf_path))

            json_path = out_dir / f"{pdf_path.stem}.json"
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(outline, f, indent=2, ensure_ascii=False)

            print(f"✔  {pdf_path.name:<40} →  {json_path.name}")
        except Exception as e:
            print(f"✘  {pdf_path.name:<40}  ({e})")

    print(f"\nProcessed {len(pdf_files)} file(s) in {time.time()-t0:.1f}s")
