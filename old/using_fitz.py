import fitz  # PyMuPDF
import os
import json
import time

def main():
    start = time.time()

    pdf_path  = "/home/sakamuri/Documents/Adobe-Hack/pdfs/mega.pdf"
    base_name = os.path.splitext(pdf_path)[1]
    json_path = f"{base_name}_words.json"
    txt_path  = f"{base_name}_words.txt"

    records = []

    # Open the PDF
    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            # Get structured text as a dict
            page_dict = page.get_text("dict")
            for block in page_dict["blocks"]:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        # Each span has text, font, size, and a bbox [x0, y0, x1, y1]
                        fontname = span.get("font", "")
                        size     = span.get("size", 0)
                        top      = span["bbox"][1]
                        colour   = span.get("color", "")
                        # Split the span text into words on whitespace
                        for word in span["text"].split():
                            records.append({
                                "page":     page_num,
                                "text":     word,
                                "fontname": fontname,
                                "size":     size,
                                "colour":   colour,
                                "top":      round(top)
                            })

    # Bulk write JSON array
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(records, jf, ensure_ascii=False)

    # Bulk write plain text
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write("\n".join(r["text"] for r in records))

    elapsed = time.time() - start
    print(f"Done! Processed {len(records)} words in {elapsed:.2f}s")

if __name__ == "__main__":
    main()
