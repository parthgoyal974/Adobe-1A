import pdfplumber
import os
import json
import time

def main():
    start = time.time()

    pdf_path   = "/home/sakamuri/Documents/Adobe-Hack/pdfs/mega_pdf"
    base_name  = os.path.splitext(pdf_path)[0]
    json_path  = f"{base_name}_words.json"
    txt_path   = f"{base_name}_words.txt"

    records = []  #one dict per word

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            words = page.extract_words(x_tolerance=1, y_tolerance=1)
            for w in words:
                records.append({
                    "page":     page_num,
                    "text":     w["text"],
                    "fontname": w.get("fontname", ""),
                    "size":     w.get("size", 0),
                    "top":      w.get("top", 0),
                    "x0":       w.get("x0", 0),
                    "x1":       w.get("x1", 0)
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
