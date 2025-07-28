import pdfplumber
import os
import json
import time

start_time = time.time()

pdf_path      = "/home/sakamuri/Documents/Adobe-Hack/pdfs/mega_pdf"
base_name   = os.path.splitext(pdf_path)[0]
jsonl_path  = f"{base_name}_words.jsonl"
txt_path    = f"{base_name}_words.txt"

# Buffers to accumulate output
jsonl_buffer = []
txt_buffer   = []

with pdfplumber.open(pdf_path) as pdf:
    for page_num, page in enumerate(pdf.pages, start=1):
        chars_buffer = []
        size_sum     = 0.0
        last_char    = None

        for char in page.chars:
            c = char.get("text", "")
            if c.isspace():
                if chars_buffer:
                    word_str   = "".join(chars_buffer)
                    char_count = len(chars_buffer)
                    record = {
                        "page": page_num,
                        "text": word_str,
                        "fontname": last_char.get("fontname", ""),
                        "avg_size": round(size_sum / char_count, 2),
                        "top": last_char.get("top", 0)
                    }
                    jsonl_buffer.append(json.dumps(record))
                    txt_buffer.append(word_str)
                    # reset
                    chars_buffer = []
                    size_sum     = 0.0
                    last_char    = None
            else:
                chars_buffer.append(c)
                size_sum += char.get("size", 0)
                last_char = char

        # Flush any word left at end of page
        if chars_buffer and last_char:
            word_str   = "".join(chars_buffer)
            char_count = len(chars_buffer)
            record = {
                "page": page_num,
                "text": word_str,
                "fontname": last_char.get("fontname", ""),
                "avg_size": round(size_sum / char_count, 2),
                "top": last_char.get("top", 0)
            }
            jsonl_buffer.append(json.dumps(record))
            txt_buffer.append(word_str)

# Write once to disk
with open(jsonl_path, "w", encoding="utf-8") as jsonl_f:
    jsonl_f.write("\n".join(jsonl_buffer) + "\n")

with open(txt_path, "w", encoding="utf-8") as txt_f:
    txt_f.write("\n".join(txt_buffer) + "\n")

end_time = time.time()
print(f"Execution time: {end_time - start_time:.4f} seconds")

# import pdfplumber
# import os
# import json
# import time

# start_time = time.time()  # ⏱ Start timer

# pdf_path = "/home/sakamuri/Documents/Adobe-Hack/pdfs/mega_pdf"
# base_name = os.path.splitext(pdf_path)[1] 
# chars_output_path = f"{base_name}_chars.jsonl"
# # sentences_output_path = f"{base_name}_sentences.txt"
# words_output_path = f"{base_name}_words.txt"

# # Open output files
# #     open(sentences_output_path, "w", encoding="utf-8") as sentences_file, \
# with open(chars_output_path, "w", encoding="utf-8") as chars_file, \
#      open(words_output_path, "w", encoding="utf-8") as words_file, \
#      pdfplumber.open(pdf_path) as pdf:

#     for page_num, page in enumerate(pdf.pages, 1):
#         # Write characters to JSONL
#         word = ""
#         netCharSize = 0
#         for char in page.chars:
#             word_info = {}
            
#             if char.get("text", "") == " ":
#                 try:
#                     word_info = {
#                         "text": word,
#                         "fontname": lastChar.get("fontname", ""),
#                         "size": netCharSize/len(word),
#                         "top": lastChar.get("top", 0)
#                     }

#                     chars_file.write(json.dumps(word_info) + "\n")

#                 except:
#                     print("first character appears to be spaced?")
#                     continue

#                 word = ""
#                 netCharSize = 0
#                 continue
            
#             word += char.get("text", "")
#             netCharSize += round(char.get("size", 0))


#             lastChar = char

#         # # Extract and write sentences
#         # page_text = page.extract_text()
#         # if page_text:
#         #     # lines = page_text.split("\n")
#         #     # for line in lines:
#         #     #     line = line.strip()
#         #     #     if line:
#         #     #         sentences_file.write(line + "\n")

#         #     # Extract and write words
#         #     words = page_text.replace("\n", " ").split(" ")
#         #     for word in words:
#         #         word = word.strip()
#         #         if word:
#         #             words_file.write(word + "\n")


# # ⏱ End timer and print
# end_time = time.time()
# print(f"Execution time: {end_time - start_time:.4f}s")



# import pdfplumber
# import os
# import json
# import time

# start_time = time.time()  # ⏱ Start timer

# pdf_path = "/home/sakamuri/Documents/Adobe-Hack/pdfs/E0CCG5S239.pdf"
# base_name = os.path.splitext(pdf_path)[1]  # Changed to [0] to get filename without extension
# chars_output_path = f"{base_name}_chars.jsonl"
# sentences_output_path = f"{base_name}_sentences.txt"
# words_output_path = f"{base_name}_words.txt"

# # Open output files
# with open(chars_output_path, "w", encoding="utf-8") as chars_file, \
#      open(sentences_output_path, "w", encoding="utf-8") as sentences_file, \
#      open(words_output_path, "w", encoding="utf-8") as words_file, \
#      pdfplumber.open(pdf_path) as pdf:

#     for page_num, page in enumerate(pdf.pages, 1):
#         # Write characters to JSONL
#         for char in page.chars:
#             char_info = {
#                 "page": page_num,
#                 "text": char.get("text", ""),
#                 "fontname": char.get("fontname", ""),
#                 "size": char.get("size", 0),
#                 "x0": char.get("x0", 0),
#                 "x1": char.get("x1", 0),
#                 "top": char.get("top", 0),
#                 "bottom": char.get("bottom", 0),
#                 "doctop": char.get("doctop", 0)
#             }
#             chars_file.write(json.dumps(char_info) + "\n")

#         # Extract and write sentences
#         page_text = page.extract_text()
#         if page_text:
#             lines = page_text.split("\n")
#             for line in lines:
#                 line = line.strip()
#                 if line:
#                     sentences_file.write(line + "\n")

#             # Extract and write words
#             words = page_text.replace("\n", " ").split(" ")
#             for word in words:
#                 word = word.strip()
#                 if word:
#                     words_file.write(word + "\n")

# # ⏱ End timer and print
# end_time = time.time()
# print(f"Done! Time taken: {end_time - start_time:.2f} seconds")

