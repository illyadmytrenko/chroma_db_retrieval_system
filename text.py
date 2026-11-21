import re

import docx
from pypdf import PdfReader


# Extracting text fynction
def extract_text_from_file(uploaded_file) -> str:
    filename = uploaded_file.name.lower()

    if filename.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        pages_text = [
            page.extract_text() or ""
            for page in reader.pages
            if page.extract_text().strip()
        ]
        return "\n".join(pages_text)

    elif filename.endswith(".docx"):
        doc = docx.Document(uploaded_file)
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

        # Handling tables in .docx
        for table in doc.tables:
            for row in table.rows:
                paragraphs.append(" | ".join(cell.text.strip() for cell in row.cells))
        return "\n".join(paragraphs)

    else:
        # Other text files
        return uploaded_file.read().decode("utf-8", errors="ignore")


# Chunker function
def smart_chunker(
    text: str, extension: str, max_chars: int = 900, overlap: int = 120
) -> list[str]:
    ext = extension.lower()

    # Picking correct chunking strategy depending on extension
    if ext == ".py":
        blocks = re.split(r"(?=\n(?:async\s+def|def|class)\s+)", text)
    elif ext == ".md":
        blocks = re.split(r"(?=\n#{1,3}\s+)", text)
    else:
        blocks = text.split("\n\n")

    chunks = []
    current_chunk = ""

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        if len(current_chunk) + len(block) + 2 <= max_chars:
            current_chunk = f"{current_chunk}\n\n{block}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
                current_chunk = ""

            # Cutting block if it's bigger than max_chars
            if len(block) > max_chars:
                lines = block.splitlines()
                sub_chunk = ""
                for line in lines:
                    if len(sub_chunk) + len(line) + 1 <= max_chars:
                        sub_chunk = f"{sub_chunk}\n{line}".strip()
                    else:
                        if sub_chunk:
                            chunks.append(sub_chunk)
                            # Overlap
                            overlap_part = sub_chunk[-overlap:]
                            sub_chunk = f"{overlap_part}\n{line}".strip()
                        else:
                            for i in range(0, len(line), max_chars - overlap):
                                chunks.append(line[i : i + max_chars])
                            sub_chunk = ""
                if sub_chunk:
                    current_chunk = sub_chunk
            else:
                current_chunk = block

    if current_chunk:
        chunks.append(current_chunk)

    return chunks
