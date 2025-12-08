#!/usr/bin/env python3
"""
Build FAISS indices for medical datasets.

Creates per-language-pair FAISS indices using sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
for dense embeddings, and a TF-IDF vectorizer for keyword matching to enable hybrid search.

Saves each FAISS index and supporting metadata with pickle for offline loading.

Usage: python scripts/build_faiss_index.py --input-dir data/ --out-dir data/faiss_out
"""
import argparse
import os
import re
import pickle
from pathlib import Path
from typing import List, Dict, Tuple

import faiss
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer


def read_text_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return path.read_text(encoding="latin-1")


def list_documents(input_dir: Path) -> List[Path]:
    docs = []
    for ext in (".txt", ".md", ".html", ".htm", ".pdf", ".csv", ".tsv"):
        docs.extend(input_dir.rglob(f"*{ext}"))
    return docs


def extract_text_from_pdf(path: Path) -> str:
    # lightweight PDF text extraction using PyPDF2 if available
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(str(path))
        pages = [p.extract_text() or "" for p in reader.pages]
        return "\n\n".join(pages)
    except Exception:
        return ""


def simple_tokenize(text: str) -> List[str]:
    # fallback whitespace + punctuation tokenizer for chunking by 'tokens'
    tokens = re.findall(r"\w+|[^\\s\w]", text, flags=re.UNICODE)
    return tokens


def detokenize(tokens: List[str]) -> str:
    return " ".join(tokens)


def chunk_text(text: str, chunk_size: int = 256, overlap: int = 64) -> List[str]:
    tokens = simple_tokenize(text)
    if not tokens:
        return []
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i : i + chunk_size]
        chunks.append(detokenize(chunk))
        i += chunk_size - overlap
    return chunks


def detect_language_from_filename(name: str) -> Tuple[str, str]:
    # naive detection: find language codes in filename like en-sw, en_sw, en-sw-lu etc.
    # returns (source, target) when found, else ('', '')
    m = re.search(r"(en)[-_](sw|sw|lu|luo|swahili|luo)", name, flags=re.IGNORECASE)
    if m:
        return (m.group(1).lower(), m.group(2).lower())
    # fallback: check individual langs
    if "sw" in name.lower() or "swahili" in name.lower():
        return ("en", "sw")
    if "luo" in name.lower():
        return ("en", "luo")
    return ("", "")


def build_indices(
    input_dir: str,
    out_dir: str,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
    chunk_size: int = 256,
    overlap: int = 64,
):
    input_dir = Path(input_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading embedding model...")
    model = SentenceTransformer(f"sentence-transformers/{model_name}")

    docs = list_documents(input_dir)
    print(f"Found {len(docs)} documents")

    # Prepare containers per language-pair
    indices = {}  # key: 'en-sw', value: list of vectors
    metadatas = {}  # key: 'en-sw', value: list of metadata dicts
    texts_for_tfidf = []
    tfidf_docs_map = []  # map doc idx to (lang_pair, idx_in_pair)

    for path in docs:
        ext = path.suffix.lower()
        name = path.stem.lower()
        text = ""
        # CSV/TSV handling: process parallel translation tables
        if ext in (".csv", ".tsv"):
            try:
                import csv as _csv
                delim = '\t' if ext == '.tsv' else ','
                with path.open(newline='', encoding='utf-8') as fh:
                    reader = _csv.DictReader(fh, delimiter=delim)
                    rows = list(reader)
                # detect language-like columns (en, sw, luo or english/swahili/luo)
                headers = [h.lower() for h in reader.fieldnames or []]
                lang_cols = {}
                for h in headers:
                    if h in ('en','english'):
                        lang_cols['en'] = h
                    if h in ('sw','swahili','kiswahili'):
                        lang_cols['sw'] = h
                    if h in ('luo','lb','luo_language'):
                        lang_cols['luo'] = h

                # fallback: try columns that look like language codes
                for h in headers:
                    if len(h) == 2 and h.isalpha() and h not in lang_cols.values():
                        lang_cols[h] = h

                # iterate rows and build chunks per language pair
                for row in rows:
                    # try producing chunks for every pair of detected language columns
                    keys = list(lang_cols.keys())
                    for i in range(len(keys)):
                        for j in range(len(keys)):
                            if i == j:
                                continue
                            src = keys[i]
                            tgt = keys[j]
                            src_col = lang_cols[src]
                            txt = (row.get(src_col) or "").strip()
                            if not txt:
                                continue
                            chunks = chunk_text(txt, chunk_size=chunk_size, overlap=overlap)
                            for chunk in chunks:
                                meta = {
                                    "source_document": str(path.relative_to(input_dir)),
                                    "source_language": src,
                                    "target_language": tgt,
                                    "medical_category": infer_medical_category_from_filename(name),
                                }
                                vec = model.encode(chunk)
                                pair = f"{src}-{tgt}"
                                indices.setdefault(pair, []).append(vec)
                                metadatas.setdefault(pair, []).append(meta)
                                texts_for_tfidf.append(chunk)
                                tfidf_docs_map.append((pair, len(metadatas[pair]) - 1))
            except Exception as e:
                print(f"Failed to process CSV {path}: {e}")
            continue
        if ext == ".pdf":
            text = extract_text_from_pdf(path)
        elif ext in (".html", ".htm"):
            raw = read_text_file(path)
            # strip tags naively
            text = re.sub(r"<[^>]+>", " ", raw)
        else:
            text = read_text_file(path)

        if not text.strip():
            continue

        # language detection from filename and basic heuristics
        src, tgt = detect_language_from_filename(name)
        lang_pair = f"{src}-{tgt}" if src and tgt else "unknown"

        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)

        for chunk in chunks:
            # build metadata
            meta = {
                "source_document": str(path.relative_to(input_dir)),
                "source_language": src or "",
                "target_language": tgt or "",
                "medical_category": infer_medical_category_from_filename(name),
            }

            vec = model.encode(chunk)

            indices.setdefault(lang_pair, []).append(vec)
            metadatas.setdefault(lang_pair, []).append(meta)

            # for TF-IDF
            texts_for_tfidf.append(chunk)
            tfidf_docs_map.append((lang_pair, len(metadatas[lang_pair]) - 1))

    # Build TF-IDF vectorizer globally for keyword search
    print("Building TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
    tfidf_matrix = tfidf.fit_transform(texts_for_tfidf)

    # Build FAISS indices per language pair
    saved = []
    for pair, vectors in indices.items():
        arr = np.vstack(vectors).astype('float32')
        dim = arr.shape[1]
        print(f"Building FAISS index for {pair} with {arr.shape[0]} vectors (dim={dim})")

        # use inner product on normalized vectors for cosine similarity
        faiss.normalize_L2(arr)
        index = faiss.IndexFlatIP(dim)
        index.add(arr)

        out_index_path = out_dir / f"faiss_{pair}.index"
        faiss.write_index(index, str(out_index_path))

        # Save metadata and a small wrapper to load everything
        meta_out = {
            "metadatas": metadatas[pair],
            "tfidf_map": tfidf_docs_map,
            "tfidf_vocab": tfidf.vocabulary_,
            "tfidf_idf": tfidf.idf_,
        }

        with open(out_dir / f"meta_{pair}.pkl", "wb") as f:
            pickle.dump(meta_out, f)

        saved.append(pair)

    # Save TF-IDF model and mapping as well
    with open(out_dir / "tfidf_global.pkl", "wb") as f:
        pickle.dump({"vectorizer": tfidf, "docs_map": tfidf_docs_map}, f)

    print("Saved indices for:", ", ".join(saved))


def infer_medical_category_from_filename(name: str) -> str:
    # crude rules to infer category
    name = name.lower()
    if any(x in name for x in ["drug", "dosage", "prescription"]):
        return "drug_prescription"
    if any(x in name for x in ["who", "guideline", "moh", "policy", "protocol"]):
        return "guideline"
    if any(x in name for x in ["translate", "translation", "parallel"]):
        return "parallel_translation"
    return "general"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, help="Directory with medical documents")
    parser.add_argument("--out-dir", required=True, help="Directory to write FAISS indices and metadata")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=64)
    args = parser.parse_args()

    build_indices(args.input_dir, args.out_dir, chunk_size=args.chunk_size, overlap=args.overlap)


if __name__ == "__main__":
    main()
