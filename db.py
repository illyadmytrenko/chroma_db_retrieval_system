import os
import time
from datetime import datetime

import chromadb
import numpy as np
import streamlit as st
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from search import get_bm25_retriever, bm25_search, get_rrf_top_n
from text import extract_text_from_file, smart_chunker


# Caching the heavy embedding model once across the entire server
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embedding_function():
    return SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")


# Caching the shared preloaded demo database
@st.cache_resource(show_spinner="Connecting to demo database...")
def get_demo_collection():
    client = chromadb.PersistentClient(path="./demo_chroma_db")
    return client.get_or_create_collection(
        name="demo_docs",
        embedding_function=get_embedding_function(),
    )


# Storing the user's temporary in-memory database per session (not cached globally)
def get_user_collection():
    if "user_chroma_client" not in st.session_state:
        st.session_state.user_chroma_client = chromadb.EphemeralClient()

    return st.session_state.user_chroma_client.get_or_create_collection(
        name="user_docs",
        embedding_function=get_embedding_function(),
    )


def add_files_to_collection(collection):
    ALLOWED_EXTENSIONS = {".md", ".txt", ".py", ".pdf", ".docx"}
    BATCH_SIZE = 500

    allowed_extensions_text = ", ".join(sorted(ALLOWED_EXTENSIONS))
    st.subheader("Add Files to Collection")
    st.caption(f"Supported file formats: {allowed_extensions_text}")

    uploaded_files = st.file_uploader(
        "Select a directory to upload:",
        accept_multiple_files="directory",
    )

    if not uploaded_files:
        return

    skipped_files = [
        f.name
        for f in uploaded_files
        if os.path.splitext(os.path.basename(f.name))[1].lower()
        not in ALLOWED_EXTENSIONS
    ]
    valid_files = [
        f
        for f in uploaded_files
        if os.path.splitext(os.path.basename(f.name))[1].lower() in ALLOWED_EXTENSIONS
    ]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Selected", len(uploaded_files))
    col2.metric("Supported Files", len(valid_files))
    col3.metric("Skipped Files", len(skipped_files))

    if skipped_files:
        with st.expander(f"View {len(skipped_files)} unsupported file(s)"):
            st.write(skipped_files)

    if not valid_files:
        st.warning("No valid files (.py, .md, .txt) found in the selected directory.")
        return

    if st.button("Start Processing & Upload", type="primary", use_container_width=True):
        all_ids = []
        all_documents = []
        all_metadatas = []
        encoding_errors = []

        progress_bar = st.progress(0, text="Initializing file processing...")
        total_valid = len(valid_files)

        # Uploading files and breaking them into chunks
        for idx_file, uploaded_file in enumerate(valid_files, start=1):
            file_path_in_dir = uploaded_file.name
            root_directory = (
                uploaded_file.name.split("/")[0]
                if "/" in valid_files[0].name
                else "uploaded_dir"
            )
            filename = os.path.basename(file_path_in_dir)
            extension = os.path.splitext(filename)[1].lower()

            progress_bar.progress(
                idx_file / total_valid,
                text=f"Processing ({idx_file}/{total_valid}): {filename}",
            )

            try:
                content = extract_text_from_file(uploaded_file)
            except UnicodeDecodeError:
                encoding_errors.append(file_path_in_dir)
                continue

            # Getting chunks
            chunks = smart_chunker(content, extension=extension)
            current_time = time.time()

            # Appending data
            for idx_chunk, chunk_text in enumerate(chunks):
                all_ids.append(f"{file_path_in_dir}_chunk_{idx_chunk}")
                all_documents.append(chunk_text)
                all_metadatas.append(
                    {
                        "source": file_path_in_dir,
                        "directory": root_directory,
                        "extension": extension,
                        "uploaded_at": current_time,
                        "chunk_index": idx_chunk,
                    }
                )

        progress_bar.empty()

        # Uploading all chunks
        if all_ids:
            with st.spinner("Saving chunks to the database..."):
                for i in range(0, len(all_ids), BATCH_SIZE):
                    collection.upsert(
                        ids=all_ids[i : i + BATCH_SIZE],
                        documents=all_documents[i : i + BATCH_SIZE],
                        metadatas=all_metadatas[i : i + BATCH_SIZE],
                    )

            # Clearing cache
            get_bm25_retriever.clear()
            st.success(
                f"Successfully added **{len(all_ids)}** chunks from **{total_valid - len(encoding_errors)}** files!"
            )

        if encoding_errors:
            with st.expander(
                f"Skipped {len(encoding_errors)} file(s) due to encoding issues"
            ):
                st.write(encoding_errors)


def query_collection(collection):
    st.subheader("Query Collection")

    total_docs = collection.count()
    if not total_docs:
        st.info("There are no documents in the collection yet. Add some files first!")
        return

    col_query, col_num = st.columns([4, 1])
    with col_query:
        query = st.text_input(
            "Search query:",
            placeholder="Enter your question or keywords",
        )
    with col_num:
        n_results = st.number_input(
            "Max results:",
            min_value=1,
            max_value=total_docs,
            value=min(3, total_docs),
            step=1,
        )

    # Choosing search option
    st.subheader("Select an appropriate type of search:")
    search_options = [
        "Hybrid search",
        "Semantic (dense) search",
        "Keyword (sparse) search",
    ]
    search_option = st.pills(
        "Search",
        options=search_options,
        label_visibility="collapsed",
        default=search_options[0],
    )

    # Metadata filters
    query_meta = {}
    with st.expander("Filter by Metadata (Optional)", expanded=False):
        all_metadatas = collection.get(include=["metadatas"])["metadatas"]

        if all_metadatas:
            categorical_fields = sorted(
                list(
                    set(
                        k
                        for m in all_metadatas
                        for k in m.keys()
                        if k not in {"chunk_index", "uploaded_at"}
                    )
                )
            )

            for field in categorical_fields:
                unique_values = sorted(
                    list(set(m[field] for m in all_metadatas if field in m))
                )
                plural_label = (
                    "All directories" if field == "directory" else f"All {field}s"
                )
                selected_vals = st.multiselect(
                    f"Filter by {field}:",
                    options=unique_values,
                    placeholder=plural_label,
                )
                if selected_vals:
                    query_meta[field] = {"$in": selected_vals}

            # Date filter
            has_uploaded_at = any("uploaded_at" in m for m in all_metadatas)
            if has_uploaded_at:
                use_date_filter = st.checkbox("Filter by upload date")
                if use_date_filter:
                    n_days = st.number_input(
                        "Uploaded within the last N days:",
                        min_value=1,
                        max_value=3650,
                        value=7,
                        step=1,
                    )
                    cutoff_timestamp = time.time() - (n_days * 86400)
                    query_meta["uploaded_at"] = {"$gte": cutoff_timestamp}

    if st.button("Search", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a search query.")
            return

        if not query_meta:
            where_filter = None
        elif len(query_meta) == 1:
            where_filter = query_meta
        else:
            where_filter = {"$and": [{k: v} for k, v in query_meta.items()]}

        with st.spinner("Searching collection..."):
            try:
                coll = collection.get(
                    where=where_filter,
                    include=["documents", "metadatas"],
                )
                all_ids = coll["ids"]
                all_docs = coll["documents"]
                all_metas = coll["metadatas"]

                if not all_ids:
                    st.warning("No matching documents found for your filters.")
                    return

                dense_results = collection.query(
                    query_texts=[query],
                    n_results=len(all_ids),
                    include=["distances"],
                    where=where_filter,
                )
                dist_map = dict(
                    zip(dense_results["ids"][0], dense_results["distances"][0])
                )
                # Dense results are not alligned with default order
                aligned_dense_distances = [dist_map[doc_id] for doc_id in all_ids]

                sparse_order, sparse_scores = bm25_search(all_docs, query)

                actual_n = min(int(n_results), len(all_ids))
                # Getting RRF scores
                top_indices, top_n_scores = get_rrf_top_n(
                    aligned_dense_distances, sparse_order, actual_n
                )

            except Exception as e:
                st.error(f"Query failed: {e}")
                return

            # Getting selected search data
            actual_indices = top_indices
            actual_scores = top_n_scores
            actual_score_name = "RRF Score (the bigger the better)"
            if search_option == search_options[0]:
                actual_indices = top_indices
                actual_scores = top_n_scores
            elif search_option == search_options[1]:
                actual_indices = np.argsort(aligned_dense_distances)[:actual_n]
                actual_scores = dense_results["distances"][0][:actual_n]
                actual_score_name = "L2 Distance (the lower the better)"
            elif search_option == search_options[2]:
                actual_indices = sparse_order[:actual_n]
                actual_scores = sparse_scores[:actual_n]
                actual_score_name = "BM25 Score (the bigger the better)"

        # Results
        docs = [all_docs[i] for i in actual_indices]
        metas = [all_metas[i] for i in actual_indices]

        st.divider()
        st.markdown(f"### Top {len(docs)} Result(s)")

        for idx, (doc, meta, score) in enumerate(
            zip(docs, metas, actual_scores), start=1
        ):
            with st.container(border=True):
                header_col, score_col = st.columns([4, 1])

                source = meta.get("source", "Unknown file")
                chunk_idx = meta.get("chunk_index", "N/A")
                uploaded_ts = meta.get("uploaded_at")
                uploaded_str = (
                    datetime.fromtimestamp(uploaded_ts).strftime("%Y-%m-%d %H:%M")
                    if uploaded_ts
                    else "N/A"
                )

                with header_col:
                    st.markdown(f"**{idx}. `{source}`** *(Chunk {chunk_idx})*")
                    st.caption(
                        f"Directory: `{meta.get('directory', 'N/A')}` | "
                        f"Extension: `{meta.get('extension', 'N/A')}` | "
                        f"Uploaded: `{uploaded_str}`"
                    )

                with score_col:
                    st.metric(actual_score_name, f"{score:.4f}")

                st.code(doc[:400] + ("..." if len(doc) > 400 else ""), language=None)
                if len(doc) > 400:
                    with st.expander("Show full chunk text"):
                        st.text(doc)


def get_files_from_collection(collection):
    if not collection.count():
        st.info("There are no documents in the collection yet. Add some files first!")
        return

    results = collection.get(include=["metadatas"])
    table_data = []

    st.subheader("Files in the collection:")
    for doc_id, metadata in zip(results["ids"], results["metadatas"]):
        uploaded_ts = metadata.get("uploaded_at")
        uploaded_str = (
            datetime.fromtimestamp(uploaded_ts).strftime("%Y-%m-%d %H:%M")
            if uploaded_ts
            else "N/A"
        )
        table_data.append(
            {
                "ID": doc_id,
                "Source": metadata.get("source", ""),
                "Directory": metadata.get("directory", ""),
                "Extension": metadata.get("extension", ""),
                "Uploaded At": uploaded_str,
            }
        )

    st.dataframe(table_data, use_container_width=True, hide_index=True)


def delete_files_from_collection(collection):
    if not collection.count():
        st.info("There are no documents in the collection yet. Add some files first!")
        return

    st.subheader("Delete files from the collection")

    all_metadatas = collection.get(include=["metadatas"])["metadatas"]
    if not all_metadatas:
        st.info("The collection is empty. Nothing to delete.")
        return

    unique_files = sorted(
        list(set(m["source"] for m in all_metadatas if "source" in m))
    )

    files_to_delete = st.multiselect(
        "Select files to delete:",
        options=unique_files,
        placeholder="Choose one or more files...",
    )

    if files_to_delete:
        st.warning(f"Selected {len(files_to_delete)} file(s) for deletion.")

        if st.button("Delete selected files", type="primary"):
            collection.delete(where={"source": {"$in": files_to_delete}})
            get_bm25_retriever.clear()
            st.success(f"Deleted: {', '.join(files_to_delete)}")
            st.rerun()


def delete_collection(collection):
    if not collection.count():
        st.info("There are no documents in the collection yet. Add some files first!")
        return

    st.subheader("Are you sure you want to delete the whole collection?")
    if st.button("Yes, delete everything", type="primary", use_container_width=True):
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)
        get_bm25_retriever.clear()
        st.success("Collection was successfully cleared.")
        st.rerun()
