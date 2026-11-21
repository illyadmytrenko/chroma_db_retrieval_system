import streamlit as st

from db import (
    get_demo_collection,
    get_user_collection,
    add_files_to_collection,
    query_collection,
    get_files_from_collection,
    delete_files_from_collection,
    delete_collection,
)

source_options = ["📚 Demo database (preloaded)", "📤 My files (temporary session)"]
mode = st.sidebar.radio(
    "Data source:",
    source_options,
)

is_demo = mode == source_options[0]
collection = get_demo_collection() if is_demo else get_user_collection()

st.title("Select an option:")
if is_demo:
    options = ["🔍 Query collection", "📄 Show files"]
    st.info(
        "ℹ️ **Demo Mode:** Database is read-only. Switch to **My files** in the"
        " sidebar to upload or delete your own documents."
    )
else:
    options = [
        "📁 Add files",
        "🔍 Query collection",
        "📄 Show files",
        "🗑️ Delete files",
        "💥 Delete whole collection",
    ]
option = st.pills(
    "Navigation",
    options=options,
    label_visibility="collapsed",
    default=options[0],
)

if option is not None:
    match option:
        case "📁 Add files":
            if is_demo:
                st.error("Adding files is disabled in Demo mode.")
            else:
                add_files_to_collection(collection)
        case "🔍 Query collection":
            query_collection(collection)
        case "📄 Show files":
            get_files_from_collection(collection)
        case "🗑️ Delete files":
            if is_demo:
                st.error("Deleting files is disabled in Demo mode.")
            else:
                delete_files_from_collection(collection)
        case "💥 Delete whole collection":
            if is_demo:
                st.error("Deleting the collection is disabled in Demo mode.")
            else:
                delete_collection(collection)
