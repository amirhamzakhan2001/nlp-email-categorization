
# streamlit_app/app_cluster_viewer.py
"""
Streamlit app — Cluster viewer (JSON names authoritative, leaf summaries from clusters CSV)
Environment variables required:
 - MASTER_CSV_PATH      -> path to email CSV / parquet
 - CLUSTERS_CSV_PATH    -> path to clusters CSV (cluster_id,label,summary) where label/summary start with **
 - CLUSTER_JSON_PATH    -> path to JSON mapping containing subcluster_label { cluster_id: "name", ... }

Behavior:
 - Dropdown lists all cluster names from JSON (cleaned).
 - On select: show name, summary (from clusters CSV), total mails (match by cluster_id if present else cluster_path), breadcrumb (root->...->leaf), direct children, and table of mails (date,name,email,body).
 - No sidebar — all controls on main page.
"""

import os
import json
from typing import Dict, List, Optional
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np


import streamlit as st


# --------------------------
# Config (read from env)
# --------------------------

MASTER_CSV_PATH = os.environ.get("MASTER_CSV_PATH", r"/Users/amirhamzakhan/Desktop/Amir_lab/NLP/email_data.csv")
CLUSTERS_CSV_PATH = os.environ.get("CLUSTERS_CSV_PATH", r"/Users/amirhamzakhan/Desktop/Amir_lab/NLP/cluster_labels.csv")
CLUSTER_JSON_PATH = os.environ.get("CLUSTER_JSON_PATH", r"/Users/amirhamzakhan/Desktop/Amir_lab/NLP/api_cluster.json")


st.set_page_config(page_title="Cluster Viewer", layout="wide")
st.title("Cluster Viewer — select a cluster to inspect")

# --------------------------
# Helpers
# --------------------------
@st.cache_data
def load_master(path: str) -> pd.DataFrame:
    """Load emails dataset. Accept CSV or parquet. Normalize common column names."""
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # normalize common column names: accept several variants
    col_map = {}
    if "name" in df.columns and "sender_name" not in df.columns:
        col_map["name"] = "sender_name"
    if "email" in df.columns and "sender_email" not in df.columns:
        col_map["email"] = "sender_email"
    if "body" in df.columns and "body_clean" not in df.columns:
        col_map["body"] = "body_clean"
    if col_map:
        df = df.rename(columns=col_map)
    # ensure expected columns exist
    for c in ["date", "sender_name", "sender_email", "body_clean", "cluster_id", "cluster_path", "cluster_leaf_label"]:
        if c not in df.columns:
            df[c] = None
    # normalize types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cluster_path"] = df["cluster_path"].fillna("").astype(str)
    df["cluster_id"] = df["cluster_id"].fillna("").astype(str)
    df["body_clean"] = df["body_clean"].fillna("").astype(str)
    df["sender_name"] = df["sender_name"].fillna("").astype(str)
    df["sender_email"] = df["sender_email"].fillna("").astype(str)
    return df

# --- robust clusters loader: dedupe & build mapping ---
@st.cache_data
def load_clusters_csv_as_map(path: str):
    """
    Load clusters CSV (cluster_id,label,summary), clean leading '**', dedupe by cluster_id,
    and return a dict mapping cluster_id -> {"label": ..., "summary": ...}
    """
    if not os.path.exists(path):
        return {}  # empty map

    df = pd.read_csv(path, dtype=str)
    # ensure necessary columns
    for c in ["cluster_id", "label", "summary"]:
        if c not in df.columns:
            df[c] = ""

    def clean_text(x: str) -> str:
        if not isinstance(x, str):
            return ""
        s = x.strip()
        # remove leading asterisks and slashes and normalize whitespace
        while s.startswith("*"):
            s = s.lstrip("* ").strip()
        s = s.rstrip("/ ").strip()
        s = " ".join(s.split())
        return s

    df["cluster_id"] = df["cluster_id"].astype(str).str.strip()
    df["label"] = df["label"].fillna("").apply(clean_text)
    df["summary"] = df["summary"].fillna("").apply(clean_text)

    # dedupe by cluster_id: keep first occurrence
    df_unique = df.groupby("cluster_id", sort=False).agg({"label": "first", "summary": "first"}).reset_index()

    # build a dict map for O(1) lookups
    clusters_map = {}
    for _, row in df_unique.iterrows():
        cid = str(row["cluster_id"])
        clusters_map[cid] = {
            "label": row.get("label", "") if not pd.isna(row.get("label", "")) else "",
            "summary": row.get("summary", "") if not pd.isna(row.get("summary", "")) else ""
        }
    return clusters_map


@st.cache_data
def load_json_labels(path: str) -> Dict[str, str]:
    """Load JSON mapping and clean names (strip stars and trailing slashes)."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    # JSON may have nested structure; try to find 'subcluster_label' key. Otherwise, assume flat mapping.
    if isinstance(j, dict) and "subcluster_label" in j and isinstance(j["subcluster_label"], dict):
        raw_map = j["subcluster_label"]
    else:
        # fallback: assume top-level mapping
        raw_map = {}
        for k, v in j.items():
            if isinstance(v, str):
                raw_map[k] = v
    def clean_name(s: str) -> str:
        if not isinstance(s, str):
            return str(s)
        s = s.replace("\n", " ").strip()
        # remove leading stars and spaces
        while s.startswith("*"):
            s = s.lstrip("* ").strip()
        # remove trailing slashes and spaces
        s = s.rstrip("/ ").strip()
        # normalize multi-spaces
        s = " ".join(s.split())
        return s
    cleaned = {}
    for k, v in raw_map.items():
        cleaned[str(k).strip()] = clean_name(v)
    return cleaned

def build_parent_prefixes(cluster_ids: List[str]) -> List[str]:
    """From list of cluster ids like 'root.1.1.2' build all prefixes (root, root.1, root.1.1, ...)"""
    s = set()
    for cid in cluster_ids:
        if not cid:
            continue
        parts = cid.split(".")
        for i in range(1, len(parts) + 1):
            s.add(".".join(parts[:i]))
    return sorted(list(s), key=lambda x: (x.count("."), x))

def get_children_keys(selected_key: str, all_keys: List[str]) -> List[str]:
    """Return immediate child keys under selected_key (one level deeper)."""
    children = set()
    prefix = selected_key + "."
    for k in all_keys:
        if k.startswith(prefix):
            rest = k[len(prefix):]
            first_seg = rest.split(".")[0]
            child_key = prefix + first_seg
            children.add(child_key)
    return sorted(children)

def clean_cluster_path_display(cluster_path_str: str) -> List[str]:
    """
    Convert cluster_path field from email CSV like:
    '** Admissions & A / ** Sub / ** Leaf'
    into a list of cleaned names ['Admissions & A', 'Sub', 'Leaf'] (strip '**' and '/')
    """
    if not isinstance(cluster_path_str, str):
        return []
    # split by '/' and clean each segment
    parts = [p.strip() for p in cluster_path_str.split("/") if p.strip() != ""]
    cleaned = []
    for p in parts:
        # remove leading stars and spaces
        s = p
        while s.startswith("*"):
            s = s.lstrip("* ").strip()
        # remove trailing slashes/spaces
        s = s.rstrip("/ ").strip()
        if s:
            cleaned.append(" ".join(s.split()))
    return cleaned



# --------------------------
# Load datasets
# --------------------------
if not os.path.exists(MASTER_CSV_PATH):
    st.error(f"Master CSV not found at: {MASTER_CSV_PATH}")
    st.stop()
if not os.path.exists(CLUSTERS_CSV_PATH):
    st.warning(f"Clusters CSV not found at: {CLUSTERS_CSV_PATH} — summaries will be empty.")
if not os.path.exists(CLUSTER_JSON_PATH):
    st.error(f"Cluster JSON not found at: {CLUSTER_JSON_PATH}")
    st.stop()

emails = load_master(MASTER_CSV_PATH)
clusters_map = load_clusters_csv_as_map(CLUSTERS_CSV_PATH)
json_labels = load_json_labels(CLUSTER_JSON_PATH)

# Build authoritative cluster list from JSON keys; JSON contains cluster_id -> name
all_cluster_keys = sorted(list(json_labels.keys()), key=lambda x: (x.count("."), x))
# Build mapping name -> key; if duplicate names exist, we keep list of keys
name_to_keys = {}
for k in all_cluster_keys:
    nm = json_labels.get(k, k)
    name_to_keys.setdefault(nm, []).append(k)

# Dropdown list (names only) — sort alphabetically for user friendliness
cluster_names_sorted = sorted(list(name_to_keys.keys()))

# --------------------------
# Page layout (main only)
# --------------------------
st.markdown("### Choose a cluster (names come from JSON mapping)")
selected_name = st.selectbox("Select cluster (by name)", options=["(none)"] + cluster_names_sorted)

if selected_name == "(none)":
    st.info("Select a cluster from the dropdown above to view details.")
    st.stop()

# Resolve the selected cluster key(s). If multiple keys share same name, choose the first (shouldn't usually happen).
selected_keys = name_to_keys.get(selected_name, [])
selected_key = selected_keys[0] if selected_keys else None

# Display cluster name (clean)
st.markdown(f"## {selected_name}")

# Get summary from clusters_df by matching cluster_id == selected_key
cluster_summary = ""
cluster_label_override = ""
if selected_key and selected_key in clusters_map:
    cluster_label_override = clusters_map[selected_key].get("label", "")
    cluster_summary = clusters_map[selected_key].get("summary", "")
else:
    # As fallback, try to match cluster label in clusters_df by label text matching cleaned name
    # (some data had labels starting with '** name' - we cleaned them already)
    matches = [v for k, v in clusters_map.items() if v.get("label") == selected_name]
    if matches:
        cluster_summary = matches[0].get("summary", "")
        cluster_label_override = matches[0].get("label", "")

# Show summary
if cluster_summary:
    st.markdown(f"**Summary:** {cluster_summary}")
else:
    st.markdown("**Summary:** (no summary found in clusters CSV for this cluster)")

# ------------- total mails in selected cluster -------------
# Prefer to match emails by 'cluster_id' column if present and populated
total_count = 0
subtree_emails = pd.DataFrame()
if "cluster_id" in emails.columns and emails["cluster_id"].notna().sum() > 0 and emails["cluster_id"].astype(bool).any():
    # match any email whose cluster_id equals selected_key or startswith selected_key (so parent selection includes child leaves)
    subtree_emails = emails[emails["cluster_id"].str.startswith(selected_key)]
    total_count = subtree_emails.shape[0]
else:
    # fallback: use cluster_path text matching (clean it and check if selected name appears in any segment)
    def path_contains_selected(row):
        cp = row.get("cluster_path", "")
        parts = clean_cluster_path_display(cp)
        # exact match to selected name in any segment
        return any(part == selected_name for part in parts)
    mask = emails.apply(path_contains_selected, axis=1)
    subtree_emails = emails[mask]
    total_count = subtree_emails.shape[0]

st.markdown(f"**Total mails in selected cluster/subtree:** {total_count}")

# ------------- Selected cluster hierarchy (breadcrumb) -------------
# Build breadcrumb from selected_key by splitting prefixes and mapping to JSON names
breadcrumb_names = []
if selected_key:
    parts = selected_key.split(".")
    prefixes = [".".join(parts[:i+1]) for i in range(len(parts))]
    for p in prefixes:
        name = json_labels.get(p, p)  # JSON authoritative; fallback to key
        # clean name (should already be cleaned)
        breadcrumb_names.append(name)
if breadcrumb_names:
    st.markdown("**Hierarchy:** " + "  →  ".join(breadcrumb_names))
else:
    st.markdown("**Hierarchy:** (no hierarchy available)")

# ------------- Children -------------
children_keys = get_children_keys(selected_key, all_cluster_keys) if selected_key else []
children_names = [json_labels.get(k, k) for k in children_keys]
if children_names:
    st.markdown("**Direct child clusters:** " + ", ".join(children_names))
else:
    st.markdown("**Direct child clusters:** (none)")

# ------------- Number of mails to show control -------------
num_to_show = st.number_input("Number of mails to show", min_value=1, max_value=200, value=10, step=1)

# ------------- Show table of mails (date, name, email, body) -------------
if subtree_emails.shape[0] == 0:
    st.info("No emails found for this selection.")
else:
    # Sort by date descending
    subtree_emails_sorted = subtree_emails.sort_values("date", ascending=False)
    # Show up to num_to_show rows
    display_df = subtree_emails_sorted.head(num_to_show)[["date", "sender_name", "sender_email", "body_clean"]].copy()
    display_df = display_df.rename(columns={
        "date": "Date",
        "sender_name": "Name",
        "sender_email": "Email",
        "body_clean": "Body (snippet)"
    })
    # show snippet (first 300 chars) for body column for compactness
    def snippet(text, n=300):
        if not isinstance(text, str):
            return ""
        s = " ".join(text.split())
        return s if len(s) <= n else s[:n].rsplit(" ", 1)[0] + "..."
    display_df["Body (snippet)"] = display_df["Body (snippet)"].apply(lambda t: snippet(t, 300))
    st.write(display_df.reset_index(drop=True))

    # allow user to pick one row to view full body
    indices = subtree_emails_sorted.head(num_to_show).index.astype(str).tolist()
    idx_choice = st.selectbox("Open full email (select index)", options=["(none)"] + indices)
    if idx_choice != "(none)":
        idx = int(idx_choice)
        row = subtree_emails_sorted.loc[idx]
        st.markdown("---")
        st.markdown(f"**Date:** {row['date']}")
        st.markdown(f"**From:** {row.get('sender_name','')}  <{row.get('sender_email','')}>")
        st.markdown("**Full body:**")
        st.write(row.get("body_clean", ""))

# ------------- Downloads -------------
if subtree_emails.shape[0] > 0:
    st.download_button("Download shown mails CSV", data=subtree_emails_sorted.head(num_to_show).to_csv(index=False).encode("utf-8"),
                       file_name=f"{selected_key.replace('.','_')}_sample.csv")
    st.download_button("Download all subtree mails CSV", data=subtree_emails_sorted.to_csv(index=False).encode("utf-8"),
                       file_name=f"{selected_key.replace('.','_')}_all.csv")







# Footer note
st.markdown("---")
st.caption("Notes: Cluster names shown in the dropdown come from the JSON mapping. Cluster summaries are fetched from the clusters CSV (matching cluster_id). Matching emails uses 'cluster_id' in the email CSV if present; otherwise it falls back to matching cleaned cluster_path segments. If names don't match exactly, consider ensuring JSON keys match the cluster_id values used in emails.")
