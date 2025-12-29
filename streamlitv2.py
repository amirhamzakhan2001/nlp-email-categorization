# streamlitv2.py
"""
Streamlit app — Cluster viewer (JSON names authoritative)

Features:
 - Structural dendrogram built from JSON cluster keys/labels (no embeddings required for structure).
 - Node counts computed from master CSV (leaf counts from each email's cluster_id; parent counts aggregated).
 - Uses precomputed 'embeddings' column (All-MiniLM) to compute cluster centroids (cached).
 - Runs UMAP on cluster centroids for 2D visualization. Tries GPU (cuML) then umap-learn, then PCA fallback.
 - Renders Plotly structural tree and UMAP scatter (centroids annotated with counts).
 - Heavy computation sits behind an expander; parse/cache steps use @st.cache_data.
"""

import os
import json
import ast
import math
import traceback
from typing import Dict, List, Tuple

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

# plotting & clustering utils
from scipy.cluster.hierarchy import dendrogram  # we won't compute linkage for structure, but keep available
import plotly.graph_objects as go
from sklearn.decomposition import PCA

# optional: UMAP CPU
_HAS_UMAP = False
try:
    import umap as umap_learn
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False

# try GPU cuml import helper (we'll attempt inside compute_umap)
def try_import_cuml_umap():
    try:
        from cuml import UMAP as cumlUMAP  # type: ignore
        # quick check for GPU presence handled by import; if no GPU, import may still succeed but operations will fail
        return cumlUMAP
    except Exception:
        return None

# --------------------------
# Config (read from env)
# --------------------------
MASTER_CSV_PATH = os.environ.get("MASTER_CSV_PATH", r"/Users/amirhamzakhan/Desktop/Amir_lab/ml_lab/NLP/email_data.csv")
CLUSTERS_CSV_PATH = os.environ.get("CLUSTERS_CSV_PATH", r"/Users/amirhamzakhan/Desktop/Amir_lab/ml_lab/NLP/cluster_labels.csv")
CLUSTER_JSON_PATH = os.environ.get("CLUSTER_JSON_PATH", r"/Users/amirhamzakhan/Desktop/Amir_lab/ml_lab/NLP/api_cluster.json")

st.set_page_config(page_title="Cluster Viewer (Hierarchy + UMAP centroids)", layout="wide")
st.title("Cluster Viewer — structural dendrogram + UMAP of cluster centroids")

# --------------------------
# I/O helpers
# --------------------------
@st.cache_data
def load_master(path: str) -> pd.DataFrame:
    """Load master CSV/parquet and normalize expected columns."""
    if path.lower().endswith(".parquet"):
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # normalize names
    col_map = {}
    if "name" in df.columns and "sender_name" not in df.columns:
        col_map["name"] = "sender_name"
    if "email" in df.columns and "sender_email" not in df.columns:
        col_map["email"] = "sender_email"
    if "body" in df.columns and "body_clean" not in df.columns:
        col_map["body"] = "body_clean"
    if col_map:
        df = df.rename(columns=col_map)
    for c in ["date", "sender_name", "sender_email", "body_clean", "cluster_id", "cluster_path", "cluster_leaf_label", "embeddings"]:
        if c not in df.columns:
            df[c] = None
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["cluster_path"] = df["cluster_path"].fillna("").astype(str)
    df["cluster_id"] = df["cluster_id"].fillna("").astype(str)
    df["body_clean"] = df["body_clean"].fillna("").astype(str)
    df["sender_name"] = df["sender_name"].fillna("").astype(str)
    df["sender_email"] = df["sender_email"].fillna("").astype(str)
    return df

@st.cache_data
def load_clusters_csv_as_map(path: str) -> Dict[str, Dict[str,str]]:
    """Load clusters CSV and return dict cluster_id -> {label, summary}"""
    if not os.path.exists(path):
        return {}
    df = pd.read_csv(path, dtype=str)
    for c in ["cluster_id", "label", "summary"]:
        if c not in df.columns:
            df[c] = ""
    def clean_text(x):
        if not isinstance(x, str):
            return ""
        s = x.strip()
        while s.startswith("*"):
            s = s.lstrip("* ").strip()
        s = s.rstrip("/ ").strip()
        s = " ".join(s.split())
        return s
    df["cluster_id"] = df["cluster_id"].astype(str).str.strip()
    df["label"] = df["label"].fillna("").apply(clean_text)
    df["summary"] = df["summary"].fillna("").apply(clean_text)
    df_unique = df.groupby("cluster_id", sort=False).agg({"label":"first","summary":"first"}).reset_index()
    clusters_map = {}
    for _, row in df_unique.iterrows():
        cid = str(row["cluster_id"])
        clusters_map[cid] = {"label": row.get("label","") or "", "summary": row.get("summary","") or ""}
    return clusters_map

@st.cache_data
def load_json_labels(path: str) -> Dict[str,str]:
    """Load JSON mapping cluster_id -> label (cleaned)."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        j = json.load(f)
    if isinstance(j, dict) and "subcluster_label" in j and isinstance(j["subcluster_label"], dict):
        raw_map = j["subcluster_label"]
    else:
        raw_map = {}
        for k, v in j.items():
            if isinstance(v, str):
                raw_map[k] = v
    def clean_name(s):
        if not isinstance(s, str):
            return str(s)
        s = s.replace("\n"," ").strip()
        while s.startswith("*"):
            s = s.lstrip("* ").strip()
        s = s.rstrip("/ ").strip()
        s = " ".join(s.split())
        return s
    cleaned = {}
    for k, v in raw_map.items():
        cleaned[str(k).strip()] = clean_name(v)
    return cleaned

# --------------------------
# small utilities
# --------------------------
def clean_cluster_path_display(cluster_path_str: str) -> List[str]:
    if not isinstance(cluster_path_str, str):
        return []
    parts = [p.strip() for p in cluster_path_str.split("/") if p.strip() != ""]
    cleaned = []
    for p in parts:
        s = p
        while s.startswith("*"):
            s = s.lstrip("* ").strip()
        s = s.rstrip("/").strip()
        if s:
            cleaned.append(" ".join(s.split()))
    return cleaned

def parse_embedding_cell(cell) -> np.ndarray:
    """Parse embeddings stored as Python list string, JSON array, or list/np.array. Return np.array or None."""
    if cell is None:
        return None
    if isinstance(cell, (list, tuple, np.ndarray)):
        try:
            return np.asarray(cell, dtype=np.float32)
        except Exception:
            return None
    if isinstance(cell, str):
        s = cell.strip()
        if s == "":
            return None
        try:
            val = ast.literal_eval(s)
            return np.asarray(val, dtype=np.float32)
        except Exception:
            try:
                val = json.loads(s)
                return np.asarray(val, dtype=np.float32)
            except Exception:
                try:
                    parts = [float(x) for x in s.replace("[","").replace("]","").split(",") if x.strip()!=""]
                    return np.asarray(parts, dtype=np.float32)
                except Exception:
                    return None
    return None

# --------------------------
# Tree building & aggregation
# --------------------------
def build_tree_from_json_labels(json_labels: Dict[str,str]) -> Tuple[Dict[str,List[str]], Dict[str, str], List[str]]:
    """
    Build children_map, parent_map, ordered_keys list from JSON label mapping that uses dotted keys.
    """
    all_keys = sorted(list(json_labels.keys()), key=lambda x: (x.count("."), x))
    children_map = {}
    parent_map = {}
    for k in all_keys:
        parts = k.split(".")
        if len(parts) == 1:
            parent = None
        else:
            parent = ".".join(parts[:-1])
        parent_map[k] = parent
        if parent is not None:
            children_map.setdefault(parent, []).append(k)
        children_map.setdefault(k, [])  # ensure present
    return children_map, parent_map, all_keys

def aggregate_counts_over_tree(children_map: Dict[str,List[str]], leaf_counts: Dict[str,int]) -> Dict[str,int]:
    """
    Post-order sum of counts. leaf_counts keys match leaf cluster ids (but may also contain intermediate nodes).
    Returns node_counts map for all nodes in children_map.
    """
    node_counts = {}
    nodes_sorted = sorted(list(children_map.keys()), key=lambda x: x.count("."), reverse=True)
    for node in nodes_sorted:
        kids = children_map.get(node, [])
        if not kids:
            node_counts[node] = int(leaf_counts.get(node, 0))
        else:
            s = 0
            for c in kids:
                s += int(node_counts.get(c, 0))
            s += int(leaf_counts.get(node, 0))
            node_counts[node] = s
    return node_counts

# --------------------------
# Embedding parsing & centroid computation (cached)
# --------------------------
@st.cache_data(ttl=3600)
def parse_all_embeddings(series: pd.Series) -> List:
    """Parse all embeddings in a pandas Series and return list (np.array or None)."""
    out = []
    for v in series.fillna("").tolist():
        out.append(parse_embedding_cell(v))
    return out

@st.cache_data(ttl=3600)
def compute_cluster_centroids_from_parsed(index_list: List[int], parsed_embeddings: List, cluster_ids_list: List[str]):
    """
    Given parsed embeddings aligned to index_list and cluster_ids_list (same ordering),
    compute centroid per cluster key and counts. Returns centroids_map and counts_map.
    """
    groups = {}
    counts = {}
    for idx, emb, cid in zip(index_list, parsed_embeddings, cluster_ids_list):
        if emb is None:
            continue
        groups.setdefault(cid, []).append(emb.astype(np.float32))
    centroids_map = {}
    counts_map = {}
    for cid, arr_list in groups.items():
        arr = np.vstack(arr_list).astype(np.float32)
        centroid = arr.mean(axis=0)
        norm = np.linalg.norm(centroid)
        if norm > 0:
            centroid = centroid / norm
        centroids_map[cid] = centroid.astype(np.float32)
        counts_map[cid] = arr.shape[0]
    return centroids_map, counts_map

# --------------------------
# UMAP runner (GPU-friendly if cuML present)
# --------------------------
@st.cache_data(ttl=3600)
def compute_umap(emb_matrix: np.ndarray, n_neighbors=15, min_dist=0.1, random_state=42, use_gpu_preference=True):
    """
    Try GPU cuML UMAP first (if available), otherwise umap-learn CPU, otherwise PCA fallback.
    Returns (coords (k,2), info dict).
    """
    emb = np.asarray(emb_matrix, dtype=np.float32)
    info = {"backend": None}
    if use_gpu_preference:
        CumlUMAP = try_import_cuml_umap()
        if CumlUMAP is not None:
            try:
                um = CumlUMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
                transformed = um.fit_transform(emb)
                try:
                    transformed = np.asarray(transformed)
                except Exception:
                    pass
                info["backend"] = "cuml_gpu"
                return transformed, info
            except Exception:
                traceback.print_exc()
                info["backend"] = "cuml_failed"
    # CPU umap-learn
    if _HAS_UMAP:
        try:
            reducer = umap_learn.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist, random_state=random_state)
            transformed = reducer.fit_transform(emb)
            info["backend"] = "umap_cpu"
            return transformed, info
        except Exception:
            traceback.print_exc()
            info["backend"] = "umap_failed"
    # fallback PCA
    transformed = PCA(n_components=2, random_state=random_state).fit_transform(emb)
    info["backend"] = "pca_cpu"
    return transformed, info

# --------------------------
# Plotting a deterministic tree layout (from explicit tree)
# --------------------------
def build_plotly_tree_from_json(children_map: Dict[str,List[str]], json_labels: Dict[str,str], node_counts: Dict[str,int], root_nodes: List[str]=None):
    """
    Build a deterministic left-to-right tree plot using depth as x and leaf-order as y.
    Returns Plotly Figure.
    """
    # choose roots (top-level nodes with no dots) if not provided
    if root_nodes is None:
        root_nodes = [k for k in children_map.keys() if '.' not in k]
        if not root_nodes:
            # fallback: nodes with no parent
            parent_of = {c for kids in children_map.values() for c in kids}
            root_nodes = [k for k in children_map.keys() if k not in parent_of]
            if not root_nodes:
                root_nodes = sorted(children_map.keys())[:1]

    positions = {}  # node -> (depth, y)
    y_counter = 0

    def dfs(node, depth):
        nonlocal y_counter
        kids = children_map.get(node, [])
        if not kids:
            positions[node] = (depth, float(y_counter))
            y_counter += 1
            return [positions[node][1]]
        child_ys = []
        for c in kids:
            ys = dfs(c, depth+1)
            child_ys.extend(ys)
        # parent y is mean of children
        positions[node] = (depth, float(sum(child_ys)/len(child_ys)))
        return child_ys

    for r in sorted(root_nodes):
        dfs(r, 0)

    fig = go.Figure()
    # lines parent->child
    for parent, kids in children_map.items():
        if parent not in positions:
            continue
        px, py = positions[parent]
        for c in kids:
            if c not in positions:
                continue
            cx, cy = positions[c]
            fig.add_trace(go.Scatter(x=[px, cx], y=[py, cy], mode='lines', line=dict(color='gray', width=1), hoverinfo='none', showlegend=False))

    # nodes with labels and sizes
    xs, ys, texts, sizes = [], [], [], []
    for node, (xpos, ypos) in positions.items():
        xs.append(xpos)
        ys.append(ypos)
        display_name = json_labels.get(node, node)
        cnt = node_counts.get(node, 0)
        texts.append(f"{display_name} (n={cnt})")
        sizes.append(max(6, 6 + math.log1p(cnt)))  # size scaled by count

    fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers+text', marker=dict(size=sizes), text=texts, textposition='right', hoverinfo='text'))
    fig.update_layout(xaxis=dict(title="depth", showgrid=False), yaxis=dict(showticklabels=False), height=700, margin=dict(l=20, r=20, t=30, b=20))
    return fig

# --------------------------
# Load data (main)
# --------------------------
if not os.path.exists(MASTER_CSV_PATH):
    st.error(f"Master CSV not found at: {MASTER_CSV_PATH}")
    st.stop()
if not os.path.exists(CLUSTER_JSON_PATH):
    st.error(f"Cluster JSON not found at: {CLUSTER_JSON_PATH}")
    st.stop()
if not os.path.exists(CLUSTERS_CSV_PATH):
    st.warning(f"Clusters CSV not found at: {CLUSTERS_CSV_PATH} — summaries may be empty.")

with st.spinner("Loading data..."):
    emails = load_master(MASTER_CSV_PATH)
    clusters_map = load_clusters_csv_as_map(CLUSTERS_CSV_PATH)
    json_labels = load_json_labels(CLUSTER_JSON_PATH)

# UI: cluster selection same as previous
all_cluster_keys = sorted(list(json_labels.keys()), key=lambda x: (x.count("."), x))
name_to_keys = {}
for k in all_cluster_keys:
    nm = json_labels.get(k, k)
    name_to_keys.setdefault(nm, []).append(k)
cluster_names_sorted = sorted(list(name_to_keys.keys()))
st.markdown("### Choose a cluster (tree names come from JSON mapping)")
selected_name = st.selectbox("Select cluster (by name)", options=["(none)"] + cluster_names_sorted)
if selected_name == "(none)":
    st.info("Select a cluster from the dropdown above to view details.")
    st.stop()
selected_keys = name_to_keys.get(selected_name, [])
selected_key = selected_keys[0] if selected_keys else None
st.markdown(f"## {selected_name}")

# show summary from clusters CSV if available
cluster_summary = ""
if selected_key and selected_key in clusters_map:
    cluster_summary = clusters_map[selected_key].get("summary", "")
if cluster_summary:
    st.markdown(f"**Summary:** {cluster_summary}")
else:
    st.markdown("**Summary:** (no summary found in clusters CSV for this cluster)")

# build subtree_emails (match by cluster_id starting with selected_key)
if "cluster_id" in emails.columns and emails["cluster_id"].astype(bool).any():
    subtree_emails = emails[emails["cluster_id"].str.startswith(selected_key)].copy()
else:
    # fallback using cluster_path segments
    def path_contains_selected(row):
        cp = row.get("cluster_path", "")
        parts = clean_cluster_path_display(cp)
        return any(part == selected_name for part in parts)
    mask = emails.apply(path_contains_selected, axis=1)
    subtree_emails = emails[mask].copy()

st.markdown(f"**Total mails in selected subtree:** {subtree_emails.shape[0]}")

# show sample table
num_to_show = st.number_input("Number of mails to show", min_value=1, max_value=200, value=10, step=1)
if subtree_emails.shape[0] == 0:
    st.info("No emails found for this selection.")
else:
    display_df = subtree_emails.sort_values("date", ascending=False).head(num_to_show)[["date","sender_name","sender_email","body_clean"]].copy()
    display_df = display_df.rename(columns={"date":"Date","sender_name":"Name","sender_email":"Email","body_clean":"Body (snippet)"})
    def snippet(text, n=300):
        if not isinstance(text, str):
            return ""
        s = " ".join(text.split())
        return s if len(s)<=n else s[:n].rsplit(" ",1)[0] + "..."
    display_df["Body (snippet)"] = display_df["Body (snippet)"].apply(lambda t: snippet(t,300))
    st.write(display_df.reset_index(drop=True))

# --------------------------
# Structural dendrogram + UMAP (expander to avoid auto-run)
# --------------------------
st.markdown("### Structural hierarchy (from JSON) & UMAP of cluster centroids")

with st.expander("Load structural dendrogram + UMAP (heavy) — click to compute"):
    # 1) Build tree
    children_map, parent_map, all_nodes = build_tree_from_json_labels(json_labels)

    # 2) Compute leaf counts from subtree_emails (prefer cluster_id)
    if "cluster_id" in subtree_emails.columns and subtree_emails["cluster_id"].astype(bool).any():
        leaf_counts = subtree_emails['cluster_id'].value_counts().to_dict()
    else:
        # fallback parse cluster_path last segment
        def extract_leaf_key(row):
            parts = clean_cluster_path_display(row.get("cluster_path",""))
            return parts[-1] if parts else ""
        leaf_counts = subtree_emails.apply(extract_leaf_key, axis=1).value_counts().to_dict()

    # 3) Aggregate counts
    node_counts = aggregate_counts_over_tree(children_map, leaf_counts)

    st.write(f"Total nodes in JSON tree: {len(children_map)}; clusters with >0 emails in subtree: {sum(1 for v in node_counts.values() if v>0)}")

    # control top-N to include in UMAP/scatter (keeps things fast)
    top_n = st.number_input("Top N clusters to include in UMAP by email count (set higher to include more)", min_value=10, max_value=5000, value=500, step=10)

    # choose clusters with non-zero counts and sorted by count desc
    clusters_with_counts = [(k, node_counts.get(k,0)) for k in children_map.keys()]
    clusters_with_counts = [(k,c) for k,c in clusters_with_counts if c>0]
    clusters_with_counts.sort(key=lambda x: x[1], reverse=True)
    chosen_clusters = [k for k,c in clusters_with_counts[:top_n]]

    st.write(f"Showing top {len(chosen_clusters)} clusters (by count).")

    # 4) Ensure parsed embeddings cached on the main emails df (one-time)
    with st.spinner("Parsing embeddings (cached) — this may take a moment the first time..."):
        if '_emb_parsed' not in emails.columns:
            parsed_all = parse_all_embeddings(emails['embeddings'])
            emails['_emb_parsed'] = parsed_all
        else:
            parsed_all = emails['_emb_parsed']

    # 5) Filter rows belonging to chosen_clusters and build lists for centroid computation
    mask = emails['cluster_id'].isin(chosen_clusters)
    sub_df = emails[mask].copy()
    if sub_df.shape[0] == 0:
        st.info("No emails found for chosen clusters — maybe cluster_id keys do not match JSON keys. Increase Top N or check mappings.")
    else:
        idxs = list(sub_df.index)
        parsed_list = [emails.at[i, '_emb_parsed'] for i in idxs]
        cluster_ids_list = sub_df['cluster_id'].tolist()

        with st.spinner("Computing centroids for chosen clusters (cached)..."):
            centroids_map, counts_map = compute_cluster_centroids_from_parsed(idxs, parsed_list, cluster_ids_list)

        # build arrays for chosen clusters (keep order)
        centroids = []
        labels_for_centroids = []
        counts_for_centroids = []
        present_cluster_keys = []
        for ck in chosen_clusters:
            if ck in centroids_map:
                centroids.append(centroids_map[ck])
                labels_for_centroids.append(json_labels.get(ck, ck))
                counts_for_centroids.append(counts_map.get(ck, 0))
                present_cluster_keys.append(ck)
        if len(centroids) == 0:
            st.info("No centroids computed — check embeddings parsing or cluster_id consistency.")
        else:
            centroids_arr = np.vstack(centroids).astype(np.float32)

            # 6) Compute UMAP on centroids (GPU preferred)
            use_gpu_pref = True
            with st.spinner("Running UMAP on centroids (tries GPU then CPU then PCA)..."):
                umap_coords, umap_info = compute_umap(centroids_arr, n_neighbors=15, min_dist=0.1, random_state=42, use_gpu_preference=use_gpu_pref)

            st.write(f"UMAP backend used: {umap_info.get('backend','unknown')}")

            # 7) Plot structural tree (deterministic positions)
            fig_tree = build_plotly_tree_from_json(children_map, json_labels, node_counts)
            st.plotly_chart(fig_tree, use_container_width=True)

            # 8) Plot UMAP scatter of centroids with parent-child connectors where both parent+child in present_cluster_keys
            # Build coords_map keyed by cluster key
            coords_map = {ck: umap_coords[i] for i, ck in enumerate(present_cluster_keys)}

            fig = go.Figure()
            # draw parent-child centroid connectors if both exist
            for parent, kids in children_map.items():
                if parent not in coords_map:
                    continue
                for c in kids:
                    if c in coords_map:
                        px, py = coords_map[parent]
                        cx, cy = coords_map[c]
                        fig.add_trace(go.Scatter(x=[px, cx], y=[py, cy], mode='lines', line=dict(width=0.6, color='gray'), hoverinfo='none', showlegend=False))
            # scatter points
            xs, ys, hover_texts, sizes = [], [], [], []
            for i, ck in enumerate(present_cluster_keys):
                x, y = umap_coords[i]
                xs.append(x); ys.append(y)
                hover_texts.append(f"{json_labels.get(ck, ck)}<br>key={ck}<br>count={counts_for_centroids[i]}")
                sizes.append(max(6, 6 + math.log1p(counts_for_centroids[i])))
            fig.add_trace(go.Scatter(x=xs, y=ys, mode='markers+text', marker=dict(size=sizes), text=[str(c) for c in counts_for_centroids], textposition='top center', hovertext=hover_texts, hoverinfo='text'))
            fig.update_layout(title=f"UMAP of cluster centroids (top {len(present_cluster_keys)} clusters)", height=700, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.caption("Notes: Structural dendrogram uses JSON hierarchy. UMAP runs on cluster centroids computed from precomputed All-MiniLM embeddings in 'embeddings' column. For per-email UMAP of 60k+ rows, run UMAP offline on GPU and load coords; do not run CPU UMAP live for 60k rows.")
