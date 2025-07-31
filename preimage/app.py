import streamlit as st
import os
import random
import time
import re
import pickle
from copy import deepcopy
from glob import glob
from faiss import read_index
import faiss
import numpy as np
from pathlib import Path
import plotly.graph_objects as go
import sys
import itertools
from utils import *
from transformers import GPT2Tokenizer

st.set_page_config(layout="wide")

for key in ["sel_act_site", "sel_subspace"]:
    if key in st.session_state:
        st.session_state[key] = st.session_state[key]

def change_key_value(key, options, increment):
    next_i = (options.index(st.session_state[key]) + increment) % len(options)
    st.session_state[key] = options[next_i]

def process_special_tokens(tokens):
    special_tokens = {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;", " ": "&nbsp;"}   # now replace whole word, might need to replace every occurrence in each token
    # return ["&nbsp;" if t == " " else t for t in tokens]
    return [special_tokens.get(t, t.replace("\n", "\\n")) for t in tokens]


@st.cache_data
def get_gpt2_tokenizer():   # used for str token processing, cannot be fasttokenizer
    return GPT2Tokenizer.from_pretrained("gpt2") 

@st.cache_data
def get_built_index(path) -> faiss.IndexFlat:
    return read_index(path)

@st.cache_data
def get_cached_input(path):
    with open(os.path.join(path, "str_tokens.pkl"), "rb") as f:
        cached_input = pickle.load(f)
    seq_lens = [len(seq) for seq in cached_input]
    seq_edges = list(itertools.accumulate(seq_lens, initial=0))
    return cached_input, seq_edges

@st.cache_data
def get_norms(path):
    return np.load(os.path.join(path, "norms.npy"))
    
@st.cache_data
def search_for_idx(_index: faiss.IndexFlat, sel_subspace, act_idx, threshold):
    query_act = np.empty((_index.d,), dtype=np.float32)
    _index.reconstruct(act_idx, query_act)

    if isinstance(_index, faiss.IndexFlatIP):
        _, D, I = _index.range_search(query_act[None, :], threshold)

    else:
        norm = np.linalg.norm(query_act)
        threshold = (norm * threshold) ** 2
        _, D, I = _index.range_search(query_act[None, :], threshold)
        D = np.sqrt(D) / norm

    temp_mask = I != act_idx
    I = I[temp_mask]
    D = D[temp_mask]

    return D, I

@st.cache_data
def make_histogram(_index: faiss.IndexFlat, sel_subspace, act_idx):
    cosine = isinstance(_index, faiss.IndexFlatIP)
    D, _ = search_for_idx(_index, sel_subspace, act_idx, -1.0 if cosine else 100000)
    D = D[np.random.permutation(D.shape[0])[:10000]]
    counts, bin_edges = np.histogram(D, bins=100, range=(-1.0, 1.0) if cosine else (0.0, 3.0))
    fig = go.Figure(go.Bar(x=bin_edges[:-1], y=counts, width=np.diff(bin_edges)))
    fig.update_layout(title=f"Histogram of {'sim' if cosine else 'dist'} given query act {act_idx} (mean {np.mean(D).item():.2f})",
                       width=600, height=300,
                    )
    return fig


def click_random(length):
    st.session_state.sel_example_idx = random.randint(0, length-1)

def span_maker(token: str, mark: bool = False):
    if mark:
        return '<span style="background-color:rgb(238,75,43); color: white">' + token + '</span>' 
    else:
        return '<span>' + token + '</span>' 


st.session_state.exp_name = sys.argv[1]

index_path = Path("..") / "visualizations" / f"index-{st.session_state.exp_name}"
model_name_site_name = [d.split("-") for d in os.listdir(index_path)]
if len(set(n[0] for n in model_name_site_name)) == 1:
    model_name = model_name_site_name[0][0]
    all_sites = sorted([n[1] for n in model_name_site_name])
else:
    raise NotImplementedError("folder contains more than one model")

cached_input, seq_edges = get_cached_input(Path("..") / "visualizations" / f"shared_acts-{model_name}")

with st.sidebar:
    # st.text(st.session_state.exp_name)

    if "sel_act_site" not in st.session_state:
        st.session_state.sel_act_site = all_sites[0]
    sel_act_site = st.selectbox("choose act site", all_sites, key="sel_act_site")

    norms = get_norms(index_path / f"{model_name}-{sel_act_site}")

    subspaces = glob("*.index", root_dir=index_path / f"{model_name}-{sel_act_site}")
    subspaces.sort(key=lambda x: int(x.split("-")[0]))
    subspaces = [s[:-6] for s in subspaces]


    if "sel_subspace" not in st.session_state or st.session_state.sel_subspace not in subspaces:
        st.session_state.sel_subspace = subspaces[0]
    sel_subspace = st.selectbox("choose subspace", subspaces, key="sel_subspace")
    cols = st.columns(2)
    cols[0].button("prev", on_click=change_key_value, args=("sel_subspace", subspaces, -1))
    cols[1].button("next", on_click=change_key_value, args=("sel_subspace", subspaces, 1))

    index = get_built_index(str(index_path / f"{model_name}-{sel_act_site}" / f"{sel_subspace}.index"))
    cosine = isinstance(index, faiss.IndexFlatIP)
    
    if ("sel_example_idx" not in st.session_state) or st.session_state.sel_example_idx >= index.ntotal:
        st.session_state.sel_example_idx = random.randint(0, index.ntotal-1)
    act_idx = st.number_input("query act idx", min_value=0, max_value=index.ntotal-1, value=st.session_state.sel_example_idx)
    st.button("random example", type="primary", on_click=click_random, args=(index.ntotal,))    # can choose the first half

    seq_idx, pos_idx = locate_str_tokens(act_idx, seq_edges)
    str_tokens = cached_input[seq_idx]


    if cosine:
        threshold = st.number_input("set threshold", min_value=-1.0, max_value=0.999, value=0.85)
    else:
        threshold = st.number_input("set threshold", min_value=0.001, max_value=1.0, value=0.25)
    num_show = st.slider("num sample shown", min_value=1, max_value=100, value=20)
    prev_ctx = st.select_slider("# prev context per sample", ["10", "25", "100", "inf"], value="10")
    futr_ctx = st.select_slider("# future context per sample", ["0", "5", "25", "inf"], value="5")

    show_norm = st.toggle("show norm")
    show_hist = st.toggle("show histogram")
    readable_tokens = st.toggle("more readable tokens")
    show_explanation = st.toggle("show explanation", value=True)

if show_hist:
    fig = make_histogram(index, model_name+sel_act_site+sel_subspace, act_idx)
    st.plotly_chart(fig, use_container_width=False)

st.markdown("##### Query Activation")

seq_len = len(str_tokens)
if prev_ctx != "inf":
    span_s = max(0, pos_idx - int(prev_ctx))
else:
    span_s = 0
if futr_ctx != "inf":
    span_e = min(seq_len, pos_idx + 1 + int(futr_ctx))
else:
    span_e = seq_len
str_tokens = str_tokens[span_s: span_e]
if readable_tokens:
    str_tokens = more_readable_gpt2_tokens(str_tokens, get_gpt2_tokenizer())
else:
    str_tokens = list(map(lambda x: x.replace("Ġ", " "), str_tokens))  # if x != "<|endoftext|>" else "[bos]"
print(str_tokens)
str_tokens = process_special_tokens(str_tokens)

if show_norm:
    norm = norms[act_idx, int(sel_subspace.split("-")[0])].item()
    norm = f"norm: {norm:.1f} "
else:
    norm = ""
if cosine:
    shown_value = f"<span><i> sim: {1.0:.3f} ; {norm}pos: {pos_idx} ; </i></span>"
else:
    shown_value = f"<span><i> dist: {0.0:.3f} ; {norm}pos: {pos_idx} ; </i></span>"
spans = [span_maker(t, i==pos_idx-span_s) for i, t in enumerate(str_tokens)]
row = f'<div>' + shown_value + "".join(spans) + '</div>'
st.markdown(row, unsafe_allow_html=True)
st.text("")


D, I = search_for_idx(index, model_name+sel_act_site+sel_subspace, act_idx, threshold)
temp_idx = np.random.permutation(D.shape[0])[:num_show]
D = D[temp_idx].tolist()
I = I[temp_idx].tolist()

st.markdown(f"##### Subspace {sel_subspace.split('-')[0]} Preimage")
for sim, input_idx in sorted(zip(D, I), key=lambda x: x[0], reverse=cosine):
    seq_idx, pos_idx = locate_str_tokens(input_idx, seq_edges)
    str_tokens = cached_input[seq_idx]

    if show_norm:
        norm = norms[input_idx, int(sel_subspace.split("-")[0])].item()
        norm = f"norm: {norm:.1f} "
    else:
        norm = ""

    seq_len = len(str_tokens)
    if prev_ctx != "inf":
        span_s = max(0, pos_idx - int(prev_ctx))
    else:
        span_s = 0
    if futr_ctx != "inf":
        span_e = min(seq_len, pos_idx + 1 + int(futr_ctx))
    else:
        span_e = seq_len
    str_tokens = str_tokens[span_s: span_e]
    if readable_tokens:
        str_tokens = more_readable_gpt2_tokens(str_tokens, get_gpt2_tokenizer())
    else:
        str_tokens = list(map(lambda x: x.replace("Ġ", " "), str_tokens))  # if x != "<|endoftext|>" else "[bos]"
    str_tokens = process_special_tokens(str_tokens)
    
    shown_value = f"<span><i> {'sim' if cosine else 'dist'}: {sim:.3f} ; {norm}pos: {pos_idx} ; </i></span>"
    spans = [span_maker(t, i==pos_idx-span_s) for i, t in enumerate(str_tokens)]
    row = f'<div style="margin-bottom: 8px;">' + shown_value + "".join(spans) + '</div>'
    st.markdown(row, unsafe_allow_html=True)

if show_explanation:
    st.write("#####")
    st.caption("""_Token in red_ is the token position where the activation is taken from. Activations are projected to selected subspace (e.g., 2-96 means subspace 2 of dimension 96),
               and then compute cosine similarity (_column "sim"_) with the query activation (its corresponding context is shown on the top).
               Position index of the token in red is also shown (_column "pos"_). You may also check the _norm_ of the projected or subspace activation by the toggle "show norm". 
               A surrounding context window of each token in red is shown, you can adjust the window size on the left. 
               Note that future tokens (tokens after the red one) are always not part of the computation, they are invisible for the model. You can move the red token by changing "query act idx" on the left.
               If you want IOI and Greater-than examples, set act idx < around 30000. You can change layer by choose a different _"act site"_, for example, "x4.post" means post MLP residual stream after layer 4.
               Show histogram will show the _histogram_ of cosine similarity values (Given a query activation, compute its similarity with all other activations).""")