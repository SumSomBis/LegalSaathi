"""
LegalSaathi — Streamlit app
Hybrid RAG (BM25 + Pinecone/Gemini dense + RRF) + NetworkX graph expansion
+ multi-hop gap detection + Groq generation + hallucination validator.

Run:  streamlit run app.py
"""
import hashlib
import json
import os
import pickle
import re
import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Set, Tuple

import networkx as nx
import streamlit as st
from rank_bm25 import BM25Okapi

st.set_page_config(page_title="LegalSaathi", page_icon="🏛️", layout="wide")

# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
DATA_DIR = os.environ.get("LEGALSAATHI_DATA_DIR", "data")

# (file prefix) — each act has <prefix>_rag_chunks / _graph_nodes / _graph_edges .ndjson
ACTS = [
    ("constitution", "constitution_rag_chunks.ndjson", "constitution_graph_nodes.ndjson", "constitution_graph_edges.ndjson"),
    ("crpc", "crpc_rag_chunks.ndjson", "crpc_graph_nodes.ndjson", "crpc_graph_edges.ndjson"),
    ("ipc", "ipc_rag_chunks.ndjson", "ipc_graph_nodes.ndjson", "ipc_graph_edges.ndjson"),
    ("pocso", "pocso_rag_chunks.ndjson", "pocso_graph_nodes.ndjson", "pocso_graph_edges.ndjson"),
    ("dvact", "dvact_rag_chunks.ndjson", "dvact_graph_nodes.ndjson", "dvact_graph_edges.ndjson"),
    ("orissa_rent", "orissa_rent_rag_chunks.ndjson", "orissa_rent_graph_nodes.ndjson", "orissa_rent_graph_edges.ndjson"),
    ("mpl", "mpl_rag_chunks.ndjson", "mpl_graph_nodes.ndjson", "mpl_graph_edges.ndjson"),
    ("ssc", "ssc_rag_chunks.ndjson", "ssc_graph_nodes.ndjson", "ssc_graph_edges.ndjson"),
    ("sma", "sma_rag_chunks.ndjson", "sma_graph_nodes.ndjson", "sma_graph_edges.ndjson"),
    ("consumer_protection_act", "consumer_protection_act_chunks.ndjson",
     "consumer_protection_act_graph_nodes.ndjson", "consumer_protection_act_graph_edges.ndjson"),
]


class Config:
    EMBED_MODEL = "models/gemini-embedding-001"
    EMBED_DIM = 3072
    LLM_MODEL = "qwen/qwen3.8-27b"
    PINECONE_INDEX = "legal-saathi-gemini"
    PINECONE_NS = "all-acts"
    BM25_TOP_K = 15
    DENSE_TOP_K = 15
    FINAL_TOP_K = 8
    MULTIHOP_ITERS = 3
    MAX_CONTEXT_CHARS = 12_000
    BM25_CACHE = os.path.join(DATA_DIR, "bm25_index.pkl")
    KEY_STATE_FILE = os.path.join(DATA_DIR, "gemini_key_state.json")
    GEMINI_DAILY_LIMIT = int(os.environ.get("GEMINI_DAILY_LIMIT", 1000))  # requests/key/day
    GEMINI_SAFETY_MARGIN = int(os.environ.get("GEMINI_SAFETY_MARGIN", 50))  # switch this early


config = Config()


def get_secret(name: str) -> Optional[str]:
    """Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return st.secrets[name]
    except Exception:
        pass
    return os.environ.get(name)


def get_entity_id(chunk: Dict) -> str:
    return (chunk.get("article_number") or chunk.get("section_number_raw")
            or chunk.get("section_number") or "")


# ──────────────────────────────────────────────────────────────────────────────
# Gemini key rotation (persisted quota tracking)
# ──────────────────────────────────────────────────────────────────────────────
def load_gemini_keys() -> List[str]:
    """GEMINI_API_KEYS may be a TOML list or a comma-separated string.
    Falls back to single GEMINI_API_KEY."""
    raw = get_secret("GEMINI_API_KEYS")
    keys: List[str] = []
    if raw:
        keys = [k.strip() for k in (raw if isinstance(raw, (list, tuple)) else str(raw).split(",")) if str(k).strip()]
    single = get_secret("GEMINI_API_KEY")
    if single and single not in keys:
        keys.append(single)
    return keys


def _key_id(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()[:16]  # never store raw keys


def _quota_day() -> str:
    # Gemini daily quotas reset at midnight Pacific time
    return datetime.now(ZoneInfo("America/Los_Angeles")).strftime("%Y-%m-%d")


class GeminiKeyRotator:
    def __init__(self, keys: List[str]):
        self.keys = keys
        self.lock = threading.Lock()
        self.state = self._load()

    def _load(self) -> Dict:
        try:
            with open(config.KEY_STATE_FILE, "r") as f:
                st_ = json.load(f)
        except Exception:
            st_ = {}
        if st_.get("day") != _quota_day():
            st_ = {"day": _quota_day(), "keys": {}}
        st_.setdefault("keys", {})
        return st_

    def _save(self):
        try:
            os.makedirs(os.path.dirname(config.KEY_STATE_FILE) or ".", exist_ok=True)
            with open(config.KEY_STATE_FILE, "w") as f:
                json.dump(self.state, f)
        except Exception:
            pass

    def _entry(self, key: str) -> Dict:
        return self.state["keys"].setdefault(_key_id(key), {"used": 0, "exhausted": False})

    def _rollover(self):
        if self.state.get("day") != _quota_day():
            self.state = {"day": _quota_day(), "keys": {}}

    def pick(self, exclude: Set[str] = frozenset()) -> Optional[str]:
        """First key under (limit - safety margin) that isn't marked exhausted."""
        with self.lock:
            self._rollover()
            cap = config.GEMINI_DAILY_LIMIT - config.GEMINI_SAFETY_MARGIN
            for k in self.keys:
                e = self._entry(k)
                if k not in exclude and not e["exhausted"] and e["used"] < cap:
                    return k
            return None

    def record(self, key: str, n: int = 1):
        with self.lock:
            self._entry(key)["used"] += n
            self._save()

    def mark_exhausted(self, key: str):
        with self.lock:
            self._entry(key)["exhausted"] = True
            self._save()

    def status(self) -> List[Dict]:
        with self.lock:
            self._rollover()
            return [{"key": f"#{i+1} ({_key_id(k)[:6]})", **self._entry(k)} for i, k in enumerate(self.keys)]


@st.cache_resource(show_spinner=False)
def get_rotator() -> GeminiKeyRotator:
    keys = load_gemini_keys()
    if not keys:
        raise RuntimeError("Missing secrets: GEMINI_API_KEYS (or GEMINI_API_KEY)")
    return GeminiKeyRotator(keys)


@st.cache_resource(show_spinner=False)
def _embed_client_for(key: str):
    from google import genai
    from google.genai import types
    return genai.Client(api_key=key, http_options=types.HttpOptions(api_version="v1"))


# ──────────────────────────────────────────────────────────────────────────────
# Clients
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_clients():
    """Returns (rotator, groq_client, pinecone_index)."""
    from groq import Groq
    from pinecone import Pinecone

    pinecone_key = get_secret("PINECONE_API_KEY")
    groq_key = get_secret("GROQ_API_KEY")
    missing = [n for n, v in [("PINECONE_API_KEY", pinecone_key), ("GROQ_API_KEY", groq_key)] if not v]
    if missing:
        raise RuntimeError("Missing secrets: " + ", ".join(missing))
    rotator = get_rotator()
    return rotator, Groq(api_key=groq_key), Pinecone(api_key=pinecone_key).Index(config.PINECONE_INDEX)


def gemini_embed_query(text: str) -> List[float]:
    """Embed with the current key; on quota errors mark/rotate and retry on the next key."""
    from google.genai import types
    rotator, _, _ = get_clients()
    tried: Set[str] = set()
    last_err = None
    while True:
        key = rotator.pick(exclude=tried)
        if key is None:
            raise RuntimeError(f"All Gemini keys exhausted for today. Last error: {last_err}")
        try:
            resp = _embed_client_for(key).models.embed_content(
                model=config.EMBED_MODEL, contents=[text],
                config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"))
            rotator.record(key)
            return list(resp.embeddings[0].values)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                if "PerDay" in err or "per_day" in err.lower():
                    rotator.mark_exhausted(key)      # out for the day
                tried.add(key)                       # per-minute: just skip it this call
                last_err = e
                continue
            raise


def llm(messages: List[Dict], temperature: float = 0.0, json_mode: bool = False) -> str:
    _, groq_client, _ = get_clients()
    kwargs = dict(model=config.LLM_MODEL, messages=messages, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    for attempt in range(3):
        try:
            return groq_client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(5 * (attempt + 1))
                continue
            raise


# ──────────────────────────────────────────────────────────────────────────────
# Indexer (BM25)
# ──────────────────────────────────────────────────────────────────────────────
def _read_ndjson(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


class LegalIndexer:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.id_to_idx: Dict[str, int] = {}
        self.art_to_ids: Dict[str, List[str]] = {}
        self.skipped: List[str] = []

    def load(self):
        for _, chunk_file, _, _ in ACTS:
            path = os.path.join(DATA_DIR, chunk_file)
            if not os.path.exists(path):
                self.skipped.append(chunk_file)
                continue
            for c in _read_ndjson(path):
                if c.get("is_omitted", False):
                    continue
                self.chunks.append(c)
                self.id_to_idx[c["chunk_id"]] = len(self.chunks) - 1
        for c in self.chunks:
            a = get_entity_id(c)
            if a:
                self.art_to_ids.setdefault(a, []).append(c["chunk_id"])
        self._load_or_build_bm25()

    def _load_or_build_bm25(self):
        n = len(self.chunks)
        if os.path.exists(config.BM25_CACHE):
            try:
                with open(config.BM25_CACHE, "rb") as f:
                    cached_n, bm25 = pickle.load(f)
                if cached_n == n:
                    self.bm25 = bm25
                    return
            except Exception:
                pass
        corpus = []
        for c in self.chunks:
            eid = get_entity_id(c)
            enriched = " ".join(filter(None, [
                c.get("chunk_text", ""), c.get("title", ""),
                " ".join(c.get("keywords", [])),
                " ".join(c.get("legalsaathi_use_cases", [])),
                c.get("source", ""),
                f"Section {eid}" if eid else "",
            ]))
            corpus.append(enriched.lower().split())
        self.bm25 = BM25Okapi(corpus)
        try:
            with open(config.BM25_CACHE, "wb") as f:
                pickle.dump((n, self.bm25), f)
        except Exception:
            pass  # read-only filesystem is fine

    def bm25_search(self, query: str, top_k: int = 15) -> List[Tuple[Dict, float]]:
        scores = self.bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_idx if scores[i] > 0]

    def get_chunk(self, cid: str) -> Optional[Dict]:
        idx = self.id_to_idx.get(cid)
        return self.chunks[idx] if idx is not None else None

    def get_chunks_for_article(self, art: str) -> List[Dict]:
        return [c for c in (self.get_chunk(i) for i in self.art_to_ids.get(art, [])) if c]


# ──────────────────────────────────────────────────────────────────────────────
# Graph
# ──────────────────────────────────────────────────────────────────────────────
class LegalGraphStore:
    def __init__(self):
        self.G = nx.DiGraph()
        self.art_to_node: Dict[str, str] = {}

    def load(self, chunks: List[Dict]):
        for _, _, nodes_file, _ in ACTS:
            p = os.path.join(DATA_DIR, nodes_file)
            if os.path.exists(p):
                for n in _read_ndjson(p):
                    self.G.add_node(n["node_id"], node_type=n["node_type"],
                                    label=n["label"], properties=n.get("properties", {}))
        for _, _, _, edges_file in ACTS:
            p = os.path.join(DATA_DIR, edges_file)
            if os.path.exists(p):
                for e in _read_ndjson(p):
                    if self.G.has_node(e["source"]) and self.G.has_node(e["target"]):
                        self.G.add_edge(e["source"], e["target"], relation=e["relation"])
        self._build_art_map()
        self._add_cross_ref(chunks)
        self._add_same_cat(chunks)

    def _build_art_map(self):
        for nid, data in self.G.nodes(data=True):
            if data.get("node_type") in ("article", "section"):
                props = data.get("properties", {})
                a = props.get("article_number") or props.get("section_number")
                if a:
                    self.art_to_node[a] = nid
                    props.setdefault("use_cases", [])

    def _add_cross_ref(self, chunks):
        for c in chunks:
            sa = get_entity_id(c)
            if not sa or sa not in self.art_to_node:
                continue
            src = self.art_to_node[sa]
            self.G.nodes[src]["properties"]["use_cases"] = c.get("legalsaathi_use_cases", [])
            self.G.nodes[src]["properties"]["article_number"] = sa
            for ref in c.get("cross_references", []):
                if self.G.has_node(ref):
                    tgt = ref
                else:
                    tgt = self.art_to_node.get(ref.replace("Article ", "").replace("article ", "").strip())
                if tgt and not self.G.has_edge(src, tgt):
                    self.G.add_edge(src, tgt, relation="CROSS_REFERENCES")

    def _add_same_cat(self, chunks):
        uc_map: Dict[str, List[str]] = {}
        for c in chunks:
            a = get_entity_id(c)
            if a:
                for uc in c.get("legalsaathi_use_cases", []):
                    uc_map.setdefault(uc, []).append(a)
        for uc, arts in uc_map.items():
            unique = list(set(arts))[:8]
            for i in range(len(unique)):
                for j in range(i + 1, len(unique)):
                    s, t = self.art_to_node.get(unique[i]), self.art_to_node.get(unique[j])
                    if s and t and not self.G.has_edge(s, t):
                        self.G.add_edge(s, t, relation=f"SAME_{uc.upper()}")

    def get_related_articles(self, art: str, depth: int = 2) -> List[str]:
        start = self.art_to_node.get(art)
        if not start or start not in self.G:
            return []
        visited, related = set(), []
        queue = [(start, 0)]
        while queue:
            curr, d = queue.pop(0)
            if curr in visited or d > depth:
                continue
            visited.add(curr)
            data = self.G.nodes[curr]
            if data.get("node_type") in ("article", "section"):
                props = data.get("properties", {})
                a = props.get("article_number") or props.get("section_number")
                if a and a != art:
                    related.append(a)
            for nb in list(self.G.successors(curr)) + list(self.G.predecessors(curr)):
                if nb not in visited:
                    queue.append((nb, d + 1))
        return related


# ──────────────────────────────────────────────────────────────────────────────
# Retrievers
# ──────────────────────────────────────────────────────────────────────────────
class HybridRetriever:
    RRF_K = 60

    def __init__(self, indexer: LegalIndexer):
        self.indexer = indexer

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        top_k = top_k or config.FINAL_TOP_K
        bm25_ranked = [c for c, _ in self.indexer.bm25_search(query, config.BM25_TOP_K)]
        dense_ranked = self._dense(query, config.DENSE_TOP_K)
        return self._rrf(bm25_ranked, dense_ranked)[:top_k]

    def _dense(self, q: str, k: int) -> List[Dict]:
        _, _, pc_index = get_clients()
        try:
            vec = gemini_embed_query(q)
            res = pc_index.query(vector=vec, top_k=k, namespace=config.PINECONE_NS,
                                 include_metadata=True)
        except Exception as e:
            st.session_state.setdefault("warnings", []).append(f"Dense search failed, BM25 only: {e}")
            return []
        out = []
        for m in res.matches:
            c = self.indexer.get_chunk(m.id)
            if c:
                c = c.copy()
                c["_dense_score"] = round(m.score, 4)
                out.append(c)
        return out

    def _rrf(self, a: List[Dict], b: List[Dict]) -> List[Dict]:
        scores, cmap = {}, {}
        for lst in (a, b):
            for rank, c in enumerate(lst, 1):
                cid = c["chunk_id"]
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.RRF_K + rank)
                cmap[cid] = c
        result = []
        for cid in sorted(scores, key=lambda x: scores[x], reverse=True):
            c = cmap[cid].copy()
            c["_rrf_score"] = round(scores[cid], 6)
            c["_source"] = "hybrid"
            result.append(c)
        return result


class GraphRetriever:
    def __init__(self, graph: LegalGraphStore, indexer: LegalIndexer):
        self.graph, self.indexer = graph, indexer

    def expand(self, initial: List[Dict], depth: int = 2, max_add: int = 5) -> List[Dict]:
        seen = {c["chunk_id"] for c in initial}
        extra = []
        for chunk in initial:
            art = get_entity_id(chunk)
            if not art:
                continue
            for ra in self.graph.get_related_articles(art, depth=depth):
                if len(extra) >= max_add:
                    break
                for rc in self.indexer.get_chunks_for_article(ra):
                    if rc["chunk_id"] not in seen:
                        rc = rc.copy()
                        rc["_source"] = "graph"
                        rc["_graph_origin"] = art
                        extra.append(rc)
                        seen.add(rc["chunk_id"])
        return initial + extra


class MultiHopRetriever:
    def __init__(self, hybrid: HybridRetriever):
        self.hybrid = hybrid

    def retrieve(self, query: str, trace: List[str], max_hops: int = None) -> List[Dict]:
        max_hops = max_hops or config.MULTIHOP_ITERS
        all_chunks, seen, curr_q = [], set(), query
        for hop in range(1, max_hops + 1):
            new = self.hybrid.retrieve(curr_q, top_k=5)
            added = 0
            for c in new:
                if c["chunk_id"] not in seen:
                    c = c.copy()
                    c["_hop"] = hop
                    all_chunks.append(c)
                    seen.add(c["chunk_id"])
                    added += 1
            trace.append(f"Hop {hop}: “{curr_q[:70]}” → {added} new chunks (total {len(all_chunks)})")
            if hop == max_hops:
                break
            nq = self._detect_gap(query, curr_q, all_chunks)
            if nq is None:
                trace.append("No further gap detected — stopping")
                break
            trace.append(f"Gap found → “{nq}”")
            curr_q = nq
        return all_chunks

    def _detect_gap(self, original: str, last_q: str, chunks: List[Dict]) -> Optional[str]:
        summary = "\n".join(
            f"- Sec/Art.{get_entity_id(c)} [{c.get('source', '')}]: {c.get('title', 'Untitled')}"
            for c in chunks[:12]
        )
        prompt = (
            f'Original legal question: "{original}"\n\n'
            f"Laws retrieved so far:\n{summary}\n\n"
            "Is there an Indian law provision still needed to fully answer the question?\n"
            "YES → reply with ONLY a short search query (under 10 words).\n"
            "NO  → reply with ONLY the word: COMPLETE"
        )
        result = llm([{"role": "user", "content": prompt}]).strip().strip('"')
        if not result or result.upper().startswith("COMPLETE") or result.lower() == last_q.lower():
            return None
        return result


# ──────────────────────────────────────────────────────────────────────────────
# Generation + validator
# ──────────────────────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are LegalSaathi, a helpful Indian legal assistant.
Given a user's legal problem and retrieved sections/articles from Indian statutes:
1. Explain each relevant law at a Class 8 reading level — no legal jargon.
2. Give one real-life example per law.
3. List 2-3 concrete action steps.
4. Respond in the SAME LANGUAGE as the user's query.
5. ONLY cite sections/articles in the provided context. Never invent section numbers.

Respond ONLY in this JSON format:
{
  "law_cards": [{
    "article_number": "84",
    "act": "Consumer Protection Act, 2019",
    "title": "Liability of product manufacturer",
    "explanation": "...",
    "example": "...",
    "action_steps": ["Step 1", "Step 2"]
  }]
}"""


def parse_json_loose(raw: str) -> Dict:
    try:
        return json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"law_cards": []}


def generate(query: str, chunks: List[Dict], lang: str, trace: List[str]) -> Dict:
    context = "\n\n".join(
        f"--- {c.get('source', 'Indian Law')} | Sec/Art.{get_entity_id(c)} | {c.get('title', '')} ---\n"
        f"{c.get('chunk_text', c.get('body_text', ''))}"
        for c in chunks
    )
    lang_str = "Hindi" if lang == "hi" else "English"
    prompt = (f'Legal problem: "{query}"\n\nContext:\n{context}\n\n'
              f"Generate law cards in {lang_str}. Only use sections/articles in the context above.")
    raw = llm([{"role": "system", "content": SYSTEM_PROMPT},
               {"role": "user", "content": prompt}], temperature=0.1, json_mode=True)
    parsed = parse_json_loose(raw)

    valid = {str(get_entity_id(c)).strip() for c in chunks if get_entity_id(c)}
    cards, removed = [], 0
    for card in parsed.get("law_cards", []):
        art = str(card.get("article_number", "") or card.get("section_number", "")).strip()
        if art in valid:
            if not isinstance(card.get("action_steps"), list):
                card["action_steps"] = []
            cards.append(card)
        else:
            removed += 1
    if removed:
        trace.append(f"Validator removed {removed} hallucinated citation(s)")
    trace.append(f"{len(cards)} validated law card(s)")
    return {"law_cards": cards}


# ──────────────────────────────────────────────────────────────────────────────
# Pipeline helpers
# ──────────────────────────────────────────────────────────────────────────────
def detect_language(text: str) -> str:
    if re.search(r"[ऀ-ॿ]", text):  # Devanagari
        return "hi"
    try:
        from langdetect import detect
        lang = detect(text)
        return lang if lang in ("en", "hi") else "en"
    except Exception:
        return "en"


def classify_query(query: str) -> str:
    prompt = (
        "Classify this Indian legal query. Reply with ONE word only.\n"
        "SIMPLE = one article/section or definition\n"
        "RELATIONAL = how laws relate to each other\n"
        "COMPLEX = real-life situation needing multiple laws or procedures\n\n"
        f'Query: "{query}"\nAnswer (SIMPLE / RELATIONAL / COMPLEX):'
    )
    r = llm([{"role": "user", "content": prompt}]).strip().upper()
    if "RELATIONAL" in r:
        return "relational"
    if "COMPLEX" in r:
        return "complex"
    return "simple"


def compress(chunks: List[Dict]) -> List[Dict]:
    total, out = 0, []
    for c in chunks:
        n = len(c.get("chunk_text", ""))
        if total + n > config.MAX_CONTEXT_CHARS:
            break
        out.append(c)
        total += n
    return out


LAWYER_MAP = {
    **{k: "constitutional" for k in ["fundamental", "writ", "constitutional_remedy", "education", "dpsp"]},
    **{k: "criminal" for k in ["criminal", "arrest", "fir", "bail", "investigation", "child_protection", "sexual_offence"]},
    **{k: "family" for k in ["family_law", "domestic_violence", "marriage", "divorce", "maintenance", "personal_law"]},
    **{k: "labour" for k in ["labour", "workplace_rights", "social_security", "gratuity", "pf", "employment"]},
    **{k: "consumer" for k in [
        "consumer_complaint", "consumer_rights", "product_liability", "product_defect", "service_deficiency",
        "refund_compensation", "e_commerce_dispute", "unfair_trade_practice", "misleading_advertisement",
        "district_forum", "state_commission", "national_commission", "mediation"]},
    **{k: "civil" for k in ["property", "housing", "rent", "landlord_tenant"]},
}


def lawyer_category(chunks: List[Dict]) -> str:
    counts: Dict[str, int] = {}
    for c in chunks:
        for uc in c.get("legalsaathi_use_cases", []):
            counts[uc] = counts.get(uc, 0) + 1
    if not counts:
        return "general"
    return LAWYER_MAP.get(max(counts, key=counts.get), "general")


@st.cache_resource(show_spinner="Loading legal corpus, BM25 index and knowledge graph…")
def load_engine():
    indexer = LegalIndexer()
    indexer.load()
    if not indexer.chunks:
        raise RuntimeError(f"No chunk files found in '{DATA_DIR}/'.")
    graph = LegalGraphStore()
    graph.load(indexer.chunks)
    hybrid = HybridRetriever(indexer)
    return indexer, graph, hybrid, GraphRetriever(graph, indexer), MultiHopRetriever(hybrid)


def run_pipeline(user_query: str, forced_type: Optional[str] = None) -> Dict:
    indexer, graph, hybrid, graph_ret, multi_hop = load_engine()
    trace: List[str] = []
    lang = detect_language(user_query)
    qtype = forced_type or classify_query(user_query)
    trace.append(f"Language: {lang} | Query type: {qtype}")

    if qtype == "simple":
        chunks = hybrid.retrieve(user_query, config.FINAL_TOP_K)
    elif qtype == "relational":
        ini = hybrid.retrieve(user_query, 6)
        chunks = graph_ret.expand(ini, 2, 4)[:config.FINAL_TOP_K]
    else:
        mhc = multi_hop.retrieve(user_query, trace, config.MULTIHOP_ITERS)
        chunks = graph_ret.expand(mhc, 1, 3)[:config.FINAL_TOP_K]

    n_before = len(chunks)
    chunks = compress(chunks)
    trace.append(f"Context: {len(chunks)}/{n_before} chunks after compression")
    result = generate(user_query, chunks, lang, trace)

    return {
        "query": user_query, "language": lang, "query_type": qtype,
        "law_cards": result["law_cards"],
        "lawyer_category": lawyer_category(chunks),
        "trace": trace,
        "sources": [{"article": get_entity_id(c), "title": c.get("title", ""),
                     "act": c.get("source", ""), "via": c.get("_source", "hybrid")} for c in chunks],
    }


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
EXAMPLES = [
    "What is the right to life under Article 21?",
    "I bought a defective phone and the seller refuses a refund. What can I do?",
    "My landlord threw me out at midnight with no notice after 8 years. What are my rights?",
    "My son was arrested 4 days ago, no FIR filed, still in custody.",
    "मुझे पुलिस ने बिना कारण गिरफ्तार किया। मेरे क्या अधिकार हैं?",
]

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1100px;}
.ls-badge {display:inline-block; padding:2px 10px; border-radius:12px; font-size:0.75rem;
           background:#eef2ff; color:#3730a3; margin-right:6px;}
</style>""", unsafe_allow_html=True)

st.title("🏛️ LegalSaathi")
st.caption("Indian legal awareness in plain language — English & हिन्दी. "
           "Hybrid RAG over the Constitution, CrPC, IPC, POCSO, DV Act, Rent, Social Security, "
           "Special Marriage, Muslim Personal Law and Consumer Protection Acts.")

with st.sidebar:
    st.header("Settings")
    mode = st.selectbox("Retrieval mode", ["Auto (classify query)", "Simple", "Relational", "Complex (multi-hop)"])
    forced = {"Simple": "simple", "Relational": "relational", "Complex (multi-hop)": "complex"}.get(mode)
    show_trace = st.checkbox("Show pipeline trace", value=False)
    st.divider()
    st.markdown("**Try an example**")
    for i, ex in enumerate(EXAMPLES):
        if st.button(ex[:55] + ("…" if len(ex) > 55 else ""), key=f"ex{i}", use_container_width=True):
            st.session_state["pending_query"] = ex
    st.divider()
    try:
        indexer, graph, _, _, _ = load_engine()
        st.metric("Chunks loaded", f"{len(indexer.chunks):,}")
        st.metric("Graph nodes / edges", f"{graph.G.number_of_nodes():,} / {graph.G.number_of_edges():,}")
        if indexer.skipped:
            st.warning("Missing data files:\n" + "\n".join(f"• {s}" for s in indexer.skipped))
    except Exception as e:
        st.error(str(e))
    try:
        with st.expander("Gemini key usage today"):
            for k in get_rotator().status():
                st.text(f"{k['key']}: {k['used']} used" + ("  [exhausted]" if k["exhausted"] else ""))
    except Exception:
        pass
    if st.button("Clear history", use_container_width=True):
        st.session_state["history"] = []
        st.rerun()

st.session_state.setdefault("history", [])

# Preflight: keys + data
try:
    get_clients()
    load_engine()
except Exception as e:
    st.error(f"Setup problem: {e}")
    st.info("Put your API keys in `.streamlit/secrets.toml` (see the example file) and the 30 NDJSON "
            f"files in `{DATA_DIR}/`.")
    st.stop()


def render_result(r: Dict):
    st.markdown(
        f'<span class="ls-badge">{r["language"].upper()}</span>'
        f'<span class="ls-badge">{r["query_type"]}</span>'
        f'<span class="ls-badge">lawyer: {r["lawyer_category"]}</span>',
        unsafe_allow_html=True)
    if not r["law_cards"]:
        st.warning("No validated law provisions found for this query. Try rephrasing it with more detail.")
    for card in r["law_cards"]:
        num = card.get("article_number") or card.get("section_number", "?")
        with st.container(border=True):
            st.subheader(f"{card.get('title', 'Untitled')}")
            st.caption(f"Sec/Art. {num} · {card.get('act', '')}")
            st.markdown(card.get("explanation", ""))
            if card.get("example"):
                st.info(f"**Example:** {card['example']}")
            if card.get("action_steps"):
                st.markdown("**What you can do**")
                for step in card["action_steps"]:
                    st.markdown(f"- {step}")
    with st.expander(f"Sources ({len(r['sources'])})"):
        for s in r["sources"]:
            icon = "🕸️" if s["via"] == "graph" else "⚡"
            st.markdown(f"{icon} **{s['article']}** — {s['title']} _{s['act']}_")
    if show_trace:
        with st.expander("Pipeline trace"):
            for t in r["trace"]:
                st.text(t)
    st.caption("⚖️ LegalSaathi provides legal awareness only. Consult a registered lawyer.")


# History
for item in st.session_state["history"]:
    with st.chat_message("user"):
        st.write(item["query"])
    with st.chat_message("assistant"):
        render_result(item)

pending = st.session_state.pop("pending_query", None)
query = st.chat_input("Describe your legal problem… / अपनी कानूनी समस्या बताइए…") or pending

if query:
    with st.chat_message("user"):
        st.write(query)
    with st.chat_message("assistant"):
        st.session_state["warnings"] = []
        try:
            with st.spinner("Searching laws and drafting your answer…"):
                result = run_pipeline(query, forced)
            for w in st.session_state.get("warnings", []):
                st.warning(w)
            render_result(result)
            st.session_state["history"].append(result)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
