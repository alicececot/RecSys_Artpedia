import os
import json
import time
import random
import secrets
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import requests

import numpy as np
import warnings
from PIL import Image, ImageOps, ImageEnhance
import hashlib
import re  
import streamlit.components.v1 as components


from openai import OpenAI


import torch

import streamlit as st
from pathlib import Path
from urllib.parse import urlparse
import unicodedata

import gspread
from google.oauth2.service_account import Credentials

IMG_ROOT = Path("./images")
APP_TITLE = "Recommender d'Arte"
DEFAULT_JSON_PATH = "./data/artpedia.json"   
EMB_NPZ_PATH = "./data/embeddings/artpedia_blip_base_all.npz"  
IMG_CACHE_DIR = "./.cache/images"
TOPK_SEED = 12  
TOPK_REC  = 12  

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets["OPENROUTER_API_KEY"]
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

ALPHA, BETA, GAMMA, DELTA = 0.50, 0.25, 0.15, 0.10

def load_css(path: str = "./style.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.warning(f"CSS non caricato ({path}): {e}")

@st.cache_data(show_spinner=False)
def load_artpedia(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: List[Dict] = []

    if isinstance(data, dict):
        looks_like_items = all(isinstance(v, dict) for v in data.values())
        if looks_like_items:
            running_id = 0
            for k, it in data.items():
                if not isinstance(it, dict):
                    continue
                it = it.copy()
                try:
                    gid = int(k)
                except Exception:
                    gid = running_id
                it.setdefault("id", gid)
                it.setdefault("title", f"Untitled #{gid}")
                it.setdefault("year", None)
                it.setdefault("img_url", None)
                it.setdefault("visual_sentences", [])
                it.setdefault("contextual_sentences", [])
                it.setdefault("split", it.get("split", "train"))
                items.append(it)
                running_id += 1
        else:
            running_id = 0
            for split_name, lst in data.items():
                if not isinstance(lst, list):
                    continue
                for it in lst:
                    if not isinstance(it, dict):
                        continue
                    it = it.copy()
                    it.setdefault("title", f"Untitled #{running_id}")
                    it.setdefault("year", None)
                    it.setdefault("img_url", None)
                    it.setdefault("visual_sentences", [])
                    it.setdefault("contextual_sentences", [])
                    it["split"] = str(it.get("split", split_name))
                    it["id"] = running_id
                    items.append(it)
                    running_id += 1

    elif isinstance(data, list):
        for i, it in enumerate(data):
            if not isinstance(it, dict):
                continue
            it = it.copy()
            it.setdefault("id", i)
            it.setdefault("title", f"Untitled #{i}")
            it.setdefault("year", None)
            it.setdefault("img_url", None)
            it.setdefault("visual_sentences", [])
            it.setdefault("contextual_sentences", [])
            it.setdefault("split", it.get("split", "train"))
            items.append(it)
    else:
        raise ValueError("Formato JSON non riconosciuto (atteso dict o list).")

    if not items:
        raise ValueError("Nessun item caricato: verifica il JSON.")

    return items


def ensure_embeddings_local():
    if os.path.exists(EMB_NPZ_PATH):
        return
    url = os.getenv("EMB_URL") or st.secrets.get("EMB_URL")
    if not url:
        st.error("Embeddings mancanti: definisci EMB_URL nei Secrets oppure carica l'NPZ in ./data/embeddings/.")
        st.stop()
    os.makedirs(os.path.dirname(EMB_NPZ_PATH), exist_ok=True)
    with st.spinner("Scarico embeddings…"):
        r = requests.get(url, stream=True, timeout=120)
        r.raise_for_status()
        with open(EMB_NPZ_PATH, "wb") as f:
            for chunk in r.iter_content(1024 * 1024):
                if chunk:
                    f.write(chunk)

@st.cache_resource
def get_gsheet_worksheet():
    sa_info = dict(st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"])
    
    if "private_key" in sa_info and isinstance(sa_info["private_key"], str):
        sa_info["private_key"] = sa_info["private_key"].replace("\\n", "\n")
    
    creds = Credentials.from_service_account_info(
        sa_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets",
                "https://www.googleapis.com/auth/drive"]
    )
    
    gc = gspread.authorize(creds)
    sh = gc.open_by_key(st.secrets["SHEETS_SPREADSHEET_ID"])
    ws = sh.sheet1
    return ws


def fuse_modalities(img_vec, vis_vec, ctx_vec, meta_vec, w: Tuple[float, float, float, float]):
    a, b, c, d = w
    return a*img_vec + b*vis_vec + c*ctx_vec + d*meta_vec


def normalize_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True) + 1e-8
    return x / n


def cosine_sim(u: np.ndarray, V: np.ndarray) -> np.ndarray:
    u = u / (np.linalg.norm(u) + 1e-8)
    Vn = normalize_rows(V)
    return Vn @ u

def sanitize_name(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[^\w\-.]+", "_", s)
    return s.strip("_") or "file"

def basename_from_url(url: str) -> str:
    p = urlparse(url)
    base = os.path.basename(p.path) or hashlib.md5(url.encode()).hexdigest()
    return sanitize_name(base)

def hashed_filename(url: str) -> str:
    h = hashlib.sha1(url.encode()).hexdigest()[:8]
    return f"{h}_{basename_from_url(url)}"

def _open_resized(path: Path, max_megapixels: int = 20, max_side: int = 2048) -> Optional[Image.Image]:
    if path is None or not path.exists():
        return None

    old_max = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None  
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  

            w, h = im.size
            s_side = min(max_side / max(w, h), 1.0)
            max_px = max_megapixels * 1_000_000
            s_mp = (max_px / (w * h)) ** 0.5 if (w * h) > max_px else 1.0
            s = min(s_side, s_mp)

            if s < 1.0:
                try:
                    im.draft("RGB", (int(w * s), int(h * s)))
                except Exception:
                    pass
                im = im.resize((max(1, int(w * s)), max(1, int(h * s))), Image.Resampling.LANCZOS)

            return im.convert("RGB")
    except Exception:
        return None
    finally:
        Image.MAX_IMAGE_PIXELS = old_max

def load_image_local(item: Dict) -> Optional[Image.Image]:
    split = str(item.get("split", "train"))
    url = item.get("img_url") or ""

    candidates = []
    if url:
        fname = hashed_filename(url)                 
        candidates.append(IMG_ROOT / split / fname)
        candidates.extend((IMG_ROOT / split).glob(f"*_{basename_from_url(url)}"))

    iid = item.get("id")
    if iid is not None:
        candidates.append(IMG_ROOT / split / f"{iid}.jpg")
        candidates.append(IMG_ROOT / split / f"{iid}.png")

    for p in candidates:
        try:
            if p and p.exists():
                im = _open_resized(p)     
                if im is not None:
                    return im
        except Exception:
            continue
    return None

def _download_remote_image(split: str, filename: str) -> Optional[Path]:
    base = os.getenv("IMAGES_BASE_URL") or st.secrets.get("IMAGES_BASE_URL")
    if not base:
        return None

    dst_dir = IMG_ROOT / split
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst_path = dst_dir / filename
    if dst_path.exists():
        return dst_path

    url = f"{base}/{split}/{filename}"
    try:
        with st.spinner(f"Scarico immagine…"):
            r = requests.get(url, stream=True, timeout=60)
            r.raise_for_status()
            with open(dst_path, "wb") as f:
                for chunk in r.iter_content(1024 * 256):
                    if chunk:
                        f.write(chunk)
        return dst_path
    except Exception as e:
        if dst_path.exists():
            try:
                dst_path.unlink()
            except Exception:
                pass
        return None


def load_image(item: Dict) -> Optional[Image.Image]:
    img = load_image_local(item)
    if img is not None:
        return img

    split = str(item.get("split", "train"))
    url = item.get("img_url") or ""
    candidates = []

    if url:
        
        candidates.append(hashed_filename(url))
        
        candidates.append(basename_from_url(url))

    iid = item.get("id")
    if iid is not None:
        candidates.append(f"{iid}.jpg")
        candidates.append(f"{iid}.png")

    for fname in candidates:
        p = _download_remote_image(split, fname)
        if p is not None and p.exists():
            img = load_image_local(item)
            if img is not None:
                return img

    return None


@dataclass
class EmbeddingPack:
    ids: np.ndarray
    img: np.ndarray
    vis: np.ndarray
    ctx: np.ndarray
    meta: np.ndarray

@st.cache_resource(show_spinner=True)
def load_embeddings_from_file(npz_path: str) -> EmbeddingPack:
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"File NPZ non trovato: {npz_path}")
    npz = np.load(npz_path)
    return EmbeddingPack(
        ids=npz["ids"],
        img=npz["img"],
        vis=npz["vis"],
        ctx=npz["ctx"],
        meta=npz["meta"],
    )


def decompose_similarity(u_vecs: Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray],
                         v_vecs: Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray],
                         w: Tuple[float,float,float,float]) -> Tuple[float, Dict[str,float]]:
    (u_img, u_vis, u_ctx, u_meta) = u_vecs
    (v_img, v_vis, v_ctx, v_meta) = v_vecs
    a,b,c,d = w
    s_img = float(cosine_sim(u_img, v_img[None,:])[0])
    s_vis = float(cosine_sim(u_vis, v_vis[None,:])[0])
    s_ctx = float(cosine_sim(u_ctx, v_ctx[None,:])[0])
    s_meta = float(cosine_sim(u_meta, v_meta[None,:])[0])
    total = a*s_img + b*s_vis + c*s_ctx + d*s_meta
    return total, {"IMG": s_img, "VIS": s_vis, "CTX": s_ctx, "META": s_meta}

def _format_list_it(names):
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} e {names[1]}"
    return f"{', '.join(names[:-1])} e {names[-1]}"

def content_based_explanation_gid(rec_gid: int,
                                  seed_gids: list,
                                  pack,
                                  w,
                                  k_refs: int = 2) -> str:
    gid_list = pack.ids.tolist()
    idx_map = {int(g): i for i, g in enumerate(gid_list)}

    item = st.session_state.id2item.get(rec_gid, {})
    rec_title = item.get("title", "questa opera")

    if rec_gid not in idx_map or not seed_gids:
        return f"Ti abbiamo raccomandato *{rec_title}* perché riteniamo che sia in linea con i tuoi gusti."

    fused = fuse_modalities(pack.img, pack.vis, pack.ctx, pack.meta, w)

    rv = fused[idx_map[rec_gid]]
    rvn = np.linalg.norm(rv) + 1e-9

    sims = []
    for sg in seed_gids:
        i = idx_map.get(int(sg))
        if i is None:
            continue
        sv = fused[i]
        sim = float(np.dot(rv, sv) / (rvn * (np.linalg.norm(sv) + 1e-9)))
        sims.append((sg, sim))

    if not sims:
        return f"Ti abbiamo raccomandato *{rec_title}* perché riteniamo che sia in linea con i tuoi gusti."

    sims.sort(key=lambda x: x[1], reverse=True)
    top_refs = [g for g, _ in sims[:max(1, k_refs)]]

    ref_titles = []
    for g in top_refs:
        it = st.session_state.id2item.get(int(g), {})
        t = (it.get("title") or "").strip()
        if t:
            ref_titles.append(t)

    ref_list = _format_list_it(ref_titles)
    if not ref_list:
        return f"Ti abbiamo raccomandato *{rec_title}* perché riteniamo che sia in linea con i tuoi gusti."

    return f"Ti abbiamo raccomandato *{rec_title}* perché hai apprezzato opere come *{ref_list}*."


def get_explanation_for_item(rec_gid: int, seed_gids: list, pack, w) -> str:
    item = st.session_state.id2item.get(rec_gid, {})
    rec_title = item.get("title", "questa opera")
    
    if st.session_state.exp_style == "CONTENT-BASED":
        return content_based_explanation_gid(rec_gid, seed_gids, pack, w, k_refs=2)
    else:  
        
        prefs = {
            "liked_titles": [],
            "liked_styles": []
        }
        
        for seed_gid in seed_gids:
            seed_item = st.session_state.id2item.get(seed_gid, {})
            if seed_item.get("title"):
                prefs["liked_titles"].append(seed_item["title"])
        
        try:
            explanation, latency_ms = generate_llm_explanation_gemini_flash(item, prefs)
            return explanation
        except Exception as e:
            return f"Ti abbiamo raccomandato *{rec_title}* perché riteniamo che sia in linea con i tuoi gusti."

def build_llm_system() -> str:
    return ("Sei un esperto di storia dell'arte che genera spiegazioni chiare e concise "
            "per raccomandazioni d'arte. Usa SOLO le informazioni fornite nel contesto. "
            "Non inventare dettagli non presenti. La spiegazione deve essere breve: massimo 2 frasi, 30–40 parole totali. "
            "Stile: professionale ma accessibile. Collega 1-2 opere apprezzate dall'utente "
            "con 1-2 caratteristiche dell'opera raccomandata.")

def build_llm_user_context(item: dict, prefs: dict) -> str:
    def _slice(lst, k): 
        lst = (lst or [])[:k]
        return [s.strip().replace("\n", " ") for s in lst if s and s.strip()]
    
    VIS = " | ".join(_slice(item.get("visual_sentences", []), 2))
    CTX = " | ".join(_slice(item.get("contextual_sentences", []), 1))
    
    liked_titles = ", ".join(_slice(prefs.get("liked_titles", []), 3))
    liked_styles = ", ".join(_slice(prefs.get("liked_styles", []), 3))
    
    return (
        "CONTESTO OPERA RACCOMANDATA:\n"
        f"TITOLO: {item.get('title', 'Senza titolo')}\n"
        f"ANNO: {item.get('year', 'Non specificato')}\n"
        f"DESCRIZIONE VISIVA: {VIS}\n"
        f"CONTESTO STORICO: {CTX}\n\n"
        "PREFERENZE UTENTE:\n"
        f"OPERE APPREZZATE: {liked_titles}\n"
        f"STILI PREFERITI: {liked_styles}\n\n"
        "COMPITO: Spiega in modo chiaro e convincente perché questa opera è stata raccomandata, "
        "collegando esplicitamente le preferenze dell'utente con le caratteristiche dell'opera."
    )

def generate_llm_explanation_gemini_flash(item: dict, prefs: dict,
                                          max_tokens: int = 80, temperature: float = 0.2):
    system = build_llm_system()
    user = build_llm_user_context(item, prefs)

    try:
        t0 = time.time()
        
        extra_headers = {
            "HTTP-Referer": "https://your-art-recommender.com",  
            "X-Title": "Art Recommendation Study"
        }
        
        resp = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            extra_headers=extra_headers
        )
        
        text = resp.choices[0].message.content.strip()
        latency_ms = (time.time() - t0) * 1000.0
        
        if "llm_requests" not in st.session_state:
            st.session_state.llm_requests = []
        
        st.session_state.llm_requests.append({
            "item_id": item.get("id", "unknown"),
            "item_title": item.get("title", "unknown"),
            "latency_ms": latency_ms,
            "timestamp": time.time()
        })
        
        return text, latency_ms
        
    except Exception as e:
        error_msg = f"Errore LLM per opera {item.get('id', 'unknown')}: {str(e)}"
        print(error_msg)  
        
        fallback_explanation = (
            f"Ti raccomandiamo '{item.get('title', 'questa opera')}' "
            f"perché presenta caratteristiche in linea con le opere che hai apprezzato. "
            f"L'opera combina elementi stilistici e tematici che riteniamo possano corrispondere ai tuoi gusti."
        )
        
        return fallback_explanation, 0

def build_user_profile(pack: EmbeddingPack, idx_list: List[int], w: Tuple[float,float,float,float]) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    if not idx_list:
        d = pack.img.shape[1]
        z = np.zeros((d,), dtype=np.float32)
        return z,z,z,z
    img = pack.img[idx_list].mean(axis=0)
    vis = pack.vis[idx_list].mean(axis=0)
    ctx = pack.ctx[idx_list].mean(axis=0)
    meta = pack.meta[idx_list].mean(axis=0)
    return img, vis, ctx, meta


def rank_items(pack: EmbeddingPack, user_vecs: Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray], w: Tuple[float,float,float,float], exclude_global_idx: List[int], topk: int) -> List[Tuple[int,float,Dict[str,float]]]:
    u_img,u_vis,u_ctx,u_meta = user_vecs
    fused_items = fuse_modalities(pack.img, pack.vis, pack.ctx, pack.meta, w)
    u_fused = fuse_modalities(u_img, u_vis, u_ctx, u_meta, w)
    sims = cosine_sim(u_fused, fused_items)
    cand = [(gi, float(sims[i])) for i, gi in enumerate(pack.ids.tolist()) if gi not in exclude_global_idx]
    cand = sorted(cand, key=lambda x: x[1], reverse=True)[:max(200, topk)]
    out = []
    id_to_local = {gid:i for i,gid in enumerate(pack.ids.tolist())}
    for gid, score in cand[:topk]:
        li = id_to_local[gid]
        total, contrib = decompose_similarity(
            user_vecs,
            (pack.img[li], pack.vis[li], pack.ctx[li], pack.meta[li]),
            w
        )
        out.append((gid, score, contrib))
    return out



HEADERS = {
    "art_recommendation_study": [
        # IDENTIFICAZIONE
        "session_id", "user_id", "timestamp", "experimental_group", "explanation_type",
        
        # DATI UTENTE
        "user_age_range", "user_gender", "user_profession", "museum_visits", "appreciate_art", "art_knowledge_level", "heard_recommenders",
        
        # SELEZIONE OPERE INIZIALI
        "initial_artwork_pool", "selected_artworks", "selection_time_ms",
        
        # RACCOMANDAZIONI GENERATE
        "recommended_artworks", "recommendation_scores", 
        "recommendation_algorithm", "weights_image", "weights_visual", 
        "weights_context", "weights_metadata",
        
        # VALUTAZIONE UTENTE (1-5)
        "rating_preference_match", "rating_diversity", "rating_novelty", 
        "rating_serendipity", "rating_transparency", "rating_trust", 
        "rating_understanding", "rating_usefulness",
        
        # TEMPI
        "total_session_time_ms",
        
        # METADATI PER ANALISI
        "selected_artwork_titles", "recommended_artwork_titles"
    ]
}

def append_to_google_sheet(row: Dict):
    ws = get_gsheet_worksheet()

    header = HEADERS["art_recommendation_study"]  
    values = []
    for col in header:
        v = row.get(col, "")
        if v is None:
            v = ""
        values.append(v)
    try:
        ws.append_row(values, value_input_option="USER_ENTERED")
    except Exception as e:
        st.sidebar.warning(f"Append su Google Sheets non riuscito: {e}")

    
def get_artwork_titles(artwork_ids: List[int]) -> str:
    titles = []
    for art_id in artwork_ids:
        item = st.session_state.id2item.get(art_id, {})
        title = item.get('title', f'Unknown_{art_id}')
        year = item.get('year', '')
        title_str = f"{title} ({year})" if year else title
        titles.append(f"{title_str} [{art_id}]")
    return " | ".join(titles)

def format_list_for_csv(data: List) -> str:
    return json.dumps(data, ensure_ascii=False)

def log_complete_study_session(rec_ids, scores, ratings, rec_duration_ms):
    
    explanation_type = "LLM" if st.session_state.exp_style == "LLM-EXPLANATION" else "CONTENT_BASED"
    
    total_duration = int((time.time() - st.session_state.seed_start_ts) * 1000)
    selection_duration = int((st.session_state.rec_start_ts - st.session_state.seed_start_ts) * 1000)
    
    row = {
        # IDENTIFICAZIONE
        "session_id": st.session_state.slate_id,
        "user_id": st.session_state.user_id,
        "timestamp": int(time.time()),
        "experimental_group": st.session_state.exp_style,
        "explanation_type": explanation_type,
        
        # DATI UTENTE
        "user_age_range": st.session_state.user_demographics.get("age_range", ""),
        "user_gender": st.session_state.user_demographics.get("gender", ""),
        "user_profession": st.session_state.user_demographics.get("profession", ""),
        "museum_visits": st.session_state.user_demographics.get("museum_visits", ""),
        "appreciate_art": st.session_state.user_demographics.get("appreciate_art", ""),
        "art_knowledge_level": st.session_state.user_demographics.get("art_knowledge_level", ""),
        "heard_recommenders": st.session_state.user_demographics.get("heard_recommenders", ""),

        # SELEZIONE OPERE INIZIALI
        "initial_artwork_pool": format_list_for_csv(st.session_state.seed_pool_ids),
        "selected_artworks": format_list_for_csv(st.session_state.seed_selected_ids),
        "selection_time_ms": selection_duration,
        
        # RACCOMANDAZIONI GENERATE
        "recommended_artworks": format_list_for_csv(rec_ids),
        "recommendation_scores": format_list_for_csv(scores),
        "recommendation_algorithm": "content_based_blip",
        "weights_image": ALPHA,
        "weights_visual": BETA,
        "weights_context": GAMMA,
        "weights_metadata": DELTA,
        
        # VALUTAZIONE UTENTE
        "rating_preference_match": ratings.get("preference", ""),
        "rating_diversity": ratings.get("diversity", ""),
        "rating_novelty": ratings.get("novelty", ""),
        "rating_serendipity": ratings.get("serendipity", ""),
        "rating_transparency": ratings.get("transparency", ""),
        "rating_trust": ratings.get("trust", ""),
        "rating_understanding": ratings.get("understanding", ""),
        "rating_usefulness": ratings.get("usefulness", ""),
        
        # TEMPI
        "total_session_time_ms": total_duration,
        
        "selected_artwork_titles": get_artwork_titles(st.session_state.seed_selected_ids),
        "recommended_artwork_titles": get_artwork_titles(rec_ids)
    }
    append_to_google_sheet(row)

def _assign_group(user_id: str) -> str:
    h = int(hashlib.sha256(user_id.encode("utf-8")).hexdigest(), 16)
    return "CONTENT-BASED" if (h % 2 == 0) else "LLM-EXPLANATION"

def _init_session():
    if "user_id" not in st.session_state:
        st.session_state.user_id = secrets.token_hex(4)
    
    if "exp_style" not in st.session_state:
        st.session_state.exp_style = _assign_group(st.session_state.user_id)

    st.session_state.setdefault("phase", "consent")
    st.session_state.setdefault("user_demographics", {})
    st.session_state.setdefault("seed_pool_ids", [])
    st.session_state.setdefault("seed_selected_ids", [])
    st.session_state.setdefault("seed_start_ts", 0)
    st.session_state.setdefault("rec_start_ts", 0)
    st.session_state.setdefault("slate_id", None)  


def screen_consent():
    st.header("Consenso informato")
    st.write("Questa raccolta è anonima. I dati registrati riguardano le tue scelte e le risposte al questionario.")
    agree = st.checkbox("Acconsento all'uso anonimo dei miei dati per fini di ricerca", value=False)
    if st.button("Inizia", width='stretch'):
        if not agree:
            st.warning("Devi acconsentire per proseguire.")
            return
        
        st.session_state.phase = "demographic"  
        st.rerun()

def screen_demographic():
    st.header("Informazioni utente")
    st.write("Prima di iniziare, aiutaci a conoscerti meglio con qualche informazione anonima.")

    age_ranges = ["", "18-26", "27-36", "37-50", "Over 50", "Preferisco non dirlo"]
    default_age = st.session_state.user_demographics.get("age_range", "")
    age_idx = age_ranges.index(default_age) if default_age in age_ranges else 0
    age_range = st.selectbox("Età*", age_ranges, index=age_idx, key="demographic_age_range")
    st.session_state.user_demographics["age_range"] = age_range


    gender = st.selectbox(
        "Genere*",
        ["", "Donna", "Uomo", "Preferisco non dirlo"],
        index=0 if "gender" not in st.session_state.user_demographics else
               ["", "Donna", "Uomo", "Preferisco non dirlo"].index(st.session_state.user_demographics.get("gender","")),
        key="demographic_gender"
    )

    profession = st.selectbox(
        "Professione/Area di studio*",
        ["", "Disoccupato", "Studente", "Impiegato/Dipendente", "Libero professionista", "Casalinga / Casalingo", "Pensionato", "Altro", "Preferisco non dirlo"],
        index=0,
        key="demographic_profession"
    )

    other_profession = ""
    if profession == "Altro":
        other_profession = st.text_input("Specifica la tua professione*", placeholder="Inserisci la tua professione", key="demographic_other_prof")

    if st.button("Continua", width='stretch'):
        errors = []
        if not age_range: errors.append("L'età è obbligatoria")
        if not gender: errors.append("Il genere è obbligatorio")
        if not profession: errors.append("La professione è obbligatoria")
        if profession == "Altro" and not (other_profession or "").strip():
            errors.append("Specifica la tua professione")

        if errors:
            for e in errors:
                st.error(e)
            return

        final_profession = other_profession.strip() if profession == "Altro" else profession
        st.session_state.user_demographics = {
            "age_range": age_range,
            "gender": gender,
            "profession": final_profession,
        }
        st.session_state.phase = "background"
        st.session_state.seed_start_ts = time.time()
        st.rerun()

def screen_background_questions():
    st.header("Domande preliminari")
    st.write("Rispondi a queste domande per aiutarci a capire il tuo rapporto con l'arte.")

    visit_opts = ["", "Mai", "1–2 volte", "3–5 volte", "Più di 5 volte"]
    default_visit = st.session_state.user_demographics.get("museum_visits", "")
    visit_museums = st.selectbox(
        "Quante volte visiti mostre/musei d’arte in un anno?*", visit_opts,
        index=visit_opts.index(default_visit) if default_visit in visit_opts else 0,
        key="q_museum_visits"
    )

    appreciate_opts = ["", "Per niente", "Poco", "Abbastanza", "Molto"]
    default_app = st.session_state.user_demographics.get("appreciate_art", "")
    appreciate_art = st.selectbox(
        "Quanto apprezzi l’arte in generale?*", appreciate_opts,
        index=appreciate_opts.index(default_app) if default_app in appreciate_opts else 0,
        key="q_appreciate_art"
    )

    knowledge_opts = ["", "Per niente", "Poco", "Abbastanza", "Molto"]
    default_kn = st.session_state.user_demographics.get("art_knowledge_level", "")
    art_knowledge_level = st.selectbox(
        "Quanto ti consideri esperto/a di arte?*", knowledge_opts,
        index=knowledge_opts.index(default_kn) if default_kn in knowledge_opts else 0,
        key="q_art_knowledge_level"
    )

    heard_opts = ["", "Si", "No"]
    default_heard = st.session_state.user_demographics.get("heard_recommenders", "")
    heard_recommenders = st.selectbox(
        "Hai mai sentito parlare dei sistemi di raccomandazione?*", heard_opts,
        index=heard_opts.index(default_heard) if default_heard in heard_opts else 0,
        key="q_heard_recs"
    )

    if st.button("Inizia", width="stretch"):
        errors = []
        if not visit_museums:      errors.append("La frequenza di visita ai musei è obbligatoria")
        if not appreciate_art:     errors.append("L'apprezzamento dell'arte è obbligatorio")
        if not art_knowledge_level:errors.append("Il livello di conoscenza dell'arte è obbligatorio")
        if not heard_recommenders: errors.append("La domanda sui sistemi di raccomandazione è obbligatoria")

        if errors:
            for e in errors:
                st.error(e)
            return

        st.session_state.user_demographics.update({
            "museum_visits": visit_museums,
            "appreciate_art": appreciate_art,
            "art_knowledge_level": art_knowledge_level,
            "heard_recommenders": heard_recommenders,
        })

        st.session_state.phase = "seed"
        st.session_state.seed_start_ts = time.time()
        st.rerun()



def find_local_image_path(item) -> Optional[Path]:
    split = str(item.get("split","train"))
    url = item.get("img_url") or ""
    candidates = []
    if url:
        h = hashlib.sha1(url.encode()).hexdigest()[:8]
        from urllib.parse import urlparse
        import os as _os, unicodedata as _uni, re as _re
        def _sanitize(s):
            s = _uni.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
            return _re.sub(r"[^\w\-.]+","_", s).strip("_") or "file"
        base = _os.path.basename(urlparse(url).path) or hashlib.md5(url.encode()).hexdigest()
        base = _sanitize(base)
        fname = f"{h}_{base}"
        candidates.append(IMG_ROOT / split / fname)
        candidates.extend((IMG_ROOT / split).glob(f"*_{base}"))
    iid = item.get("id")
    if iid is not None:
        candidates += [IMG_ROOT / split / f"{iid}.jpg", IMG_ROOT / split / f"{iid}.png"]
    for p in candidates:
        if p.exists():
            return p
    return None

def _sample_seed_pool(all_ids: List[int], k: int = TOPK_SEED) -> List[int]:
    valid = set(map(int, st.session_state.pack.ids.tolist()))   
    ids = [g for g in all_ids if g in valid]
    random.shuffle(ids)

    out, scanned = [], 0
    budget = 10 * k 

    for gid in ids:
        it = st.session_state.id2item[gid]

        if find_local_image_path(it):
            out.append(gid)
        else:
            split = str(it.get("split", "train"))
            url = it.get("img_url") or ""
            tried = False

            if url and not tried:
                fname = hashed_filename(url)
                p = _download_remote_image(split, fname)
                tried = True
                if p and p.exists():
                    out.append(gid)

            if url and (len(out) < k) and not find_local_image_path(it):
                base = basename_from_url(url)
                p2 = _download_remote_image(split, base)
                if p2 and p2.exists():
                    out.append(gid)

        scanned += 1
        if len(out) >= k or scanned >= budget:
            break

    return out

def prepare_recommendations_and_start_seq(w: Tuple[float,float,float,float], topk: int = TOPK_REC):
    pack = st.session_state.pack
    if pack is None:
        st.error("Embedding non caricati. Controlla EMB_NPZ_PATH.")
        st.stop()

    selected = st.session_state.seed_selected_ids
    gid_to_row = {int(g): i for i, g in enumerate(pack.ids.tolist())}
    seed_rows = [gid_to_row[g] for g in selected if g in gid_to_row]

    # 🎯 CALCOLO UNA SOLA VOLTA - stessa logica del Codice 1
    user_vecs = build_user_profile(pack, seed_rows, w)
    results = rank_items(pack, user_vecs, w, exclude_global_idx=selected, topk=topk)

    # 🎯 PREPARA TUTTI I DATI UNA VOLTA SOLA
    rec_ids: List[int] = []
    scores: List[float] = []
    explanations: Dict[int, str] = {}

    for gid, score, contrib in results:
        rec_ids.append(gid)
        scores.append(round(score, 6))
        explanation = get_explanation_for_item(gid, selected, pack, w)
        explanation_clean = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", explanation)
        explanation_clean = re.sub(r"\*(.+?)\*", r"<em>\1</em>", explanation_clean)
        explanations[gid] = explanation_clean

    rec_ids = rec_ids[:topk]
    scores  = scores[:topk]

    # 🎯 SALVA TUTTO NEL BUNDLE (raccomandazioni + spiegazioni)
    st.session_state.rec_bundle = {
        "ids": rec_ids,
        "scores": scores,
        "explanations": explanations,  # 🎯 SPIEGAZIONI GIÀ CALCOLATE
    }
    st.session_state.rec_idx = 0
    st.session_state.rec_ts = time.time()
    st.session_state.rec_start_ts = time.time()  # 🎯 TIMESTAMP INIZIO
    st.session_state.phase = "rec_seq"
    st.rerun()


def screen_seed_select(data: List[Dict]):
    st.subheader("Seleziona almeno 4 dipinti che ti piacciono")

    if not st.session_state.seed_pool_ids:
        all_ids = list(st.session_state.id2item.keys())
        st.session_state.seed_pool_ids = _sample_seed_pool(all_ids, TOPK_SEED)

    ids = st.session_state.seed_pool_ids[:12] 
    pre_sel = set(st.session_state.get("seed_selected_ids", []))

    with st.form("seed_pick_form", clear_on_submit=False):
        rows, cols_per_row = 4, 3
        idx = 0

        for r in range(rows):
            cols = st.columns(3, gap="small", border=True, width="stretch")
            for c in range(cols_per_row):
                if idx >= len(ids):
                    break
                gid = ids[idx]; idx += 1
                item = st.session_state.id2item[gid]

                with cols[c]:
                    st.markdown('<div class="art-card">', unsafe_allow_html=True)

                    img = load_image(item)

                    with st.popover("Ingrandisci 🔍", width="stretch"):
                        if img is not None:
                            st.image(img, width="stretch")
                            
                    if img is not None:
                        cropped_img = ImageOps.fit(
                            img, (450, 450), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5)
                        )
                        st.image(cropped_img, width="stretch")
                    else:
                        st.markdown(
                            '<div class="img-missing">Immagine non trovata</div>',
                            unsafe_allow_html=True
                        )

                    st.markdown(
                        f"<div class='titleline'><span class='title'>{item.get('title','Senza titolo')}</span> "
                        f"<span class='meta'>({item.get('year','?')})</span></div>",
                        unsafe_allow_html=True
                    )

                    default_checked = gid in pre_sel
                    st.checkbox("Seleziona", key=f"sel_{gid}", value=default_checked)

                    st.markdown("</div>", unsafe_allow_html=True)

        submitted = st.form_submit_button(
            "Genera raccomandazioni",
            type="primary",
            width="stretch"
        )


    if submitted:
        selected = [g for g in ids if st.session_state.get(f"sel_{g}", False)]
        if len(selected) < 4:
            st.error("Seleziona almeno 4 dipinti prima di proseguire.")
            return

        st.session_state.seed_selected_ids = selected
        st.session_state.slate_id = secrets.token_hex(6)
        prepare_recommendations_and_start_seq((ALPHA, BETA, GAMMA, DELTA))  

def screen_recommend_sequential(delay_ms: int = 4000):
    st.subheader("Raccomandazioni per te")

    bundle = st.session_state.get("rec_bundle")
    if not bundle:
        st.error("Nessuna raccomandazione disponibile.")
        if st.button("Torna alla selezione"):
            st.session_state.phase = "seed"
            st.rerun()
        return

    # 🎯 USA I DATI PRECALCOLATI
    rec_ids = bundle["ids"]
    explanations = bundle["explanations"]  # 🎯 SPIEGAZIONI GIÀ PRONTE
    idx = st.session_state.get("rec_idx", 0)

    if idx >= len(rec_ids):
        st.session_state.phase = "rec"  # 🎯 Va alla griglia con STESSE raccomandazioni
        st.rerun()
        return

    gid = rec_ids[idx]
    item = st.session_state.id2item.get(gid, {})
    img = load_image(item)
    exp_html = explanations.get(gid, "")  # 🎯 NO ricalcolo

    left, right = st.columns([7, 5], gap="large")
    with left:
        if img is not None:
            st.image(img, width="stretch")
        else:
            st.markdown('<div class="img-missing">Immagine non trovata</div>', unsafe_allow_html=True)

    with right:
        st.markdown(
            f"<h3 style='margin-top:0'>{item.get('title','Senza titolo')} "
            f"<span style='font-weight:400;color:var(--muted);'>({item.get('year','?')})</span></h3>",
            unsafe_allow_html=True
        )
        st.markdown(f"<div class='exp-box'><strong>Perché:</strong> {exp_html}</div>", unsafe_allow_html=True)

        elapsed_ms = int((time.time() - st.session_state.get("rec_ts", time.time())) * 1000)

        if elapsed_ms < delay_ms:
            st.button("Avanti →", width="stretch", disabled=True, key=f"next_disabled_{idx}")
            time.sleep(min((delay_ms - elapsed_ms) / 1000.0, 0.5))
            st.rerun()
        else:
            if st.button("Avanti →", width="stretch", key=f"next_enabled_{idx}"):
                st.session_state.rec_idx = idx + 1
                st.session_state.rec_ts = time.time()
                st.rerun()

def screen_recommend(data: List[Dict], w: Tuple[float, float, float, float]):
    st.subheader("Raccomandazioni per te")

    bundle = st.session_state.get("rec_bundle")
    if not bundle or not bundle.get("ids"):
        st.error("Nessuna raccomandazione disponibile. Torna alla selezione.")
        if st.button("Torna alla selezione"):
            st.session_state.phase = "seed"
            st.rerun()
        return

    rec_ids = bundle["ids"]
    scores = bundle["scores"]
    explanations = bundle["explanations"]  

    left, right = st.columns([7, 5], gap="large")

    with left:
        rows, cols_per_row = 4, 3
        idx = 0
        for r in range(rows):
            cols = st.columns(3, gap="small", border=True)
            for c in range(cols_per_row):
                if idx >= len(rec_ids):
                    break
                gid = rec_ids[idx]; idx += 1
                item = st.session_state.id2item.get(gid, {})
                img = load_image(item)
                if img is not None:
                    cropped_img = ImageOps.fit(img, (450, 450), method=Image.Resampling.LANCZOS, centering=(0.5, 0.5))
                else:
                    cropped_img = None
                with cols[c]:
                    st.markdown('<div class="art-card">', unsafe_allow_html=True)

                    with st.popover("Ingrandisci 🔍", use_container_width=True):
                        if img is not None:
                            st.image(img, use_container_width=True)

                    if img is not None:
                        st.image(cropped_img, use_container_width=True)
                    else:
                        st.markdown('<div class="img-missing">Immagine locale non trovata</div>', unsafe_allow_html=True)

                    st.markdown(
                        f"<div class='titleline'><span class='title'>{item.get('title','Senza titolo')}</span> "
                        f"<span class='meta'>({item.get('year','?')})</span></div>",
                        unsafe_allow_html=True
                    )

                    with st.popover("Perche?", use_container_width=True):
                        explanations.get(gid, '')

                    st.markdown('</div>', unsafe_allow_html=True)

    with right:
        likert_opts = ["Per niente d'accordo", "In disaccordo", "Neutrale", "D'accordo", "Totamente d'accordo"]
        with st.form("likert_form_side", clear_on_submit=False):
            st.markdown(
                """
                <div class="form-title-wrap">
                  <h2 class="form-title">Dicci cosa pensi di queste raccomandazioni</h2>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown("**Questi dipinti rispecchiano le mie preferenze e interessi personali.**")
            q1 = st.radio("accuracy", options=likert_opts, index=2, key="q1", label_visibility="collapsed")

            st.markdown("**Questi dipinti sono tra loro diversi.**")
            q2 = st.radio("diversity", options=likert_opts, index=2, key="q2", label_visibility="collapsed")

            st.markdown("**Ho scoperto dipinti che non conoscevo.**")
            q3 = st.radio("novelty", options=likert_opts, index=2, key="q3", label_visibility="collapsed")

            st.markdown("**Ho trovato dipinti sorprendentemente interessanti.**")
            q4 = st.radio("serendipity", options=likert_opts, index=2, key="q4", label_visibility="collapsed")

            st.markdown("**Ho capito chiaramente perché questi dipinti mi sono stati raccomandati.**")
            q5 = st.radio("explanation transparency", options=likert_opts, index=2, key="q5", label_visibility="collapsed")

            st.markdown("**La spiegazione ha contribuito ad aumentare la mia fiducia nelle raccomandazioni proposte.**")
            q6 = st.radio("explanation usefulness", options=likert_opts, index=2, key="q6", label_visibility="collapsed")

            st.markdown("**La spiegazione era chiara e comprensibile.**")
            q7 = st.radio("explanation clarity", options=likert_opts, index=2, key="q7", label_visibility="collapsed")

            st.markdown("**La spiegazione mi ha aiutato a capire perché l'opera era raccomandata.**")
            q8 = st.radio("trust", options=likert_opts, index=2, key="q8", label_visibility="collapsed")

            submitted = st.form_submit_button("Invia", use_container_width=True)

        if submitted:
            ratings = {
                "preference": likert_opts.index(q1) + 1,
                "diversity": likert_opts.index(q2) + 1,
                "novelty": likert_opts.index(q3) + 1,
                "serendipity": likert_opts.index(q4) + 1,
                "transparency": likert_opts.index(q5) + 1,
                "trust": likert_opts.index(q6) + 1,
                "understanding": likert_opts.index(q7) + 1,
                "usefulness": likert_opts.index(q8) + 1
            }
            dur_ms = int((time.time() - st.session_state.rec_start_ts) * 1000)  
            log_complete_study_session(rec_ids, scores, ratings, dur_ms)
            st.session_state.phase = "done"
            st.rerun()

def screen_done():
    st.success("Grazie! Hai completato il questionario.")
    st.caption(f"Il tuo codice partecipante: {st.session_state.user_id}")


def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    load_css("./style.css")

    json_path = DEFAULT_JSON_PATH
    if not os.path.exists(json_path):
        st.error(f"Percorso JSON non trovato: {json_path}")
        st.stop()
        
    data = load_artpedia(json_path)
    id2item = {it["id"]: it for it in data}
    st.session_state.id2item = id2item

    ensure_embeddings_local()
    
    if "pack" not in st.session_state:
        try:
            with st.spinner("Carico embedding precomputati…"):
                pack = load_embeddings_from_file(EMB_NPZ_PATH)

                data_ids = np.array([it["id"] for it in data], dtype=np.int64)
                if not np.array_equal(pack.ids, data_ids):
                    idx_map = {int(g): i for i, g in enumerate(pack.ids.tolist())}
                    order = [idx_map[g] for g in data_ids if g in idx_map]
                    pack = EmbeddingPack(
                        ids=np.array([g for g in data_ids if g in idx_map], dtype=np.int64),                        
                        img=pack.img[order],
                        vis=pack.vis[order],
                        ctx=pack.ctx[order],
                        meta=pack.meta[order],
                    )
                    if len(order) != len(data_ids):
                        st.warning(
                            "Attenzione: alcuni ID del JSON non sono presenti nel NPZ, verranno ignorati."
                        )
                st.session_state.pack = pack
        except Exception as e:
            st.error(f"Errore nel caricamento embeddings: {e}")
            st.stop()

    w = (ALPHA, BETA, GAMMA, DELTA)  
    _init_session()

    phase = st.session_state.phase
    if phase == "consent":
        screen_consent()
    elif phase == "demographic":
        screen_demographic()
    elif phase == "background":
        screen_background_questions()
    elif phase == "seed":
        screen_seed_select(data)
    elif phase == "rec_seq":
        screen_recommend_sequential()
    elif phase == "rec":
        screen_recommend(data, w)
    else:
        screen_done()



if __name__ == "__main__":
    main()
