import os
import json
import time
import csv
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

from openai import OpenAI


import torch

import streamlit as st
from pathlib import Path
from urllib.parse import urlparse
import unicodedata

import gspread
from google.oauth2.service_account import Credentials

# =========================
# CONFIGURAZIONE
# =========================

IMG_ROOT = Path("./images")
APP_TITLE = "Recommender d'Arte"
DEFAULT_JSON_PATH = "./data/artpedia.json"   # path al JSON unico Artpedia (MIX, con split originali)
EMB_NPZ_PATH = "./data/embeddings/artpedia_blip_base_all.npz"  # embedding pre-calcolati per TUTTO il dataset
IMG_CACHE_DIR = "./.cache/images"
LOG_DIR = "./logs"
TOPK_SEED = 12  # 12 immagini nella schermata di selezione
TOPK_REC  = 12  # 12 raccomandazioni nella schermata successiva

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or st.secrets["OPENROUTER_API_KEY"]
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

ALPHA, BETA, GAMMA, DELTA = 0.50, 0.25, 0.15, 0.10


# =========================
# CSS loader
# =========================
def load_css(path: str = "./style.css"):
    try:
        with open(path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.sidebar.warning(f"CSS non caricato ({path}): {e}")

# =========================
# Utility base
# =========================

@st.cache_data(show_spinner=False)
def load_artpedia(json_path: str) -> List[Dict]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items: List[Dict] = []

    if isinstance(data, dict):
        # Caso A: dict di item -> chiave = id
        # Caso B: dict per split -> chiave = nome split, valore = lista di item
        looks_like_items = all(isinstance(v, dict) for v in data.values())
        if looks_like_items:
            running_id = 0
            for k, it in data.items():
                if not isinstance(it, dict):
                    continue
                it = it.copy()
                # prova a usare l'id della chiave, altrimenti assegna un id progressivo
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
            # dict per split: {split: [items]}
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
        # lista MIX piatta
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



def ensure_dirs():
    os.makedirs(os.path.dirname(EMB_NPZ_PATH), exist_ok=True)
    os.makedirs(IMG_CACHE_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

def ensure_embeddings_local():
    """Scarica l'NPZ dagli URL (Secrets/ENV) solo se non esiste in locale."""
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
    """
    Apre il foglio Google usando la service account nei Secrets.
    Ritorna il primo worksheet (sheet1).
    """
    sa_info = st.secrets["GOOGLE_SERVICE_ACCOUNT_JSON"]
    
    # DEBUG: Verifica il tipo
    print(f"Tipo di sa_info: {type(sa_info)}")
    
    # Se è già un dizionario, usalo direttamente
    if isinstance(sa_info, dict):
        creds = Credentials.from_service_account_info(
            sa_info,
            scopes=["https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"]
        )
    # Se è una stringa, parsala come JSON
    elif isinstance(sa_info, str):
        creds = Credentials.from_service_account_info(
            json.loads(sa_info),
            scopes=["https://www.googleapis.com/auth/spreadsheets",
                    "https://www.googleapis.com/auth/drive"]
        )
    else:
        raise ValueError(f"Tipo non supportato per service account: {type(sa_info)}")
    
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
    """
    Apre l'immagine ignorando il limite MAX_IMAGE_PIXELS e la riduce
    entro:
      - area massima = max_megapixels (MP)
      - lato massimo = max_side (px)
    """
    if path is None or not path.exists():
        return None

    old_max = Image.MAX_IMAGE_PIXELS
    try:
        Image.MAX_IMAGE_PIXELS = None  # niente DecompressionBomb
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)

        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)  # orientamento corretto

            w, h = im.size
            # vincolo lato
            s_side = min(max_side / max(w, h), 1.0)
            # vincolo MP
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
    """
    Carica l'immagine SOLO dalla cartella locale:
    ./images/<split>/<hash8>_<basename_url>.jpg

    Tenta anche fallback utili:
    - qualunque file che finisca con _<basename_url>
    - <id>.jpg / <id>.png
    """
    split = str(item.get("split", "train"))
    url = item.get("img_url") or ""

    candidates = []
    # nome "canone" usato dallo script di preparazione: 8 hex + "_" + basename
    if url:
        fname = hashed_filename(url)                 # es. 0a4d6678_Diego_...jpg
        candidates.append(IMG_ROOT / split / fname)
        # fallback: qualunque file che termini con _<basename_url>
        candidates.extend((IMG_ROOT / split).glob(f"*_{basename_from_url(url)}"))

    # ulteriori fallback: id.jpg / id.png
    iid = item.get("id")
    if iid is not None:
        candidates.append(IMG_ROOT / split / f"{iid}.jpg")
        candidates.append(IMG_ROOT / split / f"{iid}.png")

    for p in candidates:
        try:
            if p and p.exists():
                im = _open_resized(p)     # <— usa l’opener che riduce
                if im is not None:
                    return im
        except Exception:
            continue
    return None

def _download_remote_image(split: str, filename: str) -> Optional[Path]:
    """
    Scarica un singolo file da Hugging Face Datasets in ./<split>/<filename>
    se IMAGES_BASE_URL è definita. Ritorna il Path locale se scaricato/esistente.
    """
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
        # opzionale: st.sidebar.warning(f"Download immagine fallito: {e}")
        if dst_path.exists():
            try:
                dst_path.unlink()
            except Exception:
                pass
        return None


def load_image(item: Dict) -> Optional[Image.Image]:
    """
    Prova a caricare l'immagine locale; se manca, tenta il download da HF usando
    lo stesso nome file che la tua app si aspetta (hash8_basename o id.jpg/png),
    poi riprova il caricamento locale.
    """
    # 1) tentativo locale
    img = load_image_local(item)
    if img is not None:
        return img

    # 2) se non c'è, prova a scaricare da HF con i nomi candidati
    split = str(item.get("split", "train"))
    url = item.get("img_url") or ""
    candidates = []

    if url:
        # nome canonico che già usi in locale
        candidates.append(hashed_filename(url))
        # eventuali varianti commonsense del basename (se le hai caricate così)
        candidates.append(basename_from_url(url))

    # fallback per id.jpg/png
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

    # 3) se non riesce, None (la UI mostrerà "Immagine locale non trovata")
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
    """Carica SOLO da NPZ (niente calcolo) e ritorna l'EmbeddingPack."""
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


# =========================
# Explainability
# =========================

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
    """Restituisce "A", "A e B" oppure "A, B e C"."""
    names = [n for n in names if n]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    if len(names) == 2:
        return f"{names[0]} e {names[1]}"
    return f"{', '.join(names[:-1])} e {names[-1]}"


# =========================
# CONTENT-BASED
# =========================
def content_based_explanation_gid(rec_gid: int,
                                  seed_gids: list,
                                  pack,
                                  w,
                                  k_refs: int = 2) -> str:
    """
    Crea la frase: 'Ti abbiamo raccomandato {A} perché hai apprezzato opere come {B} e {C}.'
    usando la similarità tra l'opera raccomandata e i seed selezionati.
    """
    # mappa gid -> indice di riga negli embedding
    gid_list = pack.ids.tolist()
    idx_map = {int(g): i for i, g in enumerate(gid_list)}

    # fallback se mancano dati
    item = st.session_state.id2item.get(rec_gid, {})
    rec_title = item.get("title", "questa opera")

    if rec_gid not in idx_map or not seed_gids:
        return f"Ti abbiamo raccomandato *{rec_title}* perché riteniamo che sia in linea con i tuoi gusti."

    # embedding fusi per tutto il catalogo (stessa fusione usata nel ranking)
    fused = fuse_modalities(pack.img, pack.vis, pack.ctx, pack.meta, w)

    # vettore della raccomandazione
    rv = fused[idx_map[rec_gid]]
    rvn = np.linalg.norm(rv) + 1e-9

    # calcola similarità con i seed
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

    # prendi i k riferimenti più simili (2 di default)
    sims.sort(key=lambda x: x[1], reverse=True)
    top_refs = [g for g, _ in sims[:max(1, k_refs)]]

    # titoli delle opere di riferimento
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
    """
    Restituisce la spiegazione in base al gruppo sperimentale dell'utente.
    """
    item = st.session_state.id2item.get(rec_gid, {})
    rec_title = item.get("title", "questa opera")
    
    if st.session_state.exp_style == "CONTENT-BASED":
        return content_based_explanation_gid(rec_gid, seed_gids, pack, w, k_refs=2)
    else:  # LLM-EXPLANATION
        # Prepara le preferenze dell'utente basate sulle opere seed selezionate
        prefs = {
            "liked_titles": [],
            "liked_styles": []
        }
        
        # Estrai titoli dalle opere seed
        for seed_gid in seed_gids:
            seed_item = st.session_state.id2item.get(seed_gid, {})
            if seed_item.get("title"):
                prefs["liked_titles"].append(seed_item["title"])
        
        try:
            explanation, latency_ms = generate_llm_explanation_gemini_flash(item, prefs)
            return explanation
        except Exception as e:
            # Fallback a spiegazione content-based in caso di errore
            return f"Ti abbiamo raccomandato *{rec_title}* perché riteniamo che sia in linea con i tuoi gusti."

# =========================
# LLM
# =========================



def build_llm_system() -> str:
    return ("Sei un esperto di storia dell'arte che genera spiegazioni chiare e concise "
            "per raccomandazioni d'arte. Usa SOLO le informazioni fornite nel contesto. "
            "Non inventare dettagli non presenti. La spiegazione deve essere lunga 50-100 parole. "
            "Stile: professionale ma accessibile. Collega 1-2 opere apprezzate dall'utente "
            "con 1-2 caratteristiche dell'opera raccomandata.")

def build_llm_user_context(item: dict, prefs: dict) -> str:
    def _slice(lst, k): 
        lst = (lst or [])[:k]
        return [s.strip().replace("\n", " ") for s in lst if s and s.strip()]
    
    # Estrai frasi visive e contestuali dall'opera
    VIS = " | ".join(_slice(item.get("visual_sentences", []), 2))
    CTX = " | ".join(_slice(item.get("contextual_sentences", []), 1))
    
    # Preferenze utente
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
                                          max_tokens: int = 220, temperature: float = 0.2):
    """
    Genera una spiegazione LLM per una raccomandazione d'arte usando Gemini Flash via OpenRouter.
    
    Args:
        item: Dizionario con i dati dell'opera raccomandata
        prefs: Dizionario con le preferenze dell'utente (titoli e stili apprezzati)
        max_tokens: Numero massimo di token per la risposta
        temperature: Livello di creatività (0.0-1.0)
    
    Returns:
        tuple: (testo spiegazione, latenza in millisecondi)
    """
    system = build_llm_system()
    user = build_llm_user_context(item, prefs)

    try:
        t0 = time.time()
        
        # Aggiungi headers opzionali per OpenRouter rankings
        extra_headers = {
            "HTTP-Referer": "https://your-art-recommender.com",  # Sostituisci con il tuo URL
            "X-Title": "Art Recommendation Study"  # Nome del tuo studio
        }
        
        resp = client.chat.completions.create(
            model="google/gemini-2.5-flash",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            # Aggiungi gli headers extra per OpenRouter
            extra_headers=extra_headers
        )
        
        text = resp.choices[0].message.content.strip()
        latency_ms = (time.time() - t0) * 1000.0
        
        # Log della richiesta LLM (opzionale, per debug)
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
        # Log dell'errore
        error_msg = f"Errore LLM per opera {item.get('id', 'unknown')}: {str(e)}"
        print(error_msg)  # Puoi sostituire con un logger più sofisticato
        
        # Spiegazione di fallback
        fallback_explanation = (
            f"Ti raccomandiamo '{item.get('title', 'questa opera')}' "
            f"perché presenta caratteristiche in linea con le opere che hai apprezzato. "
            f"L'opera combina elementi stilistici e tematici che riteniamo possano corrispondere ai tuoi gusti."
        )
        
        return fallback_explanation, 0

# =========================
# Recommender core
# =========================

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



def log_row(name: str, row: Dict):
    """Logga una riga nel CSV unico"""
    ensure_dirs()
    path = os.path.join(LOG_DIR, f"{name}.csv")
    is_new = not os.path.exists(path)
    
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS[name])
        if is_new:
            writer.writeheader()
        
        # Assicura che tutte le colonne siano presenti
        complete_row = {col: row.get(col, "") for col in HEADERS[name]}
        writer.writerow(complete_row)

def append_to_google_sheet(row: Dict):
    """
    Aggiunge una riga al foglio Google con le stesse colonne del CSV.
    Assume che la riga 1 del foglio contenga già l'header esatto.
    """
    ws = get_gsheet_worksheet()

    header = HEADERS["art_recommendation_study"]  # ordine colonne già definito nel tuo codice
    values = []
    for col in header:
        v = row.get(col, "")
        # Normalizza valori None
        if v is None:
            v = ""
        # Lasciamo le liste come stringhe JSON (il tuo row le ha già così via format_list_for_csv)
        values.append(v)
    try:
        ws.append_row(values, value_input_option="USER_ENTERED")
    except Exception as e:
        # Non bloccare l'app se il foglio non è raggiungibile
        st.sidebar.warning(f"Append su Google Sheets non riuscito: {e}")

    
def get_artwork_titles(artwork_ids: List[int]) -> str:
    """Converti IDs opere in stringa di titoli leggibile"""
    titles = []
    for art_id in artwork_ids:
        item = st.session_state.id2item.get(art_id, {})
        title = item.get('title', f'Unknown_{art_id}')
        year = item.get('year', '')
        title_str = f"{title} ({year})" if year else title
        titles.append(f"{title_str} [{art_id}]")
    return " | ".join(titles)

def format_list_for_csv(data: List) -> str:
    """Formatta liste in stringa JSON compatta"""
    return json.dumps(data, ensure_ascii=False)

def log_complete_study_session(rec_ids, scores, ratings, rec_duration_ms):
    """Logga tutti i dati dello studio in una riga unica"""
    
    explanation_type = "LLM" if st.session_state.exp_style == "LLM-EXPLANATION" else "CONTENT_BASED"
    
    # Calcola tempi
    total_duration = int((time.time() - st.session_state.seed_start_ts) * 1000)
    selection_duration = int((st.session_state.rec_start_ts - st.session_state.seed_start_ts) * 1000)
    
    # Prepara i dati per il CSV
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
        
        # METADATI PER ANALISI
        "selected_artwork_titles": get_artwork_titles(st.session_state.seed_selected_ids),
        "recommended_artwork_titles": get_artwork_titles(rec_ids)
    }
    append_to_google_sheet(row)
    log_row("art_recommendation_study", row)

# =========================
# UI — Stato e Schermate
# =========================

def _assign_group(user_id: str) -> str:
    """Ritorna 'CONTENT-BASED' o 'LLM-EXPLANATION' in modo deterministico dall'user_id."""
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
    st.session_state.setdefault("slate_id", None)  # Inizializza a None


def screen_consent():
    st.header("Consenso informato")
    st.write("Questa raccolta è anonima. I dati registrati riguardano le tue scelte e le risposte al questionario.")
    agree = st.checkbox("Acconsento all'uso anonimo dei miei dati per fini di ricerca", value=False)
    if st.button("Inizia", width='stretch'):
        if not agree:
            st.warning("Devi acconsentire per proseguire.")
            return
        
        st.session_state.phase = "demographic"  # Vai a demographic invece di seed
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

    # Validazione e salvataggio
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
    """
    Schermata con 4 domande di background sull'arte.
    Salva i risultati in st.session_state.user_demographics
    e, se valido, passa alla fase successiva (seed).
    """
    st.header("Domande preliminari")
    st.write("Rispondi a queste domande per aiutarci a capire il tuo rapporto con l'arte.")

    # 1) Frequenza visite a mostre/musei (obbligatoria)
    visit_opts = ["", "Mai", "1–2 volte", "3–5 volte", "Più di 5 volte"]
    default_visit = st.session_state.user_demographics.get("museum_visits", "")
    visit_museums = st.selectbox(
        "Quante volte visiti mostre/musei d’arte in un anno?*", visit_opts,
        index=visit_opts.index(default_visit) if default_visit in visit_opts else 0,
        key="q_museum_visits"
    )

    # 2) Apprezzamento dell’arte (obbligatoria)
    appreciate_opts = ["", "Per niente", "Poco", "Abbastanza", "Molto"]
    default_app = st.session_state.user_demographics.get("appreciate_art", "")
    appreciate_art = st.selectbox(
        "Quanto apprezzi l’arte in generale?*", appreciate_opts,
        index=appreciate_opts.index(default_app) if default_app in appreciate_opts else 0,
        key="q_appreciate_art"
    )

    # 3) Conoscenza/esperienza in arte (obbligatoria)
    knowledge_opts = ["", "Per niente", "Poco", "Abbastanza", "Molto"]
    default_kn = st.session_state.user_demographics.get("art_knowledge_level", "")
    art_knowledge_level = st.selectbox(
        "Quanto ti consideri esperto/a di arte?*", knowledge_opts,
        index=knowledge_opts.index(default_kn) if default_kn in knowledge_opts else 0,
        key="q_art_knowledge_level"
    )

    # 4) Conoscenza dei sistemi di raccomandazione (obbligatoria)
    heard_opts = ["", "Si", "No"]
    default_heard = st.session_state.user_demographics.get("heard_recommenders", "")
    heard_recommenders = st.selectbox(
        "Hai mai sentito parlare dei sistemi di raccomandazione?*", heard_opts,
        index=heard_opts.index(default_heard) if default_heard in heard_opts else 0,
        key="q_heard_recs"
    )

    # ---- Validazione + salvataggio ----
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

        # Salva nello stato utente
        st.session_state.user_demographics.update({
            "museum_visits": visit_museums,
            "appreciate_art": appreciate_art,
            "art_knowledge_level": art_knowledge_level,
            "heard_recommenders": heard_recommenders,
        })

        # Avanza al seed
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
    valid = set(map(int, st.session_state.pack.ids.tolist()))   # solo ID con embedding
    ids = [g for g in all_ids if g in valid]
    random.shuffle(ids)

    out, scanned = [], 0
    budget = 10 * k  # non scandire tutto il dataset

    for gid in ids:
        it = st.session_state.id2item[gid]

        # 1) se c'è già in locale, ok
        if find_local_image_path(it):
            out.append(gid)
        else:
            # 2) tenta un download "on demand" da HF
            split = str(it.get("split", "train"))
            url = it.get("img_url") or ""
            tried = False

            # prova prima il nome canonico hash8_basename
            if url and not tried:
                fname = hashed_filename(url)
                p = _download_remote_image(split, fname)
                tried = True
                if p and p.exists():
                    out.append(gid)

            # se non basta, prova anche il basename "puro"
            if url and (len(out) < k) and not find_local_image_path(it):
                base = basename_from_url(url)
                p2 = _download_remote_image(split, base)
                if p2 and p2.exists():
                    out.append(gid)

        scanned += 1
        if len(out) >= k or scanned >= budget:
            break

    return out




def screen_seed_select(data: List[Dict]):
    st.subheader("Seleziona 4 dipinti che ti piacciono")

    if not st.session_state.seed_pool_ids:
        all_ids = list(st.session_state.id2item.keys())
        st.session_state.seed_pool_ids = _sample_seed_pool(all_ids, TOPK_SEED)

    ids = st.session_state.seed_pool_ids[:12]  # 3×4
    sel = set(st.session_state.seed_selected_ids)

    rows, cols_per_row = 4, 3
    idx = 0
    for r in range(rows):
        cols = st.columns(
            3,
            gap="small",
            vertical_alignment="bottom",
            border=True,
            width="stretch",
        )
        for c in range(cols_per_row):
            if idx >= len(ids):
                break
            gid = ids[idx]; idx += 1
            item = st.session_state.id2item[gid]
            is_sel = gid in sel
            reached_limit = len(sel) >= 4
            disabled_this = (reached_limit and not is_sel)

            with cols[c]:
                st.markdown(f'<div class="art-card {"is-selected" if is_sel else ""}">', unsafe_allow_html=True)

                img = load_image(item)
                if img is not None:
                    show_img = img
                    if disabled_this and not is_sel:
                        show_img = ImageEnhance.Color(show_img).enhance(0.2)
                        show_img = ImageEnhance.Brightness(show_img).enhance(0.75)
                    st.image(show_img, width='stretch')
                else:
                    st.markdown('<div class="img-missing">Immagine locale non trovata</div>', unsafe_allow_html=True)

                st.markdown(
                    f"<div class='titleline'><span class='title'>{item.get('title','Senza titolo')}</span> "
                    f"<span class='meta'>({item.get('year','?')})</span></div>",
                    unsafe_allow_html=True
                )

                if st.button(("Deseleziona" if is_sel else "Seleziona"),
                             key=f"tap_{gid}_{r}_{c}",
                             disabled=(disabled_this and not is_sel),
                             width='stretch'):
                    
                    if is_sel: sel.remove(gid)
                    else:      sel.add(gid)
                    st.session_state.seed_selected_ids = list(sel)
                    st.rerun()

                st.markdown("</div>", unsafe_allow_html=True)

    st.write(f"Selezionati: **{len(sel)}/4**")

    if st.button("Genera raccomandazioni",
                 disabled=(len(sel) != 4),
                 width='stretch',
                 type="primary"):
        st.session_state.slate_id = secrets.token_hex(6)
        st.session_state.phase = "rec"
        st.session_state.rec_start_ts = time.time()
        st.rerun()



def screen_recommend(data: List[Dict], w: Tuple[float, float, float, float]):
    st.subheader("Raccomandazioni per te")

    pack = st.session_state.pack
    if pack is None:
        st.error("Embedding non caricati. Controlla EMB_NPZ_PATH.")
        st.stop()

    selected = st.session_state.seed_selected_ids

    # mappa gid -> indice riga negli embedding
    gid_to_row = {int(g): i for i, g in enumerate(pack.ids.tolist())}
    seed_rows = [gid_to_row[g] for g in selected if g in gid_to_row]

    # Profilo utente + ranking
    user_vecs = build_user_profile(pack, seed_rows, w)
    results = rank_items(pack, user_vecs, w, exclude_global_idx=selected, topk=TOPK_REC)

    # STEP A: accumulo
    rec_ids: List[int] = []
    scores: List[float] = []
    explanations: Dict[int, str] = {}

    for gid, score, contrib in results:
        rec_ids.append(gid)
        scores.append(round(score, 6))
        explanation = get_explanation_for_item(gid, selected, pack, w)
        # Pulisci e formatta il testo (rimuovi markdown se presente)
        explanation_clean = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", explanation)
        explanation_clean = re.sub(r"\*(.+?)\*", r"<em>\1</em>", explanation_clean)
        explanations[gid] = explanation_clean

    rec_ids = rec_ids[:TOPK_REC]
    scores  = scores[:TOPK_REC]

    rows, cols_per_row = 4, 3
    idx = 0
    for r in range(rows):
        cols = st.columns(
            3,
            gap="small",
            vertical_alignment="bottom",
            border=True,
            width="stretch",
        )
        for c in range(cols_per_row):
            if idx >= len(rec_ids):
                break
            gid = rec_ids[idx]; idx += 1
            item = st.session_state.id2item.get(gid, {})
            img = load_image(item)

            with cols[c]:
                st.markdown('<div class="art-card">', unsafe_allow_html=True)

                if img is not None:
                    st.image(img, width='stretch')
                else:
                    st.markdown('<div class="img-missing">Immagine locale non trovata</div>', unsafe_allow_html=True)

                title = item.get('title', 'Senza titolo')
                year = item.get('year', '?')
                st.markdown(
                    f"<div class='titleline'><span class='title'>{title}</span> "
                    f"<span class='meta'>({year})</span></div>",
                    unsafe_allow_html=True
                )

                st.markdown(
                    f"<div class='exp-box'><strong>Perché:</strong> {explanations.get(gid, '')}</div>",
                    unsafe_allow_html=True
                )

                st.markdown('</div>', unsafe_allow_html=True)

    # ====== BLOCCO LIKERT ======
    st.markdown("""
        <div style='border:2px solid var(--accent); padding: 16px; border-radius: 8px;'>
          <h3>Dicci cosa pensi di queste raccomandazioni</h3> 
        </div>
    """, unsafe_allow_html=True)

    likert_opts = ["Per niente d'accordo", "In disaccordo", "Neutrale", "D'accordo", "Totalmente d'accordo"]
    st.markdown('<div class="likert-block">', unsafe_allow_html=True)
    with st.form("likert_form", clear_on_submit=False):
        st.markdown("#### Questi dipinti rispecchiano le mie preferenze e interessi personali.")
        q1 = st.radio("accuracy", options=likert_opts, index=2, key="q1", label_visibility="collapsed")

        st.markdown("#### Questi dipinti sono tra loro diversi.")
        q2 = st.radio("diversity", options=likert_opts, index=2, key="q2", label_visibility="collapsed")

        st.markdown("#### Ho scoperto dipinti che non conoscevo.")
        q3 = st.radio("novelty", options=likert_opts, index=2, key="q3", label_visibility="collapsed")

        st.markdown("#### Ho trovato dipinti sorprendentemente interessanti.")
        q4 = st.radio("serendipity", options=likert_opts, index=2, key="q4", label_visibility="collapsed")

        st.markdown("#### Ho capito chiaramente perché questi dipinti mi sono stati raccomandati.")
        q5 = st.radio("explanation transparency", options=likert_opts, index=2, key="q5", label_visibility="collapsed")

        st.markdown("#### La spiegazione ha contribuito ad aumentare la mia fiducia nelle raccomandazioni proposte.")
        q6 = st.radio("explanation usefulness", options=likert_opts, index=2, key="q6", label_visibility="collapsed")

        st.markdown("#### La spiegazione era chiara e comprensibile.")
        q7 = st.radio("explanation clarity", options=likert_opts, index=2, key="q7", label_visibility="collapsed")

        st.markdown("#### La spiegazione mi ha aiutato a capire perché l’opera era raccomandata.")
        q8 = st.radio("trust", options=likert_opts, index=2, key="q8", label_visibility="collapsed")

        submitted = st.form_submit_button("Invia", width="stretch")
    st.markdown('</div>', unsafe_allow_html=True)

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


# =========================
# MAIN
# =========================

def main():
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    load_css("./style.css")


    # JSON path (puoi mettere qui il path assoluto se preferisci)
    json_path = st.sidebar.text_input("Percorso JSON Artpedia", value=DEFAULT_JSON_PATH)
    if not os.path.exists(json_path):
        st.warning("Percorso JSON non trovato. Specifica il path corretto al file Artpedia.")
        st.stop()

    # Carico Artpedia e costruisco id2item
    data = load_artpedia(json_path)
    id2item = {it["id"]: it for it in data}
    st.session_state.id2item = id2item

    ensure_embeddings_local()
    
    # Precarico embedding precomputati all'avvio (UNA SOLA VOLTA)
    if "pack" not in st.session_state:
        try:
            with st.spinner("Carico embedding precomputati…"):
                pack = load_embeddings_from_file(EMB_NPZ_PATH)

                # Riallinea l'ordine se necessario (in base all'ordine degli item nel JSON attuale)
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

    w = (ALPHA, BETA, GAMMA, DELTA)  # pesi fissati
    _init_session()

    # Router schermate
    phase = st.session_state.phase
    if phase == "consent":
        screen_consent()
    elif phase == "demographic":
        screen_demographic()
    elif phase == "background":
        screen_background_questions()
    elif phase == "seed":
        screen_seed_select(data)
    elif phase == "rec":
        screen_recommend(data, w)
    else:
        screen_done()



if __name__ == "__main__":
    main()
