import os, json, sqlite3
import numpy as np

try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

try:
    import faiss
except Exception:
    faiss = None

BASE = os.path.join('data', 'rag_index')
MODEL_BASE = os.path.join('data', 'emb_models')
os.makedirs(BASE, exist_ok=True)
os.makedirs(MODEL_BASE, exist_ok=True)

def _model_local_path(model_name):
    safe = model_name.replace('/', '_')
    return os.path.join(MODEL_BASE, safe)

def build_index(csv_path='data/sample_stats.csv', embed_model_name='all-MiniLM-L6-v2', force_rebuild=False):
    import pandas as pd
    df = pd.read_csv(csv_path)
    texts = []
    for i, row in df.iterrows():
        text = f"date: {row.get('date')} | indicator: {row.get('indicator')} | region: {row.get('region')} | value: {row.get('value')}"
        texts.append(text)

    model_local = _model_local_path(embed_model_name)
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers not installed')
    if not os.path.exists(model_local) or force_rebuild:
        model = SentenceTransformer(embed_model_name)
        try:
            model.save(model_local)
        except Exception:
            pass
    model = SentenceTransformer(model_local if os.path.exists(model_local) else embed_model_name)
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True).astype('float32')

    emb_path = os.path.join(BASE, 'embeddings.npy')
    np.save(emb_path, embeddings)

    if faiss is None:
        raise RuntimeError('faiss not installed')
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    faiss.write_index(index, os.path.join(BASE, 'index.faiss'))

    db_path = os.path.join(BASE, 'metadata.db')
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS metadata (id INTEGER PRIMARY KEY, row_index INTEGER, json TEXT)')
    c.execute('DELETE FROM metadata')
    for i, row in df.iterrows():
        c.execute('INSERT INTO metadata (row_index, json) VALUES (?,?)', (int(i), json.dumps(row.to_dict(), ensure_ascii=False)))
    conn.commit()
    conn.close()
    return index, len(df), model_local

def load_index():
    emb_path = os.path.join(BASE, 'embeddings.npy')
    idx_path = os.path.join(BASE, 'index.faiss')
    db_path = os.path.join(BASE, 'metadata.db')
    if not (os.path.exists(emb_path) and os.path.exists(idx_path) and os.path.exists(db_path)):
        return None, None
    if faiss is None:
        raise RuntimeError('faiss not installed')
    index = faiss.read_index(idx_path)
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('SELECT row_index, json FROM metadata ORDER BY id ASC')
    rows = c.fetchall()
    metadata = [json.loads(r[1]) for r in rows]
    conn.close()
    return index, metadata

def retrieve_similar(query_row_dict, k=3, embed_model_name='all-MiniLM-L6-v2'):
    index, metadata = load_index()
    if index is None or metadata is None:
        build_index(embed_model_name=embed_model_name)
        index, metadata = load_index()
    model_local = _model_local_path(embed_model_name)
    if SentenceTransformer is None:
        raise RuntimeError('sentence-transformers not installed')
    model = SentenceTransformer(model_local if os.path.exists(model_local) else embed_model_name)
    qtext = f"date: {query_row_dict.get('date')} | indicator: {query_row_dict.get('indicator')} | region: {query_row_dict.get('region')} | value: {query_row_dict.get('value')}"
    qemb = model.encode([qtext], convert_to_numpy=True).astype('float32')
    D, I = index.search(qemb, k)
    results = []
    for dist, idx in zip(D[0], I[0]):
        try:
            results.append({'distance': float(dist), 'metadata': metadata[int(idx)]})
        except Exception:
            pass
    return results
