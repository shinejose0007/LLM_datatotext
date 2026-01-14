from app.utils import fallback_template, row_to_prompt
from app.llm_interface import generate_text
from app.rag import build_index, retrieve_similar

def test_fallback_basic():
    prompt = "- date: 2024-10\n- indicator: X\n- value: 1.2\n"
    out = fallback_template(prompt)
    assert isinstance(out, str) and '2024-10' in out

def test_generate_fallback():
    out = generate_text('hello', backend='fallback')
    assert isinstance(out, str) and len(out) > 0

def test_rag_build_and_retrieve():
    idx, n, model_local = build_index(force_rebuild=True)
    assert n > 0
    res = retrieve_similar({'date':'2024-10','indicator':'Unemployment rate','region':'Luxembourg','value':'5.1'}, k=2)
    assert isinstance(res, list)
    assert len(res) == 2
