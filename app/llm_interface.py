# Simple LLM interface with OpenAI, HuggingFace, and deterministic fallback.
import os
from app.utils import fallback_template

def generate_text(prompt: str, backend="openai", api_key=None, hf_model="gpt2", max_tokens=256):
    backend = backend.lower()
    if backend == "openai" or backend == 'openai':
        return _gen_openai(prompt, api_key=api_key, max_tokens=max_tokens)
    elif backend in ("hf", 'local', 'huggingface'):
        return _gen_hf(prompt, model_name=hf_model, max_tokens=max_tokens)
    else:
        return _gen_fallback(prompt)

def _gen_openai(prompt, api_key=None, max_tokens=256):
    try:
        import openai
    except Exception as e:
        raise RuntimeError("openai library not installed. Install via `pip install openai`.") from e
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _gen_fallback(prompt)
    openai.api_key = api_key
    try:
        resp = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role":"user","content":prompt}],
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        resp = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.2,
        )
        return resp.choices[0].text.strip()

def _gen_hf(prompt, model_name="gpt2", max_tokens=256):
    try:
        from transformers import pipeline, set_seed
    except Exception as e:
        raise RuntimeError("transformers library not installed. Install via `pip install transformers`") from e
    pipe = pipeline("text-generation", model=model_name, device=-1)
    set_seed(0)
    out = pipe(prompt, max_length=len(prompt.split())+max_tokens, do_sample=False, num_return_sequences=1)
    generated = out[0].get('generated_text', '')
    if generated.startswith(prompt):
        generated = generated[len(prompt):]
    return generated.strip()

def _gen_fallback(prompt):
    return fallback_template(prompt)
