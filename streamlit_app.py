import streamlit as st
import pandas as pd
from app.llm_interface import generate_text
from app.utils import row_to_prompt
from app.rag import build_index, retrieve_similar

st.set_page_config(page_title='LLM Data Text Prototype', layout='wide')
st.title('LLM Data Text Prototype')

with st.sidebar:
    st.header('Settings')
    model_option = st.selectbox('Model backend', ['Fallback', 'Local HF', 'OpenAI'])
    use_rag = st.checkbox('Use RAG context', value=False)
    if st.button('Build RAG index (if not exists)'):
        with st.spinner('Building RAG index...'):
            try:
                build_index()
                st.success('RAG index built')
            except Exception as e:
                st.error(str(e))

df = pd.read_csv('data/sample_stats.csv')
st.dataframe(df)
idx = st.number_input('Row index', min_value=0, max_value=len(df)-1, value=0)
row = df.iloc[int(idx)].to_dict()
st.json(row)
prompt = row_to_prompt(row, 'date', 'indicator', 'value', 'region')
st.text_area('Prompt', value=prompt, height=160)
if st.button('Generate'):
    backend = 'fallback' if model_option=='Fallback' else ('hf' if model_option=='Local HF' else 'openai')
    out = generate_text(prompt, backend=backend)
    st.write('Generated:')
    st.write(out)
    if use_rag:
        with st.expander('RAG results'):
            try:
                res = retrieve_similar(row, k=3)
                for r in res:
                    st.write(r['metadata'])
            except Exception as e:
                st.error('RAG failed: ' + str(e))
