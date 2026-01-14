import streamlit as st
import pandas as pd
import csv, os
from app.utils import row_to_prompt
from app.llm_interface import generate_text

st.title('Human Eval UI')
df = pd.read_csv('data/sample_stats.csv')
idx = st.number_input('Row index', min_value=0, max_value=len(df)-1, value=0)
row = df.iloc[int(idx)].to_dict()
st.json(row)
if st.button('Generate'):
    out = generate_text(row_to_prompt(row, 'date','indicator','value','region'), backend='fallback')
    st.write(out)
    st.session_state['latest'] = out
if st.session_state.get('latest'):
    st.radio('Factuality', [1,2,3,4,5], key='fact')
    st.radio('Clarity', [1,2,3,4,5], key='clar')
    if st.button('Save'):
        os.makedirs('data', exist_ok=True)
        with open('data/human_evals.csv','a', newline='', encoding='utf-8') as f:
            f.write(f"{idx},\"{st.session_state['latest'].replace('"','\'') }\",{st.session_state['fact']},{st.session_state['clar']}\n")
        st.success('Saved')
