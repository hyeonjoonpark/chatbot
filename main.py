import json
import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

@st.cache(allow_output_mutation=True)
def cached_model():
    model = SentenceTransformer('jhgan/ko-sroberta-multitask')
    return model

@st.cache(allow_output_mutation=True)
def get_dataset():
    df = pd.read_csv('챗봇.csv')
    df['embedding'] = df['embedding'].apply(json.loads)
    return df

model = cached_model()
df = get_dataset()

st.header('ㅂㅅ소마고 챗봇')
st.subheader("ㅂㅅ소마고 챗봇임")




with st.form('form', clear_on_submit=True):
    user_input = st.text_input('사용자 : ', '')
    submitted = st.form_submit_button('전송')
# 유저의 상태를 체크(세션 생성)
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
# 유저의 상태를 채크
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 제출과 
if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    if answer['distance'] > 0.5:
        st.session_state.generated.append(answer['챗봇'])
    else:
        st.session_state.generated.append('히히 무슨 말인지 모름')

for i in range(len(st.session_state['past'])):
    message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    if len(st.session_state['generated']) > i:
        message(st.session_state['generated'][i], key=str(i) + '_bot')



# picture = st.camera_input("Take a picture")



img_file_buffer = st.camera_input("Take a picture!")

if img_file_buffer:
    st.image(img_file_buffer)

if img_file_buffer is not None:
    bytes_data = img_file_buffer.getvalue()
    # st.write(type(bytes_data))