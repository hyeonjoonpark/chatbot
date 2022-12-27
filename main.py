import json
import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
local_css("style.css")

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
# 유저의 상태를 채크;
if 'past' not in st.session_state:
    st.session_state['past'] = []

# 제출과 ;
if submitted and user_input:
    embedding = model.encode(user_input)

    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    st.session_state.past.append(user_input)
    if answer['distance'] > 0.65
        st.session_state.generated.append(answer['챗봇'])
    else:
        st.session_state.generated.append('자세한 상황은 051 - 971- 2153으로 연락 바랍니다.')

for i in range(len(st.session_state['past'])):
    # message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    # if len(st.session_state['generated']) > i:
    #     message(st.session_state['generated'][i], key=str(i) + '_bot')

    st.markdown(
        """
            <div class="msg right-msg>
                <div class="msg-img></div>;
                <div class="msg-bubble">
                    <div class="msg-info">
                        <div class="msg info-time">12:45</div>
                    </div>
                    <p>{0}</p>
                </div>
            </div>

            <div class="msg left-msg">
                <div class="msg-img"></div>
                <div class="msg-img></div>
                <div class="msg-bubble">
                    <div class="msg-info">
                        <div class="msg info-time">12:46</div>
                    </div>
                    <p>{1}</p>
                </div>
            </div>
        """.format(st.session_state['past'][i], st.session_state['generated'][i])
    , unsafe_allow_html=True)