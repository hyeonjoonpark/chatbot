from datetime import datetime
import json
import streamlit as st
from streamlit_chat import message
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(layout = "wide")


tab1, tab2, tab3, tab4= st.tabs(["학교소개", "입학안내", "문의", "챗봇 주의사항"])

with tab1 :
    st.markdown(
    """
        부산소마고를 소개합니다
    """
    )


with tab2 :
    st.markdown(
    """
        입학안내
    """
    )


with tab3 :
    st.markdown(
    """
        문의하러 가기 : 051 - 971- 2153
    """
    )

with tab4 :
    st.markdown(
        """
            주의사항 : 챗봇이 질문을 이해 잘 못할 수도 있습니다.
                     답을 얻지 못하면 옆에 문의하기를 눌러서 문의해주시기 바랍니다.
        """
    )


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

st.header('부산소마고 챗봇')




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
    if answer['distance'] > 0.7 :
        st.session_state.generated.append(answer['챗봇'])
    else:
        st.session_state.generated.append('이해를 하지 못했어요. 문의를 할려면 위쪽 상단에 문의를 눌러주시기 바랍니다.')


for i in range(len(st.session_state['past'])):
    # message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
    # if len(st.session_state['generated']) > i:
    #     message(st.session_state['generated'][i], key=str(i) + '_bot')

    st.markdown(
        """
            <div class="msg right-msg>
                <div class="msg-img></div>
                <div class="msg-bubble">
                    <div class="msg-info">
                        <div class="msg info-time">유저</div>
                    </div>
                    <p>{0}</p>
                </div>
            </div>

            <div class="msg left-msg">
                <div class="msg-img"></div>
                <div class="msg-img></div>
                <div class="msg-bubble">
                    <div class="msg-info">
                        <div class="msg info-time">챗봇</div>
                    </div>
                    <p>{1}</p>
                </div>
            </div>
        """.format(st.session_state['past'][i], st.session_state['generated'][i])
    , unsafe_allow_html=True)



st.sidebar.title("부산소마고 SNS")
st.sidebar.info(
    """
        [HomePage](https://school.busanedu.net/bssm-h/main.do)
        [Instagram](https://www.instagram.com/bssm.hs/)
        [Facebook](https://www.facebook.com/BusanSoftwareMeisterHighschool/)
        
    """
)

st.sidebar.title("학생 제작 사이트")
st.sidebar.info(
    """
        [BSM](https://bssm.kro.kr/)
        [BSMboo](https://bsmboo.kro.kr/)
    """
)

st.sidebar.title("contact")
st.sidebar.info(
    """
        Phone : 051-971-2153
    """
)

st.sidebar.title("Dev Github")
st.sidebar.info(
    """
        [Developer Github](https://github.com/hyeonjoonpark)
    """
)