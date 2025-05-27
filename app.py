import json
import streamlit as st
import pandas as pd  # type: ignore
from pycaret.clustering import load_model, predict_model  # type: ignore
import plotly.express as px  # type: ignore

# streamlit run app.py
# conda activate od_zera_do_ai

MODEL_NAME = 'welcome_survey_clustering_pipeline_v2'
DATA = 'welcome_survey_simple_v2.csv'
CLUSTER_NAMES_AND_DESCRIPTIONS = 'welcome_survey_cluster_names_and_descriptions_v2.json'

st.set_page_config(
    page_title="Znajomi wg AI",      
    page_icon="👋",                  
    layout="centered"         # "wide", "centered", "compressed"            
)

@st.cache_data
def get_model():
    return load_model(MODEL_NAME)


@st.cache_data
def get_cluster_names_and_descriptions():
    with open(CLUSTER_NAMES_AND_DESCRIPTIONS, "r", encoding='utf-8') as f:
        return json.loads(f.read())


with st.sidebar:
    st.header("Powiedz nam coś o sobie")
    st.markdown("Pomożemy Ci znaleźć osoby, które mają podobne zainteresowania")
    age = st.selectbox("Wiek", ['<18', '18-24', '25-34', '35-44', '45-54', '55-64', '>=65'])
    edu_level = st.selectbox("Wykształcenie", ['Podstawowe', 'Średnie', 'Wyższe'])
    fav_animals = st.selectbox("Ulubione zwierzęta", ['Brak ulubionych', 'Psy', 'Koty'])
    fav_place = st.selectbox("Ulubione miejsce", ['Nad wodą', 'W lesie', 'W górach'])
    gender = st.radio("Płeć", ['Kobieta','Mężczyzna'])

    person_df = pd.DataFrame([
        {
            'age': age,
            'edu_level': edu_level,
            'fav_animals': fav_animals,
            'fav_place': fav_place,
            'gender': gender,
        }
    ])


@st.cache_data
def get_all_participants():
    all_df = pd.read_csv(DATA, sep=';')
    df_with_clusters = predict_model(model, data=all_df)
    return df_with_clusters


model = get_model()
all_df = get_all_participants()
cluster_names_and_descriptions = get_cluster_names_and_descriptions()

predicted_cluster_id = predict_model(model, data=person_df)["Cluster"].values[0]
predicted_cluster_data = cluster_names_and_descriptions.get(predicted_cluster_id, {
    "name": "Nieznana grupa",
    "description": "Nie udało się dopasować do żadnej znanej grupy."
})

st.header(f"*Najbliżej Ci do grupy:* \n **{predicted_cluster_data['name']}**")

with st.container(border=True):
    c0, c1 = st.columns([1,4])
    with c0:
        same_cluster_df = all_df[all_df["Cluster"] == predicted_cluster_id]
        st.metric("Liczba znajomych", len(same_cluster_df))
        
    with c1:
        st.write(f"**Opis:** \n {predicted_cluster_data['description']}")
        

    # st.write("DEBUG:", predicted_cluster_id, cluster_names_and_descriptions)



st.header("Osoby z grupy")
c0,c1 = st.columns([1,1])
with c0:

    fig = px.histogram(same_cluster_df.sort_values("age"), x="age")
    fig.update_layout(
        title="Rozkład wieku w grupie",
        xaxis_title="Wiek",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with c1:
    fig = px.histogram(same_cluster_df, x="edu_level")
    fig.update_layout(
        title="Rozkład wykształcenia w grupie",
        xaxis_title="Wykształcenie",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

c0,c1 = st.columns([1,1])

with c0:
    fig = px.histogram(same_cluster_df, x="fav_animals")
    fig.update_layout(
        title="Rozkład ulubionych zwierząt w grupie",
        xaxis_title="Ulubione zwierzęta",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)

with c1:
    fig = px.histogram(same_cluster_df, x="fav_place")
    fig.update_layout(
        title="Rozkład ulubionych miejsc w grupie",
        xaxis_title="Ulubione miejsce",
        yaxis_title="Liczba osób",
    )
    st.plotly_chart(fig)


fig = px.histogram(same_cluster_df, x="gender")
fig.update_layout(
    title="Rozkład płci w grupie",
    xaxis_title="Płeć",
    yaxis_title="Liczba osób",
)
st.plotly_chart(fig)