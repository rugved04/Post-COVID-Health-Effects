import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Post-COVID Health Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    return pd.read_csv("post_covid_health_effects.csv")

df = load_data()
binary_map = {
    "Yes": 1,
    "No": 0
}

binary_cols = [
    "Brain_Fog",
    "Breathing_Issue",
    "Loss_of_Taste_Smell"
]

for col in binary_cols:
    df[col] = df[col].map(binary_map)

numeric_cols = [
    "Fatigue_Level",
    "Brain_Fog",
    "Breathing_Issue",
    "Loss_of_Taste_Smell",
    "Mental_Health_Impact",
    "Days_to_Recovery"
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# ---------------- SIDEBAR FILTERS ----------------
st.sidebar.title("Filters")

age_range = st.sidebar.slider(
    "Age Range",
    int(df["Age"].min()),
    int(df["Age"].max()),
    (20, 70)
)

gender = st.sidebar.multiselect(
    "Gender",
    df["Gender"].unique(),
    default=df["Gender"].unique()
)

severity = st.sidebar.multiselect(
    "COVID Severity",
    df["COVID_Severity"].unique(),
    default=df["COVID_Severity"].unique()
)

hospitalized = st.sidebar.multiselect(
    "Hospitalized",
    df["Hospitalized"].unique(),
    default=df["Hospitalized"].unique()
)

if not gender:
    gender = df["Gender"].unique()

if not severity:
    severity = df["COVID_Severity"].unique()

if not hospitalized:
    hospitalized = df["Hospitalized"].unique()


# Apply filters
filtered_df = df[
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Gender"].isin(gender)) &
    (df["COVID_Severity"].isin(severity)) &
    (df["Hospitalized"].isin(hospitalized))
]

# ---------------- TITLE ----------------
st.title("Post-COVID Health Outcomes Dashboard")
st.markdown(
    "Analyzing recovery patterns, symptoms, mental health impact, and Long COVID risk."
)

# ---------------- KPIs ----------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Patients", filtered_df.shape[0])
col2.metric(
    "Avg Recovery Days",
    round(filtered_df["Days_to_Recovery"].mean(), 1)
)
col3.metric(
    "High Long COVID Risk %",
    f"{round((filtered_df['Long_COVID_Risk'] == 'High').mean() * 100, 2)}%"
)
col4.metric(
    "Hospitalized %",
    f"{round((filtered_df['Hospitalized'] == 'Yes').mean() * 100, 2)}%"
)

st.divider()

col5, col6 = st.columns(2)

# Severity vs Recovery
with col5:
    st.subheader("COVID Severity vs Recovery Time")
    fig, ax = plt.subplots()
    sns.barplot(
        x="COVID_Severity",
        y="Days_to_Recovery",
        data=filtered_df,
        ax=ax
    )
    ax.set_ylabel("Avg Recovery Days")
    st.pyplot(fig)

# Long COVID Risk Distribution
with col6:
    st.subheader("Long COVID Risk Distribution")
    fig, ax = plt.subplots()
    filtered_df["Long_COVID_Risk"].value_counts().plot(
        kind="pie",
        autopct="%1.1f%%",
        startangle=90,
        ax=ax
    )
    ax.set_ylabel("")
    st.pyplot(fig)

st.divider()


col7, col8 = st.columns(2)

# Symptoms Analysis
with col7:
    st.subheader("Symptoms by Long COVID Risk")

    symptoms = [
        "Fatigue_Level",
        "Brain_Fog",
        "Breathing_Issue",
        "Loss_of_Taste_Smell"
    ]

    if filtered_df.empty:
        st.warning("No data available for the selected filters.")
    else:
        plot_df = (
            filtered_df
            .groupby("Long_COVID_Risk", as_index=False)[symptoms]
            .mean()
        )

        fig, ax = plt.subplots()

        plot_df.set_index("Long_COVID_Risk")[symptoms].plot(
            kind="bar",
            stacked=True,
            colormap="tab10",
            ax=ax
        )

        ax.set_ylabel("Average Severity")
        ax.set_xlabel("Long COVID Risk")
        ax.legend(
            title="Symptoms",
            bbox_to_anchor=(1.05, 1),
            loc="upper left"
        )

        st.pyplot(fig)


# Mental Health Impact
with col8:
    st.subheader("Mental Health Impact vs Long COVID Risk")
    fig, ax = plt.subplots()
    sns.barplot(
        x="Long_COVID_Risk",
        y="Mental_Health_Impact",
        data=filtered_df,
        ax=ax
    )
    st.pyplot(fig)

st.divider()


st.subheader("Physical Activity vs Recovery Time")

fig, ax = plt.subplots(figsize=(6, 4))
sns.boxplot(
    x="Physical_Activity_Level",
    y="Days_to_Recovery",
    data=filtered_df,
    ax=ax
)

ax.set_ylabel("Days to Recovery")
ax.set_xlabel("Physical Activity Level")

st.pyplot(fig)
st.divider()

st.subheader("Key Insights")
st.markdown("""
- Severe COVID cases show significantly longer recovery times  
- Brain fog and fatigue are strongest indicators of Long COVID  
- Mental health impact increases sharply for high-risk patients  
- Higher physical activity is associated with faster recovery  
""")

# ---------------- FOOTER ----------------
st.caption("Healthcare analytics dashboard built using Python & Streamlit")
