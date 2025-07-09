import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow import keras
import streamlit as st
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder


# Aggiungi CSS per sfondo gradiente blu-azzurro e colori testi
st.markdown(
    """
    <style>
    /* Sfondo gradiente blu -> azzurro -> bianco */
    .main {
    min-height: 100vh;
    background: linear-gradient(135deg, #003366, #66ccff, #ffffff);
    background-repeat: no-repeat;
    background-attachment: fixed;
    color: #000000;
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    padding: 2rem 3rem;
    }


    /* Titoli */
    h1, h2, h3, h4, h5, h6 {
        color: #e0f7ff;
    }

    /* Testo normale */
    .stText, .stMarkdown {
        color: #f0faff;
    }

    /* DataFrame */
    .dataframe tbody tr th {
        color: #eef6fb;
    }
    .dataframe thead tr th {
        background-color: rgba(255, 255, 255, 0.2);
        color: white;
    }

    /* Bottone file uploader */
    div.stFileUploader > label {
        background-color: #00509e;
        color: white;
        border-radius: 5px;
        padding: 8px 15px;
        font-weight: bold;
    }
    div.stFileUploader > label:hover {
        background-color: #007bff;
        cursor: pointer;
    }

    /* Barra laterale (sidebar) */
    .css-1d391kg {
        background: linear-gradient(135deg, #004080, #3399ff);
        color: white;
    }

    /* Grafici matplotlib/seaborn */
    .stPlotlyChart > div > div > div {
        background-color: transparent !important;
    }

    /* Scrollbar colore personalizzato */
    ::-webkit-scrollbar {
        width: 10px;
    }
    ::-webkit-scrollbar-track {
        background: #003366;
    }
    ::-webkit-scrollbar-thumb {
        background: #66ccff;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


def process_uploaded_data(df_cla2_continuo, df_cla_passaggi, df_tiri):
    # === Step 1: Rendi i tempi continui ===
    df_cla2_continuo['t'] = np.arange(len(df_cla2_continuo)) * (1000 / 100)  # 100 Hz
    df_cla_passaggi['t'] = np.arange(len(df_cla_passaggi)) * (1000 / 100)
    df_tiri['t'] = np.arange(len(df_tiri)) * (1000 / 100)

    # === Step 2: Calcolo del modulo accelerazione ===
    for df in [df_cla2_continuo, df_cla_passaggi, df_tiri]:
        df["Acc_Modulo"] = np.sqrt(df["ax"]**2 + df["ay"]**2 + df["az"]**2)

    # === Step 3: Estrazione "Fermo" e "Corsa" da cla2_continuo ===
    df_fermo = df_cla2_continuo[df_cla2_continuo["t"] <= 120000].copy()
    df_fermo["Class"] = "Fermo"

    df_corsa = df_cla2_continuo[(df_cla2_continuo["t"] > 122000) & (df_cla2_continuo["t"] <= 250000)].copy()
    df_corsa["Class"] = "Corsa"

    # === Step 4: Identificazione picchi Passaggi ===
    threshold_fixed = 2.7
    new_distance = 20
    peaks_passaggi, _ = find_peaks(df_cla_passaggi["Acc_Modulo"], height=threshold_fixed, distance=new_distance)

    # === Step 5: Finestratura attorno ai picchi Passaggi ===
    time_window = 40
    windows_passaggi = []
    for peak_idx in peaks_passaggi:
        peak_time = df_cla_passaggi.iloc[peak_idx]["t"]
        start_time = peak_time - time_window
        end_time = peak_time + time_window
        window = df_cla_passaggi[(df_cla_passaggi["t"] >= start_time) & (df_cla_passaggi["t"] <= end_time)]

        if len(window) > 1:
            mean_modulo = window["Acc_Modulo"].mean()
            std_modulo = window["Acc_Modulo"].std()
            windows_passaggi.append({
                "Mean_Modulo": mean_modulo,
                "Std_Modulo": std_modulo,
                "Class": "Passaggio"
            })
    df_features_passaggi = pd.DataFrame(windows_passaggi)

    # === Analisi Tiri ===
    percentile_99 = np.percentile(df_tiri["Acc_Modulo"], 99)
    threshold_tiro = 4.8  # basato su 99° percentile
    peaks_tiri, _ = find_peaks(df_tiri["Acc_Modulo"], height=threshold_tiro, distance=new_distance)

    windows_tiri = []
    for peak_idx in peaks_tiri:
        peak_time = df_tiri.iloc[peak_idx]["t"]
        start_time = peak_time - time_window
        end_time = peak_time + time_window
        window = df_tiri[(df_tiri["t"] >= start_time) & (df_tiri["t"] <= end_time)]

        if len(window) > 1:
            mean_modulo = window["Acc_Modulo"].max()
            std_modulo = window["Acc_Modulo"].std()
            windows_tiri.append({
                "Mean_Modulo": mean_modulo,
                "Std_Modulo": std_modulo,
                "Class": "Tiro"
            })
    df_features_tiri = pd.DataFrame(windows_tiri)

    # === Step 8: Finestratura per "Fermo" e "Corsa" ===
    sampling_rate = 100
    overlap = 0.92
    window_duration = 2.56
    window_size = int(window_duration * sampling_rate)
    overlap_size = int(window_size * overlap)
    step_size = window_size - overlap_size

    windows_fermo_corsa = []
    for movimento_df, movimento in zip([df_fermo, df_corsa], ["Fermo", "Corsa"]):
        if movimento_df.empty:
            continue
        for start in range(0, len(movimento_df) - window_size + 1, step_size):
            end = start + window_size
            window = movimento_df.iloc[start:end]
            mean_modulo = window["Acc_Modulo"].mean()
            std_modulo = window["Acc_Modulo"].std()
            windows_fermo_corsa.append({
                "Mean_Modulo": mean_modulo,
                "Std_Modulo": std_modulo,
                "Class": movimento
            })
    df_features_fermo_corsa = pd.DataFrame(windows_fermo_corsa)

    # === Step 9: Unione dataset finali ===
    dataset_finale = pd.concat([df_features_fermo_corsa, df_features_passaggi, df_features_tiri], ignore_index=True)

    return dataset_finale


st.title("Classificazione Movimenti con Accelerometro")

st.write("""
Carica i file CSV.
Il sistema estrarrà le caratteristiche e applicherà il modello per la classificazione.
""")

file_cla2 = st.file_uploader("Carica Fermo e Corsa", type=["csv"])
file_passaggi = st.file_uploader("Carica Passaggi", type=["csv"])
file_tiri = st.file_uploader("Carica Tiri", type=["csv"])

if file_cla2 and file_passaggi and file_tiri:
    df_cla2 = pd.read_csv(file_cla2)
    df_passaggi = pd.read_csv(file_passaggi)
    df_tiri = pd.read_csv(file_tiri)

    with st.spinner("Preprocessing e analisi dati..."):
        dataset_finale = process_uploaded_data(df_cla2, df_passaggi, df_tiri)

    st.success(f"Dati processati: {len(dataset_finale)} finestre estratte")
    st.dataframe(dataset_finale.head())

    # Prepara dati per la classificazione
    X = dataset_finale[["Mean_Modulo", "Std_Modulo"]].values
    y = dataset_finale["Class"].values
    class_names = ["Fermo", "Corsa", "Passaggio", "Tiro"]

    #le = LabelEncoder()
    #y_enc = le.fit_transform(y)

    # Carica modello
    model = load_model("modello_neural_network.h5")

    # Predizioni
    y_pred_probs = model.predict(X)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_pred_labels = [class_names[i] for i in y_pred]

    dataset_finale["Predicted_Class"] = y_pred_labels

    st.subheader("Risultati della classificazione")
    st.dataframe(dataset_finale[["Mean_Modulo", "Std_Modulo", "Class", "Predicted_Class"]])

    st.subheader("Distribuzione delle classi predette")
    st.bar_chart(dataset_finale["Predicted_Class"].value_counts())

    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    cm = confusion_matrix(dataset_finale["Class"], dataset_finale["Predicted_Class"], labels=class_names)

    fig, ax = plt.subplots(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax,
                cbar_kws={"shrink": 0.7}, linewidths=0.5, linecolor="white")
    ax.set_xlabel("Classe Predetta")
    ax.set_ylabel("Classe Reale")
    ax.set_title("Matrice di Confusione")
    st.pyplot(fig)

    fig2, ax2 = plt.subplots(figsize=(8,4))
    df_counts = pd.DataFrame({
        "Reale": dataset_finale["Class"].value_counts(),
        "Predetta": dataset_finale["Predicted_Class"].value_counts()
    }).fillna(0)

    df_counts.plot(kind="bar", ax=ax2)
    ax2.set_title("Confronto Classi Reali vs Predette")
    ax2.set_ylabel("Conteggio")
    ax2.set_xlabel("Classe")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(1,2, figsize=(14,5))
    sns.scatterplot(data=dataset_finale, x="Mean_Modulo", y="Std_Modulo", hue="Class",
                    palette="coolwarm", ax=ax3[0], edgecolor="k", alpha=0.9)

    sns.scatterplot(data=dataset_finale, x="Mean_Modulo", y="Std_Modulo", hue="Predicted_Class",
                    palette="coolwarm", ax=ax3[1], edgecolor="k", alpha=0.9)

    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(1,2, figsize=(14,5))

# Boxplot
    sns.boxplot(x="Predicted_Class", y="Mean_Modulo", data=dataset_finale, ax=ax4[0], palette="Blues")
    ax4[0].set_title("Distribuzione Mean_Modulo per Classe Predetta")
    ax4[0].tick_params(axis='x', rotation=45)

    sns.boxplot(x="Predicted_Class", y="Std_Modulo", data=dataset_finale, ax=ax4[1], palette="Blues")     
    ax4[1].set_title("Distribuzione Std_Modulo per Classe Predetta")
    ax4[1].tick_params(axis='x', rotation=45)

    st.pyplot(fig4)

    selected_classes = st.multiselect(
        "Seleziona classi predette da visualizzare",
        options=class_names,
        default=class_names
    )

    df_filtered = dataset_finale[dataset_finale["Predicted_Class"].isin(selected_classes)]

    fig_filtered = px.scatter(
        df_filtered,
        x="Mean_Modulo",
        y="Std_Modulo",
        color="Predicted_Class",
        symbol="Class",
        hover_data=["Class", "Predicted_Class"],
        title="Scatter Plot filtrato per classi predette",
        labels={"Mean_Modulo": "Media Accelerazione", "Std_Modulo": "Dev. Std Accelerazione"},
        template="plotly_dark"  # tema scuro coerente con lo sfondo
    )
    st.plotly_chart(fig_filtered, use_container_width=True)

else:
    st.info("Carica tutti e tre i file CSV per abilitare l'analisi.")
