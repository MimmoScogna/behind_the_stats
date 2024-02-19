import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import altair as alt
import plotly.graph_objects as go

st.set_page_config(
     page_title='Behind The Stats', page_icon="soccer",
     layout="wide",
     initial_sidebar_state="expanded",
        menu_items={
        'About': "# Behind The Stats | Serie A Edition"
    }
)
hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''

st.markdown(hide_img_fs, unsafe_allow_html=True)

players_pos_df = 'df_players/players_pos_data.json'
df = pd.read_json(players_pos_df)

# Calcoliamo i Gol/Partita
def calculate_metric(row):
    if row['matches'] > 0:
        return round(row['Gol Fatti'] / row['matches'], 2)
    else:
        return 0

df_result = df.apply(calculate_metric, axis=1)
df['Gol Fatti/Partite Giocate'] = df_result  
df = pd.concat([df, df_result], axis=1)
df = df.drop(columns=[0])

# Calcoliamo gli Assist/Partita
def calculate_metric_2(row):
    if row['matches'] > 0:
        return round(row['Assist Vincenti'] / row['matches'], 2)
    else:
        return 0

df_result_2 = df.apply(calculate_metric_2, axis=1)
df['Assist Vincenti/Partite Giocate'] = df_result_2 
df = pd.concat([df, df_result_2], axis=1)
df = df.drop(columns=[0])

# Calcoliamo i Gol/P90
def calculate_metric_3(row):
    if row['minutesOnField'] > 0:
        return round(row['Gol Fatti'] / row['minutesOnField'] * 90, 2)
    else:
        return 0

df_result_3 = df.apply(calculate_metric_3, axis=1)
df['Gol Fatti/P90'] = df_result_3  
df = pd.concat([df, df_result_3], axis=1)
df = df.drop(columns=[0])

# Calcoliamo gli Assist/P90
def calculate_metric_4(row):
    if row['minutesOnField'] > 0:
        return round(row['Assist Vincenti'] / row['minutesOnField'] * 90, 2)
    else:
        return 0

df_result_4 = df.apply(calculate_metric_4, axis=1)
df['Assist Vincenti/P90'] = df_result_4  
df = pd.concat([df, df_result_4], axis=1)
df = df.drop(columns=[0])

# Tiri/Partita
def calculate_metric_5(row):
    if row['matches'] > 0:
        return round(row['Tiri Totali Fatti'] / row['matches'], 2)
    else:
        return 0

df_result_5 = df.apply(calculate_metric_5, axis=1)
df['Tiri Totali Fatti/Partite Giocate'] = df_result_5 
df = pd.concat([df, df_result_5], axis=1)
df = df.drop(columns=[0])

# Tiri/P90
def calculate_metric_6(row):
    if row['minutesOnField'] > 0:
        return round(row['Tiri Totali Fatti'] / row['minutesOnField'] * 90, 2)
    else:
        return 0

df_result_6 = df.apply(calculate_metric_6, axis=1)
df['Tiri Totali Fatti/P90'] = df_result_6 
df = pd.concat([df, df_result_6], axis=1)
df = df.drop(columns=[0])

# Second Assist/Partita
def calculate_metric_9(row):
    if row['matches'] > 0:
        return round(row['Assist di Seconda'] / row['matches'], 2)
    else:
        return 0

df_result_9 = df.apply(calculate_metric_9, axis=1)
df['Assist di Seconda/Partite Giocate'] = df_result_9  
df = pd.concat([df, df_result_9], axis=1)
df = df.drop(columns=[0])

# Second Assist/P90
def calculate_metric_10(row):
    if row['minutesOnField'] > 0:
        return round(row['Assist di Seconda'] / row['minutesOnField'] * 90, 2)
    else:
        return 0

df_result_10 = df.apply(calculate_metric_10, axis=1)
df['Assist di Seconda/P90'] = df_result_10  
df = pd.concat([df, df_result_10], axis=1)
df = df.drop(columns=[0])

# Tiri di Testa Fatti/Partita
def calculate_metric_7(row):
    if row['matches'] > 0:
        return round(row['Tiri di Testa Fatti'] / row['matches'], 2)
    else:
        return 0

df_result_7 = df.apply(calculate_metric_7, axis=1)
df['Tiri di Testa Fatti/Partite Giocate'] = df_result_7  
df = pd.concat([df, df_result_7], axis=1)
df = df.drop(columns=[0])

# Tiri di Testa Fatti/P90
def calculate_metric_8(row):
    if row['minutesOnField'] > 0:
        return round(row['Tiri di Testa Fatti'] / row['minutesOnField'] * 90, 2)
    else:
        return 0

df_result_8 = df.apply(calculate_metric_8, axis=1)
df['Tiri di Testa Fatti/P90'] = df_result_8  
df = pd.concat([df, df_result_8], axis=1)
df = df.drop(columns=[0])

# Assist per Tiro/Partita
def calculate_metric_11(row):
    if row['matches'] > 0:
        return round(row['Assist per Tiro'] / row['matches'], 2)
    else:
        return 0

df_result_11 = df.apply(calculate_metric_11, axis=1)
df['Assist per Tiro/Partite Giocate'] = df_result_11  
df = pd.concat([df, df_result_11], axis=1)
df = df.drop(columns=[0])

# xG Tiro/P90
def calculate_metric_13(row):
    if row['minutesOnField'] > 0:
        return round(row['xG Totale'] / row['minutesOnField'] * 90, 2)
    else:
        return 0

df_result_13 = df.apply(calculate_metric_13, axis=1)
df['xG Tiro/P90'] = df_result_13 
df = pd.concat([df, df_result_13], axis=1)
df = df.drop(columns=[0])

# xG Tiro/Partite
def calculate_metric_15(row):
    if row['minutesOnField'] > 0:
        return round(row['xG Totale'] / row['matches'], 2)
    else:
        return 0

df_result_15 = df.apply(calculate_metric_15, axis=1)
df['xG Tiro/Partite Giocate'] = df_result_15 
df = pd.concat([df, df_result_15], axis=1)
df = df.drop(columns=[0])

# xG Assist/Partita
def calculate_metric_14(row):
    if row['matches'] > 0:
        return round(row['xgAssist'] / row['matches'], 2)
    else:
        return 0

df_result_14 = df.apply(calculate_metric_14, axis=1)
df['xG Assist/Partite Giocate'] = df_result_14  
df = pd.concat([df, df_result_14], axis=1)
df = df.drop(columns=[0])

# xG Assist/P90
def calculate_metric_16(row):
    if row['minutesOnField'] > 0:
        return round(row['xgAssist'] / row['minutesOnField'] * 90, 2)
    else:
        return 0

df_result_16 = df.apply(calculate_metric_16, axis=1)
df['xG Assist/P90'] = df_result_16   
df = pd.concat([df, df_result_16], axis=1)
df = df.drop(columns=[0])

# Assist per Tiro/P90
def calculate_metric_12(row):
    if row['minutesOnField'] > 0:
        return round(row['Assist per Tiro'] / row['minutesOnField'] * 90, 2)
    else:
        return 0

df_result_12 = df.apply(calculate_metric_12, axis=1)
df['Assist per Tiro/P90'] = df_result_12  
df = pd.concat([df, df_result_12], axis=1)
df = df.drop(columns=[0])

# Tiri Porta/P90
def calculate_metric_17(row):
    if row['minutesOnField'] > 0:
        return round(row['Tiri in Porta Fatti'] / row['minutesOnField'] * 90, 2)
    else:
        return 0

df_result_17 = df.apply(calculate_metric_17, axis=1)
df['Tiri in Porta Fatti/P90'] = df_result_17 
df = pd.concat([df, df_result_17], axis=1)
df = df.drop(columns=[0])

# Tiri Porta/Partite
def calculate_metric_18(row):
    if row['minutesOnField'] > 0:
        return round(row['Tiri in Porta Fatti'] / row['matches'], 2)
    else:
        return 0

df_result_18 = df.apply(calculate_metric_18, axis=1)
df['Tiri in Porta Fatti/Partite Giocate'] = df_result_18
df = pd.concat([df, df_result_18], axis=1)
df = df.drop(columns=[0])

#Creiamo la colonna "Altri Ruoli" e Convertiamo i Dati del DataFrame in float, cancellando le righe vuote
new_column_data = {'name_rol_4': 'Altri Ruoli'}
new_column_df = pd.DataFrame(new_column_data, index=df.index)

# Unione dei due dataframe
df = pd.concat([df, new_column_df], axis=1)

# Conversione delle colonne percentuali a numeriche
df['percent_rol_1'] = pd.to_numeric(df['percent_rol_1'], errors='coerce')
df['percent_rol_2'] = pd.to_numeric(df['percent_rol_2'], errors='coerce')
df['percent_rol_3'] = pd.to_numeric(df['percent_rol_3'], errors='coerce')

# Riempimento dei valori NaN con 0
df['percent_rol_1'].fillna(0, inplace=True)
df['percent_rol_2'].fillna(0, inplace=True)
df['percent_rol_3'].fillna(0, inplace=True)

# Calcolo della nuova colonna 'percent_rol_4' senza condizione
df['percent_rol_4'] = 100 - df['percent_rol_1'] - df['percent_rol_2'] - df['percent_rol_3']

# Riempimento dei valori NaN in 'percent_rol_4' con 0
df['percent_rol_4'].fillna(0, inplace=True)

# Riempimento dei valori NaN con stringhe vuote
df['name_rol_4'].fillna('', inplace=True)

# Sposta le colonne 'name_rol_4' e 'percent_rol_4' alle nuove posizioni
col_name = 'name_rol_4'
new_position = 13
col_to_move = df[col_name]
df = df.drop(columns=[col_name])
df.insert(new_position, col_name, col_to_move)

col_name_2 = 'percent_rol_4'
new_position_2 = 14
col_to_move = df[col_name_2]
df = df.drop(columns=[col_name_2])
df.insert(new_position_2, col_name_2, col_to_move)

df = df.drop(columns=['minutesTagged'])
df = df.drop(columns=['wyId_player'])
df['team_name'] = df['team_name'].replace('Hellas', 'Hellas Verona')
df['role_base'] = df['role_base'].replace('Goalkeeper', 'Portiere')
df['role_base'] = df['role_base'].replace('Defender', 'Difensore')
df['role_base'] = df['role_base'].replace('Midfielder', 'Centrocampista')
df['role_base'] = df['role_base'].replace('Forward', 'Attaccante')

# Logo che appare sopra i menu
st.sidebar.image("Logo BTS.png", use_column_width=True)

#Divisione in Schede
c30, c31, c32 = st.columns([0.2, 0.1, 3])

with c32:

    st.title("🧔🏻‍♂️ Player Stats Insight")

st.write(
    "Naviga tra le schede per visualizzare tutte le informazioni sui calciatori selezionati"
)

tabPos, tabStats, tabDataViz_Bar, tabDataViz_Radar, tabNext = st.tabs(["Posizioni in Campo", "Stats", "DataViz | Bar Chart", "DataViz | Radar Chart","Still to Come"])

# Menù a tendina per la selezione del team
team_name_selected = st.sidebar.multiselect('Seleziona una o più Squadre', df['team_name'].unique(), placeholder="Seleziona...")

# Filtra il dataframe in base ai team selezionati
filtered_df = df[df['team_name'].isin(team_name_selected)]

# Menù a tendina per la selezione della colonna "role_base"
role_base_selected = st.sidebar.multiselect('Seleziona uno o più Ruoli', sorted(filtered_df['role_base'].unique()), placeholder="Seleziona...")

# Filtra il dataframe in base ai ruoli selezionati
filtered_df = filtered_df[filtered_df['role_base'].isin(role_base_selected)]

# Menù a tendina per la selezione della colonna "Giocatore"
short_names_selected = st.sidebar.multiselect('Seleziona uno o più Giocatori', sorted(filtered_df['Giocatore'].unique()), placeholder="Seleziona...")

colonne_non_selezionabili = ['name_rol_1', 'role_base','name_rol_2', 'name_rol_3', 'name_rol_4', 'code_rol_1', 'code_rol_2', 'code_rol_3', 'code_rol_4', 'team_name', 'percent_rol_1', 'percent_rol_2', 'percent_rol_3', 'percent_rol_4', 'minutesOnField', 'matches', 'matchesComingOff', 'matchesInStart', 'matchesSubstituted', 'minutesOnField']
colonne_disponibili = [col for col in filtered_df.columns if col not in colonne_non_selezionabili]

# Aggiungi la possibilità di selezionare tutte le statistiche disponibili
colonne_selezionate = st.sidebar.multiselect('Seleziona i Dati da Visualizzare in Stats', sorted(colonne_disponibili), default=['Giocatore'], placeholder="Seleziona...")

# Sposta la checkbox sotto il menu a tendina
seleziona_tutte_le_statistiche = st.sidebar.checkbox("Seleziona tutti i Dati disponibili")
if seleziona_tutte_le_statistiche:
    # Assicurati che 'Giocatore' sia sempre la prima colonna
    colonne_selezionate = ['Giocatore'] + sorted([col for col in colonne_disponibili if col != 'Giocatore'])

# Testo sotto ai menu a tendina
st.sidebar.caption(
    """This webapp is created for demonstration and educational purposes only""")
st.sidebar.caption(
    """Last update: 20/02/2024""")

# Copyright
st.sidebar.markdown(
    '<h6>Made in &nbsp<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" alt="Streamlit logo" height="16">&nbsp by <a href="https://www.domenicoscognamiglio.it/">Domenico Scognamiglio</a></h6>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.image("Logo v4 bianco.png", use_column_width=True)

# Verifica se sono stati selezionati giocatori prima di filtrare il dataframe
if short_names_selected:
    # Filtra ulteriormente il dataframe in base ai giocatori selezionati
    filtered_df = filtered_df[filtered_df['Giocatore'].isin(short_names_selected)]

    # Verifica se sono state selezionate tutte le colonne
    if seleziona_tutte_le_statistiche:
        # Seleziona tutte le colonne disponibili tranne 'Giocatore'
        colonne_selezionate = ['Giocatore'] + sorted([col for col in colonne_disponibili if col != 'Giocatore'])

    with tabStats:
        if colonne_selezionate:
            df_visualizzato = filtered_df[colonne_selezionate]
            st.info('👈 Seleziona quali dati vuoi visualizzare')
            st.dataframe(df_visualizzato, hide_index=True)
            with st.expander("Interattività **Tabella Stats**", expanded=True):
                    st.write(
                        """
                    - Puoi ridimensionare le colonne portandoti con il cursore sui limiti verticali;
                    - Puoi ordinare i valori delle colonne in ordine crescente/decrescente cliccando sul nome della statistica nella prima riga;
                    - Tramite il menu che viene fuori portandosi con il cursore sull'angolo alto a destra della tabella puoi:
                        - Scaricare tutta la tabella in formato *csv* portando il cursone nell'angolo destro alto della tabella e cliccando sulla freccia che va verso il basso;
                        - Ingrandire la tabella;
                        - Cercare appositi valori/parametri con l'apposita funzione.
                        """
                    )
                    st.write("")    
        else:
            # Se nessuna colonna è stata selezionata, visualizza il DataFrame completo
            st.info("👈 Ora seleziona quali dati vuoi inserire nel report")

# Verifica se sono stati selezionati giocatori prima di filtrare il dataframe
if short_names_selected:
    for selected_player in short_names_selected:
        # Filtra ulteriormente il dataframe per il giocatore corrente
        player_df = filtered_df[filtered_df['Giocatore'] == selected_player]

        # Creazione del dizionario per le coppie "chiave:valore"
        data_dict = {
            player_df['name_rol_1'].values[0]: player_df['percent_rol_1'].values[0],
            player_df['name_rol_2'].values[0]: player_df['percent_rol_2'].values[0],
            player_df['name_rol_3'].values[0]: player_df['percent_rol_3'].values[0],
            player_df['name_rol_4'].values[0]: player_df['percent_rol_4'].values[0]
        }

        # Rimuovi valori zero dal dizionario
        data_dict = {key: value for key, value in data_dict.items() if value != 0}

        # Troviamo i dati per usarli nella creazione del grafico a torta
        matches_value = player_df['matches'].values[0]
        matches_instart = player_df['matchesInStart'].values[0]
        matches_sub = player_df['matchesSubstituted'].values[0]
        matches_comingoff = player_df['matchesComingOff'].values[0]
        minutes = player_df['minutesOnField'].values[0]
        team_name = player_df['team_name'].values[0]

        with tabPos:
            if minutes == 0 or pd.isna(minutes):
                st.warning(f"**{selected_player}** in questa stagione non è mai sceso in campo")
            else:
                st.info(
                    f"Queste sono le posizioni occupate da **{selected_player}** in questa stagione")
                
                # Creazione del grafico a torta
                fig = px.pie(values=list(data_dict.values()), names=list(data_dict.keys()),
                            template="simple_white",
                            title=f'{selected_player} - {team_name} | Partite Giocate: {matches_value} (Tit: {matches_instart} | Ris: {matches_comingoff} | Sost: {matches_sub}) | Minuti Giocati: {minutes}',
                            labels=list(data_dict.values()))
                fig.update_traces(textposition='inside', textinfo='percent+label')
                fig.update_layout(legend={
                "title": "Posizione",})
            
                # Mostra il grafico nella web app
                st.plotly_chart(fig, use_column_width=True)

                # Creazione dell'expander per la legenda dei ruoli
                with st.expander("Legenda Ruoli"):
                    # Dividi la riga in tre colonne
                    col1, col2, col3 = st.columns(3)

                    # Primo blocco (prime 10 righe)
                    with col1:
                        st.write("Difesa")
                        data_block1 = pd.DataFrame({
                            "Ruolo in Inglese": ["Goalkeeper (GK)", "Right Centre Back (RCB)", "Left Centre Back (LCB)",
                                                 "Right Back (RB)", "Left Back (LB)",
                                                 "Right Centre Back (3 at the back) (RCB3)",
                                                 "Left Centre Back (3 at the back) (LCB3)", "Centre Back (CB)",
                                                 "Right Back (5 at the back) (RB5)", "Left Back (5 at the back) (LB5)"],
                            "Ruolo in Italiano": ["Portiere", "Difensore Centrale Destro", "Difensore Centrale Sinistro",
                                                  "Terzino Destro", "Terzino Sinistro", "Braccetto Destro (dif. a 3)",
                                                  "Braccetto Sinistro (dif. a 3)", "Difensore Centrale (dif.a 3)",
                                                  "Quinto a Destra (dif. a 5)", "Quinto a Sinistra (dif. a 5)"]
                        })

                        # Nascondi l'indice (index)
                        st.table(data_block1.set_index("Ruolo in Inglese", drop=True, inplace=False))

                    with col2:
                        st.write("Centrocampo")
                        data_block2 = pd.DataFrame({
                            "Ruolo in Inglese": ["Right Centre Midfielder (RCMF)", "Left Centre Midfielder (LCMF)",
                                                 "Right Centre Midfielder (RCMF3)", "Left Centre Midfielder (LCMF3)",
                                                 "Defensive Midfielder (DMF)", "Right Defensive Midfielder (RDMF)",
                                                 "Left Defensive Midfielder (LDMF)", "Right Attacking Midfielder (RAMF)",
                                                 "Left Attacking Midfielder (LAMF)", "Attacking Midfielder (AMF)",
                                                 "Right Wingback (RWB)", "Left Wingback (LWB)"],
                            "Ruolo in Italiano": ["Centrocampista Destro", "Centrocampista Sinistro", "Mezzala Sinistra",
                                                  "Mezzala Destra", "Mediano", "Mediano di Destra", "Mediano di Sinistra",
                                                  "Centrocampista Offensivo Destro", "Centrocampista Offensivo Sinistro",
                                                  "Centrocampista Offensivo", "Quinto a Sinistra", "Quinto a Destra"]
                        })

                        # Nascondi l'indice (index)
                        st.table(data_block2.set_index("Ruolo in Inglese", drop=True, inplace=False))

                    with col3:
                        st.write("Attacco")
                        data_block3 = pd.DataFrame({
                            "Ruolo in Inglese": ["Right Wing Forward (RWF)", "Left Wing Forward (LWF)",
                                                 "Second Striker (SS)",
                                                 "Striker (CF)"],
                            "Ruolo in Italiano": ["Ala Offensiva Destra", "Ala Offensiva Sinistra", "Seconda Punta",
                                                  "Attaccante Centrale"]
                        })                    
                        # Nascondi l'indice (index)
                        st.table(data_block3.set_index("Ruolo in Inglese", drop=True, inplace=False))
                        st.write("**Altri Ruoli**: posizioni non specificate dal provider dei dati.\nSolitamente rappresenta l'insieme di una o più posizioni, diverse da quelle indicate negli altri spicchi del grafico, che il giocatore ha occupato in totale per pochissimi minuti e che quindi non sono state ritenute principali.")

else:
    # Mostra un messaggio di avviso se nessun giocatore è stato selezionato
    st.info("👈 Seleziona almeno un giocatore per iniziare")

# Verifica se sono stati selezionati giocatori prima di filtrare il dataframe
if short_names_selected:
    # Filtra ulteriormente il dataframe in base ai giocatori selezionati
    filtered_df = filtered_df[filtered_df['Giocatore'].isin(short_names_selected)]

    # Verifica se sono state selezionate tutte le colonne
    if seleziona_tutte_le_statistiche:
        # Seleziona tutte le colonne disponibili tranne 'Giocatore'
        colonne_selezionate = ['Giocatore'] + sorted([col for col in colonne_disponibili if col != 'Giocatore'])

    with tabDataViz_Bar:
  # Creazione della struttura a due colonne
        with st.expander("Interattività **Bar Chart**", expanded=True):
            st.write("""
            - Passando il cursore sulle singole barre è possibile visualizzare tutte le informazioni relative a quella rappresentazione;
            - Tramite il menu che si apre portando il cursore nello spazio sopra alla legenda e cliccando sui tre puntini puoi scaricare il grafico in formato *png* o *svg*;
            - Consiglio: per sfruttare al meglio questa visualizzazione, seleziona almeno due/tre giocatori.
            """)    
        col1, col2 = st.columns(2)

        # Numero totale di colonne selezionate
        num_colonne_selezionate = len(colonne_selezionate[1:])

        # Calcola il numero massimo di grafici per colonna
        max_grafici_per_colonna = num_colonne_selezionate // 2 + num_colonne_selezionate % 2

        # Ordina le colonne selezionate in ordine alfabetico
        colonne_selezionate_sorted = sorted(colonne_selezionate[1:])

        # Altezza del grafico
        grafico_altezza = 400  # Modifica l'altezza a tuo piacimento

        # Utilizzo del primo spazio per i grafici
        with col1:
            if colonne_selezionate_sorted:
                for i, col in enumerate(colonne_selezionate_sorted[:max_grafici_per_colonna]):
                    chart = alt.Chart(filtered_df).mark_bar().encode(
                        x=alt.X('Giocatore', sort='-y'),
                        y=col,
                        color=alt.Color('Giocatore', scale=alt.Scale(scheme='tableau10')),
                        tooltip=['Giocatore', 'team_name', col]
                    ).interactive().properties(
                        title={'text': f'{col}', 'fontSize': 16},  # Modifica la dimensione del titolo
                        height=grafico_altezza
                    )
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.info("👈 Ora seleziona quali dati vuoi inserire nel report")

        # Utilizzo del secondo spazio per i grafici
        with col2:
            if colonne_selezionate_sorted:
                for i, col in enumerate(colonne_selezionate_sorted[max_grafici_per_colonna:]):
                    chart = alt.Chart(filtered_df).mark_bar().encode(
                        x=alt.X('Giocatore', sort='-y'),
                        y=col,
                        color=alt.Color('Giocatore', scale=alt.Scale(scheme='tableau10')),
                        tooltip=['Giocatore', 'team_name', col]
                    ).interactive().properties(
                        title={'text': f'{col}', 'fontSize': 16},  # Modifica la dimensione del titolo
                        height=grafico_altezza
                    )
                    st.altair_chart(chart, use_container_width=True)

    with tabDataViz_Radar:
        with st.expander("Interattività **Radar Chart**", expanded=True):
                st.write(
                """
            - Singolo clic sul nome nella legenda: accendi/spegni la traccia del giocatore (di default sono tutte accese);
            - Doppio clic sul nome nella legenda: isola la traccia del giocatore su cui si è cliccato;
            - Tramite il menu che appare portando il cursore nello spazio sopra alla legenda puoi:
                - Scaricare il grafico in formato *png*;
                - Ingrandire il grafico selezionando anche solamente una parte di esso.
            """
                    )
                st.write("") 

        # Seleziona solo i giocatori e i parametri desiderati
        df_selezionati = filtered_df[['Giocatore'] + colonne_selezionate[1:]]

        # Creazione del grafico radar con Plotly
        fig = go.Figure()

        # Aggiungi tracce per i giocatori
        for index, row in df_selezionati.iterrows():
            giocatore = row['Giocatore']
            valori_parametri = row[colonne_selezionate[1:]].rank(pct=True)
            
            fig.add_trace(go.Scatterpolar(
                r=valori_parametri,
                theta=colonne_selezionate[1:],
                fill='toself',
                name=giocatore
            ))

        # Aggiunta del layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True),
                bgcolor='rgba(0,0,0,0)',  # Colore di sfondo
                angularaxis=dict(linecolor='rgba(255,255,255,0.5)')  # Colore delle linee dell'asse
            ),
            showlegend=True,
            legend={
            "title": "Giocatore",},
            template="plotly_dark",  # Utilizza il tema scuro
            height=750,  # Altezza personalizzata
            width=1150    # Larghezza personalizzata
)
        # Mostra il grafico
        st.plotly_chart(fig, use_container_width=True)   

        with st.expander("Come si legge questo grafico"):
            st.write("""
                Il *Radar Chart* rappresenta visivamente le prestazioni del giocatore selezionato in varie statistiche, rispetto ad altri giocatori. 
                Ciascun asse del grafico indica una statistica specifica e l'area tracciata illustra il punteggio del giocatore all'interno dell'intervallo percentile di tutti i giocatori.\\
                **Percentile**: un posizionamento più alto su un asse indica una prestazione in un percentile più alto, il che significa che il giocatore ha un rendimento migliore in quella statistica rispetto a una percentuale maggiore di giocatori.
            """)       

with tabNext:
    with st.expander("Cosa verrà inserito in futuro", expanded=True):
        st.write(
            """
        - Nuove statistiche avanzate
        - Nuovi grafici
            """
        )
        st.write("")    
