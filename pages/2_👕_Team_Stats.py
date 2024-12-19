import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
import re
import altair as alt
import matplotlib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import ssl
from PIL import Image
from matplotlib.colors import to_rgba
from mplsoccer import Pitch, FontManager, add_image
from plottable import ColumnDefinition, Table
from plottable.cmap import normed_cmap
from plottable.plots import image

ssl._create_default_https_context = ssl._create_unverified_context

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
hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(hide_img_fs, unsafe_allow_html=True)

# Logo che appare sopra i menu
st.sidebar.image("Logo BTS.png", use_container_width=True)

#Divisione in Schede
c30, c31, c32 = st.columns([0.2, 0.1, 3])

with c32:

    st.title("👕 Team Stats Insight")

st.write(
    "Naviga tra le schede per visualizzare i diversi contenuti"
)

tabStats, tabDataViz, tabPass, tabTocchi, tabNext = st.tabs(["Stats", "DataViz | Line Chart", "DataViz | Passing Networks", "DataViz | Tocchi Palla per Terzo Campo", "Still to Come"])



with tabTocchi:
    #Intestazione
    path_to_image = "tocchi_opta.png"

    # Mostra l'immagine
    st.image(path_to_image, use_container_width=True)   

    #Usiamo Pandas per prendere i dati
    df = pd.read_html('https://fbref.com/en/comps/11/Serie-A-Stats/', attrs={'id': "stats_squads_possession_for"})[0]
    df = df.drop(columns=["Carries"])
    df = df[['Unnamed: 0_level_0', 'Touches']].copy()
    df.columns = df.columns.droplevel()
    df = df.drop(["Def Pen", "Att Pen", "Live"], axis = 1)
    touches_cols = ['Def 3rd', 'Mid 3rd', 'Att 3rd']
    df_total = pd.DataFrame(df[touches_cols].sum())
    df_total.columns = ['total']
    df_total = df_total.T
    df_total = df_total.divide(df_total.sum(axis=1), axis=0) * 100
    df[touches_cols] = df[touches_cols].divide(df[touches_cols].sum(axis=1), axis=0) * 100.
    df.sort_values(['Att 3rd', 'Def 3rd'], ascending=[True, False], inplace=True)
    fm = FontManager(("https://github.com/google/fonts/blob/main/apache/opensanshebrew/OpenSansHebrew-Bold.ttf?raw=true"))
    path_eff = [path_effects.Stroke(linewidth=3, foreground='#252525'),
            path_effects.Normal()]
    
    # setup a mplsoccer pitch
    pitch = Pitch(line_zorder=2, line_color='black', pad_top=20)

    bin_statistic = pitch.bin_statistic([0], [0], statistic='count', bins=(3, 1))

    GRID_HEIGHT = 0.8
    CBAR_WIDTH = 0.03
    fig, axs = pitch.grid(nrows=4, ncols=5, figheight=20,
                        # leaves some space on the right hand side for the colorbar
                        grid_width=0.88, left=0.025,
                        endnote_height=0.03, endnote_space=0,
                        axis=False,
                        title_space=0.00, title_height=0.01, grid_height=GRID_HEIGHT)
    fig.set_facecolor('white')

    teams = sorted(df['Squad'].values)
    vmin = df[touches_cols].min().min()
    vmax = df[touches_cols].max().max()
    for i, ax in enumerate(axs['pitch'].flat[:len(teams)]):
        ax.text(60, -10, teams[i],
                ha='center', va='center', fontsize=50,
                fontproperties=fm.prop)

        # fill in the bin statistics from df and plot the heatmap
        bin_statistic['statistic'] = df.loc[df.Squad == teams[i], touches_cols].values
        heatmap = pitch.heatmap(bin_statistic, ax=ax, cmap='coolwarm', vmin=vmin, vmax=vmax)
        annotate = pitch.label_heatmap(bin_statistic, color='white', fontproperties=fm.prop,
                                    path_effects=path_eff, fontsize=50, ax=ax,
                                    str_format='{0:.0f}%', ha='center', va='center')

    #if its the Serie A remove the two spare pitches
    if len(teams) == 18:
        for ax in axs['pitch'][-1, 3:]:
            ax.remove()

    #add cbar axes
    cbar_bottom = axs['pitch'][-1, 0].get_position().y0
    cbar_left = axs['pitch'][0, -1].get_position().x1 + 0.01
    ax_cbar = fig.add_axes((cbar_left, cbar_bottom, CBAR_WIDTH,
                            # take a little bit off the height because of padding
                            GRID_HEIGHT - 0.036))
    cbar = plt.colorbar(heatmap, cax=ax_cbar)
    for label in cbar.ax.get_yticklabels():
        label.set_fontproperties(fm.prop)
        label.set_fontsize(50)

    #Copyright
    copyright_text = "Created by Domenico Scognamiglio | Behind The Stats"
    plt.figtext(0.897, 0.10, copyright_text, color="#9EA3B0", fontsize=12, ha='right')
    st.pyplot(fig)
    with st.expander("**Nota Bene**", expanded=True):
            st.write("""
            I dati in questa scheda si riferiscono a tutte le partite giocate in stagione, a prescindere da quelle selezionate dalla barra laterale 
            """)

with tabStats:
    @st.cache_data
    def load_dataframes():
        # Ottieni il percorso completo della cartella "df_teams"
        df_teams_path = os.path.join(os.getcwd(), "df_teams")

        # Carica i dataframe da file JSON
        dataframes = {}
        for filename in os.listdir(df_teams_path):
            if filename.endswith(".json"):
                dataframe_name = os.path.splitext(filename)[0]
                file_path = os.path.join(df_teams_path, filename)
                dataframes[dataframe_name] = pd.read_json(file_path)

        return dataframes

    # Carica i dataframe utilizzando la funzione st.cache
    dataframes = load_dataframes()

    # Menù a tendina per la selezione del team
    selected_team = st.sidebar.selectbox('Seleziona la Squadra', sorted(list(dataframes.keys())))

    # Filtra il dataframe in base alla squadra selezionata
    team_df = dataframes[selected_team]
    
    # Crea l'opzione "Seleziona tutte"
    select_all_option = st.sidebar.checkbox('Seleziona tutte le Partite Giocate')

    # Seleziona le partite tramite un checkbox multiselezione
    team_df = team_df.sort_index(ascending=True)
    if select_all_option:
        selected_matches = team_df['Partita'].unique()
    else:
        selected_matches = st.sidebar.multiselect('Seleziona una o più Partite', team_df['Partita'].unique(), placeholder="Seleziona...")

    # Filtra ulteriormente il dataframe in base alle partite selezionate
    final_df = team_df[team_df['Partita'].isin(selected_matches)]

    colonne_disponibili = final_df.columns
    # Aggiungi la possibilità di selezionare tutte le colonne
    seleziona_tutte_le_colonne = st.sidebar.checkbox("Seleziona tutti i Dati disponibili")
    colonne_selezionate = st.sidebar.multiselect('Seleziona uno o più Dati', sorted(colonne_disponibili), default=['Partita'], placeholder="Selezona...")

    if seleziona_tutte_le_colonne:
        # Assicurati che 'Partita' sia sempre la prima colonna
        colonne_selezionate = ['Partita'] + sorted([col for col in colonne_disponibili if col != 'Partita'])

    if colonne_selezionate:
        df_visualizzato = final_df[colonne_selezionate]

        # Aggiungi la riga "Totale" escludendo la colonna 'Partita' e 'PPDA'
        total_row = {}
        for col in colonne_selezionate:
            if col not in ['Partita', 'PPDA']:
                total_row[col] = df_visualizzato[col].sum()
            elif col == 'PPDA':
                total_row[col] = ''
            elif col == 'Partita':
                total_row[col] = ''

        total_row = pd.DataFrame(total_row, index=['Totale'])
        df_visualizzato = pd.concat([df_visualizzato, total_row])

        # Aggiungi la colonna "mediana"
        if not df_visualizzato.empty:
            mediana_values = df_visualizzato.select_dtypes(include=['number']).median()
            mediana_values['Partita'] = ''  # Sostituisci "None" con uno spazio vuoto
            mediana_row = pd.DataFrame(mediana_values, columns=['Mediana']).transpose()
            df_visualizzato = pd.concat([df_visualizzato, mediana_row])

        st.info("👈 Seleziona squadra, partite e dati")
        st.dataframe(df_visualizzato, 
                     column_config={
                         "Partita": st.column_config.Column(
                         width="medium")
        })
        with st.expander("Interattività **Tabella Stats**", expanded=True):
                    st.write(
                        """
                    - Puoi ridimensionare le colonne portandoti con il cursore sui limiti verticali (solo da pc);
                    - Puoi ordinare i valori delle colonne in ordine crescente/decrescente cliccando sul nome della statistica nella prima riga;
                    - Tramite il menu che viene fuori portandosi con il cursore sull'angolo alto a destra della tabella puoi:
                        - Scaricare tutta la tabella in formato *csv* portando il cursore nell'angolo destro alto della tabella e cliccando sulla freccia che va verso il basso;
                        - Mettere la tabella a schermo intero;
                        - Cercare valori/parametri specifici con l'apposita funzione.
                        """
                    )
                    st.write("")  

with tabDataViz:
    with st.expander("Interattività **Line Chart**", expanded=True):
            st.write("""
            - Passando il cursore/facendo tap sui singoli punti di intersezione è possibile visualizzare tutte le informazioni relative a quel punto;
            - Tramite il menu che si apre portando il cursore nello spazio sopra alla legenda è possibile: 
                - Scaricare il grafico in formato *png*;
                - Ingrandire/Rimpicciolire il grafico;
                - Selezionare e visionare solamente una parte del grafico.
            - Consiglio: per sfruttare al meglio questa visualizzazione, seleziona almeno due/tre partite.     
            """)
    if not colonne_selezionate:
        st.info("👈 Seleziona squadra, partite e dati per generare un grafico")
    else:
        # Rimuovi 'Partita' dalla lista delle colonne per il grafico a linee
        colonne_grafico = [col for col in colonne_selezionate if col != 'Partita']

        # Crea un DataFrame per il grafico a linee
        if 'Partita' in colonne_selezionate:
            df_grafico = df_visualizzato[['Partita'] + colonne_grafico]
            # Filtra il dataframe per escludere "Totale" e "Mediana"
            df_grafico = df_grafico.loc[~df_grafico.index.isin(['Totale', 'Mediana'])]
        else:
            df_grafico = df_visualizzato[colonne_grafico]
            # Filtra il dataframe per escludere "Totale" e "Mediana"
            df_grafico = df_grafico.loc[~df_grafico.index.isin(['Totale', 'Mediana'])]

        # Imposta 'Partita' come indice per il DataFrame
        df_grafico.set_index('Partita', inplace=True)

        # Ordina il DataFrame in base alla colonna 'Partita'
        df_grafico.sort_index(ascending=True)

        # Itera attraverso le colonne selezionate e crea un grafico per ognuna
        for colonna in colonne_grafico:
            # Crea il grafico a linee con Plotly Express
            fig = px.line(df_grafico, x=df_grafico.index, y=colonna, title=f'{colonna}')
            fig.update_layout(title_font=dict(size=18), height=600)

            # Calcola la mediana come un array ripetuto per ogni riga del DataFrame
            median_values = np.full_like(df_grafico[colonna], df_grafico[colonna].median())

            # Aggiungi una linea per rappresentare la mediana
            median_line = go.Scatter(x=df_grafico.index, y=median_values,
                                    mode='lines', line=dict(color='orange', dash='dash'),
                                    name='Mediana',
                                    hovertemplate=f'Mediana: {df_grafico[colonna].median():.2f}')
            fig.add_trace(median_line)
            fig.update_xaxes(tickangle=90) 
            
            # Aggiungi una linea verticale e punti di intersezione per ogni valore sull'asse x
            for x_value in df_grafico.index:
                fig.add_shape(go.layout.Shape(type="line",
                                            x0=x_value, x1=x_value,
                                            y0=min(df_grafico[colonna]),
                                            y1=max(df_grafico[colonna]),
                                            line=dict(color="rgba(169,169,169,0.09)", dash="dash")))

                intersection_point = df_grafico[df_grafico.index == x_value][colonna].values[0]
                fig.add_trace(go.Scatter(x=[x_value], y=[intersection_point],
                                        mode='markers',
                                        marker=dict(color='white', size=5),
                                        name=f'',
                                        showlegend=False,
                                        hovertemplate=f'Partita: {x_value}<br>Valore: {intersection_point}'))
            
            st.plotly_chart(fig, use_container_width=True)

        else:
            st.info("👈 Seleziona squadra, partite e dati per generare un grafico")
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
st.sidebar.image("Logo v4 bianco.png", use_container_width=True)

with tabPass:
    #Intestazione
    path_to_image = "passing_opta.png"

    # Mostra l'immagine
    st.image(path_to_image, use_container_width=True)

    st.info("👈 Seleziona squadra e partite per visualizzare")
     
    @st.cache_data 
    def extract_json_from_html(html_path, save_output=False):
        html_file = open(html_path, 'r')
        html = html_file.read()
        html_file.close()
        regex_pattern = r'(?<=require\.config\.params\["args"\].=.)[\s\S]*?;'
        data_txt = re.findall(regex_pattern, html)[0]

       
        data_txt = data_txt.replace('matchId', '"matchId"')
        data_txt = data_txt.replace('matchCentreData', '"matchCentreData"')
        data_txt = data_txt.replace('matchCentreEventTypeJson', '"matchCentreEventTypeJson"')
        data_txt = data_txt.replace('formationIdNameMappings', '"formationIdNameMappings"')
        data_txt = data_txt.replace('};', '}')

        if save_output:
            
            output_file = open(f"{html_path}.txt", "wt")
            n = output_file.write(data_txt)
            output_file.close()

        return data_txt
         
    @st.cache_data 
    def extract_data_from_dict(data):
       
        event_types_json = data["matchCentreEventTypeJson"]
        formation_mappings = data["formationIdNameMappings"]
        events_dict = data["matchCentreData"]["events"]
        teams_dict = {data["matchCentreData"]['home']['teamId']: data["matchCentreData"]['home']['name'],
                    data["matchCentreData"]['away']['teamId']: data["matchCentreData"]['away']['name']}
        players_dict = data["matchCentreData"]["playerIdNameDictionary"]
        
        players_home_df = pd.DataFrame(data["matchCentreData"]['home']['players'])
        players_home_df["teamId"] = data["matchCentreData"]['home']['teamId']
        players_away_df = pd.DataFrame(data["matchCentreData"]['away']['players'])
        players_away_df["teamId"] = data["matchCentreData"]['away']['teamId']
        players_df = pd.concat([players_home_df, players_away_df])
        players_ids = data["matchCentreData"]["playerIdNameDictionary"]
        return events_dict, players_df, teams_dict

    # Itera sui percorsi dei file HTML
    match_html_paths = []
    for match in selected_matches:
        path = f'passing_network_files/{match}.html'
        match_html_paths.append(path)

    for html_path in match_html_paths:
        json_data_txt = extract_json_from_html(html_path)
        data = json.loads(json_data_txt)
        events_dict, players_df, teams_dict = extract_data_from_dict(data)

        players_df.head()

        @st.cache_data 
        def get_passes_df(events_dict):
            df = pd.DataFrame(events_dict)
            df['eventType'] = df.apply(lambda row: row['type']['displayName'], axis=1)
            df['outcomeType'] = df.apply(lambda row: row['outcomeType']['displayName'], axis=1)

            df["receiver"] = df["playerId"].shift(-1)
            
            passes_ids = df.index[df['eventType'] == 'Pass']
            df_passes = df.loc[
                passes_ids, ["id", "x", "y", "endX", "endY", "teamId", "playerId", "receiver", "eventType", "outcomeType"]]

            return df_passes

        passes_df = get_passes_df(events_dict)
        passes_df.head()

        @st.cache_data 
        def get_passes_between_df(team_id, passes_df, players_df):
            
            passes_df = passes_df[passes_df["teamId"] == team_id]

            passes_df = passes_df.merge(players_df[["playerId", "isFirstEleven"]], on='playerId', how='left')

            passes_df = passes_df[passes_df['isFirstEleven'] == True]

            average_locs_and_count_df = (passes_df.groupby('playerId')
                                        .agg({'x': ['mean'], 'y': ['mean', 'count']}))
            average_locs_and_count_df.columns = ['x', 'y', 'count']
            average_locs_and_count_df = average_locs_and_count_df.merge(players_df[['playerId', 'name', 'shirtNo', 'position']],
                                                                        on='playerId', how='left')
            average_locs_and_count_df = average_locs_and_count_df.set_index('playerId')

            passes_player_ids_df = passes_df.loc[:, ['id', 'playerId', 'receiver', 'teamId']]
            passes_player_ids_df['pos_max'] = (passes_player_ids_df[['playerId', 'receiver']].max(axis='columns'))
            passes_player_ids_df['pos_min'] = (passes_player_ids_df[['playerId', 'receiver']].min(axis='columns'))

            passes_between_df = passes_player_ids_df.groupby(['pos_min', 'pos_max']).id.count().reset_index()
            passes_between_df.rename({'id': 'pass_count'}, axis='columns', inplace=True)

            passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_min', right_index=True)
            passes_between_df = passes_between_df.merge(average_locs_and_count_df, left_on='pos_max', right_index=True,
                                                        suffixes=['', '_end'])
            return passes_between_df, average_locs_and_count_df

        home_team_id = list(teams_dict.keys())[0]  # selected home team
        home_passes_between_df, home_average_locs_and_count_df = get_passes_between_df(home_team_id, passes_df, players_df)

        away_team_id = list(teams_dict.keys())[1]  # selected home team
        away_passes_between_df, away_average_locs_and_count_df = get_passes_between_df(away_team_id, passes_df, players_df)

        def pass_network_visualization(ax, passes_between_df, average_locs_and_count_df, flipped=False):
            MAX_LINE_WIDTH = 10
            MAX_MARKER_SIZE = 3000
            passes_between_df['width'] = (passes_between_df.pass_count / passes_between_df.pass_count.max() *
                                        MAX_LINE_WIDTH)
            average_locs_and_count_df['marker_size'] = (average_locs_and_count_df['count']
                                                        / average_locs_and_count_df['count'].max() * MAX_MARKER_SIZE)

            MIN_TRANSPARENCY = 0.3
            color = np.array(to_rgba('#00a8e5'))
            color = np.tile(color, (len(passes_between_df), 1))
            c_transparency = passes_between_df.pass_count / passes_between_df.pass_count.max()
            c_transparency = (c_transparency * (1 - MIN_TRANSPARENCY)) + MIN_TRANSPARENCY
            color[:, 3] = c_transparency

            pitch = Pitch(pitch_type='opta', pitch_color='#0e1117', line_color='#c7d5cc')
            pitch.draw(ax=ax)

            if flipped:
                passes_between_df['x'] = pitch.dim.right - passes_between_df['x']
                passes_between_df['y'] = pitch.dim.right - passes_between_df['y']
                passes_between_df['x_end'] = pitch.dim.right - passes_between_df['x_end']
                passes_between_df['y_end'] = pitch.dim.right - passes_between_df['y_end']
                average_locs_and_count_df['x'] = pitch.dim.right - average_locs_and_count_df['x']
                average_locs_and_count_df['y'] = pitch.dim.right - average_locs_and_count_df['y']

            pass_lines = pitch.lines(passes_between_df.x, passes_between_df.y,
                                    passes_between_df.x_end, passes_between_df.y_end, lw=passes_between_df.width,
                                    color=color, zorder=1, ax=ax)
            pass_nodes = pitch.scatter(average_locs_and_count_df.x, average_locs_and_count_df.y,
                                    s=average_locs_and_count_df.marker_size, marker='h',
                                    color='#FEFEFC', edgecolors='#0D182E', linewidth=1.1, alpha=1, ax=ax)
            for index, row in average_locs_and_count_df.iterrows():
                player_name = row["name"].split()
                player_initials = "".join(word[0] for word in player_name).upper()
                pitch.annotate(player_initials, xy=(row.x, row.y), c='#403F4C', va='center',
                            ha='center', size=14, ax=ax)

            return pitch

        # Crea un nuovo plot per ogni partita
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        axes = axes.flat
        plt.tight_layout()
        fig.set_facecolor("#0e1117")

        main_color = '#FBFAF5'
        font_bold = FontManager(("https://github.com/google/fonts/blob/main/apache/opensanshebrew/OpenSansHebrew-Bold.ttf?raw=true"))

        pass_network_visualization(axes[0], home_passes_between_df, home_average_locs_and_count_df)
        axes[0].set_title(teams_dict[home_team_id], color=main_color, fontsize=14, fontproperties=font_bold.prop)

        pass_network_visualization(axes[1], away_passes_between_df, away_average_locs_and_count_df, flipped=True)
        axes[1].set_title(teams_dict[away_team_id], color=main_color, fontsize=14, fontproperties=font_bold.prop)

        plt.suptitle(f"{teams_dict[home_team_id]} - {teams_dict[away_team_id]}", color=main_color, fontsize=42, fontproperties=font_bold.prop)
        subtitle = "Reti di passaggi e combinazioni più cercate: più è spessa e marcata la linea, più quella traiettoria/combinazione è stata cercata"
        plt.text(-10, 120, subtitle, horizontalalignment='center', verticalalignment='center', color=main_color, fontsize=14, fontproperties=font_bold.prop)
        # Aggiungi il copyright in basso a destra
        copyright_text = "Created by Domenico Scognamiglio | Behind The Stats"
        plt.figtext(0.97, 0.22, copyright_text, color="#9EA3B0", fontsize=9, fontproperties=font_bold.prop, ha='right')

        st.pyplot(fig)

    with st.expander("Scaricare i **Passing Networks**", expanded=True):
        st.write("""
        - Puoi scaricare i grafici visualizzati in questa scheda facendo clic con il tasto destro su esso e selezionando "Salva Immagine con Nome..." 
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
