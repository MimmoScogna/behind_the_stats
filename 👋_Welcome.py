import streamlit as st
from streamlit.logger import get_logger

LOGGER = get_logger(__name__)

def run():
    st.set_page_config(
        page_title="Behind The Stats",
        page_icon="soccer",
        layout="wide",
        menu_items={
        'About': "# Behind The Stats | Serie A Edition"
    }
    )

    st.write("# Benvenuti su Behind The Stats | Serie A Edition 👋")
    # Logo che appare sopra i menu
    st.sidebar.image("Logo BTS.png", use_column_width=True)

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
    #st.sidebar.success("Seleziona Player Stats o Team Stats in base a cosa vuoi analizzare")

    st.markdown(
        """
        Dove le cifre diventano una potente narrativa! \n    """
    )
    st.info("👈 Seleziona Player Stats o Team Stats dalla barra laterale ed inizia subito ad analizzare")
    st.warning("Per sfruttare al meglio tutte le funzionalità della piattaforma, consigliamo di utilizzarla tramite PC o Tablet")


if __name__ == "__main__":
    run()

hide_img_fs = '''
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
</style>
'''

st.markdown(hide_img_fs, unsafe_allow_html=True)
