from streamlit_option_menu import option_menu
import joblib
import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import warnings
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

import time

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Peringkasan Dan Klasifikasi",
    page_icon='https://img.freepik.com/free-vector/people-showcasing-different-types-ways-access-news_53876-66059.jpg?t=st=1739836314~exp=1739839914~hmac=97374f842066f72c85c3294967a4dfc6e8e88ada3d8ae4992dd0ca8cad16f61f&w=900',
    layout='centered',
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
st.write("""
<center><h2 style = "text-align: justify;">Peringkasan Artikel Berita Menggunakan Metode Maximum Marginal Relevance (MMR) Dan Klasifikasi Berita Dengan Metode Support Vector Machine (SVM)</h2></center>
""",unsafe_allow_html=True)

with st.container():
    with st.sidebar:
        selected = option_menu(
        st.write("""<h3 style = "text-align: center;"></h3>""",unsafe_allow_html=True), 
        ["Home", "Dataset","Preprocessing","Demo Klasifikasi", "Demo Peringkasan"], 
            icons=['house', 'bar-chart','check2-square'], menu_icon="cast", default_index=0,
            styles={
                "container": {"padding": "0!important", "background-color": "#3674B5"},
                "icon": {"color": "white", "font-size": "18px"}, 
                "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "color":"white"},
                "nav-link-selected":{"background-color": "#D91656"}
            }
        )
    if selected == "Home" :
        st.write("""<h3 style="text-align: center;">
        <img src="https://img.freepik.com/free-vector/people-showcasing-different-types-ways-access-news_53876-66059.jpg?t=st=1739836314~exp=1739839914~hmac=97374f842066f72c85c3294967a4dfc6e8e88ada3d8ae4992dd0ca8cad16f61f&w=900" width="500" height="300">
        </h3>""", unsafe_allow_html=True)
    if selected == "Dataset":
        st.write("Data Berita :")
        file_path = 'Data_BeritaDetik.csv'  # Ganti dengan path ke file Anda
        data = pd.read_csv(file_path,delimiter=';')
        st.write(data)
    if selected == "Preprocessing":
        tab1, tab2, tab3, tab4 = st.tabs(["Cleaning", "Case Folding", "Tokenizing", "Stopword Removal"])

        with tab1:
            st.write("Cleaning")
            try:
                file_path = 'cleaning.csv'  # Ganti dengan path ke file Anda
                data = pd.read_csv(file_path)
                st.write(data)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        with tab2:
            st.write("Case Folding")
            try:
                file_path = 'case folding.csv'  # Ganti dengan path ke file Anda
                data = pd.read_csv(file_path)
                st.write(data)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        with tab3:
            st.write("Tokenizing")
            try:
                file_path = 'tokenizing.csv'  # Ganti dengan path ke file Anda
                data = pd.read_csv(file_path)
                st.write(data)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
        with tab4:
            st.write("Stopword Removal")
            try:
                file_path = 'stopword.csv'  # Ganti dengan path ke file Anda
                data = pd.read_csv(file_path)
                st.write(data)
            except FileNotFoundError:
                st.error("File tidak ditemukan. Pastikan path file benar.")
            except Exception as e:
                st.error(f"Terjadi kesalahan: {e}")
    if selected == "Demo Klasifikasi":
        import streamlit as st
        import joblib
        import numpy as np
        from sklearn.feature_extraction.text import TfidfVectorizer

        # Load the trained model and vectorizer
        model_filename = "svm_ovr_fold_5.pkl"
        vectorizer_filename = "vectorizer (4).pkl"

        try:
            svm_classifier = joblib.load(model_filename)
            vectorizer = joblib.load(vectorizer_filename)
            model_loaded = True
        except Exception as e:
            st.error(f"Error loading model or vectorizer: {e}")
            model_loaded = False

        # Streamlit UI
        st.write("Masukkan teks pada kolom di bawah ini untuk diklasifikasikan :")

        # Input text from user
        user_input = st.text_area("Input Teks")

        if st.button("Prediksi"):
            if model_loaded:
                if user_input.strip():
                    # Transform input text
                    new_X = vectorizer.transform([user_input]).toarray()
                    
                    # Predict
                    prediction = svm_classifier.predict(new_X)[0]
                    st.success(f"Predicted Class: {prediction}")
                else:
                    st.warning("Mohon masukkan teks terlebih dahulu.")
            else:
                st.error("Model atau vectorizer tidak ditemukan.")


    if selected == "Demo Peringkasan":
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        import pandas as pd
        import streamlit as st

        def ringkas_berita_mmr(df, artikel_col='Artikel', judul_col='Judul', max_summary_size=4, lambda_mmr=0.8):
            summaries = []

            for _, row in df.iterrows():
                documents = row[artikel_col]
                query = row[judul_col]

                if isinstance(documents, list):
                    documents = [str(doc) for doc in documents]
                else:
                    documents = [sent.strip() for sent in documents.split('. ') if sent]
                    documents = [sent if sent.endswith('.') else sent + '.' for sent in documents]

                query = str(query)

                if not documents or not query:
                    summaries.append("Tidak ada konten untuk diringkas.")
                    continue

                vectorizer = TfidfVectorizer()
                doc_vectors = vectorizer.fit_transform(documents)
                query_vector = vectorizer.transform([query])

                doc_query_similarities = cosine_similarity(doc_vectors, query_vector).flatten()

                selected_sentences = []
                remaining_indices = list(range(len(documents)))

                while len(selected_sentences) < max_summary_size and remaining_indices:
                    mmr_values = []
                    for idx in remaining_indices:
                        sim_to_query = doc_query_similarities[idx]
                        if selected_sentences:
                            selected_vectors = doc_vectors[selected_sentences]
                            sim_to_selected = max(cosine_similarity(doc_vectors[idx], selected_vectors).flatten())
                        else:
                            sim_to_selected = 0

                        mmr = lambda_mmr * sim_to_query - (1 - lambda_mmr) * sim_to_selected
                        mmr_values.append((mmr, idx))

                    best_mmr_idx = max(mmr_values, key=lambda x: x[0])[1]
                    selected_sentences.append(best_mmr_idx)
                    remaining_indices.remove(best_mmr_idx)

                selected_sentences.sort()
                summary = " ".join([documents[idx] for idx in selected_sentences])
                summaries.append(summary)

            df['Ringkasan'] = summaries
            return df

        def input_artikel():
            judul = st.text_input("Masukkan judul artikel:")
            artikel = st.text_area("Masukkan isi artikel (pisahkan kalimat dengan titik):")
            
            if st.button("Ringkas Artikel"):
                data = pd.DataFrame({
                    'Judul': [judul],
                    'Artikel': [artikel]
                })
                return data
            return None

        #st.title("Aplikasi Peringkasan Artikel dengan MMR")
        data = input_artikel()

        if data is not None:
            hasil_ringkasan = ringkas_berita_mmr(data)
            
            st.subheader("Ringkasan Artikel:")
            st.write(hasil_ringkasan['Ringkasan'].iloc[0])


