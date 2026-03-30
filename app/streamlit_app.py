import streamlit as st
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import os
import sys

# Correction importante pour trouver le dossier src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.sentiment_analysis import load_sentiment_model, analyze_text

# Configuration de la page
st.set_page_config(
    page_title="Analyse de Sentiments",
    page_icon="💬",
    layout="wide"
)

st.title("💬 Analyse de Sentiments des Avis Clients")
st.markdown("**Modèle :** distilbert-base-uncased-finetuned-sst-2-english (Hugging Face)")

# Charger le modèle
@st.cache_resource
def get_classifier():
    return load_sentiment_model()

classifier = get_classifier()

# Sidebar
st.sidebar.header("Options")
mode = st.sidebar.radio(
    "Choisissez le mode :",
    ["Analyse d'un texte", "Analyse d'un fichier CSV", "Statistiques sur sample Amazon"]
)

# ====================== MODE 1 : Analyse d'un texte ======================
if mode == "Analyse d'un texte":
    st.subheader("Entrez un avis client")
    
    text_input = st.text_area(
        "Avis :", 
        height=150,
        placeholder="J'adore ce téléphone, il est super rapide et la batterie dure longtemps !"
    )
    
    if st.button("Analyser le sentiment", type="primary"):
        if text_input.strip():
            with st.spinner("Analyse en cours..."):
                sentiment, confidence = analyze_text(text_input, classifier)
            
            if sentiment == "POSITIVE":
                st.success(f"**POSITIVE** 😊 ({confidence*100:.1f}%)")
            else:
                st.error(f"**NEGATIVE** 😔 ({confidence*100:.1f}%)")
            
            st.write("**Texte analysé :**")
            st.write(text_input)
        else:
            st.warning("Veuillez entrer un texte.")

# ====================== MODE 2 : Analyse d'un fichier CSV ======================
elif mode == "Analyse d'un fichier CSV":
    st.subheader("Uploader un fichier CSV d'avis clients")
    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("**Aperçu des données :**")
        st.dataframe(df.head())
        
        text_column = st.selectbox("Sélectionnez la colonne contenant les avis :", df.columns)
        
        if st.button("Analyser tout le fichier", type="primary"):
            with st.spinner("Analyse en cours..."):
                results = []
                for text in df[text_column]:
                    sent, conf = analyze_text(str(text), classifier)
                    results.append({"sentiment": sent, "confidence": conf})
                
                result_df = pd.DataFrame(results)
                df = pd.concat([df.reset_index(drop=True), result_df], axis=1)
            
            st.success(f"✅ Analyse terminée ! {len(df)} avis traités.")
            st.dataframe(df)
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(df, names='sentiment', title="Répartition des Sentiments")
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig2 = px.histogram(df, x='sentiment', color='sentiment', title="Distribution des Sentiments")
                st.plotly_chart(fig2, use_container_width=True)

# ====================== MODE 3 : Sample Amazon ======================
else:
    st.subheader("📊 Analyse sur un échantillon d'Avis Amazon")
    
    sample_path = "data/amazon_reviews_sample.csv"
    
    if not os.path.exists(sample_path) or os.path.getsize(sample_path) == 0:
        st.info("Création de l'échantillon...")
        sample_data = {
            'review': [
                "This phone is absolutely amazing! Best purchase I've made this year.",
                "The battery lasts forever, I'm very impressed.",
                "Terrible quality, broke after one week. Waste of money.",
                "Fast delivery and the product is exactly as described. Love it!",
                "Camera quality is disappointing for the price.",
                "Best headphones ever! Sound is crystal clear.",
                "Screen started flickering after 10 days. Very disappointed.",
                "Super fast charging and beautiful design. Highly recommend.",
                "Not worth the money. Performance is average at best.",
                "The product arrived damaged. Terrible experience."
            ] * 4,
            'product': ["iPhone", "Samsung", "Xiaomi", "Sony", "Huawei"] * 8
        }
        os.makedirs('data', exist_ok=True)
        pd.DataFrame(sample_data).to_csv(sample_path, index=False)
        st.success("Échantillon créé avec succès !")
    
    df = pd.read_csv(sample_path)
    
    if st.button("🚀 Analyser l'échantillon Amazon", type="primary"):
        with st.spinner("Analyse en cours..."):
            results = [analyze_text(text, classifier) for text in df['review']]
            df['sentiment'] = [r[0] for r in results]
            df['confidence'] = [r[1] for r in results]
        
        st.success("Analyse terminée !")
        st.dataframe(df)
        
        col1, col2 = st.columns(2)
        with col1:
            fig1 = px.pie(df, names='sentiment', title="Répartition des Sentiments")
            st.plotly_chart(fig1, use_container_width=True)
        with col2:
            fig2 = px.bar(df, x='product', color='sentiment', title="Sentiment par Produit")
            st.plotly_chart(fig2, use_container_width=True)

st.caption("Projet NLP - Analyse de Sentiments | Farah - Étudiante en Data Science")