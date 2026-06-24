import streamlit as st
from PyPDF2 import PdfReader
import pandas as pd
import base64
import os
import asyncio
import re
from datetime import datetime

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())


    page_title="Rag pipeline",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:ital,wght@0,400;0,600;1,400&family=DM+Sans:wght@300;400;500&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #08080f !important;
    color: #e8e6e0 !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 60% at 50% -10%, rgba(180,140,80,.13) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 100% 100%, rgba(100,80,180,.07) 0%, transparent 60%),
        #08080f !important;
}

/* hide streamlit chrome */
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="collapsedControl"] { color: #a89060 !important; }
[data-testid="collapsedControl"] { color: #a89060 !important; }
[data-testid="collapsedControl"] { display: none !important; }
             
/* ── Sidebar ── */
[data-testid="stSidebar"] {
    min-width: 320px !important;
    max-width: 320px !important;
    width: 320px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
    background: rgba(12,11,20,.95) !important;
    border-right: 1px solid rgba(180,140,60,.14) !important;
    backdrop-filter: blur(20px);
}

/* hide collapse button completely */
[data-testid="collapsedControl"] {
    display: none !important;
}
                        
/* ── Wordmark ── */
.lexis-wordmark {
    font-family: 'Playfair Display', serif;
    font-size: 1.6rem;
    font-weight: 600;
    letter-spacing: .04em;
    color: #d4aa60;
    display: flex;
    align-items: center;
    gap: .5rem;
    margin-bottom: .25rem;
    animation: fadeSlideDown .6s ease both;
}
.lexis-tagline {
    font-size: .72rem;
    font-weight: 300;
    letter-spacing: .18em;
    text-transform: uppercase;
    color: rgba(200,185,150,.45);
    margin-bottom: 2rem;
    animation: fadeSlideDown .7s ease both;
}
.diamond { color: #d4aa60; font-size: 1.1rem; }

/* ── Section labels ── */
.sidebar-label {
    font-size: .68rem;
    font-weight: 500;
    letter-spacing: .2em;
    text-transform: uppercase;
    color: rgba(200,185,140,.38);
    margin: 1.6rem 0 .6rem;
}

/* ── Inputs ── */
[data-testid="stTextInput"] input,
.stTextInput input {
    background: rgba(255,255,255,.035) !important;
    border: 1px solid rgba(180,150,70,.2) !important;
    border-radius: 10px !important;
    color: #e0d8c8 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: .82rem !important;
    padding: .65rem 1rem !important;
    transition: border-color .25s, box-shadow .25s;
}
[data-testid="stTextInput"] input:focus {
    border-color: rgba(212,170,96,.55) !important;
    box-shadow: 0 0 0 3px rgba(212,170,96,.08) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label,
.stTextInput label {
    font-size: .72rem !important;
    letter-spacing: .12em !important;
    text-transform: uppercase !important;
    color: rgba(200,185,140,.5) !important;
    margin-bottom: .4rem !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    border: 1px dashed rgba(212,170,96,.22) !important;
    border-radius: 12px !important;
    background: rgba(212,170,96,.025) !important;
    padding: 1rem !important;
    transition: border-color .25s;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(212,170,96,.45) !important;
}
[data-testid="stFileUploader"] label {
    color: rgba(200,185,140,.6) !important;
    font-size: .78rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: transparent !important;
    border: 1px solid rgba(212,170,96,.35) !important;
    color: #d4aa60 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: .78rem !important;
    font-weight: 500 !important;
    letter-spacing: .1em !important;
    text-transform: uppercase !important;
    border-radius: 8px !important;
    padding: .55rem 1.2rem !important;
    transition: all .22s ease !important;
}
.stButton > button:hover {
    background: rgba(212,170,96,.1) !important;
    border-color: rgba(212,170,96,.7) !important;
    box-shadow: 0 0 18px rgba(212,170,96,.12) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* Process primary button */
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, rgba(212,170,96,.2), rgba(180,130,60,.35)) !important;
    border-color: rgba(212,170,96,.6) !important;
}