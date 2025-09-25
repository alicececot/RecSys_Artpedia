# RecSys_Artpedia

Questo progetto implementa un sistema di raccomandazione di opere d’arte basato su BLIP (Bootstrapping Language-Image Pretraining).
L’obiettivo è studiare tecniche di **Explainable AI** (XAI) nel contesto dei sistemi di raccomandazione artistici, confrontando **spiegazioni tradizionali (content-based)** con spiegazioni generate da **Large Language Models (LLM)**.

L’app permette di:
1. Mostrare un set iniziale di opere d’arte.  
2. L’utente seleziona 4 preferite.  
3. Il sistema genera 12 raccomandazioni personalizzate.  
4. Ogni raccomandazione è accompagnata da:
   - Spiegazione tradizionale.  
   - Spiegazione LLM.  
5. L’utente valuta con scala Likert (1–5) diverse metriche:
   - Accuratezza, diversità, novità, serendipità
   - Trasparenza, fiducia, chiarezza, utilità  
