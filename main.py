import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Configuração da página do Streamlit
st.set_page_config(
    page_title="Detector de Spam - TensorFlow",
    page_icon="🛡️",
    layout="centered"
)

# --- 1. DADOS DE TREINAMENTO (Exemplos) ---
# Em um cenário de produção, estes dados viriam de um dataset como o SMS Spam Collection
mensagens_treino = [
    # Exemplos de Spam (1)
    "GANHE DINHEIRO AGORA! Clique no link e mude de vida hoje mesmo!!!",
    "Você ganhou um prêmio de R$ 10.000! Inscreva seus dados bancários aqui.",
    "URGENTE: Sua conta foi bloqueada. Acesse o link para desbloquear.",
    "Compre bitcoin com desconto garantido de 50% apenas hoje.",
    "Oferta exclusiva: Viagra e medicamentos sem receita médica.",
    # Exemplos de Ham/Não Spam (0)
    "Oi mãe, tudo bem? Consigo passar aí para jantar hoje?",
    "Segue em anexo o relatório trimestral para revisão da equipe.",
    "Lembrete: Sua consulta ao dentista está marcada para amanhã às 14h.",
    "Gostei muito da reunião de ontem, vamos alinhar os próximos passos.",
    "Pode me enviar o link do repositório do GitHub, por favor?"
]

# Labels: 1 para Spam, 0 para Não Spam
labels_treino = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])

# --- 2. PRÉ-PROCESSAMENTO E MODELAGEM (TensorFlow) ---
@st.cache_resource
def inicializar_modelo_ia():
    # Configurações do Tokenizer
    vocab_size = 500
    embedding_dim = 16
    max_length = 20
    trunc_type = 'post'
    padding_type = 'post'
    oov_tok = "<OOV>"

    # Tokenização do texto
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(mensagens_treino)
    
    sequencias = tokenizer.texts_to_sequences(mensagens_treino)
    sequencias_padded = pad_sequences(sequencias, maxlen=max_length, padding=padding_type, truncating=trunc_type)

    # Construção do modelo sequencial simples (Ideal para o ambiente do Render)
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # Compilação
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Treinamento rápido (Simulação)
    model.fit(sequencias_padded, labels_treino, epochs=30, verbose=0)
    
    return model, tokenizer, max_length, padding_type, trunc_type

# Inicializa o modelo usando o cache do Streamlit para performance
model, tokenizer, max_length, padding_type, trunc_type = inicializar_modelo_ia()

# --- 3. INTERFACE DO USUÁRIO (Streamlit) ---
st.title("🛡️ Detector de Spam Inteligente")
st.subheader("Análise de e-mails em tempo real com TensorFlow")
st.write("Insira o conteúdo do e-mail recebido abaixo para verificar se ele possui características de fraude ou spam.")

# Área de input do texto
email_input = st.text_area("Cole a mensagem do e-mail aqui:", placeholder="Ex: Olá, segue a ata da reunião...", height=150)

if st.button("Analisar Mensagem", use_container_width=True):
    if email_input.strip() == "":
        st.warning("Por favor, digite ou cole alguma mensagem para que a IA possa analisar.")
    else:
        # Pré-processamento do input do usuário
        input_seq = tokenizer.texts_to_sequences([email_input])
        input_padded = pad_sequences(input_seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
        
        # Predição da IA
        predicao = model.predict(input_padded)[0][0]
        
        st.write("---")
        # Exibição dos resultados com base nas estruturas do Streamlit
        if predicao > 0.5:
            st.error(f"🚨 **Alerta de Spam!** Esta mensagem tem {predicao*100:.2f}% de chance de ser maliciosa.")
            st.warning("Recomendações: Não clique em links, não baixe anexos e bloqueie o remetente.")
        else:
            st.success(f"✅ **E-mail Seguro.** Esta mensagem é limpa (Apenas {(predicao)*100:.2f}% de similaridade com spams conhecidos).")
            st.info("A mensagem parece legítima e pode ser respondida com segurança.")