import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from huggingface_hub import login
import warnings

# Suppress specific FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, message="`resume_download` is deprecated and will be removed in version 1.0.0.")

# Set page configuration
st.set_page_config(page_title="FBRInsight Chatbot", layout="wide")

# Initialize GPU device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_answer(context):
    context = context[context.find('Answer', 10):]
    context = context[:context.find('Question')]
    return context

def get_completion(query, model, tokenizer):
    inputs = tokenizer(query, return_tensors='pt', add_special_tokens=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    
    generated_ids = model.generate(
        **inputs,
        do_sample=True,
        max_new_tokens=200,
        pad_token_id=tokenizer.eos_token_id
    )
    
    decoded = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return decoded

def answer(question, model, tokenizer):
    prompt = '''
    Answer the following questions, based on the provided context only, do not hallucinate
    
    Context : {}
    
    Question : {}
    '''
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    vc = FAISS.load_local('vc', embeddings=embeddings, allow_dangerous_deserialization=True)
    similar_docs = vc.similarity_search(question)
    context = ' '.join([element.page_content for element in similar_docs])
    
    result = get_completion(query=prompt.format(context, question), model=model, tokenizer=tokenizer)
    result = extract_answer(result)
    return result

# Ensure the model and tokenizer are loaded once
@st.cache_resource
def load_model():
    login('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')
    
    def format_path(path):
        return path.replace('\n', '').replace('\t', '').replace(' ', '')
    
    model_name = 'meta-llama/Llama-2-7b-hf'
    
    tokenizer = AutoTokenizer.from_pretrained(
        format_path(model_name),
        padding=True,
        truncation=True,
        token='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'
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        format_path(model_name),
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=False
        )
    )
    
    lora_config = LoraConfig.from_pretrained('llama2_instruct')
    model_tuned = get_peft_model(model, lora_config)
    
    return model_tuned, tokenizer

model_tuned, tokenizer = load_model()

# Main UI
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "About", "Settings"])

# Chat page
if page == "Chat":
    st.title("FBRInsight Chatbot")
    st.write("Ask me anything about tax")

    def check_prompt(prompt):
        try:
            prompt.replace('', '')
            return True
        except:
            return False
    
    def check_message():
        if 'messages' not in st.session_state:
            st.session_state.messages = []

    def clear_chat():
        st.session_state.messages = []
    
    check_message()
    
    if st.button("Clear Chat"):
        clear_chat()

    for message in st.session_state.messages:
        with st.chat_message(message['role']):
            st.markdown(message['content'])
    
    prompt = st.chat_input('Ask me anything')

    if check_prompt(prompt):
        with st.chat_message('user'):
            st.markdown(prompt)
        
        st.session_state.messages.append({'role': 'user', 'content': prompt})
        
        if prompt:
            response = answer(question=prompt, model=model_tuned, tokenizer=tokenizer)
            
            with st.chat_message('assistant'):
                st.markdown(response)
            
            st.session_state.messages.append({'role': 'assistant', 'content': response})

# About page
elif page == "About":
    st.title("About FBRInsight Chatbot")
    st.write("""
    This project is built to provide information on FBR rules and regulations about tax.
    You can ask any questions related to tax and get accurate responses based on the provided context.
    """)

# Settings page
elif page == "Settings":
    st.title("Settings")
    
    # About us option
    if st.button("About Us"):
        st.write("""
        This chatbot is developed to assist users with tax-related queries. 
        Our goal is to provide accurate and up-to-date information on FBR rules and regulations.
        """)
