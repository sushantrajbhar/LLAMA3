import streamlit as st
import transformers
from transformers import pipeline
from datasets import load_metric
import torch
import io
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import utils
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import Paragraph, SimpleDocTemplate
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import re
import math
from dotenv import load_dotenv
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
import pandas as pd

nltk.download('punkt')

# Set up Streamlit page
st.set_page_config(page_title="Medical Assistant", page_icon='✅', layout='centered', initial_sidebar_state='expanded')

load_dotenv()

# Dark theme
st.markdown("""
    <style>
    body {
        background-color: #1E1E1E;
        color: #FFFFFF;
    }
    .sidebar .sidebar-content {
        background-color: #333333;
        color: #FFFFFF;
    }
    </style>
    """, unsafe_allow_html=True)

# Model configuration
model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'
device = f'cuda:{torch.cuda.current_device()}' if torch.cuda.is_available() else 'cpu'

# Quantization configuration
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

hf_auth = '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'

@st.cache_resource
def load_model():
    model_config = transformers.AutoConfig.from_pretrained(model_id, use_auth_token=hf_auth)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, config=model_config,
        quantization_config=bnb_config, device_map='auto', use_auth_token=hf_auth
    )
    model.eval()
    return model

model = load_model()
tokenizer = transformers.AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_auth)

# Stopping criteria
stop_list = ['\nHuman:', '\nQuestion:', '\nAnswer:', '\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]

class StopOnTokens(transformers.StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if input_ids.shape[1] >= len(stop_ids):
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
        return False

stopping_criteria = transformers.StoppingCriteriaList([StopOnTokens(stop_token_ids)])

# Define examples and prompt templates
examples = [
    {"query": "What are the side effects of ibuprofen?", "answer": "Common side effects of ibuprofen include stomach pain, heartburn, nausea, vomiting, bloating, gas, diarrhea, and constipation."},
    {"query": "What are the symptoms of a common cold?", "answer": "Symptoms of a common cold include runny or stuffy nose, sore throat, cough, congestion, slight body aches or a mild headache, sneezing, and low-grade fever."},
    {"query": "What are the symptoms of high blood pressure?","answer":"Symptoms of high blood pressure may include headaches, shortness of breath, nosebleeds, and dizziness. However, it often has no symptoms, which is why it's important to get regular check-ups."},
    {"query": "What is the recommended dosage of acetaminophen for adults?","answer":" For adults, the recommended dosage of acetaminophen is usually 500 mg every 4 to 6 hours as needed, not exceeding 4,000 mg per day. Always follow the dosing instructions on the label or given by a healthcare provider."},
    {"query": "What are the side effects of antibiotics?","answer":"Common side effects of antibiotics include nausea, diarrhea, rash, and yeast infections. Some antibiotics can also cause allergic reactions, including itching, swelling, or difficulty breathing."},
    {"query": "What are the causes of a sore throat?","answer":"A sore throat can be caused by viral infections like the common cold or flu, bacterial infections such as strep throat, allergies, dry air, or irritants like smoke."},
    {"query": "How can I prevent seasonal allergies?","answer":"To prevent seasonal allergies, try to stay indoors during high pollen counts, keep windows closed, use air purifiers, and take allergy medications as prescribed. Avoid touching your face and wash hands frequently."}
]

example_template = """
Question: {query}
Answer: {answer}
"""

example_prompt = PromptTemplate(input_variables=["query", "answer"], template=example_template)
example_selector = LengthBasedExampleSelector(examples=examples, example_prompt=example_prompt, max_length=1000)
prompt_template = FewShotPromptTemplate(example_selector=example_selector, example_prompt=example_prompt,
                                        prefix="You are a medical assistant. Answer the following questions based on your knowledge.",
                                        suffix="\n\nQuestion: {query}\nAnswer:", input_variables=["query"])

# Function to get the model response
def getLLMResponse(query, temperature, top_p, top_k):
    generate_text = pipeline(
        model=model, tokenizer=tokenizer, return_full_text=False, task='text-generation',
        temperature=temperature, top_p=top_p, top_k=top_k, max_new_tokens=500,
        stopping_criteria=stopping_criteria, repetition_penalty=1.1
    )
    llm = HuggingFacePipeline(pipeline=generate_text)
    prompt = prompt_template.format(query=query)
    response = llm(prompt)
    return response.strip()

# Preprocessing function
def preprocess_text(text):
    text = text.lower()  # Lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

# Evaluation metrics functions
def calculate_bleu(reference, hypothesis):
    reference = [nltk.word_tokenize(reference)]
    hypothesis = nltk.word_tokenize(hypothesis)
    return sentence_bleu(reference, hypothesis)

def calculate_rouge(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

def calculate_perplexity(model, tokenizer, sentence):
    inputs = tokenizer.encode(sentence, return_tensors='pt')
    outputs = model(inputs, labels=inputs)
    loss = outputs.loss
    perplexity = torch.exp(loss)
    return perplexity.item()

def calculate_exact_match(reference, hypothesis):
    return reference.strip() == hypothesis.strip()

def calculate_prf(reference, hypothesis):
    ref_tokens = set(nltk.word_tokenize(reference))
    hyp_tokens = set(nltk.word_tokenize(hypothesis))
    common_tokens = ref_tokens & hyp_tokens
    precision = len(common_tokens) / len(hyp_tokens) if hyp_tokens else 0
    recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1

# Function to evaluate responses
def evaluate_response(ground_truth, generated_responses):
    metrics = []
    for gt, gen in zip(ground_truth, generated_responses):
        answer = gt.split('Answer: ')[-1].strip()
        answer = preprocess_text(answer)  # Preprocess ground truth answer
        gen = preprocess_text(gen)  # Preprocess generated response
        
        bleu = calculate_bleu(answer, gen)
        rouge = calculate_rouge(answer, gen)
        perplexity = calculate_perplexity(model, tokenizer, gen)
        exact_match = calculate_exact_match(answer, gen)
        precision, recall, f1 = calculate_prf(answer, gen)
        metrics.append({
            'BLEU': bleu,
            'ROUGE-1': rouge['rouge1'].fmeasure,
            'ROUGE-L': rouge['rougeL'].fmeasure,
            'Perplexity': perplexity,
            'Exact Match': exact_match,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        })
    return pd.DataFrame(metrics)

# Function to save chat history as PDF
def save_as_pdf(chat_history):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    elements = []
    
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='QuestionStyle', fontSize=12, leading=14, spaceAfter=10, textColor='black'))
    styles.add(ParagraphStyle(name='AnswerStyle', fontSize=12, leading=14, spaceAfter=20, textColor='blue'))

    for chat in chat_history:
        question = Paragraph(f"**Question:** {chat['query']}", styles['QuestionStyle'])
        answer = Paragraph(f"**Answer:** {chat['response']}", styles['AnswerStyle'])
        elements.append(question)
        elements.append(answer)

    doc.build(elements)
    buffer.seek(0)
    return buffer

# Streamlit app layout
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Chat", "Performance Metrics", "Hyperparameters", "Chat History"])

# Chat page
if page == "Chat":
    st.header("Medical Assistant")
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    form_input = st.text_area('Enter your medical query', height=300, key='input_query')
    tasktype_option = st.selectbox('Please select the action to be performed?', ('Act as medical assistant', 'Act as general chatbot'))
    role_option = st.selectbox('For which role?', ('Medical professional', 'General human'))
    numberOfWords = st.slider('Words limit', 1, 1000, 500)

    submit = st.button("Generate")
    if submit:
        response = getLLMResponse(form_input, temperature=0.2, top_p=0.9, top_k=50)
        st.write(response)
        st.session_state.chat_history.append({"query": form_input, "response": response})
        form_input = ''  # Clear the input field after submission


# Performance Metrics page
elif page == "Performance Metrics":
    st.header("Performance Metrics")
    uploaded_file = st.file_uploader("Choose a .txt file with ground truth questions and answers", type="txt")
    if uploaded_file is not None:
        ground_truth = uploaded_file.read().decode("utf-8").splitlines()

    if st.button('Calculate Metrics'):
        if uploaded_file:
            st.write("## Evaluation Metrics")
            metrics_df = evaluate_response(ground_truth, [chat['response'] for chat in st.session_state.chat_history])
            st.table(metrics_df)
            st.write("---")

# Hyperparameters page
elif page == "Hyperparameters":
    st.header("Hyperparameters")
    temperature = st.slider('Temperature', 0.0, 1.0, 0.2)
    top_p = st.slider('Top-p (nucleus sampling)', 0.0, 1.0, 0.9)
    top_k = st.slider('Top-k', 1, 100, 50)

# Chat History page
elif page == "Chat History":
    st.header("Chat History")
    if st.button('Show Chat History'):
        st.session_state.show_chat_history = not st.session_state.get('show_chat_history', False)

    if st.session_state.get('show_chat_history', False):
        for chat in st.session_state.chat_history:
            st.write(f"**Question:** {chat['query']}")
            st.write(f"**Answer:** {chat['response']}")
            st.write("---")

    if st.button('Download Chat History as PDF'):
        pdf_buffer = save_as_pdf(st.session_state.chat_history)
        st.download_button(label="Download PDF", data=pdf_buffer, file_name="chat_history.pdf", mime="application/pdf")
