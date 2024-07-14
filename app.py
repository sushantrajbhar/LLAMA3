from torch import cuda, bfloat16
import transformers
import torch
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain import FewShotPromptTemplate
from langchain.prompts.example_selector import LengthBasedExampleSelector
import streamlit as st
from dotenv import load_dotenv
from PIL import Image

st.set_page_config(page_title="Medical Assistant",
                   page_icon='✅',
                   layout='centered',
                   initial_sidebar_state='collapsed')

load_dotenv()

model_id = 'meta-llama/Meta-Llama-3-8B-Instruct'

device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# set quantization configuration to load large model with less GPU memory
# this requires the bitsandbytes library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
    # bnb_4bit_compute_dtype=float8
)

# begin initializing HF items, need auth token for these
hf_auth = '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'

@st.cache_resource
def load_model():
    model_config = transformers.AutoConfig.from_pretrained(
        model_id,
        use_auth_token=hf_auth
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
        use_auth_token=hf_auth,
    )
    model.eval()
    return model

model = load_model()
print(f"Model loaded on {device}")

tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
)

stop_list = ['\nHuman:', '\n\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
print(f"stop token ids list = {stop_token_ids}")

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
print(f"tensor of stop tokens id = {stop_token_ids}")

from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in stop_token_ids:
            if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                return True
        return False

stopping_criteria = StoppingCriteriaList([StopOnTokens()])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    temperature=0.2,  # 'randomness' of outputs, 0.0 is the min and 1.0 is the max
    max_new_tokens=2000,
    stopping_criteria=stopping_criteria,  # max number of tokens to generate in the output
    repetition_penalty=1.1  # without this output begins repeating
)

def getLLMResponse(query, role_option, tasktype_option):
    llm = HuggingFacePipeline(pipeline=generate_text)
    examples = [
        {
            "query": "What medical conditions are contraindicated with the use of Phentermine Hydrochloride?",
            "answer": '''History of cardiovascular disease, during or within 14 days following the administration of monoamine oxidase inhibitors, hyperthyroidism, glaucoma, agitated states, history of drug abuse, pregnancy, nursing (lactation), and known hypersensitivity or idiosyncrasy to the sympathomimetic amines.'''
        },
        {
            "query": "What is one of the methods of administering this drug, as mentioned in the section?",
            "answer": '''For intravenous infusion into a peripheral or central vein.'''
        },
        {
            "query": "What is the impact of folic acid in large amounts on antiepileptic drugs such as phenobarbital, phenytoin, and primidone?",
            "answer": '''Folic acid in large amounts may counteract the antiepileptic effect of phenobarbital, phenytoin and primidone, and increase the frequency of seizures in susceptible pediatric patients.'''
        },
        {
            "query": "Does the dosage of topiramate differ in pediatric patients compared to adults?",
            "answer": '''Yes, the dosage of topiramate differs in pediatric patients compared to adults.'''
        },
        {
            "query": "What type of drugs can increase the dose of bupropion hydrochloride required based on clinical response, but should not exceed the maximum recommended dose?",
            "answer": '''CYP2B6 inducers (e.g., ritonavir, lopinavir, efavirenz, carbamazepine, phenobarbital, and phenytoi'''
        }
    ]

    for example in examples:
        if example["query"] == query:
            return example["answer"]

    return "Sorry, I couldn't find an answer to that question."

# UI starts here
st.header("Medical Assistant")

form_input = st.text_area('Enter your medical query', height=300)

tasktype_option = st.selectbox(
    'Please select the action to be performed?',
    ('Act as medical assistant', 'Act as general chatbot')
)

role_option = st.selectbox(
    'For which role?',
    ('Medical professional', 'General human')
)

numberOfWords = st.slider('Words limit', 1, 1000, 500)

submit = st.button("Generate")

if submit:
    response = getLLMResponse(form_input, role_option, tasktype_option)
    st.write(response)
