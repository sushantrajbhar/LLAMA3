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
# this requires the `bitsandbytes` library
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
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

stop_list = ['\nHuman:', '\n```\n']

stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
print(f"stop token ids list = {stop_token_ids}")

stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
print(f" tensor of stop tokens id = {stop_token_ids}")

from transformers import StoppingCriteria, StoppingCriteriaList

# define custom stopping criteria object
class StopOnTokens(StoppingCriteria):
    def __init__(self, stop_token_ids):
        self.stop_token_ids = stop_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_ids in self.stop_token_ids:
            if input_ids.shape[1] >= len(stop_ids):
                if torch.eq(input_ids[0][-len(stop_ids):], stop_ids).all():
                    return True
        return False

stop_list = ['\nHuman:', '\nQuestion:', '\nAnswer:', '\n```\n']
stop_token_ids = [tokenizer(x)['input_ids'] for x in stop_list]
stop_token_ids = [torch.LongTensor(x).to(device) for x in stop_token_ids]
stopping_criteria = StoppingCriteriaList([StopOnTokens(stop_token_ids)])

generate_text = transformers.pipeline(
    model=model, tokenizer=tokenizer,
    return_full_text=False,  # Adjust this if LangChain needs full text
    task='text-generation',
    temperature=0.2,
    max_new_tokens=200,
    stopping_criteria=stopping_criteria,
    repetition_penalty=1.1
)

def getLLMResponse(query, role_option, tasktype_option):
    llm = HuggingFacePipeline(pipeline=generate_text)

    prompt = f"You are a {role_option}, and {tasktype_option}. Answer the following question:\n\nQuestion: {query}\nResponse:"

    response = llm(prompt)
    return response.strip()  # Strip any extraneous whitespace

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