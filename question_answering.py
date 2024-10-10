import torch
from transformers import AutoModelForCausalLM, AutoTokenizer,pipeline
import streamlit as st
torch.manual_seed(0)
st.write("Hello World")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Phi-3.5-mini-instruct", 
    device_map = device,
    torch_dtype = "auto",
    trust_remote_code = True,
)
print(torch.cuda.is_available())  # Should return True if CUDA is enabled
tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")

# messages = [
#     {"role": "system", "content": "You are a helpful AI assistant."},
#     {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
#     {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
#     {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
# ]
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    # {"role": "user", "content": "Can you provide ways to eat combinations of bananas and dragonfruits?"},
    {"role": "assistant", "content": "Sure! Here are some ways to eat bananas and dragonfruits together: 1. Banana and dragonfruit smoothie: Blend bananas and dragonfruits together with some milk and honey. 2. Banana and dragonfruit salad: Mix sliced bananas and dragonfruits together with some lemon juice and honey."},
    {"role": "user", "content": "What about solving an 2x + 3 = 7 equation?"},
]

if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role":"system","content":"You are a helpfull AI assistant."}
    ]

# Funtion to generate AI response
def generated_response():
    formatted_prompt = ""
    for message in st.session_state['messages']:
        formatted_prompt += f"{message['role']}: {message['content']}\n"

    pipe = pipeline("text-generation",model=model, tokenizer = tokenizer)

    generation_args = {
    "max_new_tokens":500,
    "return_full_text":False,
    "temperature":0.0,
    "do_sample":False,
    }

    output = pipe(formatted_prompt,**generation_args)

    return output[0]['generated_text']


# Display conversation history
for message in st.session_state['messages']:
    st.write(f"**{message['role'].capitalize()}**: {message['content']}")

# User input
user_input = st.text_input("You:")
if user_input:
    # Append user's input to the conversation history
    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    # Generate AI response
    assistant_response = generate_response()
    
    # Append assistant's response to the conversation history
    st.session_state['messages'].append({"role": "assistant", "content": assistant_response})
    
    # Refresh to show updated conversation
    st.experimental_rerun()