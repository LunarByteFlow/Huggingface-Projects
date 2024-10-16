import streamlit as st
from huggingface_hub import InferenceClient

# Hugging Face Inference API Client
client = InferenceClient(api_key="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")

st.title("Your Personal AI Chatbot: ")

# Initialize the conversation history if not already present
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful AI assistant."}
    ]

# Function to generate AI response using Hugging Face Inference API
def generate_response():

    # Simple greeting response logic
    if user_input.lower() in ["hello", "hi", "hey"]:
        return "Hello! How can I assist you today?"
        
    # Format the conversation history as a prompt
    formatted_prompt = ""
    for message in st.session_state['messages']:
        formatted_prompt += f"{message['role']}: {message['content']}\n"

    # API call to Hugging Face Inference endpoint for text generation
    response = client.text_generation(
        model="microsoft/Phi-3.5-mini-instruct",
        prompt=formatted_prompt,
        max_new_tokens= 1500,
        temperature= 0.1,
        do_sample= False,
        # parameters={
        #     "max_new_tokens": 500,
        #     "temperature": 0.0,
        #     "do_sample": False,
        # }
    )

    # Return the generated text from the API response
    st.write(response)

    # Check if response is a dictionary and contains 'generated_text'
    if isinstance(response, dict) and 'generated_text' in response:
        return response['generated_text']
    else:
        return "Error: Unexpected response format."
    # return response['generated_text']

# Display conversation history
for message in st.session_state['messages']:
    st.write(f"**{message['role'].capitalize()}**: {message['content']}")

# User input
user_input = st.text_input("You:",key="user_input")
if user_input:
    # Append user's input to the conversation history
    st.session_state['messages'].append({"role": "user", "content": user_input})
    
    # Generate AI response using Hugging Face Inference API
    assistant_response = generate_response()
    
    # Append assistant's response to the conversation history
    st.session_state['messages'].append({"role": "assistant", "content": assistant_response})
    
    # Refresh to show updated conversation
    st.experimental_get_query_params()

# Auto-scroll to bottom on load
# st.experimental_set_query_params(scroll_to_bottom="true")

    
    # Append assistant's response to the conversation history
    st.session_state['messages'].append({"role": "assistant", "content": assistant_response})
    
    # Refresh to show updated conversation
    st.experimental_get_query_params()
