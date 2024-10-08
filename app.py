import PyPDF2
from PyPDF2 import PdfReader
from transformers import pipeline
import markdown
import re
import streamlit as st
import requests

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large-cnn"
headers = {"Authorization": "Bearer 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"}
def query_inference_api(text):
  input_text = text.get('input',"")
  payload = {"inputs":input_text}
  print(f"Payload before sending: {payload}") 
  response = requests.post(API_URL,headers = headers, json = payload)
  return response.json()

  
def extract_text_pdf(pdf_path):
  with open(pdf_path,'rb') as pdf_file:
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
      text +=page.extract_text()
    return text
def clean_text(text):
  cleaned_text = re.sub(r"(\*|\**|\n|\r|\[|\]|\\|\/|\|)", "", text)
  cleaned_text = re.sub(r"(\s+\n\s+)", "\n", cleaned_text)
  return cleaned_text
# pdf_path = 'C:\Users\Mahnoor\Desktop\huggingface_projects\dummy_data.pdf'

# pdf_path = './dummy_data.pdf'
# text = extract_text_pdf(pdf_path)
# cleaned_text = clean_text(text)
# print(f"The text after cleaning: {cleaned_text}")
# summarizer = pipeline('summarization',model="facebook/bart-large-cnn")
# ARTICLE = cleaned_text
# print(summarizer(ARTICLE,max_length = 130,min_length = 30,do_sample = False))
# html = markdown.markdown(ARTICLE)
# print(html)

# Streamlit App
def main():
  st.title('Summarize Your PDF')
  # uploaded_file = st.file_uploader("Upload your PDF file",type='pdf')
  # Create a text area for user input
  user_input = st.text_area("Enter your text here:", height=200)
  if st.button('Summarize Text'):
    if user_input:
      cleaned_text = clean_text(user_input)
      payload = {"input":cleaned_text}
      summary_response = query_inference_api(payload)
      # Check data type (optional)

      print(type(summary_response))
      # Extract summary from the respise
      summary_text = summary_response[0].get("summary_text","summary_response")
      # Check if summary text is available
      if isinstance(summary_response, list) and len(summary_response) > 0:
        st.write(f"<p style='font-size:20px'>Summary:</p>", unsafe_allow_html=True)
        st.write(f"<p>{summary_text}</p>", unsafe_allow_html=True)
        
      else:
        st.write("Unable to generate a summary.")
      # print(summary_response)
      # summary = summary_response[0]["inputs"]
      # summary = summarizer(cleaned_text, max_length = 250, min_length = 100)
      # st.markdown(f"<p style='font-size:50px'>Summary:</p>", unsafe_allow_html=True)
      # st.markdown(f"<p style='font-size:20px'>{summary[0]['summary_text']}</p>", unsafe_allow_html=True)


# st.text(summary[0]['summary_text'],height=200)
  # if uploaded_file is not None:
  #   pdf_path = uploaded_file.name
  #   text = extract_text_pdf(pdf_path)



if __name__ == "__main__":
  main()
