import streamlit as st # To build Web app
from PyPDF2 import PdfReader # For reading the PDF
from langchain.text_splitter import RecursiveCharacterTextSplitter # For splitting the long pdf text and make it to chunks due to limition of input token of model
import os # For importing the env variable
from langchain_google_genai import GoogleGenerativeAIEmbeddings # For embedding the chunks
import google.generativeai as genai # For gemini
from langchain_community.vectorstores import FAISS # For storing the embedding into vector database
from langchain_google_genai import ChatGoogleGenerativeAI # For Gemini Gen AI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv # For loading the env variable

load_dotenv()
os.getenv("GOOGLE_API_KEY") # For google api key go to the makersuite google api key website
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to read the all pdf page wise and and save it to the text variable
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text


# Function to convert text into chunks by overlapping
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks

# Funtion to convert chunks into embeddings and save it the FAISS vector database locally in folder faiss_index
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Function for creating the prompt and conversational chain. Chain_type = stuff is for internal text summarization
def get_conversational_chain():

    prompt_template = """
    Extract the most detailed and accurate information from the given context to answer the following question fully and comprehensively.

    Provide all relevant facts, figures, and explanations, even if they are not explicitly stated. If the answer is not found within the context, state clearly that it's unavailable.
    
    And also try to summarize the oploaded pdf files.

    **Context:**
    {context}


    {question}

    **Answer:**
    """


    model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

    prompt = PromptTemplate(template = prompt_template, input_variables = ["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain


# Function for similarity search of user questions and generate the reply
def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    
    new_db = FAISS.load_local("faiss_index", embeddings)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])



# Function to create the streamlit app/ web app
def main():
    st.set_page_config("Chat with multiple PDF")
    st.header("Chat with multiples PDF using Gemini💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
