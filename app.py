from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.vectorstores import Chroma,FAISS
#from langchain.chains import RetrievalQA, LLMChain
#from langchain_community.retrievers import BM25Retriever, EnsembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_huggingface import HuggingFaceEndpoint
from langchain.chains import RetrievalQA
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os
from pdf2image import convert_from_path
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
import pytesseract
from dotenv import load_dotenv
from PIL import Image
import PyPDF2
from PyPDF2 import PdfReader
import streamlit as st
import io
# Load the .env file
load_dotenv()

# Set user agent and paths
os.environ['USER_AGENT'] = 'MyApp/1.0'
os.environ['TESSDATA_PREFIX'] = "D:/ML/assignments/cerebral/tessdata/"
HF_TOKEN = os.getenv('HF_TOKEN')

def is_scanned_pdf(pdf_file):
    """Check if the PDF is scanned by attempting to extract text."""
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text = page.extract_text()
        if text:  # If any page has extractable text, it's not scanned
            return False
    return True  # No extractable text found, likely scanned


def extract_text_from_pdf(pdf_files):
    extracted_text = ""
    for pdf_file in pdf_files:
        pdf_bytes = pdf_file.read()  # Read the file content as bytes
        pdf_stream = io.BytesIO(pdf_bytes)  # Convert to BytesIO
        pdf_stream.seek(0)  # Reset the stream to the start

        if is_scanned_pdf(pdf_stream):
            # For scanned PDF, convert pages to images for OCR
            pdf_stream.seek(0)  # Reset stream for OCR usage
            pages = convert_from_path(io.BytesIO(pdf_bytes), 500)  # Handle file content in memory
            for i, page in enumerate(pages):
                filename = f"page_{i+1}.jpg"
                page.save(filename, 'JPEG')
                try:
                    text = pytesseract.image_to_string(Image.open(filename), lang='ben+hin+eng+chi_sim')
                    text = text.replace('-\n', '')  # Remove hyphenated line breaks
                    extracted_text += text
                except FileNotFoundError:
                    st.error(f"File {filename} not found, skipping...")
        else:
            pdf_stream.seek(0)  # Reset stream for text extraction
            pdf_reader = PyPDF2.PdfReader(pdf_stream)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text = page.extract_text()
                extracted_text += text if text else ""

    return extracted_text if extracted_text else "No text found."

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
        # Initialize the embeddings model
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5")
    # Set up the vector store
    vectorstore = FAISS.from_texts(text_chunks, embeddings)
        # Ensure the directory exists for saving the FAISS index
    if not os.path.exists("faiss_index"):
        os.makedirs("faiss_index")
    vectorstore.save_local("faiss_index")
    return vectorstore

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible using the provided context. If the answer is not available in the context, try to infer it based on the information you have. Provide a detailed response, considering both explicit and implicit details.

    Context: 
    {input_documents}

    Question: 
    {question}

    Answer:
    """

    
    llm = HuggingFaceEndpoint(repo_id="microsoft/Phi-3-mini-4k-instruct", huggingfacehub_api_token=HF_TOKEN)

    prompt = PromptTemplate(template=prompt_template, input_variables=["input_documents", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt, document_variable_name="input_documents")

    return chain


def user_input(user_question):
    embeddings = HuggingFaceInferenceAPIEmbeddings(api_key=HF_TOKEN, model_name="BAAI/bge-base-en-v1.5")

        # Ensure that the FAISS index file exists before attempting to load
    if os.path.exists("faiss_index/index.faiss"):
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization = True)
    else:
        st.error("The FAISS index was not found. Please upload and process the PDFs first.")
        return
    docs = new_db.similarity_search(user_question,k=5)

    chain = get_conversational_chain()

    
    response = chain(
        {"input_documents":docs, "question": user_question}
        , return_only_outputs=True)

    print(response)
    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Chat with PDF 💁")

    user_question = st.text_input("Ask a Question from the PDF Files")

    # Upload and process PDF documents
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", 
                                    accept_multiple_files=True, type=["pdf"])

        if st.button("Submit & Process") and pdf_docs:
            with st.spinner("Processing..."):
                # Extract and chunk the text from uploaded PDFs
                # Convert each uploaded file to a readable format
                pdf_paths = [pdf for pdf in pdf_docs]  # List of uploaded PDFs
                raw_text = extract_text_from_pdf(pdf_paths)  # Pass the list of PDFs
                text_chunks = get_text_chunks(raw_text)
                get_vectorstore(text_chunks)
                st.success("Processing completed.")

    # Process the user question when provided
    if user_question and pdf_docs:
        user_input(user_question)


if __name__ == "__main__":
    main()




