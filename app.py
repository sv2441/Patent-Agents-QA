import streamlit as st
import pandas as pd
import os
import time
import json
import shutil  
from io import StringIO
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import WebBaseLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import OpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
# from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()


# Function to process URL and query
def process_url_and_query(name, url, query, selected_llm, temperature, k_value):
    # Load data from the specified URL
    loader = WebBaseLoader(url)
    data = loader.load()

    # Split the loaded data
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=40)
    docs = text_splitter.split_documents(data)

    # Create embeddings using HuggingFaceEmbeddings
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create a directory for embeddings
    db_path = f"./db_embeddings/{name.replace(' ', '_')}"
    if not os.path.exists(db_path):
        os.makedirs(db_path)

    # Create a Chroma vector database
    vectordb = Chroma.from_documents(
        documents=docs, embedding=embedding_function, persist_directory=db_path
    )
    vectordb.persist()

    # Create a retriever
    retriever = vectordb.as_retriever(search_kwargs={"k": k_value})

    # Select LLM based on user input
    llm = None
    if selected_llm == "gpt-4o-mini (OpenAI)":
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=temperature)
    elif selected_llm == "gpt-4o (OpenAI)":
        llm = ChatOpenAI(model_name="gpt-4o", temperature=temperature)
    elif selected_llm == "Gemini-1.5-pro (Google)":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=temperature)
    elif selected_llm == "Gemini-1.5-flash (Google)":
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=temperature)
    elif selected_llm == "mixtral (Groq)":
        llm = ChatGroq(model="mixtral-8x7b-32768", temperature=temperature)
    elif selected_llm == "llama 3.1 (Groq)":
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=temperature)
    else:
        raise ValueError("Unsupported LLM selected.")

    # Prompt template setup
    system_template = """Use the following pieces of context to answer the user's question.
    If you don't know the answer, just say that you don't know; don't try to make up an answer."""
    messages = [
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}"),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    chain_type_kwargs = {"prompt": prompt}

    # Create a RetrievalQA chain
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
    )

    # Run the query and get a response
    response = qa.run(query)

    return response

# Streamlit app title
st.title("Patent QA Application")

# Sidebar for LLM selection and settings
st.sidebar.header("Application Settings")

# LLM model selection
llm_options = [
    "gpt-4o-mini (OpenAI)",
    "gpt-4o (OpenAI)",
    "gemini-1.5-pro (Google)",
    "Gemini-1.5-flash (Google)",
    "mixtral (Groq)",
    "llama 3.1 (Groq)",
]
selected_llm = st.sidebar.selectbox("Select LLM Model", llm_options)

# Temperature setting
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7)

# Retrieval 'k' value setting
k_value = st.sidebar.number_input("Retrieval k Value", min_value=1, max_value=100, value=20)

# Main section for file uploads
st.header("Upload CSV Files")

# File uploader for the main CSV file (Name and Url)
uploaded_csv = st.file_uploader("Upload CSV file with columns 'Name' and 'Url'", type=["csv"])
# uploaded_csv=pd.read_csv("Sample.csv")
# File uploader or editor for the prompts file (Aspect and Updated ideas)


st.subheader("Prompts File (Agents and Prompts)")
prompts_file_option = st.radio(
    "Do you want to upload a prompts file or edit manually?",
    ("Edit Manually","Upload CSV File", ),
)

if prompts_file_option == "Upload CSV File":
    uploaded_prompts_csv = st.file_uploader(
        "Upload CSV file with columns 'Aspect' and 'Updated ideas'", type=["csv"]
    )
    if uploaded_prompts_csv is not None:
        prompts_df = pd.read_csv(uploaded_prompts_csv)
else:
    # Initialize an empty DataFrame or with default values
    prompts_df = pd.read_csv("prompts.csv")
    prompts_df = st.data_editor(prompts_df, num_rows="dynamic")

# Button to start processing
if st.button("Start Processing"):
    if uploaded_csv is not None and not prompts_df.empty:
        # Read the main CSV file
        main_df = pd.read_csv(uploaded_csv)

        # Initialize output data list
        output_data = []

        # Progress bar
        progress_bar = st.progress(0)
        total_iterations = len(main_df) * len(prompts_df)
        current_iteration = 0

        # Spinner
        with st.spinner("Processing..."):
            for index, row in main_df.iterrows():
                name = row["Name"]
                url = row["Url"]

                # Process each Aspect and Updated ideas (query)
                for _, prompt_row in prompts_df.iterrows():
                    aspect_name = prompt_row["Aspect"]
                    query = prompt_row["Updated ideas"]

                    # Process the URL and query with retries
                    response = None
                    for attempt in range(3):
                        try:
                            response = process_url_and_query(
                                name, url, query, selected_llm, temperature, k_value
                            )
                            break  # Break if successful
                        except Exception as e:
                            st.error(f"Error: {e}. Retry {attempt + 1}/3.")
                            time.sleep(2)  # Wait before retrying

                    if response is None:
                        response_text = "Failed after 3 attempts"
                    else:
                        response_text = response

                    # Append to output data
                    output_data.append(
                        {
                            "Name": name,
                            "Url": url,
                            "Aspect": aspect_name,
                            "Response": response_text,
                        }
                    )

                    # Update progress bar
                    current_iteration += 1
                    progress_bar.progress(current_iteration / total_iterations)

                    # Delete embeddings for the URL
                    db_path = f"./db_embeddings/{name.replace(' ', '_')}"
                    if os.path.exists(db_path):
                        try:
                            shutil.rmtree(db_path)  # This will delete the directory and all its contents
                        except Exception as e:
                            st.error(f"Error deleting embeddings: {e}")

            # Convert output data to DataFrame and display
            output_df = pd.DataFrame(output_data)
            st.success("Processing complete!")
            st.dataframe(output_df)

            # Provide option to download the final CSV
            csv = output_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download Output CSV",
                data=csv,
                file_name="final_output.csv",
                mime="text/csv",
            )
    else:
        st.error("Please upload all required files and ensure they are not empty.")

