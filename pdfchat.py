import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama
from langchain_core.runnables import RunnablePassthrough
import os
import shutil
import re
from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import PyPDFLoader

# Initialize Streamlit app
st.title("Welcome to NotebookLM💁")

# Initialize Ollama with Phi-3 model
llm = ChatOllama(model="phi3")

# Initialize session state and chat history
if 'vector_dbs' not in st.session_state:
    st.session_state.vector_dbs = {}
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'embedding' not in st.session_state:
    st.session_state.embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'questions' not in st.session_state:
    st.session_state.questions = {}

# Function to process a single PDF file
def process_pdf(file):
    temp_dir = "temp"
    os.makedirs(temp_dir, exist_ok=True)
    temp_file_path = os.path.join(temp_dir, file.name)
    with open(temp_file_path, "wb") as f:
        f.write(file.getbuffer())
    loader = PyPDFLoader(file_path=temp_file_path)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=7500, chunk_overlap=100)
    chunks = text_splitter.split_documents(data)
    return chunks

def sanitize_collection_name(name):
    sanitized = re.sub(r'[^\w\-]', '_', name)
    sanitized = re.sub(r'^[^\w]', '', sanitized)
    sanitized = sanitized[:63]
    sanitized = re.sub(r'[^\w]$', '', sanitized)
    return sanitized

def update_vector_dbs(files):
    new_files = [file for file in files if file.name not in st.session_state.uploaded_files]
    removed_files = set(st.session_state.uploaded_files) - set(file.name for file in files)
    for file_name in removed_files:
        if file_name in st.session_state.vector_dbs:
            del st.session_state.vector_dbs[file_name]
        if file_name in st.session_state.summaries:
            del st.session_state.summaries[file_name]
        if file_name in st.session_state.questions:
            del st.session_state.questions[file_name]
    for file in new_files:
        chunks = process_pdf(file)
        collection_name = sanitize_collection_name(f"local-rag-{file.name}")
        st.session_state.vector_dbs[file.name] = Chroma.from_documents(
            documents=chunks,
            embedding=st.session_state.embedding,
            collection_name=collection_name
        )
    st.session_state.uploaded_files = {file.name: file for file in files}
    if os.path.exists("temp"):
        shutil.rmtree("temp")

def handle_query(user_question, selected_pdfs):
    all_context = []
    
    for pdf in selected_pdfs:
        db = st.session_state.vector_dbs.get(pdf)
        if db:
            results = db.similarity_search_with_score(user_question, k=5)
            context = "\n".join(f"[From {pdf}]: {doc.page_content}" for doc, _ in results)
            all_context.append(context)
    
    if all_context:
        combined_context = "\n\n".join(all_context)
        template = """
        You are an AI assistant tasked with answering questions based on the provided context from PDF documents. 
        Provide a direct, concise answer to the question using only the information in the context. If the answer cannot be fully determined from the context, clearly state what is known and what is uncertain. Keep your response brief and to the point.

        Context:
        {context}

        Question: {question}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(template)
        chain = (
            {"context": lambda _: combined_context, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        response = chain.invoke(user_question)
        
        return response.strip()
    else:
        return "No relevant information found in the selected PDFs."

def generate_summary(file_name):
    db = st.session_state.vector_dbs.get(file_name)
    if db:
        results = db.similarity_search_with_score("", k=5)
        full_text = "\n\n".join([doc.page_content for doc, _ in results])
        
        prompt = f"""Provide a concise summary of the following document content in 3-4 sentences. Focus on the main ideas and key points only.

        Document content:
        {full_text}

        Concise Summary:"""
        
        response = llm.invoke(prompt)
        
        summary = re.sub(r'^(Concise )?Summary:\s*', '', response.content, flags=re.IGNORECASE)
        return summary.strip()
    else:
        return "Summary not available for this file."

def generate_questions(summary, pdf_name):
    prompt = f"""Based on the following summary of the document '{pdf_name}', generate 2 specific and distinct questions that can be directly and accurately answered using the information in the summary. Follow these guidelines:

    1. Make the questions as specific and detailed as possible.
    2. Ensure the questions focus on key facts, figures, or main ideas presented in the summary.
    3. Avoid general or vague questions that could have multiple interpretations.
    4. Prefer questions that require precise information or understanding of the content.
    5. If possible, include numerical data or specific terms mentioned in the summary.
    6. The two questions must be different from each other and cover distinct aspects of the summary.

    Summary: {summary}

    Generated Questions:
    1.
    2.
    """
    
    response = llm.invoke(prompt)
    
    # Extract questions
    questions = re.findall(r'(?:^|\n)(?:\d+\.\s*)(.+?(?:\?|$))', response.content)
    
    # If we don't have exactly 2 questions, generate more
    while len(questions) < 2:
        additional_prompt = f"""The previous response did not provide 2 distinct questions. Please generate {2 - len(questions)} more question(s) based on the summary, ensuring they are different from any previously generated questions. Focus on different aspects or potential implications of the information in the summary.

        Summary: {summary}

        Additional Question(s):
        {len(questions) + 1}.
        """
        additional_response = llm.invoke(additional_prompt)
        additional_questions = re.findall(r'(?:^|\n)(?:\d+\.\s*)(.+?(?:\?|$))', additional_response.content)
        questions.extend(additional_questions)
    
    return questions[:2]  # Return exactly 2 questions

# Main app workflow
def main():
    uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    current_file_names = [file.name for file in uploaded_files] if uploaded_files else []
    if set(current_file_names) != set(st.session_state.uploaded_files):
        if uploaded_files:
            with st.spinner('Processing PDFs and updating vector databases...'):
                update_vector_dbs(uploaded_files)
            st.success("Vector databases updated successfully!")
        else:
            st.session_state.vector_dbs = {}
            st.session_state.uploaded_files = {}
            st.session_state.summaries = {}
            st.session_state.questions = {}
            st.warning("No files uploaded. Vector databases have been cleared.")

    # PDF selection with checkboxes
    selected_pdfs = []
    if st.session_state.uploaded_files:
        st.sidebar.write("Select PDFs for Answer:")
        for pdf in st.session_state.uploaded_files.keys():
            if st.sidebar.checkbox(pdf, key=f"checkbox_{pdf}"):
                selected_pdfs.append(pdf)

    # Display summaries and generate questions for selected PDFs
    for pdf in selected_pdfs:
        st.subheader(f"Summary of {pdf}")
        if pdf not in st.session_state.summaries:
            with st.spinner(f'Generating summary for {pdf}...'):
                try:
                    summary = generate_summary(pdf)
                    st.session_state.summaries[pdf] = summary
                except Exception as e:
                    st.error(f"Error generating summary for {pdf}: {str(e)}")
                    st.session_state.summaries[pdf] = "Error generating summary."
        st.write(st.session_state.summaries[pdf])

        # Generate questions for this PDF
        if pdf not in st.session_state.questions:
            with st.spinner(f'Generating questions for {pdf}...'):
                try:
                    questions = generate_questions(st.session_state.summaries[pdf], pdf)
                    st.session_state.questions[pdf] = questions
                except Exception as e:
                    st.error(f"Error generating questions for {pdf}: {str(e)}")
                    st.session_state.questions[pdf] = [
                        "Error generating first question.",
                        "Error generating second question."
                    ]

        # Display questions for this PDF
        st.write(f"Questions for {pdf}:")
        for i, question in enumerate(st.session_state.questions[pdf]):
            with st.expander(f"Q{i+1}: {question}"):
                if st.button(f"Ask this question", key=f"{pdf}_Q{i+1}_ask"):
                    st.session_state.auto_query = question

    # Display chat history
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.chat_message("user").markdown(message["content"])
        else:
            st.chat_message("assistant").markdown(message["content"])

    # Chat input widget for custom questions
    prompt = st.chat_input("Ask a custom question about the selected PDFs")
    if 'auto_query' in st.session_state:
        prompt = st.session_state.auto_query
        del st.session_state.auto_query

    if prompt and selected_pdfs:
        st.chat_message("user").markdown(prompt)
        with st.spinner('Generating answer...'):
            response = handle_query(prompt, selected_pdfs)
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()