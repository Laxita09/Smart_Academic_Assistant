import streamlit as st
# -------------------- Page Configuration --------------------
st.set_page_config(page_title="Smart Academic Assistant", layout="centered")

import sys
try:
    import pysqlite3
    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except ImportError:
    st.warning("pysqlite3 could not be loaded. ChromaDB may not work.")

from urllib import response
import tempfile
import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyMuPDFLoader , TextLoader
from langchain_community.document_loaders import UnstructuredWordDocumentLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

load_dotenv()
hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")


# -------------------- Page Configuration --------------------
#st.set_page_config(page_title="Smart Academic Assistant", layout="centered")

# -------------------- Title --------------------
st.title("📚 Smart Academic Assistant")
st.write("Upload your academic documents and ask questions to get structured answers.")

# -------------------- File Upload Section --------------------
uploaded_files = st.file_uploader(
    "Upload academic documents (PDF, DOCX, or TXT):",
    type=["pdf", "docx", "txt"],
    accept_multiple_files=True
)

# -------------------- Question Input --------------------
question = st.text_input("Enter your academic question:")

# -------------------- Submit Button --------------------
if st.button("Get Answer"):
    if not uploaded_files or not question:
        st.warning("Please upload at least one document and enter a question.")
    else:
        # -------------------- PLACEHOLDER: RAG Pipeline Logic --------------------
        # TODO:
        # 1. Load documents using LangChain document loaders

         # Rebuild trigger
        all_docs = []
        # Rebuild trigger

        for file in uploaded_files:
            file_name = file.name
            extension = file_name.split(".")[-1].lower()

            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{extension}") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            if extension == "pdf":
                loader = PyMuPDFLoader(tmp_path)
            elif extension == "txt":
                loader = TextLoader(tmp_path, encoding="utf-8")
            elif extension == "docx":
                loader = UnstructuredWordDocumentLoader(tmp_path)
            else:
                st.warning(f"Unsupported file type: {extension}")
                continue

            docs = loader.load()
            all_docs.extend(docs)

        
        st.session_state["all_docs"] = all_docs



        # 2. Split documents using RecursiveCharacterTextSplitter or similar
        # Rebuild trigger


        splitter = RecursiveCharacterTextSplitter(
            chunk_size=100,
            chunk_overlap=0,
        )

        chunks = splitter.split_documents(all_docs)



        # 3. Create embeddings and store in vector store (e.g., FAISS, Chroma)
        # Rebuild trigger


        vector_store = Chroma(
            embedding_function=HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-MiniLM-L6-v2'),
            persist_directory= 'my_chroma_db',
            collection_name='sample'
        )

        vector_store.add_documents(chunks)


        # # 4. Retrieve relevant chunks based on the question
        # Rebuild trigger


        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

        retrieved_docs    = retriever.invoke(question)

        context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)


        #using vector store to find similarity score
        # Rebuild trigger


        similarity_score = vector_store.similarity_search_with_score(query = question)
        
        scores = [item[1] for item in similarity_score]

        average_score = sum(scores) / len(scores)         #finding avergae of all the chunk's score

        
        # 5. Use Groq-hosted LLM via LangChain (e.g., Mixtral, Gemma, Llama3)
        # Rebuild trigger


        llm = ChatGroq(
            api_key = os.getenv("GROQ_API_KEY"),
            model_name = "llama3-8b-8192"                 
        )

        st.session_state["llm"] = llm


        prompt = PromptTemplate(
            template="""
                You are a helpful assistant.
                Answer ONLY from the provided context.
                If the context is insufficient, just say you don't know.

                {context}
                Question: {question}
            """,
            input_variables = ['context', 'question']
        )

        final_prompt = prompt.invoke({"context": context_text, "question": question})

        answer = llm.invoke(final_prompt)


        # 6. Use Output Parser to format structured response
        # Rebuild trigger


        parser = StrOutputParser()

        result = parser.invoke(answer)

        file in uploaded_files

        response = {
            "question": question,
            "answer": result,
            "source_document": file.name,
            "confidence_score": average_score
        }


        # Example output format (replace this with actual output):
        # response = {
        #     "question": question,
        #     "answer": "Your answer here",
        #     "source_document": "Document Name",
        #     "confidence_score": "0.93"
        # }
        # Rebuild trigger

        
        st.subheader("📄 Answer:")
        st.json(response)

        #st.info("Implement your RAG logic above and display the final structured response here.")


# -------------------- Bonus Section: Agent Tools --------------------
st.markdown("---")
st.subheader("🧠 Extra Tools ")

col1, col2, col3 = st.columns(3)

with col1:
    if st.button("Summarize Document"):
        # TODO: Implement SummarizeDocumentTool using LangChain agent
        # Rebuild trigger


        all_docs = st.session_state['all_docs']
        context_text = "\n\n".join(doc.page_content for doc in all_docs)
        llm = st.session_state["llm"]


        @tool
        def summary(context_text :str )  ->str:
            """Summarize the given document content using LLM."""

            prompt = PromptTemplate(
            template="""
                You are a helpful assistant.
                Answer ONLY from the provided context.
                Summarise the given document content .

                {context}
            """,
            input_variables = ['context']
            )

            summary_prompt = prompt.invoke({"context": context_text})
            
            return(llm.invoke(summary_prompt))
        
        context_text = "\n\n".join(doc.page_content for doc in all_docs)
        summary_result = summary.invoke(context_text)
        
        st.info("Summary will be shown here.")
        st.write(summary_result.content)

with col2:
    if st.button("Generate MCQs"):
        # TODO: Implement GenerateMCQsTool using LangChain agent
        # Rebuild trigger


        all_docs = st.session_state['all_docs']
        context_text = "\n\n".join(doc.page_content for doc in all_docs)
        llm = st.session_state["llm"]

        @tool
        def mcq_generator(context_text :str )  ->str:
            """From the given content generate 5 Multiple Choice Question"""

            prompt = PromptTemplate(
            template="""
                You are a helpful assistant.
                Answer ONLY from the provided context.
                From the given content generate 5 multiple choice questions with each question question having 4 options.
                along with the question provide the correct answer.  

                {context}

            """,
            input_variables = ['context']
            )

            mcq_prompt = prompt.invoke({"context": context_text})
            
            return(llm.invoke(mcq_prompt))
        
        context_text = "\n\n".join(doc.page_content for doc in all_docs)
        mcq_result = mcq_generator.invoke(context_text)
        
        st.info("Generated MCQs will appear here.")
        st.write(mcq_result.content)

        

with col3:
    if st.button("Topic-wise Explanation"):
        # TODO: Implement TopicWiseExplanationTool using LangChain agent
        # Rebuild trigger


        all_docs = st.session_state['all_docs']
        context_text = "\n\n".join(doc.page_content for doc in all_docs)
        llm = st.session_state["llm"]

        @tool
        def topic_wise_explain(context_text :str )  ->str:
            """From the given content extract all the main topics and expalin then in detail using examples"""

            prompt = PromptTemplate(
            template="""
                You are a helpful assistant.
                Answer ONLY from the provided context.
                From the given content extract all the main topics , then explain them in 
                detailed form along with examples for better understanding. 

                {context}

            """,
            input_variables = ['context']
            )

            explain_prompt = prompt.invoke({"context": context_text})
            
            return(llm.invoke(explain_prompt))
        
        context_text = "\n\n".join(doc.page_content for doc in all_docs)
        explain_result = topic_wise_explain.invoke(context_text)

        st.info("Topic-wise explanation will be displayed here.")
        st.write(explain_result.content)


# -------------------- Footer --------------------
# Rebuild trigger

st.markdown("---")
#st.caption("Mentox Bootcamp · Final Capstone Project · Phase 1")
