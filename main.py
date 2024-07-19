from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import faiss

from langchain_community.llms import HuggingFacePipeline
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.llms import ollama
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Update this with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# sample document
loader = TextLoader("sample_document.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# Set up embeddings and vector store
#embeddings = HuggingFaceEmbeddings()
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cuda"})
#vectorstore = FAISS.from_documents(texts, embeddings)

vectorstore = FAISS.from_documents(texts, embeddings)
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    vectorstore.index = faiss.index_cpu_to_gpu(res, 0, vectorstore.index)
# Initialize local LLM
# Initialize local LLM
# llm = HuggingFacePipeline.from_model_id(
#     model_id="gpt2-large",
#     task="text-generation",
#     pipeline_kwargs={"max_new_tokens": 100, "do_sample": True, "temperature": 0.7},
# )
#llm = ollama.Ollama(model="llama2")
#llm = ollama.Ollama(model="llama2")
llm = ollama.Ollama(
    model="llama2",  # Use quantized model

)
# Custom prompt template
prompt_template = """You are an AI assistant for a bookstore. Use the following pieces of context to answer the customer's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context: {context}

Customer: {question}
AI Assistant: """

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)

# Create a memory object
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)

# Create the conversational chain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory,
    combine_docs_chain_kwargs={"prompt": PROMPT}
)

class ChatInput(BaseModel):
    message: str

class ChatOutput(BaseModel):
    response: str

class ChatHistory(BaseModel):
    history: List[tuple]

chat_history = []

@app.post("/chat", response_model=ChatOutput)
async def chat(input: ChatInput):
    global chat_history
    try:
        result = qa_chain({"question": input.message})
        response = result['answer']
        chat_history.append((input.message, response))
        return ChatOutput(response=response)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 

@app.get("/history", response_model=ChatHistory)
async def get_history():
    global chat_history
    return ChatHistory(history=chat_history)