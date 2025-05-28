from huggingface_hub import login
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from transformers import pipeline, AutoTokenizer
from langchain_community.vectorstores import FAISS
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_community.document_loaders import CSVLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Login to Hugging Face Hub
# login(token="")

# Load CSV data
loader = CSVLoader(file_path="esg_data/27dd16b6-7c0f-4af5-925f-3893373a5d9c_Data.csv")
docs = loader.load()

# Preprocessing documents to handle ".." as None
for doc in docs:
    doc.page_content = doc.page_content.replace("..", "None")

# Embedding model from Hugging Face
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Creating vector store using FAISS
vectorstore = FAISS.from_documents(docs, embedding_model)

# Creating retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})  

# Loading the LLM model from Hugging Face
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
qa_model = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct", max_new_tokens=200, temperature=0.001, device="cuda", pad_token_id=tokenizer.eos_token_id)
llm = HuggingFacePipeline(pipeline=qa_model)

# Defining a custom prompt for additional control over the output 
prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
                You are a question answering system providing simple, straightforward and reliable answers.
                In case a value is None or you are unsure, please say so. Use the following CSV snippet to answer the question.

                CSV Context:
                {context}

                Question:
                {question}

                Answer:
            """)

# Creating RetrievalQA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",  # Use 'stuff' for simple context stuffing
    chain_type_kwargs={"prompt": prompt_template}
)

# 8. Running the QA loop until the user decides to exit
# query = "What percentage of the land area of Greece was agricultural in 1963?"
query = ""

while query.lower() != "exit":
    query = input("Enter your question (or type 'exit' to quit): ")
    if query.lower() == "exit":
        break
    answer = qa_chain.invoke(query)
    print("Answer with RAG:", answer["result"])
    print("Answer without RAG:", qa_model(query)[0]["generated_text"])
