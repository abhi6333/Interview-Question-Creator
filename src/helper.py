from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import TokenTextSplitter
from langchain.docstore.document import Document
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from src.prompt import *
import os
from dotenv import load_dotenv


# OpenAI authentication
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY



def file_processing(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load()

    question_gen = ""

    for page in data:
        question_gen += page.page_content

    splitter_question_gen = TokenTextSplitter(
    model_name= "gpt-3.5-turbo",
    chunk_size = 10000, #1st give bigger no 
    chunk_overlap = 200
    )

    chunk_question_gen = splitter_question_gen.split_text(question_gen)

    document_question_gen = [Document(page_content = t) for t in chunk_question_gen]

    splitter_ans_gen = TokenTextSplitter(
    model_name= "gpt-3.5-turbo",
    chunk_size = 1000, #1st give bigger no 
    chunk_overlap = 100
    )

    document_ans_gen = splitter_ans_gen.split_documents(document_question_gen)

    return document_question_gen , document_ans_gen



def llm_pipeline(file_path):

    document_question_gen , document_ans_gen = file_processing(file_path)

    llm_question_gen_pipeline = ChatOpenAI(
    model = 'gpt-3.5-turbo',
    temperature = 0.3,
    
    )

    
    PROMPT_QUESTIONS = PromptTemplate(template=prompt_template, input_variables=['text'])

    REFINE_PROMPT_QUESTIONS = PromptTemplate(
    input_variables=["existing_answer", "text"],
    template=refine_template,
    )

    ques_gen_chain = load_summarize_chain(llm = llm_question_gen_pipeline, 
                                          chain_type = "refine", 
                                          verbose = True, 
                                          question_prompt=PROMPT_QUESTIONS, 
                                          refine_prompt=REFINE_PROMPT_QUESTIONS)
    
    ques = ques_gen_chain.run(document_question_gen)

    embeddings = OpenAIEmbeddings()

    vector_store = FAISS.from_documents(document_ans_gen, embeddings)

    llm_answer_gen = ChatOpenAI(temperature=0.1, model="gpt-3.5-turbo")

    ques_list = ques.split("\n")

    filtered_ques_list = [element for element in ques_list if element.endswith('?') or element.endswith('.')]


    answer_generation_chain = RetrievalQA.from_chain_type(llm=llm_answer_gen, 
                                               chain_type="stuff", 
                                               retriever=vector_store.as_retriever())
    
    return answer_generation_chain , filtered_ques_list




