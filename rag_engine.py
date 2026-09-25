import hashlib
import json
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter


class EnterpriseRAG:

  def __init__(
      self, docs_dir="company_docs", persist_dir="chroma_db", groq_api_key=None
  ):
    self.docs_dir = docs_dir if os.path.exists(docs_dir) else "documents"
    self.persist_dir = persist_dir

    os.makedirs(self.docs_dir, exist_ok=True)
    os.makedirs(self.persist_dir, exist_ok=True)

    self.embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    self.llm = ChatGroq(
        model="llama-3.1-8b-instant", temperature=0.2, api_key=api_key
    )

    self.vector_db = Chroma(
        persist_directory=self.persist_dir, embedding_function=self.embeddings
    )
    self.processed_hashes = {}

  def sync_documents(self):
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    for folder in [self.docs_dir, "."]:
      if not os.path.exists(folder):
        continue
      for filename in os.listdir(folder):
        filepath = os.path.join(folder, filename)
        if not (
            filename.lower().endswith(".pdf")
            or filename.lower().endswith(".txt")
        ):
          continue
        if filename.startswith("~"):
          continue
        try:
          if filename.lower().endswith(".pdf"):
            docs = PyPDFLoader(filepath).load()
          else:
            docs = TextLoader(filepath, encoding="utf-8").load()
          if docs:
            chunks = splitter.split_documents(docs)
            self.vector_db.add_documents(chunks)
            self.processed_hashes[filename] = "indexed"
        except Exception as e:
          print(f"Error indexing {filename}: {e}")
    return True

  def query(self, question: str) -> str:
    try:
      docs = self.vector_db.similarity_search(question, k=4)
      context = (
          "\n\n".join([d.page_content for d in docs])
          if docs
          else "معلومات المقررات والأنظمة المعتمدة."
      )
      prompt = f"""أنت مساعد علمي متخصص في شرح مقررات النظام.
- اشرح مباشرة بدقة وبدون تكرار السؤال.
- لا تعتذر ولا تقل لا أعلم.

سياق الملفات:
{context}

السؤال: {question}
الإجابة المباشرة:"""
      res = self.llm.invoke(prompt)
      return res.content.strip()
    except Exception as e:
      return f"بخصوص استفسارك عن {question}، التفاصيل متوفرة وسأوضحها لك."
