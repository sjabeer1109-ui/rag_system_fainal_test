import hashlib
import json
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq


class EnterpriseRAG:

  def __init__(
      self, docs_dir="company_docs", persist_dir="chroma_db", groq_api_key=None
  ):
    self.docs_dir = docs_dir if os.path.exists(docs_dir) else "documents"
    self.persist_dir = persist_dir

    os.makedirs(self.docs_dir, exist_ok=True)
    os.makedirs(self.persist_dir, exist_ok=True)

    # قراءة مفتاح Groq سواء من بيئة السيرفر أو من Streamlit Secrets
    self.api_key = groq_api_key or os.getenv("GROQ_API_KEY")
    if not self.api_key:
      try:
        import streamlit as st

        self.api_key = st.secrets.get("GROQ_API_KEY")
      except Exception:
        pass

    self.vector_db = None
    self._init_db()

  def _init_db(self):
    try:
      from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
      from langchain_community.vectorstores import Chroma

      embeddings = FastEmbedEmbeddings(
          model_name="sentence-transformers/all-MiniLM-L6-v2"
      )
      self.vector_db = Chroma(
          persist_directory=self.persist_dir, embedding_function=embeddings
      )
    except Exception as e:
      print(f"Vector DB init warning: {e}")
      self.vector_db = None

  def sync_documents(self):
    if not self.vector_db:
      self._init_db()
    if not self.vector_db:
      return False, "فشل الاتصال بقاعدة المتجهات."

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)
    count = 0
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
        if filename.startswith("~") or filename.lower() == "requirements.txt":
          continue
        try:
          docs = (
              PyPDFLoader(filepath).load()
              if filename.lower().endswith(".pdf")
              else TextLoader(filepath, encoding="utf-8").load()
          )
          if docs:
            chunks = splitter.split_documents(docs)
            self.vector_db.add_documents(chunks)
            count += 1
        except Exception as e:
          print(f"Error {filename}: {e}")
    return True, f"تمت فهرسة {count} ملف بنجاح."

  def query(self, question: str) -> str:
    if not self.api_key:
      return "يرجى إضافة مفتاح GROQ_API_KEY في إعدادات السيرفر أو في Streamlit Secrets."

    context = ""
    try:
      if self.vector_db:
        docs = self.vector_db.similarity_search(question, k=4)
        if docs:
          context = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
      print(f"Search warning: {e}")

    if not context:
      context = "محتوى مقررات ووثائق النظام المعتمدة (تنظيم وتصميم الحاسوب، الذاكرة، والأمن السيبراني)."

    try:
      llm = ChatGroq(
          model="llama-3.1-8b-instant", temperature=0.2, api_key=self.api_key
      )
      prompt = f"""أنت مساعد علمي متخصص في شرح وثائق ومقررات المادة.
قواعد صارمة:
- ممنوع تكرار السؤال أو كتابة مقدمات مثل "بخصوص استفسارك".
- ابدأ بالحل والشرح المباشر فوراً.
- لا تعتذر ولا تقل لا أعلم.

سياق الملفات:
{context}

السؤال المطلوب حله: {question}
الإجابة العلمية المباشرة:"""

      res = llm.invoke(prompt)
      ans = res.content.strip()

      # تنظيف أي تكرار
      lines = ans.split("\n")
      if (
          lines
          and question.strip().rstrip("؟?.: ").lower()
          in lines[0].strip().rstrip("؟?.: ").lower()
      ):
        ans = "\n".join(lines[1:]).strip()

      return ans
    except Exception as e:
      return f"خطأ في توليد الإجابة: {str(e)}"
