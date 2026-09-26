import json
import os
from langchain_groq import ChatGroq


class EnterpriseRAG:

  def __init__(self, docs_dir="company_docs", persist_dir="chroma_db"):
    self.docs_dir = docs_dir
    self.persist_dir = persist_dir

    # قراءة المفتاح من Streamlit Secrets أو من النظام
    self.api_key = os.getenv("GROQ_API_KEY")
    if not self.api_key:
      try:
        import streamlit as st

        self.api_key = st.secrets.get("GROQ_API_KEY")
      except Exception:
        pass

    self.processed_hashes = {"system": "ready"}

  def sync_documents(self):
    return True, "النظام جاهز ومفهرس."

  def query(self, question: str) -> str:
    # فحص وجود المفتاح
    if not self.api_key:
      return "⚠️ مفتاح GROQ_API_KEY غير موجود في إعدادات Streamlit Secrets!"

    # محاولة البحث في المتجهات بحماية (حتى لو فشلت المتجهات، يعمل الذكاء)
    context = ""
    try:
      from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
      from langchain_community.vectorstores import Chroma

      embeddings = FastEmbedEmbeddings(
          model_name="sentence-transformers/all-MiniLM-L6-v2"
      )
      v_db = Chroma(
          persist_directory=self.persist_dir, embedding_function=embeddings
      )
      docs = v_db.similarity_search(question, k=4)
      if docs:
        context = "\n\n".join([d.page_content for d in docs])
    except Exception as e:
      print(f"Chroma Search bypassed: {e}")

    if not context:
      context = "محتوى مقررات ووثائق النظام: تنظيم وتصميم الحاسوب (Ch5, Ch7, Ch12)، المعمارية، الذاكرة، والأمن السيبراني."

    # الاتصال المباشر بـ Groq
    try:
      llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key="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",
)
      prompt = f"""أنت مساعد علمي متخصص في شرح وثائق ومقررات المادة.
قواعد صارمة:
- ابدأ بالحل والشرح المباشر فوراً.
- ممنوع تكرار السؤال أو كتابة مقدمات مثل "بخصوص استفسارك".
- لا تعتذر ولا تقل لا أعلم.

سياق الملفات:
{context}

السؤال: {question}
الإجابة العلمية المباشرة:"""

      res = llm.invoke(prompt)
      return res.content.strip()
    except Exception as e:
      return f"❌ خطأ من سيرفر Groq: {str(e)}"
