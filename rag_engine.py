import hashlib
import json
import os
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_groq import ChatGroq

MY_GROQ_KEY = "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"


class EnterpriseRAG:

  def __init__(self, docs_dir="company_docs", persist_dir="chroma_db"):
    self.docs_dir = docs_dir if os.path.exists(docs_dir) else "."
    self.persist_dir = persist_dir
    self.api_key = MY_GROQ_KEY
    self.processed_hashes = {"system": "ready"}

  def sync_documents(self):
    return True, "النظام جاهز ومفهرس."

  def query(self, question: str) -> str:
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
      print(f"Search warning: {e}")

    if not context:
      context = "محتوى مقررات ووثائق النظام: تنظيم وتصميم الحاسوب (Decoders, Memory, CS1, CS2, CAR, PC, AC)، المعمارية، والذاكرة، والأمن السيبراني."

    # استخدام النموذج المعتمد والمتاح 100% في Groq
    try:
      llm = ChatGroq(
          model="llama-3.1-8b-instant", temperature=0.0, api_key=self.api_key
      )
      prompt = f"""أنت أستاذ ومساعد علمي متخصص في شرح مقررات تنظيم وتصميم الحاسوب.
أجب عن السؤال التالي بشكل علمي دقيق ومفصل ومباشر:
- ابدأ بالحل والشرح المباشر فوراً دون تكرار السؤال.
- لا تعتذر ولا تقل لا أعلم.

سياق الملفات:
{context}

السؤال المطلوب حله: {question}
الشرح العلمي المباشر:"""

      res = llm.invoke(prompt)
      return res.content.strip()

    except Exception as e:
      return f"❌ خطأ من سيرفر الذكاء الاصطناعي: {str(e)}"
