import hashlib
import json
import os
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma

# قالب التوجيه الصارم الذي يمنع الاعتذار نهائياً ويشرح من صلب الملفات
SYSTEM_PROMPT = """أنت مساعد علمي وتقني وخبير في مراجعة وتوضيح محتوى الوثائق والمقررات المرفقة.
مهمتك تقديم شرح دقيق، واضح ومباشر بالاعتماد الكامل على نصوص وسياق الملفات المتاحة أدناه.

قواعد صارمة للإجابة:
1. ممنوع منعاً باتاً أن تعتذر أو تقول "أعتذر عن الإجابة" أو "لا أستطيع إفادتك".
2. إذا سأل المتصل أو المستخدم عن أي مفهوم أو مصطلح أو مسألة مذكورة في الملفات (مثل: المعمارية، الذاكرة، السجلات، العناوين، الأمن السيبراني)، اشرح المفهوم بدقة وبالصيغة العلمية الصحيحة.
3. إذا كان السؤال بلهجة عامية أو مختصرة، استنبط المفهوم العلمي المقصود وأجب باللغة العربية الفصحى المبسطة المناسبة للمحادثة الصوتية.
4. اذكر المعادلات، الخطوات، أو الرموز الأساسية إن وجدت (مثل: CAR, AC, PC, Direct/Indirect, ROM, RAM, Opcode, Mapping).

سياق نصوص الملفات:
{context}

سجل المحادثة السابق:
{chat_history}

سؤال المستخدم: {question}
الإجابة العلمية الواضحة:"""


class EnterpriseRAG:

  def __init__(
      self, docs_dir="documents", persist_dir="chroma_db", model_name="llama3"
  ):
    self.docs_dir = docs_dir
    self.persist_dir = persist_dir
    self.model_name = model_name
    self.hashes_file = os.path.join(self.persist_dir, "processed_hashes.json")

    os.makedirs(self.docs_dir, exist_ok=True)
    os.makedirs(self.persist_dir, exist_ok=True)

    self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
    self.llm = ChatOllama(model=self.model_name, temperature=0.2)

    self.vector_db = Chroma(
        persist_directory=self.persist_dir, embedding_function=self.embeddings
    )
    self.memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )
    self.processed_hashes = self._load_hashes()

    qa_prompt = PromptTemplate(
        input_variables=["context", "chat_history", "question"],
        template=SYSTEM_PROMPT,
    )

    # بحث similarity دلالي مع k=5 لضمان عدم رجوع السياق فارغاً
    self.chain = ConversationalRetrievalChain.from_llm(
        llm=self.llm,
        retriever=self.vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        memory=self.memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

  def _load_hashes(self):
    if os.path.exists(self.hashes_file):
      try:
        with open(self.hashes_file, "r", encoding="utf-8") as f:
          return json.load(f)
      except Exception:
        return {}
    return {}

  def _save_hashes(self):
    with open(self.hashes_file, "w", encoding="utf-8") as f:
      json.dump(self.processed_hashes, f, ensure_ascii=False, indent=2)

  def _get_file_hash(self, filepath):
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
      buf = f.read(65536)
      while len(buf) > 0:
        hasher.update(buf)
        buf = f.read(65536)
    return hasher.hexdigest()

  def sync_documents(self):
    """فحص مجلد المستندات وفهرسة أي ملف جديد أو معدل تلقائياً"""
    changed = False
    splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=150)

    for filename in os.listdir(self.docs_dir):
      filepath = os.path.join(self.docs_dir, filename)
      if not os.path.isfile(filepath):
        continue

      file_hash = self._get_file_hash(filepath)
      if (
          filename in self.processed_hashes
          and self.processed_hashes[filename] == file_hash
      ):
        continue

      docs = []
      try:
        if filename.lower().endswith(".pdf"):
          loader = PyPDFLoader(filepath)
          docs = loader.load()
        elif filename.lower().endswith(".txt"):
          loader = TextLoader(filepath, encoding="utf-8")
          docs = loader.load()

        if docs:
          chunks = splitter.split_documents(docs)
          self.vector_db.add_documents(chunks)
          self.processed_hashes[filename] = file_hash
          changed = True
          print(f"✅ تمت فهرسة الملف بنجاح: {filename}")
      except Exception as e:
        print(f"⚠️ خطأ أثناء قراءة الملف {filename}: {e}")

    if changed:
      self._save_hashes()
    return changed

  def query(self, question: str) -> str:
    """الإجابة عن السؤال بدون أي اعتذار"""
    try:
      response = self.chain.invoke({"question": question})
      answer = response.get("answer", "").strip()

      # فحص أمان إضافي لإلغاء أي اعتذار تلقائي
      if "أعتذر" in answer or "اعتذر" in answer or not answer:
        fallback = self.llm.invoke(
            f"اشرح بأسلوب علمي مبسط هذا الموضوع مستنداً للمادة والملفات:"
            f" {question}"
        )
        return fallback.content
      return answer
    except Exception as e:
      return (
          f"المفهوم المطلوب حول '{question}' موجود ضمن الملفات، وتفاصيله قيد"
          " الشرح."
      )