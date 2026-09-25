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

# برومبت 
SYSTEM_PROMPT = """أنت مساعد علمي وتقني وخبير في مراجعة وتوضيح محتوى الوثائق والمقررات المرفقة.
مهمتك تقديم شرح دقيق، واضح ومباشر بالاعتماد الكامل على نصوص وسياق الملفات المتاحة أدناه.

قواعد صارمة للإجابة:
1. ممنوع منعاً باتاً تكرار أو إعادة كتابة السؤال في بداية الإجابة، وابدأ بالشرح والجواب مباشرة دون أي مقدمات مثل "سؤالك هو" أو "بخصوص سؤالك".
2. ممنوع منعاً باتاً أن تعتذر أو تقول "أعتذر عن الإجابة" أو "لا أستطيع إفادتك".
3. اشرح المفهوم بدقة وبالصيغة العلمية الصحيحة مستنداً إلى الملفات (مثل: المعمارية، الذاكرة، السجلات، العناوين، والأمن السيبراني).
4. اذكر المعادلات، الخطوات، أو الرموز الأساسية إن وجدت (مثل: CAR, AC, PC, Direct/Indirect, ROM, RAM, Opcode, Mapping).

سياق نصوص الملفات:
{context}

سجل المحادثة السابق:
{chat_history}

سؤال المستخدم: {question}
الشرح المباشر (ابدأ بالحل فوراً بدون ذكر السؤال):"""


class EnterpriseRAG:

  def __init__(
        self,
        docs_dir="documents",
        persist_dir="chroma_db",
        model_name="llama3",
    ):
      self.docs_dir = docs_dir
      self.persist_dir = persist_dir
      self.model_name = model_name
      self.hashes_file = os.path.join(self.persist_dir, "processed_hashes.json")

      os.makedirs(self.docs_dir, exist_ok=True)
      os.makedirs(self.persist_dir, exist_ok=True)

      self.embeddings = OllamaEmbeddings(model="nomic-embed-text")
      self.llm = ChatOllama(model=self.model_name, temperature=0.2)

      # فحص وإنشاء قاعدة المتجهات 
      try:
        self.vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )
      except Exception as e:
        import shutil

        print("🔄 جاري تجهيز قاعدة البينات ...")
        shutil.rmtree(self.persist_dir, ignore_errors=True)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.vector_db = Chroma(
            persist_directory=self.persist_dir,
            embedding_function=self.embeddings,
        )

      self.memory = ConversationBufferMemory(
          memory_key="chat_history", return_messages=True
      )
      self.processed_hashes = self._load_hashes()

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
      """قم برفع الملفات """
      changed = False
      indexed_count = 0
      errors = []
      splitter = RecursiveCharacterTextSplitter(
          chunk_size=900, chunk_overlap=150
      )

      search_dirs = [self.docs_dir, "."]

      for folder in search_dirs:
        if not os.path.exists(folder):
          continue
        for filename in os.listdir(folder):
          filepath = os.path.join(folder, filename)
          if not os.path.isfile(filepath):
            continue

          if not (
              filename.lower().endswith(".pdf")
              or filename.lower().endswith(".txt")
          ):
            continue

          # تجنب الملفات المؤقتة
          if filename.startswith("~"):
            continue

          file_hash = self._get_file_hash(filepath)
          if (
              filename in self.processed_hashes
              and self.processed_hashes[filename] == file_hash
          ):
            indexed_count += 1
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
              indexed_count += 1
              print(f"✅ تمت قراءة الملف: {filename}")
          except Exception as e:
            err_msg = f"خطأ في قراءة {filename}: {str(e)}"
            print(f"⚠️ {err_msg}")
            errors.append(err_msg)

      if changed:
        self._save_hashes()

      status_info = f"تمت قرائة {indexed_count} الملف بنجاح."
      if errors:
        status_info += f" (تنبيهات: {', '.join(errors)})"

      return (True if indexed_count > 0 else False), status_info

  def clean_echo(self, answer: str, question: str) -> str:
    """..................."""
    text = answer.strip()

    lines = text.split("\n")
    if lines:
      first_line = lines[0].strip().rstrip("؟?.: ")
      q_clean = question.strip().rstrip("؟?.: ")
      if first_line.lower() == q_clean.lower() or q_clean.lower() in first_line.lower():
        text = "\n".join(lines[1:]).strip()

    prefixes = [
        "سؤال المستخدم:",
        "سؤالك:",
        "السؤال:",
        "الإجابة المباشرة:",
        "الإجابة على الاستفسار:",
        "الإجابة:",
        "الجواب:",
        "الشرح المباشر:",
    ]
    for p in prefixes:
      if text.startswith(p):
        text = text[len(p) :].strip()

    return text

  def query(self, question: str) -> str:
      """استرجاع مباشر وشرح مفصل من الملفات"""
      try:
        # 1. البحث الدلالي المباشر 
        docs = self.vector_db.similarity_search(question, k=5)

        if not docs:
          # في حال كانت قاعدة البيانات لم تفهرس الملفات بعد، نقرأ من الملفات مباشرة كخيار بديل
          context = "ملخص محتويات المقررات: تنظيم وتصميم الحاسوب، سجلات المعالجة CAR و AC، الذاكرة، والأمن السيبراني."
        else:
          context = "\n\n".join([d.page_content for d in docs])

        # 2. بناء التوجيه المباشر
        prompt = f"""أنت مساعد علمي متخصص في شرح مقررات ووثائق النظام.
اشرح وأجب عن السؤال التالي بشكل علمي دقيق، واضح ومفصل بالاعتماد على سياق النصوص المرفقة.
قواعد صارمة:
- لا تكرر السؤال في بداية الإجابة، وابدأ بالشرح فوراً.
- لا تعتذر ولا تقل "لا أعلم".
- اشرح المفاهيم والمعادلات والمصطلحات الإنجليزية المرتبطة بها بدقة.

سياق نصوص الملفات:
{context}

السؤال المطلوب شرحه: {question}
الإجابة العلمية المباشرة:"""

        # 3. استدعاء النموذج
        response = self.llm.invoke(prompt)
        answer = response.content.strip()

        # 4. تنظيف أي تكرار للسؤال
        return self.clean_echo(answer, question)

      except Exception as e:
        # طباعة الخطأ الحقيقي في التيرمينال لمعرفة سببه بدقة
        import traceback

        print("\n❌ حدث خطأ أثناء توليد الإجابة:")
        traceback.print_exc()
        return f"حدث خطأ تقني في الاتصال بنموذج الذكاء الاصطناعي: {str(e)}"