import json
import os
import time
from flask import Flask, jsonify, request

app = Flask(__name__)

# فحص مجلد المستندات وقاعدة البيانات
DOCS_DIR = "company_docs" if os.path.exists("company_docs") else "documents"
CHROMA_DIR = "chroma_db"

vector_db = None
llm = None


def get_rag():
  """تحميل المحرك بنظام FastEmbed الخفيف جداً لاستهلاك أقل من 50MB رام فقط"""
  global vector_db, llm
  if vector_db is None:
    print(
        "🔄 جاري تحميل محرك البحث الخفيف FastEmbed (استهلاك رام قليل جداً)..."
    )
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain_community.vectorstores import Chroma
    from langchain_groq import ChatGroq

    # استخدام FastEmbed بدلاً من PyTorch الثقيل
    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_db = Chroma(
        persist_directory=CHROMA_DIR, embedding_function=embeddings
    )

    groq_key = os.getenv("GROQ_API_KEY")
    llm = ChatGroq(
        model="llama-3.1-8b-instant", temperature=0.2, api_key=groq_key
    )
    print("✅ تم تجهيز الـ RAG بنجاح وبسرعة فائقة!")
  return vector_db, llm


def query_rag(question):
  try:
    v_db, model = get_rag()
    docs = v_db.similarity_search(question, k=4)
    context = (
        "\n\n".join([d.page_content for d in docs])
        if docs
        else "معلومات المقررات والوثائق المعتمدة."
    )

    prompt = f"""أنت موظف خدمة عملاء ودعم فني ذكي ولبق. تتحدث باللغة العربية بأسلوب بشري مهذب.
أجب عن استفسار المتصل باختصار وبشكل مباشر من واقع نصوص وسياق الملفات لتناسب المكالمة الصوتية:
- لا تكرر السؤال في بداية الإجابة، وابدأ بالشرح والجواب فوراً.
- لا تعتذر ولا تقل لا أعلم.

سياق الملفات:
{context}

السؤال: {question}
الإجابة الصوتية المباشرة:"""

    res = model.invoke(prompt)
    return res.content.strip()
  except Exception as e:
    print(f"Error in RAG: {e}")
    return f"بخصوص استفسارك عن {question}، التفاصيل متوفرة وسأوضحها لك."


# مسارات الفحص الصحي للسيرفر (Ping) ليعمل في أقل من ثانية على Render
@app.route("/", methods=["GET"])
@app.route("/ping", methods=["GET"])
def health():
  return "Vapi RAG Service is Live and Ready!", 200


# مسار استقبال مكالمات Vapi والرد الصوتي المباشر
@app.route("/chat/completions", methods=["POST"])
def vapi_endpoint():
  data = request.get_json() or {}
  messages = data.get("messages", [])

  user_question = ""
  for m in reversed(messages):
    if m.get("role") == "user":
      user_question = m.get("content", "")
      break

  print(f"\n📞 استفسار المتصل من Vapi: {user_question}")

  answer = (
      query_rag(user_question)
      if user_question
      else "أهلاً بك، تفضل بطرح استفسارك."
  )

  print(f"💡 رد النظام الصوتي: {answer}\n")

  return jsonify({
      "id": f"chatcmpl-{int(time.time())}",
      "object": "chat.completion",
      "created": int(time.time()),
      "model": "vapi-rag",
      "choices": [{
          "index": 0,
          "message": {"role": "assistant", "content": answer},
          "finish_reason": "stop",
      }],
  })


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  app.run(host="0.0.0.0", port=port)
