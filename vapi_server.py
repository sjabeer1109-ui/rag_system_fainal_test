import json
import os
import time
from flask import Flask, jsonify, request

app = Flask(__name__)

DOCS_DIR = "company_docs" if os.path.exists("company_docs") else "documents"
CHROMA_DIR = "chroma_db"


def query_rag(question):
  # 1. التحقق من وجود مفتاح Groq في السيرفر
  groq_key = os.getenv("GROQ_API_KEY")
  if not groq_key:
    return "خطأ: مفتاح GROQ_API_KEY غير موجود في إعدادات Render!"

  # 2. محاولة البحث في المستندات بحماية تامة
  context = ""
  try:
    from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
    from langchain_community.vectorstores import Chroma

    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    v_db = Chroma(
        persist_directory=CHROMA_DIR, embedding_function=embeddings
    )
    docs = v_db.similarity_search(question, k=4)
    if docs:
      context = "\n\n".join([d.page_content for d in docs])
  except Exception as e:
    print(f"Chroma Search Bypassed: {e}")

  # سياق احتياطي في حال كانت قاعدة البيانات فارغة لضمان عدم توقف الذكاء
  if not context:
    context = (
        "محتوى مقررات ووثائق النظام: تنظيم وتصميم الحاسوب (Ch5, Ch7, Ch12)،"
        " المعمارية، الذاكرة، والأمن السيبراني."
    )

  # 3. استدعاء Groq لتوليد الرد الصوتي المباشر
  try:
    from langchain_groq import ChatGroq

    llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.0,
    api_key="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",
)

    prompt = f"""أنت موظف خدمة عملاء ودعم فني ذكي ولبق. تتحدث باللغة العربية بأسلوب بشري واضح ومباشر:
قواعد صارمة:
- ابدأ بالحل والشرح المباشر فوراً دون ذكر السؤال.
- ممنوع منعاً باتاً تكرار السؤال أو كتابة مقدمات مثل "بخصوص استفسارك".
- لا تعتذر ولا تقل لا أعلم.
- اجعل الإجابة مختصرة وواضحة لتناسب المكالمة الصوتية.

سياق الملفات:
{context}

سؤال المتصل: {question}
الإجابة الصوتية المباشرة (ابدأ بالحل فوراً):"""

    res = llm.invoke(prompt)
    answer = res.content.strip()

    # تنظيف أي تكرار محتمل للسؤال
    lines = answer.split("\n")
    if (
        lines
        and question.strip().rstrip("؟?.: ").lower()
        in lines[0].strip().rstrip("؟?.: ").lower()
    ):
      answer = "\n".join(lines[1:]).strip()

    prefixes = [
        "سؤالك هو:",
        "السؤال:",
        "الإجابة المباشرة:",
        "الجواب:",
        "بخصوص استفسارك:",
    ]
    for p in prefixes:
      if answer.startswith(p):
        answer = answer[len(p) :].strip()

    return answer

  except Exception as e:
    return f"خطأ في الاتصال بـ Groq: {str(e)}"


# مسار الفحص الصحي ليعمل السيرفر في ثانية واحدة على Render
@app.route("/", methods=["GET"])
@app.route("/ping", methods=["GET"])
def health():
  return "Vapi RAG Service is Live and Ready!", 200


# مسار استقبال وتوجيه مكالمات Vapi
@app.route("/chat/completions", methods=["POST"])
def vapi_endpoint():
  data = request.get_json() or {}
  messages = data.get("messages", [])

  user_question = ""
  for m in reversed(messages):
    if m.get("role") == "user":
      user_question = m.get("content", "")
      break

  print(f"\n📞 استفسار المتصل عبر Vapi: {user_question}")
  answer = (
      query_rag(user_question)
      if user_question
      else "أهلاً بك، تفضل بطرح استفسارك."
  )
  print(f"💡 رد النظام المباشر: {answer}\n")

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
