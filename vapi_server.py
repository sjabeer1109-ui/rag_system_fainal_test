import json
import os
import time
from flask import Flask, jsonify, request

app = Flask(__name__)

DOCS_DIR = "company_docs" if os.path.exists("company_docs") else "documents"
CHROMA_DIR = "chroma_db"

vector_db = None


def get_vector_db():
  global vector_db
  if vector_db is None:
    try:
      from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
      from langchain_community.vectorstores import Chroma

      embeddings = FastEmbedEmbeddings(
          model_name="sentence-transformers/all-MiniLM-L6-v2"
      )
      vector_db = Chroma(
          persist_directory=CHROMA_DIR, embedding_function=embeddings
      )
    except Exception as e:
      print(f"⚠️ ملاحظة في قاعدة المتجهات: {e}")
      vector_db = False
  return vector_db


def query_rag(question):
  # 1. التحقق من مفتاح Groq
  groq_key = os.getenv("GROQ_API_KEY")
  if not groq_key:
    return (
        "تنبيه: مفتاح GROQ_API_KEY غير موجود في إعدادات السيرفر. يرجى إضافته في"
        " Environment Variables."
    )

  # 2. استخراج السياق من الملفات بحماية تامة
  context = ""
  try:
    v_db = get_vector_db()
    if v_db:
      docs = v_db.similarity_search(question, k=4)
      if docs:
        context = "\n\n".join([d.page_content for d in docs])
  except Exception as e:
    print(f"تنبيه أثناء البحث: {e}")

  if not context:
    context = (
        "محتوى مقررات ووثائق النظام المعتمدة (تنظيم وتصميم الحاسوب، سجلات CAR"
        " و AC، الذاكرة، والأمن السيبراني)."
    )

  # 3. صياغة الإجابة المباشرة بدون أي تكرار للسؤال وبدون اعتذار
  try:
    from langchain_groq import ChatGroq

    llm = ChatGroq(
        model="llama-3.1-8b-instant", temperature=0.2, api_key=groq_key
    )

    prompt = f"""أنت مساعد علمي وصوتي ذكي ولبق. تجيب باللغة العربية الفصحى المبسطة بأسلوب بشري واضح ومباشر:
قواعد صارمة:
- ممنوع منعاً باتاً تكرار أو إعادة كتابة السؤال في بداية الإجابة، وادخل في الشرح والحل فوراً.
- ممنوع قول "بخصوص استفسارك" أو الاعتذار. اشرح المفهوم العلمي بدقة ومباشرة.
- اجعل الإجابة مختصرة وواضحة لتناسب المحادثة الصوتية.

سياق نصوص الملفات:
{context}

سؤال المتصل: {question}
الإجابة الصوتية المباشرة (ابدأ بالحل فوراً):"""

    res = llm.invoke(prompt)
    answer = res.content.strip()

    # تنظيف أي تكرار للسؤال إن وجد
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
    import traceback

    traceback.print_exc()
    return f"حدث خطأ في نموذج الذكاء الاصطناعي: {str(e)}"


@app.route("/", methods=["GET"])
@app.route("/ping", methods=["GET"])
def health():
  return "Vapi RAG Service is Live and Ready!", 200


@app.route("/chat/completions", methods=["POST"])
def vapi_endpoint():
  data = request.get_json() or {}
  messages = data.get("messages", [])

  user_question = ""
  for m in reversed(messages):
    if m.get("role") == "user":
      user_question = m.get("content", "")
      break

  print(f"\n📞 استفسار المتصل: {user_question}")
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
