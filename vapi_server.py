import json
import os
import time
from flask import Flask, jsonify, request
from langchain_groq import ChatGroq

app = Flask(__name__)

# مفتاحك الحقيقي مدمج مباشرة
MY_GROQ_KEY = "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"
CHROMA_DIR = "chroma_db"


def query_rag(question):
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
    print(f"Chroma Bypassed: {e}")

  if not context:
    context = (
        "محتوى مقررات تنظيم وتصميم الحاسوب: Decoders, Memory Chips, CS1, CS2,"
        " CAR, PC, AC."
    )

  try:
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", temperature=0.0, api_key=MY_GROQ_KEY
    )
    prompt = f"""أنت موظف دعم فني ومساعد علمي ذكي ولبق تجيب في مكالمة صوتية باللغة العربية الفصحى:
- ابدأ بالحل والشرح المباشر فوراً دون ذكر السؤال.
- ممنوع منعاً باتاً تكرار السؤال أو كتابة مقدمات مثل "بخصوص استفسارك".
- لا تعتذر ولا تقل لا أعلم.
- اجعل الإجابة مركزة وواضحة لتناسب المكالمة الصوتية.

سياق الملفات:
{context}

سؤال المتصل: {question}
الإجابة الصوتية المباشرة (ابدأ بالحل فوراً):"""

    res = llm.invoke(prompt)
    answer = res.content.strip()

    # تنظيف أي تكرار
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
    return f"خطأ في الاتصال: {str(e)}"


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

  print(f"\n📞 مكالمة Vapi: {user_question}")
  answer = (
      query_rag(user_question)
      if user_question
      else "أهلاً بك، تفضل بطرح استفسارك."
  )
  print(f"💡 رد الموديل 70B: {answer}\n")

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
