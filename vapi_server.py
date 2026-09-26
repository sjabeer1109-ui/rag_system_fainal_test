import json
import os
import time
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

MY_GROQ_KEY = "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"
CHROMA_DIR = "chroma_db"


def get_context(question):
  """جلب السياق من ملفات النظام"""
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
    context = "محتوى مقررات ووثائق تنظيم وتصميم الحاسوب: Direct vs Indirect Addressing, Memory, CAR, PC, Decoders, والأمن السيبراني."
  return context


@app.route("/", methods=["GET"])
@app.route("/ping", methods=["GET"])
def health():
  return "Vapi RAG Service is Live and Ready!", 200


@app.route("/chat/completions", methods=["POST"])
def vapi_endpoint():
  data = request.get_json() or {}
  messages = data.get("messages", [])
  is_streaming = data.get("stream", True)  # Vapi يطلب stream افتراضياً

  user_question = ""
  for m in reversed(messages):
    if m.get("role") == "user":
      user_question = m.get("content", "")
      break

  print(f"\n📞 استفسار المتصل عبر Vapi: {user_question}")
  if not user_question:
    user_question = "أهلاً بك"

  context = get_context(user_question)

  prompt = f"""أنت موظفة خدمة عملاء ودعم فني ذكية ولبقة تجيبين في مكالمة صوتية باللغة العربية الفصحى المبسطة:
قواعد صارمة:
- ابدئي بالشرح المباشر فوراً دون ذكر السؤال ودون مقدمات مثل "بخصوص استفسارك".
- لا تعتذري واشرحي المفهوم العلمي بدقة ومباشرة.
- اجعلي الإجابة مركزة وسلسة لتناسب المحادثة الصوتية.

سياق الملفات:
{context}

سؤال المتصل: {user_question}
الإجابة الصوتية المباشرة (ابدئي بالحل فوراً):"""

  # مولد الـ Streaming المتوافق 100% مع Vapi
  def generate_sse():
    chunk_id = f"chatcmpl-{int(time.time())}"
    from langchain_groq import ChatGroq

    supported_models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "qwen/qwen3.8-27b",
    ]

    llm = None
    for m_name in supported_models:
      try:
        llm = ChatGroq(model=m_name, temperature=0.0, api_key=MY_GROQ_KEY)
        # تجربة الاتصال بالتدفق
        stream_iter = llm.stream(prompt)
        for chunk in stream_iter:
          token = chunk.content
          if token:
            payload = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": m_name,
                "choices": [{
                    "index": 0,
                    "delta": {"content": token},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(payload)}\n\n"
        break
      except Exception as err:
        print(f"Error with model {m_name}: {err}")
        continue

    # إرسال إشارة اكتمال الإجابة لـ Vapi
    stop_payload = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "vapi-rag",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    yield f"data: {json.dumps(stop_payload)}\n\n"
    yield "data: [DONE]\n\n"

  # إرجاع الرد كتدفق حقيقي (Event Stream) لـ Vapi
  return Response(
      generate_sse(),
      mimetype="text/event-stream",
      headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
  )


if __name__ == "__main__":
  port = int(os.environ.get("PORT", 10000))
  app.run(host="0.0.0.0", port=port)
