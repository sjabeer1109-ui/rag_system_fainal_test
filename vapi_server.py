import json
import os
import time
from flask import Flask, Response, jsonify, request

app = Flask(__name__)

# حد أقصى لحجم الطلب لمنع هجمات حجب الخدمة (1 ميجابايت)
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

MY_GROQ_KEY = os.getenv("GROQ_API_KEY")
VAPI_SECRET = os.getenv("VAPI_SECRET")
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
    # التحقق الأمني من الرمز السري الخاص بـ Vapi لمنع أي استهلاك غير مصرح به
    if VAPI_SECRET:
        auth_header = request.headers.get("Authorization", "")
        custom_header = request.headers.get("X-Vapi-Secret", "")
        expected_bearer = f"Bearer {VAPI_SECRET}"
        if auth_header != expected_bearer and custom_header != VAPI_SECRET and auth_header != VAPI_SECRET:
            return jsonify({"error": "Unauthorized - Invalid Vapi Secret"}), 401

    if not MY_GROQ_KEY:
        return jsonify({"error": "GROQ_API_KEY is not configured on server"}), 500

    data = request.get_json(silent=True) or {}
    messages = data.get("messages", [])

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
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "openai/gpt-oss-120b",
            "openai/gpt-oss-20b",
            "qwen/qwen3.8-27b",
        ]

        for m_name in supported_models:
            try:
                llm = ChatGroq(model=m_name, temperature=0.0, api_key=MY_GROQ_KEY)
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

    return Response(
        generate_sse(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
