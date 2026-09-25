import json
import time
from flask import Flask, jsonify, request
from rag_engine import EnterpriseRAG

app = Flask(__name__)

# تهيئة محرك الـ RAG الخاص بملفاتك
print("🔄 جاري تحميل ملفات النظام...")
rag = EnterpriseRAG()
rag.sync_documents()
print("✅ محرك الـ RAG جاهز لاستقبال مكالمات Vapi!")


@app.route("/chat/completions", methods=["POST"])
def vapi_chat_completion():
  data = request.get_json()

  # استخراج آخر ما قاله المتصل في المكالمة
  messages = data.get("messages", [])
  user_message = ""
  for m in reversed(messages):
    if m.get("role") == "user":
      user_message = m.get("content", "")
      break

  print(f"\n📞 المتصل قال: {user_message}")

  # البحث في صلب ملفاتك وتوليد الإجابة
  ai_answer = rag.query(user_message)
  print(f"💡 رد النظام من الملفات: {ai_answer}")

  # إرجاع الإجابة بصيغة متوافقة 100% مع معايير Vapi
  response_payload = {
      "id": f"chatcmpl-{int(time.time())}",
      "object": "chat.completion",
      "created": int(time.time()),
      "model": "local-rag",
      "choices": [{
          "index": 0,
          "message": {"role": "assistant", "content": ai_answer},
          "finish_reason": "stop",
      }],
  }

  return jsonify(response_payload)


@app.route("/", methods=["GET"])
def health_check():
  return "Vapi RAG Webhook is Live!"


if __name__ == "__main__":
  # تشغيل السيرفر على المنفذ 5001
  app.run(host="0.0.0.0", port=5001, debug=False)