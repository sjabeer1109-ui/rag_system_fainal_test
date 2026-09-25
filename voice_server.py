from flask import Flask, request
from twilio.twiml.voice_response import VoiceResponse, Gather
from twilio.rest import Client
from rag_engine import EnterpriseRAG
from call_center import CallCenterEngine

app = Flask(__name__)

# تهيئة محركات الذكاء الاصطناعي والمكالمات
rag = EnterpriseRAG()
rag.sync_documents()
call_center = CallCenterEngine(rag)

# --- بيانات حسابك في TWILIO (ضع بياناتك هنا) ---
TWILIO_ACCOUNT_SID = "AC4a2573dc112a71dadbc73a4f8636a940"
TWILIO_AUTH_TOKEN = "bff021ea449211d2546992dfa37e1297"
TWILIO_PHONE_NUMBER = "+16204136727"  # مثل: +1234567890

# --- 1. بداية المكالمة: الترحيب بالعميل والاستماع له بالعربية ---
@app.route("/voice", methods=["POST", "GET"])
def voice():
    response = VoiceResponse()
    
    # رسالة ترحيبية بصوت عربي طبيعي
    gather = Gather(
        input="speech",
        action="/handle-speech",
        method="POST",
        language="ar-JO",  # الاستماع باللغة العربية للأردن
        timeout=4,
        speechTimeout="auto"
    )
    gather.say(
        "أهلاً بك في شركة الخدمات الذكية. تفضل، كيف أستطيع مساعدتك اليوم؟",
        language="ar-XA",
        voice="Polly.Zeina"
    )
    response.append(gather)
    
    # إذا لم يتكلم المتصل
    response.redirect("/voice")
    return str(response)

# --- 2. معالجة كلام المتصل والرد من مستندات الشركة أو التحويل ---
@app.route("/handle-speech", methods=["POST", "GET"])
def handle_speech():
    response = VoiceResponse()
    
    # استخراج ما قاله المتصل بدقة
    user_speech = request.form.get("SpeechResult", "")
    caller_phone = request.form.get("From", "غير معروف")
    
    if not user_speech:
        response.say("عذراً، لم أسمعك جيداً. يرجى إعادة سؤالك.", language="ar-XA", voice="Polly.Zeina")
        response.redirect("/voice")
        return str(response)

    print(f"📞 العميل قال: {user_speech}")

    # تمرير المكالمة لمحرك call_center الذكي
    result = call_center.handle_call_interaction(
        customer_name="متصل هاتفياً",
        customer_phone=caller_phone,
        inquiry_text=user_speech
    )

    # إذا طلب العميل التحويل لموظف
    if result["action"] == "transferred":
        response.say(
            f"تم تحويلك إلى موظف الدعم الفني {result['assigned_agent']}. جميع بياناتك أصبحت على شاشته وسيتحدث معك الآن.",
            language="ar-XA",
            voice="Polly.Zeina"
        )
        # هنا يمكن ربط المكالمة برقم هاتف الموظف مباشرة عبر response.dial(...)
    else:
        # إذا كانت إجابة من الـ RAG
        ai_reply = result["message"]
        gather = Gather(
            input="speech",
            action="/handle-speech",
            method="POST",
            language="ar-JO",
            timeout=4
        )
        gather.say(ai_reply, language="ar-XA", voice="Polly.Zeina")
        gather.say("هل لديك أي استفسار آخر؟", language="ar-XA", voice="Polly.Zeina")
        response.append(gather)

    return str(response)

# --- دالة للاتصال برقمك الأردني ليرن هاتفك فوراً ---
def call_my_phone(target_phone_number):
    client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
    # استبدل هذا الرابط برابط ngrok الخاص بك متبوعاً بـ /voice
    ngrok_url = "https://gating-grab-front.ngrok-free.dev"
    
    call = client.calls.create(
        to=target_phone_number,        # رقمك الأردني مثل: +962791234567
        from_=TWILIO_PHONE_NUMBER,     # رقم Twilio
        url=ngrok_url
    )
    print(f"🚀 جاري الاتصال بهاتفك الآن! معرف المكالمة: {call.sid}")

if __name__ == "__main__":
    # تشغيل خادم المكالمات على منفذ 5000
    app.run(port=5000)