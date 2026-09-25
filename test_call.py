from twilio.rest import Client

# 1. بيانات Twilio الخاصة بك
TWILIO_ACCOUNT_SID = "AC4a2573dc112a71dadbc73a4f8636a940"
TWILIO_AUTH_TOKEN = "bff021ea449211d2546992dfa37e1297"
TWILIO_PHONE_NUMBER = "+16204136727"  # مثل: +1205xxxxxxx

# 2. رابط ngrok الذي نسخته من الخطوة 2 (تأكد من إضافة /voice في آخره)
NGROK_URL = "https://gating-grab-front.ngrok-free.dev"

# 3. رقم هاتفك الأردني بالصيغة الدولية (+962)
MY_JORDAN_PHONE = "+962791887540"

# تنفيذ الاتصال
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
print(f"🚀 جاري الاتصال بهاتفك الأردني ({MY_JORDAN_PHONE})...")

call = client.calls.create(
    to=MY_JORDAN_PHONE, from_=TWILIO_PHONE_NUMBER, url=NGROK_URL
)

print(f"✅ تم إرسال المكالمة بنجاح! راقب هاتفك الآن. معرف المكالمة: {call.sid}")