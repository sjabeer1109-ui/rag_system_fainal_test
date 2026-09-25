from twilio.rest import Client

TWILIO_ACCOUNT_SID = "AC4a2573dc112a71dadbc73a4f8636a940"
TWILIO_AUTH_TOKEN = "bff021ea449211d2546992dfa37e1297"

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
call_sid = "CA1dbdb1d8daa0dd142755848ee80ded29"

# جلب الإشعارات والأخطاء المسجلة للمكالمة
notifications = client.calls(call_sid).notifications.list()

print("\n" + "=" * 45)
if notifications:
  for n in notifications:
    print(f"⚠️ رمز الخطأ: {n.error_code}")
    print(f"📝 رسالة الخطأ: {n.message_text}")
    print(f"🔗 للمزيد من التفاصيل: {n.more_info}")
else:
  # فحص سجل الأخطاء العام (Debugger)
  alerts = client.monitor.v1.alerts.list(limit=3)
  for a in alerts:
    print(f"⚠️ خطأ مسجل: {a.error_code} - {a.alert_text}")
print("=" * 45 + "\n")