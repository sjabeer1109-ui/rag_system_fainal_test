from pyngrok import ngrok

# 1. ضع رمز التوثيق الذي نسخته هنا بين علامتي التنصيص
ngrok.set_auth_token("3JKk94F7SNc90kcu298sVoYTHoj_aa9WzvfjSwZz5mZ3EBrX")

# 2. تشغيل النفق
tunnel = ngrok.connect(5000)
print("\n" + "=" * 50)
print(f"🚀 الرابط العام لموقعك هو:\n{tunnel.public_url}")
print("=" * 50 + "\n")

ngrok_process = ngrok.get_ngrok_process()
try:
  ngrok_process.proc.wait()
except KeyboardInterrupt:
  print("تم إغلاق النفق.")