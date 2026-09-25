import io
import json
import os
import time

try:
  from gtts import gTTS
except ImportError:
  gTTS = None

CALLS_FILE = "calls.json"


class CallCenterEngine:

  def __init__(self, rag_engine):
    self.rag = rag_engine

  def handle_call_interaction(self, name, phone, inquiry):
    """معالجة استفسار المتصل وتوجيهه للـ RAG وفحص طلب التحويل"""
    answer = self.rag.query(inquiry)

    transfer_keywords = [
        "تحويل",
        "موظف",
        "إنسان",
        "بشري",
        "مشكلة معقدة",
        "دعم فني",
        "أحمد",
    ]
    is_transfer = any(kw in inquiry for kw in transfer_keywords)

    action = "transferred" if is_transfer else "answered"
    assigned_agent = "أحمد علي (دعم فني)" if is_transfer else None

    return {
        "action": action,
        "message": answer,
        "assigned_agent": assigned_agent,
    }

  def text_to_speech(self, text: str) -> bytes:
    """تحويل النص إلى صوت عربي واضح للمتصل"""
    if gTTS is None:
      return b""
    try:
      clean_text = (
          text.replace("*", "").replace("#", "").replace("_", " ").strip()
      )
      if not clean_text:
        clean_text = "أهلاً بك، تفضل بطرح استفسارك."

      tts = gTTS(text=clean_text[:450], lang="ar")
      fp = io.BytesIO()
      tts.write_to_fp(fp)
      fp.seek(0)
      return fp.read()
    except Exception as e:
      print(f"TTS Error: {e}")
      return b""