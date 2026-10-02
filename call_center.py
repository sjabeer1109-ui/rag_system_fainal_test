import io
import json
import os
import re
import time

try:
    from gtts import gTTS
except ImportError:
    gTTS = None

CALLS_FILE = "calls.json"
USERS_FILE = "users.json"


class CallCenterEngine:

    def __init__(self, rag_engine):
        self.rag = rag_engine

    def classify_intent_and_department(self, inquiry: str):
        """تحليل نية المتصل بذكاء وتحديد القسم المناسب للتحويل"""
        t = inquiry.strip().lower()

        # 1. فحص الكلمات الدلالية للأقسام
        sales_keywords = ["سعر", "أسعار", "تكلفة", "شراء", "اشتراك", "باقة", "عروض", "مبيعات", "خصم"]
        tech_keywords = ["عطل", "مشكلة", "خراب", "سيستم", "نظام", "خطأ", "كود", "دعم فني", "تطبيق", "سيرفر", "مش شغال", "مشكلة معقدة"]
        support_keywords = ["شكوى", "استفسار", "خدمة عملاء", "موظف", "بشري", "إنسان", "تحويل", "احكي مع حد", "كلمني", "مدير"]

        department = "خدمة عملاء"
        if any(kw in t for kw in tech_keywords):
            department = "دعم فني"
        elif any(kw in t for kw in sales_keywords):
            department = "مبيعات"
        elif any(kw in t for kw in support_keywords):
            department = "خدمة عملاء"

        # 2. تحديد ما إذا كان المتصل يطلب التحدث مع شخص بشري
        handoff_patterns = [
            r"تحويل",
            r"موظف",
            r"إنسان",
            r"بشري",
            r"أحكي\s+(مع\s+)?(حدا|شخص|واحد|حد)",
            r"كلم\s+موظف",
            r"خدمة\s+العملاء",
            r"دعم\s+فني",
            r"مشكلة\s+معقدة",
        ]
        is_handoff_requested = any(re.search(pat, t) for pat in handoff_patterns)

        return is_handoff_requested, department

    def find_best_agent(self, department: str, inquiry: str):
        """البحث الديناميكي في قائمة الموظفين لاختيار الموظف الأنسب للطلب"""
        try:
            # قراءة المستخدمين من الملف أو السحاب
            from app import load_json, default_users
            users = load_json(USERS_FILE, default_users)
        except Exception:
            users = [
                {"username": "ahmad", "name": "أحمد علي", "role": "employee", "job_title": "دعم فني"},
                {"username": "sara", "name": "سارة محمود", "role": "employee", "job_title": "خدمة عملاء"},
            ]

        employees = [u for u in users if u.get("role") == "employee"]
        if not employees:
            return "ahmad", "أحمد علي (دعم فني)"

        # إذا طلب المتصل موظفاً بالاسم (مثلاً "أحمد" أو "سارة")
        for emp in employees:
            first_name = emp.get("name", "").split()[0].lower()
            if first_name and first_name in inquiry.lower():
                return emp.get("username"), f"{emp.get('name')} ({emp.get('job_title', 'موظف')})"

        # اختيار الموظف المتخصص في نفس القسم
        dept_match = [e for e in employees if department in e.get("job_title", "")]
        if dept_match:
            chosen = dept_match[0]
            return chosen.get("username"), f"{chosen.get('name')} ({chosen.get('job_title')})"

        # إذا لم نجد تطابقاً مع القسم، نختار أول موظف متاح
        fallback_emp = employees[0]
        return fallback_emp.get("username"), f"{fallback_emp.get('name')} ({fallback_emp.get('job_title', 'خدمة عملاء')})"

    def handle_call_interaction(self, name, phone, inquiry):
        """معالجة المكالمة الصوتية بالذكاء الاصطناعي وتحديد التحويل التلقائي"""
        # جلب الإجابة من قاعدة المعرفة RAG
        answer = self.rag.query(inquiry)

        # تحليل نية المتصل والقسم المستهدف
        is_transfer_requested, department = self.classify_intent_and_department(inquiry)

        # التحويل التلقائي في حال عدم معرفة الذكاء الاصطناعي أو حدوث خطأ
        ai_failed = any(phrase in answer for phrase in ["لا أعلم", "اعتذر", "خطأ من سيرفر", "تعذر الحصول"])
        should_transfer = is_transfer_requested or ai_failed

        action = "answered"
        assigned_username = None
        assigned_agent = None

        if should_transfer:
            action = "transferred"
            assigned_username, assigned_agent = self.find_best_agent(department, inquiry)
            if ai_failed and not is_transfer_requested:
                answer += f"\n\n📞 جاري تحويل استفسارك تلقائياً إلى الزميل {assigned_agent} لمساعدتك بشكل دقيق."

        return {
            "action": action,
            "message": answer,
            "assigned_agent": assigned_agent,
            "assigned_username": assigned_username,
            "department": department,
        }

    def text_to_speech(self, text: str) -> bytes:
        """تحويل النص إلى صوت عربي واضح للمتصل"""
        if gTTS is None:
            return b""
        try:
            # تنظيف الرموز والتنسيقات ليكون النطق الصوتي طبيعياً
            clean_text = re.sub(r"[*#_`\[\]()]", " ", text).strip()
            if not clean_text:
                clean_text = "أهلاً بك، تفضل بطرح استفسارك وسأساعدك فوراً."

            tts = gTTS(text=clean_text[:500], lang="ar")
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as e:
            print(f"TTS Error: {e}")
            return b""
