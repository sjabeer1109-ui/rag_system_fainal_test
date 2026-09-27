import base64
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import json
import os
import smtplib
import time
from call_center import CallCenterEngine
import pandas as pd
from rag_engine import EnterpriseRAG
import streamlit as st

st.set_page_config(
    page_title="نظام الإدارة وخدمة العملاء المركزي", layout="wide"
)


@st.cache_resource
def get_systems():
  rag = EnterpriseRAG()
  rag.sync_documents()
  cc = CallCenterEngine(rag)
  return rag, cc


rag, call_center = get_systems()

USERS_FILE = "users.json"
TASKS_FILE = "tasks.json"
CALLS_FILE = "calls.json"
CHATS_FILE = "chats.json"
EMAIL_CONFIG_FILE = "email_config.json"
FAQ_CACHE_FILE = "smart_faq.json"
RECORDINGS_DIR = "recordings"
DOCS_DIR = "documents"

os.makedirs(RECORDINGS_DIR, exist_ok=True)
os.makedirs(DOCS_DIR, exist_ok=True)


def load_json(filepath, default_val):
  if os.path.exists(filepath):
    try:
      with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)
    except Exception:
      return default_val
  return default_val


def save_json(filepath, data):
  with open(filepath, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)


# دالة فحص الملفات
def get_physical_documents():
  found_files = {}
  search_folders = [DOCS_DIR, "."]
  for folder in search_folders:
    if os.path.exists(folder):
      for f in os.listdir(folder):
        if (
            f.lower().endswith(".pdf") or f.lower().endswith(".txt")
        ) and not f.startswith("~"):
          full_p = os.path.join(folder, f)
          if os.path.isfile(full_p) and f not in found_files:
            size_kb = round(os.path.getsize(full_p) / 1024, 1)
            found_files[f] = {
                "path": full_p,
                "size": f"{size_kb} KB",
                "folder": folder,
            }
  return found_files


# ==============================================================================
# الأسئلة المتوقعة الذكية (مبنية على تحليل الملفات فعلياً عبر الـ RAG)
# ==============================================================================
def get_docs_signature():
  """بصمة نصية لمجموعة الملفات الحالية، تُستخدم لمعرفة إذا تغيّرت الملفات."""
  files = get_physical_documents()
  sig_parts = sorted([f"{name}:{info['size']}" for name, info in files.items()])
  return "|".join(sig_parts)


def _extract_json_object(raw_text):
  cleaned = raw_text.strip()
  if cleaned.startswith("```"):
    cleaned = cleaned.strip("`")
    if cleaned.lower().startswith("json"):
      cleaned = cleaned[4:]
  start = cleaned.find("{")
  end = cleaned.rfind("}")
  if start != -1 and end != -1:
    cleaned = cleaned[start:end + 1]
  return json.loads(cleaned)


def generate_smart_faq(rag, num_categories=5, questions_per_category=4):
  """يحلل الملفات المفهرسة عبر الـ RAG، يستنتج تصنيفات وأسئلة متوقعة من العملاء
  لكل تصنيف، ثم يستخرج إجابة فعلية لكل سؤال ويخزن كل شيء (النتيجة جاهزة/محفوظة)."""
  signature = get_docs_signature()
  if not signature:
    return False, "⚠️ لا توجد ملفات مفهرسة بعد لتوليد أسئلة منها.", None

  try:
    meta_prompt = (
        "بناءً على كل الوثائق والملفات المتوفرة لديك في قاعدة المعرفة فقط،"
        f" ولّد {num_categories} تصنيفات (فئات مواضيع) تغطي أهم محتويات هذه"
        f" الملفات، ولكل تصنيف اكتب {questions_per_category} أسئلة يُتوقع أن"
        " يطرحها عميل أو موظف حول هذا الموضوع تحديداً. أجب فقط بصيغة JSON"
        " صحيحة وصارمة بدون أي نص أو شرح أو علامات ``` قبلها أو بعدها، بالشكل"
        ' التالي بالضبط: {"اسم التصنيف الأول": ["السؤال 1", "السؤال 2"],'
        ' "اسم التصنيف الثاني": ["السؤال 1", "السؤال 2"]}'
    )
    raw = rag.query(meta_prompt)
    categories_questions = _extract_json_object(raw)
  except Exception as e:
    return False, f"❌ تعذر توليد التصنيفات والأسئلة من الملفات: {str(e)}", None

  result = {}
  for cat, qs in categories_questions.items():
    qa_list = []
    for q in qs:
      try:
        ans = rag.query(q)
      except Exception as e:
        ans = f"⚠️ تعذر استخراج الإجابة تلقائياً: {str(e)}"
      qa_list.append({"question": q, "answer": ans})
    result[cat] = qa_list

  cache = {
      "signature": signature,
      "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
      "categories": result,
  }
  save_json(FAQ_CACHE_FILE, cache)
  return True, "✅ تم توليد الأسئلة الذكية وإجاباتها بنجاح من الملفات الحالية.", cache


def load_smart_faq():
  return load_json(FAQ_CACHE_FILE, None)


def render_smart_faq(rag, allow_generate=False):
  cache = load_smart_faq()
  current_sig = get_docs_signature()

  if allow_generate:
    col_g1, col_g2 = st.columns([3, 1])
    with col_g1:
      if cache:
        st.caption(f"🕒 آخر توليد: {cache.get('generated_at', '-')}")
        if cache.get("signature") != current_sig:
          st.warning(
              "⚠️ الملفات تغيّرت منذ آخر توليد للأسئلة، يُفضل إعادة التوليد"
              " ليتم تحديث الأسئلة والإجابات."
          )
      else:
        st.info("لم يتم توليد أسئلة ذكية بعد من الملفات.")
    with col_g2:
      if st.button("🧠 توليد / تحديث الأسئلة من الملفات"):
        with st.spinner("جاري تحليل الملفات وتوليد الأسئلة المتوقعة وإجاباتها..."):
          ok, msg, new_cache = generate_smart_faq(rag)
        if ok:
          st.success(msg)
          st.rerun()
        else:
          st.error(msg)
    st.markdown("---")

  if not cache or not cache.get("categories"):
    st.info("📭 لا توجد أسئلة ذكية متاحة حالياً.")
    return

  for cat_name, qa_list in cache["categories"].items():
    with st.expander(f"📂 {cat_name}", expanded=True):
      for qa in qa_list:
        st.markdown(f"**❓ {qa['question']}**")
        st.info(qa["answer"])


# ==============================================================================
# الشات الذكي المحفوظ (مشترك بين المدير والموظفين، لكل مستخدم جلساته الخاصة)
# ==============================================================================
def render_chat_tab(rag, username):
  if username not in st.session_state.chats_db:
    st.session_state.chats_db[username] = []
  my_sessions = st.session_state.chats_db[username]

  if not my_sessions:
    first_id = f"session_1_{int(datetime.now().timestamp())}"
    my_sessions.append({"id": first_id, "title": "محادثة عامة", "messages": []})
    save_json(CHATS_FILE, st.session_state.chats_db)

  col_side, col_chat = st.columns([1, 2])

  with col_side:
    st.write("#### 📑 سجل جلسات محادثاتك:")
    if st.button("➕ محادثة جديدة", key=f"newchat_{username}"):
      new_sess_id = (
          f"session_{len(my_sessions) + 1}_{int(datetime.now().timestamp())}"
      )
      my_sessions.append({
          "id": new_sess_id,
          "title": f"محادثة #{len(my_sessions) + 1}",
          "messages": [],
      })
      save_json(CHATS_FILE, st.session_state.chats_db)
      st.rerun()

    session_dict = {s["id"]: s["title"] for s in my_sessions}
    selected_session_id = st.radio(
        "اختر الجلسة:",
        list(session_dict.keys()),
        format_func=lambda x: f"🗨️ {session_dict[x]}",
        key=f"radio_{username}",
    )

  with col_chat:
    curr_session = next(s for s in my_sessions if s["id"] == selected_session_id)
    st.write(f"### 📌 {curr_session['title']}")

    for m in curr_session["messages"]:
      with st.chat_message(m["role"]):
        st.write(m["content"])

    prompt = st.chat_input("اكتب سؤالك هنا...", key=f"chatinput_{username}")
    if prompt:
      if not curr_session["messages"]:
        curr_session["title"] = " ".join(prompt.split()[:5])

      curr_session["messages"].append({"role": "user", "content": prompt})
      with st.chat_message("user"):
        st.write(prompt)

      with st.chat_message("assistant"):
        with st.spinner("جاري استخراج الإجابة..."):
          ans = rag.query(prompt)
        st.write(ans)

      curr_session["messages"].append({"role": "assistant", "content": ans})
      save_json(CHATS_FILE, st.session_state.chats_db)
      st.rerun()


def send_employee_email(
    to_email, employee_name, username, pin, job_title, action="update"
):
  cfg = st.session_state.email_config
  sender = cfg.get("sender_email", "").strip()
  password = cfg.get("sender_password", "").strip()
  server = cfg.get("smtp_server", "smtp.gmail.com").strip()
  port = int(cfg.get("smtp_port", 587))

  if not sender or not password:
    return (
        False,
        "⚠️ لم يتم ضبط بريد الإدارة وكلمة مرور التطبيقات في تبويب 'إعدادات"
        " البريد الإلكتروني'.",
    )
  if not to_email or "@" not in to_email:
    return False, "⚠️ البريد الإلكتروني للموظف غير صالح أو فارغ."

  try:
    msg = MIMEMultipart()
    msg["From"] = sender
    msg["To"] = to_email

    if action == "create":
      msg["Subject"] = "بيانات حسابك الجديد في النظام المركزي"
      body = f"""مرحباً {employee_name}،
تم إنشاء حساب عمل جديد لك في النظام المركزي بالبيانات التالية:
- اسم المستخدم: {username}
- رمز الدخول (PIN): {pin}
- المسمى الوظيفي: {job_title}
رابط النظام: http://localhost:8501
مع تحيات الإدارة العامة."""
    else:
      msg["Subject"] = "تحديث بيانات حسابك في النظام المركزي"
      body = f"""مرحباً {employee_name}،
نود إعلامك بأنه تم تحديث بيانات حسابك من قبل الإدارة:
- اسم المستخدم: {username}
- رمز الدخول (PIN): {pin}
- المسمى الوظيفي: {job_title}
إذا لم تكن على علم بهذا التغيير، يرجى مراجعة إدارة النظام فوراً."""

    msg.attach(MIMEText(body, "plain", "utf-8"))
    with smtplib.SMTP(server, port) as s:
      s.starttls()
      s.login(sender, password)
      s.send_message(msg)
    return True, f"✅ تم إرسال الإيميل بنجاح إلى: {to_email}"
  except Exception as e:
    return False, f"❌ فشل إرسال البريد: {str(e)}"


default_users = [
    {
        "id": 1,
        "username": "admin",
        "pin": "0000",
        "name": "المدير العام",
        "role": "admin",
        "job_title": "مدير النظام",
        "email": "admin@company.com",
    },
    {
        "id": 2,
        "username": "ahmad",
        "pin": "1234",
        "name": "أحمد علي",
        "role": "employee",
        "job_title": "دعم فني",
        "email": "ahmad@company.com",
    },
    {
        "id": 3,
        "username": "sara",
        "pin": "5678",
        "name": "سارة محمود",
        "role": "employee",
        "job_title": "خدمة عملاء",
        "email": "sara@company.com",
    },
]

if "users_db" not in st.session_state:
  st.session_state.users_db = load_json(USERS_FILE, default_users)
  save_json(USERS_FILE, st.session_state.users_db)

if "tasks_db" not in st.session_state:
  st.session_state.tasks_db = load_json(TASKS_FILE, [])

if "calls_db" not in st.session_state:
  st.session_state.calls_db = load_json(CALLS_FILE, [])

if "chats_db" not in st.session_state:
  st.session_state.chats_db = load_json(CHATS_FILE, {})

if "email_config" not in st.session_state:
  st.session_state.email_config = load_json(
      EMAIL_CONFIG_FILE,
      {
          "sender_email": "",
          "sender_password": "",
          "smtp_server": "smtp.gmail.com",
          "smtp_port": 587,
      },
  )

if "logged_user" not in st.session_state:
  st.session_state.logged_user = None

if "real_admin_user" not in st.session_state:
  st.session_state.real_admin_user = None

if "admin_qa_result" not in st.session_state:
  st.session_state.admin_qa_result = None

# شاشة تسجيل الدخول
if st.session_state.logged_user is None:
  st.title("🔐 تسجيل الدخول إلى النظام ")
  c1, c2, c3 = st.columns(3)
  with c2:
    with st.form("login_form"):
      u = st.text_input("اسم المستخدم:")
      p = st.text_input("رمز الدخول (PIN):", type="password")
      if st.form_submit_button("تسجيل الدخول"):
        matched = next(
            (
                x
                for x in st.session_state.users_db
                if x["username"].lower() == u.strip().lower()
                and x["pin"] == p.strip()
            ),
            None,
        )
        if matched:
          st.session_state.logged_user = matched["username"]
          if matched["role"] == "admin":
            st.session_state.real_admin_user = matched["username"]
          st.rerun()
        else:
          st.error("بيانات الدخول غير صحيحة.")
    st.info("💡 ادخل بحسابك تم ارسال معلومات الحساب عبر ايميل الشركة ")
  st.stop()

current_user = next(
    u
    for u in st.session_state.users_db
    if u["username"] == st.session_state.logged_user
)

# ==============================================================================
# الشريط الجانبي (SIDEBAR)
# ==============================================================================
with st.sidebar:
  st.write(f"👤 المستخدم الحالي: **{current_user['name']}**")
  role_display = "مدير عام" if current_user["role"] == "admin" else "موظف"
  st.caption(f"الصلاحية: {role_display} ({current_user.get('job_title', '')})")

  if (
      st.session_state.real_admin_user
      and st.session_state.logged_user != st.session_state.real_admin_user
  ):
    st.warning("⚠️ وضع التصفح كـ موظف (حساب المدير)")
    if st.button("🔙 العودة لحساب المدير"):
      st.session_state.logged_user = st.session_state.real_admin_user
      st.rerun()

  elif current_user["role"] == "admin":
    st.markdown("---")
    st.subheader("⚡ الدخول السريع لحساب الموظف:")
    emp_list = [u for u in st.session_state.users_db if u["role"] == "employee"]
    for emp in emp_list:
      if st.button(
          f"👤 فتح واجهة {emp['name']}", key=f"quick_to_{emp['username']}"
      ):
        st.session_state.logged_user = emp["username"]
        st.rerun()

  st.markdown("---")
  if st.button("🚪 تسجيل الخروج"):
    st.session_state.logged_user = None
    st.session_state.real_admin_user = None
    st.rerun()

# ==============================================================================
#                      1. واجهة المدير (ADMIN DASHBOARD)
# ==============================================================================
if current_user["role"] == "admin":
  st.title("🛡️ لوحة تحكم الإدارة العامة ومتابعة التسجيلات")

  (
      tab_faq,
      tab_mychat,
      tab_recordings,
      tab_mgmt,
      tab_email,
      tab_docs,
  ) = st.tabs([
      "💡 الأسئلة الذكية",
      "💬 شاتي",
      "🎧 سجل المكالمات ",
      "👥 إدارة الموظفين والمهام",
      "⚙️ إعدادات البريد الإلكتروني",
      "📁 ملفات الشركة ",
  ])

  # --- تبويب 1: الأسئلة المتوقعة الذكية (مولّدة تلقائياً من الملفات) ---
  with tab_faq:
    st.subheader("💡 الأسئلة المتوقعة الذكية (مبنية على تحليل ملفات الشركة)")
    st.caption(
        "يتم توليد هذه الأسئلة وإجاباتها تلقائياً من تحليل الملفات المرفوعة،"
        " وتظهر نفسها للموظفين أيضاً."
    )
    render_smart_faq(rag, allow_generate=True)

    st.markdown("---")
    st.subheader("🔍 استعلام مباشر (غير محفوظ في الأسئلة الشائعة)")
    user_custom_q = st.text_input(
        "اكتب السؤال هنا:", placeholder="ب ماذا تفكر  "
    )
    if st.button("🔍 تنفيذ الاستعلام الآن"):
      if user_custom_q.strip():
        with st.spinner("جاري استخراج الإجابة..."):
          ans = rag.query(user_custom_q.strip())
          st.session_state.admin_qa_result = {
              "question": user_custom_q.strip(),
              "answer": ans,
          }

    if st.session_state.admin_qa_result:
      res = st.session_state.admin_qa_result
      st.markdown("---")
      st.markdown("### 📋الإجابة:")
      st.success(f"**السؤال:** {res['question']}")
      st.info(f"💡 **الشرح المباشر:**\n\n{res['answer']}")
      if st.button("مسح الإجابة"):
        st.session_state.admin_qa_result = None
        st.rerun()

  # --- تبويب 2: شات المدير الخاص المحفوظ ---
  with tab_mychat:
    st.subheader("💬 شاتي الخاص")
    st.caption("محادثاتك محفوظة هنا، وتقدر تفتح أكثر من محادثة وترجعلها بأي وقت.")
    render_chat_tab(rag, current_user["username"])

  # --- تبويب 3: سجل المكالمات والتسجيل الحقيقي ---
  with tab_recordings:
    st.subheader("🎧 سجل المكالمات  ")
    calls = load_json(CALLS_FILE, [])
    if not calls:
      st.info(
          "لا توجد مكالمات مسجلة بعد. عند إجراء مكالمة ستظهر هنا"
          " فوراً."
      )

    for call in reversed(calls):
      call_id = call.get("id")
      with st.expander(
          f"📞 مكالمة: {call.get('customer_name')} ({call.get('customer_phone')}) |"
          f" الحالة: [{call.get('status')}] - بتوقيت: {call.get('timestamp')}"
      ):
        st.write(f"**نص حديث العميل:** {call.get('inquiry')}")
        st.write(
            f"**إجابة الذكاء الاصطناعي:**\n{call.get('ai_initial_answer')}"
        )

        audio_file = call.get("audio_file")
        st.markdown("#### 🎙️ الاستماع للتسجيل الصوتي المباشر للمكالمة:")
        if audio_file and os.path.exists(audio_file):
          with open(audio_file, "rb") as af:
            st.audio(af.read(), format="audio/webm")
          st.caption(
              "✅ هذا التسجيل يتضمن صوت المتصل وصوت الذكاء الاصطناعي معاً بجودة"
              " كاملة."
          )
        else:
          st.warning("جاري معالجة التسجيل أو لم يتم إغلاق المكالمة بعد.")

        if call.get("employee_note"):
          st.info(f"📝 **رد وملاحظة الموظف:**\n{call.get('employee_note')}")

        if st.button("🗑️ حذف هذه المكالمة", key=f"del_c_{call_id}"):
          calls = [c for c in calls if c.get("id") != call_id]
          save_json(CALLS_FILE, calls)
          st.success("تم الحذف.")
          st.rerun()

  # --- تبويب 4: إدارة الموظفين وتحديث بياناتهم وإرسال الإيميل ---
  with tab_mgmt:
    with st.expander("➕ إضافة موظف جديد إلى النظام", expanded=False):
      with st.form("add_emp_form", clear_on_submit=True):
        col_a1, col_a2 = st.columns(2)
        with col_a1:
          new_name = st.text_input("اسم الموظف الكامل:")
          new_uname = st.text_input("اسم المستخدم (Username):")
          new_pin = st.text_input("رمز الدخول (PIN):")
        with col_a2:
          new_job = st.selectbox(
              "المسمى الوظيفي:",
              ["دعم فني", "خدمة عملاء", "مبيعات", "إدارة"],
          )
          new_email = st.text_input("البريد الإلكتروني للموظف:")
          send_welcome_mail = st.checkbox(
              "📧 إرسال بيانات الدخول لإيميل الموظف فوراً", value=True
          )

        if st.form_submit_button("💾 حفظ وإضافة الموظف الآن"):
          if new_name.strip() and new_uname.strip() and new_pin.strip():
            clean_un = new_uname.strip().lower()
            if any(
                u["username"].lower() == clean_un
                for u in st.session_state.users_db
            ):
              st.error("اسم المستخدم مسجل مسبقاً! اختر اسماً آخر.")
            else:
              new_id = (
                  max([u["id"] for u in st.session_state.users_db], default=0)
                  + 1
              )
              new_emp = {
                  "id": new_id,
                  "username": clean_un,
                  "pin": new_pin.strip(),
                  "name": new_name.strip(),
                  "role": "employee",
                  "job_title": new_job,
                  "email": new_email.strip(),
              }
              st.session_state.users_db.append(new_emp)
              save_json(USERS_FILE, st.session_state.users_db)
              st.success(f"تمت إضافة الموظف '{new_name}' بنجاح!")

              if send_welcome_mail and new_email.strip():
                ok, msg_mail = send_employee_email(
                    new_email.strip(),
                    new_name.strip(),
                    clean_un,
                    new_pin.strip(),
                    new_job,
                    action="create",
                )
                if ok:
                  st.success(msg_mail)
                else:
                  st.warning(msg_mail)
              st.rerun()

    st.markdown("---")
    st.subheader("سجلات وبطاقات الموظفين وتعديل البيانات:")
    for emp in st.session_state.users_db:
      if emp["role"] == "employee":
        emp_id = emp["id"]
        emp_name = emp["name"]
        with st.expander(
            f"👤 ملف: {emp_name} | الوظيفة: {emp['job_title']}", expanded=False
        ):
          c1, c2 = st.columns(2)
          with c1:
            u_val = st.text_input(
                "اسم المستخدم:", value=emp["username"], key=f"u_{emp_id}"
            )
            p_val = st.text_input(
                "رمز الدخول (PIN):", value=emp["pin"], key=f"p_{emp_id}"
            )
          with c2:
            e_val = st.text_input(
                "البريد الإلكتروني:",
                value=emp.get("email", ""),
                key=f"e_{emp_id}",
            )
            j_val = st.text_input(
                "المسمى الوظيفي:", value=emp["job_title"], key=f"j_{emp_id}"
            )

          btn_col1, btn_col2, btn_col3 = st.columns(3)
          with btn_col1:
            if st.button(
                "💾 حفظ وإرسال إيميل بالبيانات", key=f"save_email_{emp_id}"
            ):
              emp["username"] = u_val.strip()
              emp["pin"] = p_val.strip()
              emp["email"] = e_val.strip()
              emp["job_title"] = j_val.strip()
              save_json(USERS_FILE, st.session_state.users_db)
              st.success("تم حفظ التعديلات في النظام!")

              ok, msg_info = send_employee_email(
                  emp["email"],
                  emp_name,
                  emp["username"],
                  emp["pin"],
                  emp["job_title"],
                  action="update",
              )
              if ok:
                st.success(msg_info)
              else:
                st.warning(msg_info)
              st.rerun()

          with btn_col2:
            if st.button("📧 إرسال إيميل فقط", key=f"send_only_{emp_id}"):
              ok, msg_info = send_employee_email(
                  emp.get("email", ""),
                  emp_name,
                  emp["username"],
                  emp["pin"],
                  emp["job_title"],
                  action="update",
              )
              if ok:
                st.success(msg_info)
              else:
                st.warning(msg_info)

          with btn_col3:
            if st.button("🗑️ حذف الموظف", key=f"del_{emp_id}"):
              st.session_state.users_db = [
                  u for u in st.session_state.users_db if u["id"] != emp_id
              ]
              save_json(USERS_FILE, st.session_state.users_db)
              st.warning(f"تم حذف {emp_name}.")
              st.rerun()

          st.markdown("---")

          task_c1, task_c2 = st.columns(2)
          with task_c1:
            task_text = st.text_input(
                "إسناد مهمة جديدة للموظف:", key=f"task_in_{emp_id}"
            )
          with task_c2:
            st.write("")
            st.write("")
            if st.button("➕ إرسال المهمة", key=f"btn_task_{emp_id}"):
              if task_text.strip():
                new_tid = (
                    max(
                        [t["id"] for t in st.session_state.tasks_db], default=0
                    )
                    + 1
                )
                st.session_state.tasks_db.append({
                    "id": new_tid,
                    "username": emp["username"],
                    "المهمة": task_text.strip(),
                    "الحالة": "قيد التنفيذ",
                    "تنبيه": f"🔔 مهمة من الإدارة: {task_text.strip()}",
                })
                save_json(TASKS_FILE, st.session_state.tasks_db)
                st.success("تم إرسال المهمة بنجاح!")
                st.rerun()

          st.markdown("##### المهام المسندة إليه:")
          emp_tasks = [
              t
              for t in st.session_state.tasks_db
              if t["username"] == emp["username"]
          ]
          if not emp_tasks:
            st.caption("لا توجد مهام مسندة.")
          for t in emp_tasks:
            task_desc = t.get("المهمة", t.get("task", ""))
            task_status = t.get("الحالة", t.get("status", "قيد التنفيذ"))
            st.write(f"- **{task_desc}** | الحالة: `{task_status}`")

  # --- تبويب 5: إعدادات البريد الإلكتروني (SMTP) ---
  with tab_email:
    st.subheader("⚙️ إعدادات البريد الإلكتروني للإدارة (SMTP)")
    st.caption("يتم استخدام هذه الإعدادات لإرسال بيانات الحسابات للموظفين آلياً:")
    with st.form("smtp_config_form"):
      s_email = st.text_input(
          "بريدك الإلكتروني (Gmail):",
          value=st.session_state.email_config.get("sender_email", ""),
      )
      s_pass = st.text_input(
          "كلمة مرور التطبيقات (App Password):",
          type="password",
          value=st.session_state.email_config.get("sender_password", ""),
          help="كلمة مرور التطبيقات المكونة من 16 حرفاً من حساب جوجل",
      )
      s_server = st.text_input(
          "خادم SMTP:",
          value=st.session_state.email_config.get(
              "smtp_server", "smtp.gmail.com"
          ),
      )
      s_port = st.number_input(
          "منفذ SMTP:",
          value=int(st.session_state.email_config.get("smtp_port", 587)),
      )
      if st.form_submit_button("💾 حفظ إعدادات البريد"):
        st.session_state.email_config = {
            "sender_email": s_email.strip(),
            "sender_password": s_pass.strip(),
            "smtp_server": s_server.strip(),
            "smtp_port": s_port,
        }
        save_json(EMAIL_CONFIG_FILE, st.session_state.email_config)
        st.success("تم حفظ إعدادات البريد بنجاح!")

  # --- تبويب 6: المستندات والمستودع وإظهار الملفات الموجودة فعلياً على القرص ---
  with tab_docs:
    st.subheader("📁 ملفات الشركة")

    st.markdown("#### 📥 رفع ملفات جديدة وحفظها فوراً في مجلد documents:")
    uploaded_files = st.file_uploader(
        "اختر ملفات PDF أو TXT لإضافتها إلى النظام:",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )
    if uploaded_files:
      if st.button("🚀 بدء حفظ وفهرسة الملفات المرفوعة"):
        with st.spinner("جاري حفظ الملفات وتحديث الفهرس..."):
          for uf in uploaded_files:
            target_path = os.path.join(DOCS_DIR, uf.name)
            with open(target_path, "wb") as f_out:
              f_out.write(uf.read())
          sync_res = rag.sync_documents()
          st.success("تم حفظ الملفات وتحديث الفهرس بنجاح!")

    st.markdown("---")
    # فحص وعرض الملفات الموجودة فعلياً على القرص
    physical_files = get_physical_documents()
    st.markdown("#### 📂 الملفات الموجودة فعلياً في مجلدات المشروع:")
    if physical_files:
      for fname, finfo in physical_files.items():
        is_indexed = fname in rag.processed_hashes
        status_tag = (
            "✅ مفهرس في قاعدة البيانات"
            if is_indexed
            else "⏳ بانتظار إتمام الفهرسة"
        )
        st.write(
            f"- 📄 **`{fname}`** ({finfo['size']}) — المسار:"
            f" `{finfo['folder']}` | الحالة: **{status_tag}**"
        )
    else:
      st.error(
          "⚠️ لم يتم العثور على أي ملفات PDF أو TXT داخل المجلد الرئيسي أو داخل"
          " مجلد documents."
      )

    st.markdown("---")
    if st.button("🔄 فحص وتحديث فهرس الملفات الآن"):
      with st.spinner(
          "جاري فحص المجلد الرئيسي ومجلد المستندات وتحديث الفهرس..."
      ):
        sync_res = rag.sync_documents()
        if isinstance(sync_res, tuple):
          success, msg_sync = sync_res
        else:
          success = bool(sync_res)
          msg_sync = (
              "تم تحديث الفهرس بنجاح."
              if success
              else "جميع الملفات مفهرسة مسبقاً."
          )

      st.success(f"نتيجة الفحص: {msg_sync}")
      st.rerun()
# ==============================================================================
#                      2. واجهة الموظف (EMPLOYEE DASHBOARD)
# ==============================================================================
else:
  st.title(f"💼 واجهة عمل الموظف: {current_user['name']}")
  t0, t1, t2, t3 = st.tabs([
      "💡 الأسئلة الشائعة",
      "📞 المكالمات المحولة إليّ ",
      "📌 مهامي وتنبيهات الإدارة",
      "💬 الشات الذكي لك",
  ])

  # --- تبويب 0: الأسئلة الشائعة الذكية (نفس أسئلة المدير، للعرض فقط) ---
  with t0:
    st.subheader("💡 الأسئلة الشائعة")
    render_smart_faq(rag, allow_generate=False)

  # --- تبويب 1: مكالمات الموظف ---
  with t1:
    calls = load_json(CALLS_FILE, [])
    transferred = [
        c
        for c in calls
        if c.get("assigned_to") == current_user["username"]
        and c.get("status") == "محولة للموظف"
    ]
    if not transferred:
      st.info("🟢 لا توجد مكالمات محولة إليك حالياً.")
    for c in transferred:
      cid = c.get("id")
      st.error(
          f"🚨 مكالمة محولة من: {c.get('customer_name')} | هاتف:"
          f" {c.get('customer_phone')}"
      )
      st.write(f"**سؤال العميل:** {c.get('inquiry')}")
      st.info(f"**الحل المستخرج من الملفات:**\n{c.get('ai_initial_answer')}")

      if c.get("audio_file") and os.path.exists(c.get("audio_file")):
        st.write("🎧 استمع لتسجيل صوت العميل والذكاء الاصطناعي قبل الرد:")
        with open(c.get("audio_file"), "rb") as f:
          st.audio(f.read(), format="audio/webm")

      emp_reply_note = st.text_area(
          "ملاحظات أو رد الموظف على المكالمة:", key=f"reply_{cid}"
      )

      if st.button("✅ تم الرد وحل المشكلة للعميل", key=f"done_c_{cid}"):
        c["status"] = "تم الحل بواسطة الموظف"
        c["employee_note"] = emp_reply_note
        save_json(CALLS_FILE, calls)
        st.success("تم إغلاق التذكرة وتحديث السجل للمدير!")
        st.rerun()

  # --- تبويب 2: مهام الموظف ---
  with t2:
    my_tasks = [
        t
        for t in st.session_state.tasks_db
        if t["username"] == current_user["username"]
    ]
    if not my_tasks:
      st.info("لا توجد مهام مسندة إليك.")

    for t in my_tasks:
      t_id = t["id"]
      task_name = t.get("المهمة", t.get("task", ""))
      task_status = t.get("الحالة", t.get("status", "قيد التنفيذ"))
      st.write(f"📌 **{task_name}** - الحالة: `{task_status}`")
      if task_status != "تم":
        if st.button(f"✅ تأكيد إنجاز المهمة", key=f"finish_t_{t_id}"):
          t["الحالة"] = "تم"
          t["status"] = "تم"
          save_json(TASKS_FILE, st.session_state.tasks_db)
          st.success("تم إرسال تأكيد الإنجاز للمدير!")
          st.rerun()

  # --- تبويب 3: الشات الذكي المحفوظ الخاص بالموظف ---
  with t3:
    st.subheader("💬 الشات الذكي للاستعلام")
    st.caption("محادثاتك محفوظة هنا، وتقدر تفتح أكثر من محادثة وترجعلها بأي وقت.")
    render_chat_tab(rag, current_user["username"])
