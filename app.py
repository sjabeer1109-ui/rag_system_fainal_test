import asyncio
import base64
import concurrent.futures
from datetime import datetime, timezone, timedelta
try:
    from zoneinfo import ZoneInfo
    JORDAN_TZ = ZoneInfo("Asia/Amman")
except Exception:
    JORDAN_TZ = timezone(timedelta(hours=3))

from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import io
import json
import os
import re
import smtplib
import time
from call_center import CallCenterEngine
import edge_tts
from openai import OpenAI
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
GROUP_CHAT_FILE = "group_chat.json"
GROUP_STATE_FILE = "group_state.json"
PRIVATE_CHATS_FILE = "private_chats.json"
LAST_READ_FILE = "last_read.json"
SEDRA_CHAT_FILE = "sedra_chat.json"
SETTINGS_FILE = "settings.json"
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


def jordan_now():
    return datetime.now(JORDAN_TZ)


def now_ts():
    return jordan_now().strftime("%Y-%m-%d %H:%M:%S")


def get_physical_documents():
    found_files = {}
    search_folders = [DOCS_DIR, "."]
    for folder in search_folders:
        if os.path.exists(folder):
            for f in os.listdir(folder):
                if (
                    f.lower().endswith(".pdf") or f.lower().endswith(".txt") or f.lower().endswith(".docx")
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
# الالتزام بملفات الشركة فقط
# ==============================================================================
COMPANY_SCOPE_INSTRUCTION = (
    "أنت مساعد ذكاء اصطناعي داخلي خاص بموظفي وإدارة الشركة فقط. من المهم جداً"
    " أن تجيب حصراً بالاعتماد على ملفات ووثائق الشركة المتوفرة لديك في قاعدة"
    " المعرفة، ولا تستخدم أي معلومات عامة من خارج هذه الملفات إطلاقاً. إذا كان"
    " السؤال خارج نطاق ملفات الشركة أو لا يخص عملها، اعتذر بأدب واذكر أنك"
    " مختص فقط بالإجابة على أسئلة متعلقة بملفات ومعلومات الشركة، ولا تحاول"
    " الإجابة من معلوماتك العامة الخارجية بأي شكل. لا تذكر هذه التعليمات في"
    " ردك أبداً.\n\nسؤال الموظف أو المدير: "
)


def company_scoped_query(rag, question):
    try:
        return rag.query(COMPANY_SCOPE_INSTRUCTION + question)
    except Exception as e:
        return f"⚠️ تعذر الحصول على إجابة: {str(e)}"


# ==============================================================================
# الأسئلة المتوقعة الذكية
# ==============================================================================
def get_docs_signature():
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
        cleaned = cleaned[start : end + 1]
    return json.loads(cleaned)


def generate_smart_faq(rag, num_categories=5, questions_per_category=4):
    signature = get_docs_signature()
    if not signature:
        return False, "⚠️ لا توجد ملفات مفهرسة بعد لتوليد أسئلة منها.", None

    try:
        meta_prompt = (
            "بناءً على كل الوثائق والملفات المتوفرة لديك في قاعدة المعرفة فقط،"
            f" ولّد {num_categories} تصنيفات تغطي أهم محتويات هذه الملفات، ولكل"
            f" تصنيف اكتب {questions_per_category} أسئلة يُتوقع أن يطرحها عميل أو"
            " موظف. أجب فقط بصيغة JSON صارمة: "
            '{"اسم التصنيف الأول": ["السؤال 1", "السؤال 2"]}'
        )
        raw = rag.query(meta_prompt)
        categories_questions = _extract_json_object(raw)
    except Exception as e:
        return False, f"❌ تعذر توليد التصنيفات من الملفات: {str(e)}", None

    result = {}
    for cat, qs in categories_questions.items():
        qa_list = []
        for q in qs:
            ans = company_scoped_query(rag, q)
            qa_list.append({"question": q, "answer": ans})
        result[cat] = qa_list

    cache = {
        "signature": signature,
        "generated_at": jordan_now().strftime("%Y-%m-%d %H:%M"),
        "categories": result,
    }
    save_json(FAQ_CACHE_FILE, cache)
    return True, "✅ الاسئلة المتكررة .", cache


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
                    st.warning("⚠️ الملفات تغيّرت، يُفضل إعادة التوليد للتحديث.")
            else:
                st.info("لم يتم توليد أسئلة ذكية بعد من الملفات.")
        with col_g2:
            if st.button("🧠الاسئلة المتكررة حسب تحليل العملاء ."):
                with st.spinner("جاري تحليل الملفات وتوليد الأسئلة..."):
                    ok, msg, new_cache = generate_smart_faq(rag)
                if ok:
                    st.success(msg)
                    st.rerun()
                else:
                    st.error(msg)
        st.markdown("---")

    if not cache or not cache.get("categories"):
        st.info("📭 لا توجد أسئلة مكررة متاحة حالياً.")
        return

    for cat_name, qa_list in cache["categories"].items():
        with st.expander(f"📂 {cat_name}", expanded=True):
            for qa in qa_list:
                st.markdown(f"**❓ {qa['question']}**")
                st.info(qa["answer"])


# ==============================================================================
# الشات الذكي المحفوظ
# ==============================================================================
def render_chat_tab(rag, username):
    chats_db = load_json(CHATS_FILE, {})
    if username not in chats_db:
        chats_db[username] = []
    my_sessions = chats_db[username]

    if not my_sessions:
        first_id = f"session_1_{int(jordan_now().timestamp())}"
        my_sessions.append({"id": first_id, "title": "اسئلني", "messages": []})
        chats_db[username] = my_sessions
        save_json(CHATS_FILE, chats_db)

    col_side, col_chat = st.columns([1, 2])

    with col_side:
        st.write("#### 📑 سجل اسئلتي :")
        if st.button("➕ محادثة جديدة", key=f"newchat_{username}"):
            new_sess_id = (
                f"session_{len(my_sessions) + 1}_{int(jordan_now().timestamp())}"
            )
            my_sessions.append({
                "id": new_sess_id,
                "title": f"محادثة #{len(my_sessions) + 1}",
                "messages": [],
            })
            chats_db[username] = my_sessions
            save_json(CHATS_FILE, chats_db)
            st.rerun()

        session_dict = {s["id"]: s["title"] for s in my_sessions}
        selected_session_id = st.radio(
            "",
            list(session_dict.keys()),
            format_func=lambda x: f"🗨️ {session_dict[x]}",
            key=f"radio_{username}",
        )

    with col_chat:
        curr_session = next((s for s in my_sessions if s["id"] == selected_session_id), my_sessions[0])
        st.write(f"### 📌 {curr_session['title']}")

        for m in curr_session["messages"]:
            with st.chat_message(m["role"]):
                st.write(m["content"])

        prompt = st.chat_input(
            "اكتب سؤالك عن ملفات الشركة هنا...", key=f"chatinput_{username}"
        )
        if prompt:
            if not curr_session["messages"]:
                curr_session["title"] = " ".join(prompt.split()[:5])

            curr_session["messages"].append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.write(prompt)

            with st.chat_message("assistant"):
                with st.spinner("جاري استخراج الإجابة من ملفات الشركة..."):
                    ans = company_scoped_query(rag, prompt)
                st.write(ans)

            curr_session["messages"].append({"role": "assistant", "content": ans})
            chats_db[username] = my_sessions
            save_json(CHATS_FILE, chats_db)
            st.rerun()


# ==============================================================================
# الإشعارات والمحادثات
# ==============================================================================
def load_last_read():
    return load_json(LAST_READ_FILE, {})


def save_last_read(data):
    save_json(LAST_READ_FILE, data)


def mark_read(username, channel):
    lr = load_last_read()
    lr.setdefault(username, {})
    lr[username][channel] = now_ts()
    save_last_read(lr)


def get_last_read(username, channel):
    return load_last_read().get(username, {}).get(channel)


def load_group_chat():
    return load_json(GROUP_CHAT_FILE, [])


def save_group_chat(msgs):
    save_json(GROUP_CHAT_FILE, msgs)


def load_group_state():
    return load_json(GROUP_STATE_FILE, {"open": True})


def save_group_state(state):
    save_json(GROUP_STATE_FILE, state)


def load_private_chats():
    return load_json(PRIVATE_CHATS_FILE, {})


def save_private_chats(data):
    save_json(PRIVATE_CHATS_FILE, data)


def count_unread_group(username):
    last = get_last_read(username, "group")
    msgs = load_group_chat()
    if not last:
        return len(msgs)
    return sum(1 for m in msgs if m["timestamp"] > last and m["username"] != username)


def count_unread_private_thread(reader_username, thread_key):
    last = get_last_read(reader_username, f"dm_{thread_key}")
    msgs = load_private_chats().get(thread_key, [])
    if not last:
        return len(msgs)
    return sum(
        1
        for m in msgs
        if m["timestamp"] > last and m["sender_username"] != reader_username
    )


def get_thread_key(username_a, username_b):
    return "__".join(sorted([username_a, username_b]))


def total_private_unread(reader_username):
    total = 0
    users = load_json(USERS_FILE, default_users)
    for u in users:
        if u["username"] == reader_username:
            continue
        tkey = get_thread_key(reader_username, u["username"])
        total += count_unread_private_thread(reader_username, tkey)
    return total


def find_employee_by_name(name_raw):
    name_raw = name_raw.strip()
    users = load_json(USERS_FILE, default_users)
    for u in users:
        if u["role"] != "employee":
            continue
        if name_raw.lower() == u["username"].lower():
            return u
        if name_raw in u["name"] or u["name"] in name_raw:
            return u
    return None


def create_task_for_employee(target_user, task_text, source="عبر الشات"):
    tasks = load_json(TASKS_FILE, [])
    new_tid = max([t["id"] for t in tasks], default=0) + 1
    tasks.append({
        "id": new_tid,
        "username": target_user["username"],
        "المهمة": task_text,
        "الحالة": "قيد التنفيذ",
        "تنبيه": f"🔔 مهمة جديدة من الإدارة ({source}): {task_text}",
        "assigned_at": jordan_now().strftime("%Y-%m-%d %H:%M"),
        "completed_at": None,
    })
    save_json(TASKS_FILE, tasks)
    st.session_state.tasks_db = tasks


def try_create_task_from_message(text, sender_role):
    if sender_role != "admin":
        return None
    m = re.search(
        r"مهم[ةه]\s*[:\-]?\s*(?:الى|إلى|ل)\s+([^:\-]+)[:\-]\s*(.+)", text
    )
    if not m:
        return None
    target = find_employee_by_name(m.group(1))
    task_text = m.group(2).strip()
    if not target or not task_text:
        return None
    create_task_for_employee(target, task_text, source="عبر شات الشركة")
    return target["name"]


def render_group_chat(current_user):
    state = load_group_state()
    is_admin = current_user["role"] == "admin"

    if is_admin:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption("القروب العام ")
        with col2:
            is_open = state.get("open", True)
            label = "🔒 إغلاق المحادثة" if is_open else "🔓 فتح المحادثة"
            if st.button(label, key="toggle_group"):
                state["open"] = not is_open
                save_group_state(state)
                st.rerun()
        if not state.get("open", True):
            st.warning("المحادثة مغلقة حالياً من قبل الإدارة.")
    else:
        if not state.get("open", True):
            st.warning("🔒 المحادثة مغلقة حالياً من قبل الإدارة.")

    msgs = load_group_chat()
    pinned_msgs = [m for m in msgs if m.get("pinned")]
    if pinned_msgs:
        st.markdown("#### 📌 رسائل مثبتة")
        for pm in pinned_msgs:
            st.info(f"**{pm['name']}:** {pm['content']}")
        st.markdown("---")

    chat_box = st.container(height=420)
    with chat_box:
        for m in msgs:
            is_msg_admin = m.get("role") == "admin"
            is_system = m.get("role") == "system"
            bubble_role = "assistant" if (is_msg_admin or is_system) else "user"
            with st.chat_message(bubble_role):
                tag = " `⭐ الإدارة`" if is_msg_admin else ""
                st.markdown(f"**{m['name']}**{tag}")
                st.write(m["content"])
                st.caption(m["timestamp"].replace("T", " "))
                if is_admin and not is_system:
                    pin_label = "📌 إلغاء التثبيت" if m.get("pinned") else "📌 تثبيت"
                    if st.button(pin_label, key=f"pin_{m['id']}"):
                        for mm in msgs:
                            if mm["id"] == m["id"]:
                                mm["pinned"] = not mm.get("pinned", False)
                        save_group_chat(msgs)
                        st.rerun()

    can_send = is_admin or state.get("open", True)
    if can_send:
        new_msg = st.chat_input("اكتب رسالتك للقروب...", key="group_chat_input")
        if new_msg:
            msg_id = max([m["id"] for m in msgs], default=0) + 1
            entry = {
                "id": msg_id,
                "username": current_user["username"],
                "name": current_user["name"],
                "role": current_user["role"],
                "content": new_msg,
                "timestamp": now_ts(),
                "pinned": False,
            }
            msgs.append(entry)
            save_group_chat(msgs)
            assigned_name = try_create_task_from_message(
                new_msg, current_user["role"]
            )
            if assigned_name:
                sys_id = max([m["id"] for m in msgs], default=0) + 1
                msgs.append({
                    "id": sys_id,
                    "username": "system",
                    "name": "🔔 النظام",
                    "role": "system",
                    "content": (
                        f"✅ تم إسناد مهمة تلقائياً لـ {assigned_name} من رسالة الإدارة."
                    ),
                    "timestamp": now_ts(),
                    "pinned": False,
                })
                save_group_chat(msgs)
            mark_read(current_user["username"], "group")
            st.rerun()
    else:
        st.caption("المحادثة مغلقة، لا يمكنك إرسال رسائل حالياً.")

    mark_read(current_user["username"], "group")


def render_private_chat(current_user, thread_key, thread_title):
    chats = load_private_chats()
    thread = chats.get(thread_key, [])

    chat_box = st.container(height=380)
    with chat_box:
        if not thread:
            st.caption("لا توجد رسائل بعد في هذه المحادثة الخاصة.")
        for m in thread:
            is_admin_msg = m.get("is_admin")
            with st.chat_message("assistant" if is_admin_msg else "user"):
                tag = " `⭐ الإدارة`" if is_admin_msg else ""
                st.markdown(f"**{m['sender_name']}**{tag}")
                st.write(m["content"])
                st.caption(m["timestamp"].replace("T", " "))

    new_msg = st.chat_input(
        f"رسالة خاصة إلى {thread_title}...",
        key=f"dm_input_{thread_key}_{current_user['username']}",
    )
    if new_msg:
        entry = {
            "sender_username": current_user["username"],
            "sender_name": current_user["name"],
            "is_admin": current_user["role"] == "admin",
            "content": new_msg,
            "timestamp": now_ts(),
        }
        thread.append(entry)
        chats[thread_key] = thread
        save_private_chats(chats)
        mark_read(current_user["username"], f"dm_{thread_key}")
        st.rerun()

    mark_read(current_user["username"], f"dm_{thread_key}")


def render_company_chats_tab(current_user):
    sub_group, sub_private = st.tabs(["🗨️ القروب العام", "✉️ محادثات خاصة"])

    with sub_group:
        render_group_chat(current_user)

    with sub_private:
        users = load_json(USERS_FILE, default_users)
        others = [
            u
            for u in users
            if u["username"] != current_user["username"]
        ]
        if not others:
            st.info("لا يوجد أشخاص آخرين للمحادثة معهم بعد.")
            return

        def _contact_label(u):
            tkey = get_thread_key(current_user["username"], u["username"])
            unread = count_unread_private_thread(current_user["username"], tkey)
            role_tag = " ⭐ (الإدارة)" if u["role"] == "admin" else ""
            badge = f" 🔴 {unread}" if unread else ""
            return f"{u['name']}{role_tag}{badge}"

        name_map = {u["username"]: _contact_label(u) for u in others}
        selected_un = st.selectbox(
            "اختر الشخص للمحادثة الخاصة معه:",
            list(name_map.keys()),
            format_func=lambda x: name_map[x],
            key=f"dm_target_select_{current_user['username']}",
        )
        target_user = next(u for u in others if u["username"] == selected_un)
        thread_key = get_thread_key(current_user["username"], selected_un)
        render_private_chat(current_user, thread_key, target_user["name"])


# ==============================================================================
# محرك صوت سيدرا المزدوج
# ==============================================================================
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        settings = load_json(SETTINGS_FILE, {})
        api_key = settings.get("openai_api_key")
    if api_key and api_key.strip().startswith("sk-"):
        return OpenAI(api_key=api_key.strip())
    return None


def generate_speech_audio(text):
    clean_text = re.sub(r"[^\w\s\u0600-\u06FF،.؟]", "", text).strip()
    if not clean_text:
        return None

    client = get_openai_client()
    if client:
        try:
            response = client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=clean_text[:4096],
            )
            return response.read()
        except Exception as e:
            st.session_state["openai_voice_error"] = str(e)

    try:
        def _worker():
            new_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(new_loop)

            async def _tts_task():
                communicate = edge_tts.Communicate(clean_text, voice="ar-JO-SanaNeural")
                audio_bytes = b""
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        audio_bytes += chunk["data"]
                return audio_bytes

            try:
                return new_loop.run_until_complete(_tts_task())
            finally:
                new_loop.close()

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            return executor.submit(_worker).result()
    except Exception:
        return None


# ==============================================================================
# سيدرا — المساعد الصوتي والوكيل التنفيذي للشركة
# ==============================================================================
def load_sedra_sessions():
    return load_json(SEDRA_CHAT_FILE, {"sessions": []})


def save_sedra_sessions(data):
    save_json(SEDRA_CHAT_FILE, data)


def sedra_handle_command(text, rag, admin_user):
    t = text.strip()

    # 1. إرسال لقروب الشركة
    m = re.search(
        r"(?:ابعث|ارسل|إبعث|أرسل|اكتب|انشر)\s+(?:رسال[ةه]?\s*)?(?:على|في|إلى|الى)?\s*(?:القروب|الجروب|قروب الشركة|جروب الشركة|المجموعة|الشات العام)\s*[:\-]?\s*(.+)",
        t,
    )
    if m:
        content = m.group(1).strip()
        if content:
            msgs = load_group_chat()
            mid = max([mm["id"] for mm in msgs], default=0) + 1
            msgs.append({
                "id": mid,
                "username": admin_user["username"],
                "name": admin_user["name"],
                "role": "admin",
                "content": content,
                "timestamp": now_ts(),
                "pinned": False,
            })
            save_group_chat(msgs)
            return f'حاضر يا مديرنا، تم إرسال رسالتك لقروب الشركة: "{content}"'

    # 2. إرسال رسالة خاصة لموظف
    m = re.search(
        r"(?:ابعث|ارسل|إبعث|أرسل|اكتب)\s+(?:رسال[ةه]?\s*)?(?:ل|إلى|الى)\s*([^:\-]+)[:\-]\s*(.+)",
        t,
    )
    if m:
        emp_name_query = m.group(1).strip()
        content = m.group(2).strip()
        target = find_employee_by_name(emp_name_query)
        if target and content:
            chats = load_private_chats()
            thread_key = get_thread_key(admin_user["username"], target["username"])
            thread = chats.get(thread_key, [])
            thread.append({
                "sender_username": admin_user["username"],
                "sender_name": admin_user["name"],
                "is_admin": True,
                "content": content,
                "timestamp": now_ts(),
            })
            chats[thread_key] = thread
            save_private_chats(chats)
            return f'تم إرسال الرسالة الخاصة للموظف {target["name"]}: "{content}"'
        elif not target:
            return f'لم أجد موظفاً باسم "{emp_name_query}" في النظام.'

    # 3. إسناد مهمة لموظف
    m = re.search(
        r"(?:مهم[ةه]|اعطي مهم[ةه]|اسند مهم[ةه]|كلف|تكليف)\s*(?:ل|إلى|الى)?\s*([^:\-]+)[:\-]\s*(.+)",
        t,
    )
    if not m:
        m = re.search(r"(?:اعطي|اسند|كلف)\s+([^:\-]+)\s+مهم[ةه]\s*[:\-]?\s*(.+)", t)

    if m:
        emp_name_query = m.group(1).strip()
        task_text = m.group(2).strip()
        target = find_employee_by_name(emp_name_query)
        if target and task_text:
            create_task_for_employee(target, task_text, source="عبر أوامر سيدرا")
            return f'تم تكليف الموظف {target["name"]} بالمهمة: "{task_text}"'
        elif not target:
            return f'لم يتم العثور على الموظف "{emp_name_query}".'

    # 4. استفسارات إدارية
    if "مين الموظفين" in t or "قائمة الموظفين" in t:
        users = load_json(USERS_FILE, default_users)
        emps = [
            f"• {u['name']} ({u['job_title']})"
            for u in users
            if u["role"] == "employee"
        ]
        return (
            "الموظفون المسجلون حالياً:\n" + "\n".join(emps)
            if emps
            else "لا يوجد موظفون حالياً."
        )

    if "المهام المعلقة" in t or "مهام قيد التنفيذ" in t:
        tasks = load_json(TASKS_FILE, [])
        pending = [
            f"• {task.get('المهمة', '')} (مسندة لـ: {task.get('username')})"
            for task in tasks
            if task.get("الحالة") != "تم"
        ]
        return (
            "المهام قيد التنفيذ:\n" + "\n".join(pending)
            if pending
            else "جميع المهام مكتملة ولا توجد مهام معلقة."
        )

    return company_scoped_query(rag, t)


def render_sedra_tab(rag, admin_user):
    st.subheader("🎙️ سيدرا — المساعد الصوتي والوكيل التنفيذي ")

    settings = load_json(SETTINGS_FILE, {})
    current_key = os.environ.get("OPENAI_API_KEY") or settings.get(
        "openai_api_key", ""
    )

    col_s1, col_s2 = st.columns([2, 1])
    with col_s1:
        with st.expander(
            "🔑 ضبط مفتاح OpenAI API", expanded=not bool(current_key)
        ):
            new_key = st.text_input(
                "أدخل مفتاح  API Key الخاص بك:",
                value=current_key,
                type="password",
                help="إذا كان المفتاح بدون رصيد، سيعمل الصوت الاحتياطي المجاني تلقائياً دون انقطاع.",
            )
            if st.button("💾 حفظ المفتاح"):
                settings["openai_api_key"] = new_key.strip()
                save_json(SETTINGS_FILE, settings)
                os.environ["OPENAI_API_KEY"] = new_key.strip()
                st.success("✅ تم الحفظ بنجاح!")
                st.rerun()

    with col_s2:
        st.write("")
        if st.button("🔊 تجربة صوت سيدرا الآن", use_container_width=True):
            test_audio = generate_speech_audio(
                "أهلاً بك! صوت سيدرا يعمل بنجاح وجاهزة لتلقي أوامرك صوتياً."
            )
            if test_audio:
                st.session_state["sedra_audio_play"] = test_audio
                st.rerun()

    if st.session_state.get("openai_voice_error"):
        st.warning(
            "⚠️ تنبيه بخصوص مفتاح OpenAI: "
            + st.session_state["openai_voice_error"]
            + " (تم تشغيل الصوت البديل لضمان الرد الصوتي تلقائياً)."
        )
        del st.session_state["openai_voice_error"]

    data = load_sedra_sessions()
    sessions = data.get("sessions", [])
    if not sessions:
        sessions.append({
            "id": f"s1_{int(jordan_now().timestamp())}",
            "title": "محادثة تنفيذية",
            "messages": [{
                "role": "assistant",
                "content": (
                    f"أهلاً بك! أنا سيدرا. اضغط على الدائرة وتكلم معي"
                    " وسأجيبك بالصوت وأدير شؤون الشركة."
                ),
            }],
        })
        data["sessions"] = sessions
        save_sedra_sessions(data)

    col_side, col_main = st.columns([1, 2])

    with col_side:
        st.write("#### 📑 سجل المحادثات:")
        if st.button("➕ محادثة جديدة", key="sedra_new", use_container_width=True):
            new_sess = {
                "id": f"s{len(sessions) + 1}_{int(jordan_now().timestamp())}",
                "title": f"محادثة #{len(sessions) + 1}",
                "messages": [{
                    "role": "assistant",
                    "content": (
                        "بدأت محادثة جديدة ، أنا جاهزة للاستماع ."
                    ),
                }],
            }
            sessions.insert(0, new_sess)
            data["sessions"] = sessions
            save_sedra_sessions(data)
            st.rerun()

        sess_map = {s["id"]: s["title"] for s in sessions}
        selected_id = st.radio(
            "اختر المحادثة:",
            list(sess_map.keys()),
            format_func=lambda x: f"🗨️ {sess_map[x]}",
            key="sedra_radio",
        )

        if st.button(
            "🗑️ حذف المحادثة الحالية",
            key="del_sedra_session",
            use_container_width=True,
        ):
            if len(sessions) > 1:
                data["sessions"] = [s for s in sessions if s["id"] != selected_id]
                save_sedra_sessions(data)
                st.success("تم الحذف.")
                st.rerun()

    with col_main:
        curr = next(
            (s for s in sessions if s["id"] == selected_id),
            sessions[0] if sessions else None,
        )
        if not curr:
            return

        st.write(f"### 📌 {curr['title']}")

        incoming = st.query_params.get("sedra_voice_q")
        if incoming:
            question = incoming
            st.query_params.clear()

            if len(curr["messages"]) <= 1:
                curr["title"] = " ".join(question.split()[:5])

            curr["messages"].append({"role": "user", "content": question})

            with st.spinner("سيدرا تراجع ملفات الشركة وتجهز الرد..."):
                answer = sedra_handle_command(question, rag, admin_user)

            curr["messages"].append({"role": "assistant", "content": answer})
            save_sedra_sessions(data)

            audio_data = generate_speech_audio(answer)
            if audio_data:
                st.session_state["sedra_audio_play"] = audio_data

            st.rerun()

        if st.session_state.get("sedra_audio_play"):
            st.audio(
                st.session_state["sedra_audio_play"],
                format="audio/mp3",
                autoplay=True,
            )
            del st.session_state["sedra_audio_play"]

        chat_box = st.container(height=360)
        with chat_box:
            for m in curr["messages"]:
                with st.chat_message(m["role"]):
                    st.write(m["content"])

        chatgpt_orb_html = """
            <div style="direction: rtl; font-family: system-ui, sans-serif; display: flex; flex-direction: column; align-items: center; justify-content: center; background: radial-gradient(circle at 50% 50%, #1e1b4b 0%, #090d16 100%); border-radius: 24px; padding: 22px; border: 1px solid #312e81; box-shadow: 0 10px 30px rgba(0,0,0,0.6);">
                
                <style>
                    @keyframes pulseGlow {
                        0% { transform: scale(0.97); box-shadow: 0 0 30px rgba(99, 102, 241, 0.5), inset 0 0 20px rgba(236, 72, 153, 0.4); }
                        50% { transform: scale(1.06); box-shadow: 0 0 60px rgba(168, 85, 247, 0.8), inset 0 0 35px rgba(59, 130, 246, 0.6); }
                        100% { transform: scale(0.97); box-shadow: 0 0 30px rgba(99, 102, 241, 0.5), inset 0 0 20px rgba(236, 72, 153, 0.4); }
                    }
                    @keyframes ripple {
                        0% { transform: scale(1); opacity: 0.8; }
                        100% { transform: scale(1.6); opacity: 0; }
                    }
                    .listening-active {
                        animation: pulseGlow 1.4s ease-in-out infinite !important;
                        background: radial-gradient(circle at 35% 35%, #ec4899, #8b5cf6, #3b82f6) !important;
                    }
                </style>

                <div style="position: relative; display: flex; align-items: center; justify-content: center; margin-bottom: 12px;">
                    <div id="rippleRing" style="position: absolute; width: 110px; height: 110px; border-radius: 50%; border: 2px solid #818cf8; opacity: 0; pointer-events: none;"></div>
                    
                    <button id="orbBtn" style="
                        width: 95px; height: 95px; border-radius: 50%; border: none;
                        background: radial-gradient(circle at 35% 35%, #6366f1, #a855f7 60%, #3b82f6);
                        cursor: pointer; position: relative; z-index: 10;
                        box-shadow: 0 0 35px rgba(129, 140, 248, 0.6), inset 0 0 15px rgba(255, 255, 255, 0.4);
                        transition: all 0.3s ease; outline: none;
                        display: flex; align-items: center; justify-content: center;">
                        <span style="font-size: 34px; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.3));">🎙️</span>
                    </button>
                </div>

                <div id="orbStatus" style="color: #e2e8f0; font-size: 14px; font-weight: 600; text-align: center;">
                    اضغط على الدائرة وتكلم مع سيدرا بصوتك
                </div>
                <div id="subStatus" style="color: #94a3b8; font-size: 11px; margin-top: 4px;">
                    الرد الصوتي المباشر مفعل بالكامل وموصول بملفات الشركة
                </div>
            </div>

            <script>
                const orb = document.getElementById('orbBtn');
                const status = document.getElementById('orbStatus');
                const ring = document.getElementById('rippleRing');
                let isRec = false;
                let recog = null;

                const win = window.parent || window;
                const SpeechAPI = win.SpeechRecognition || win.webkitSpeechRecognition || window.SpeechRecognition || window.webkitSpeechRecognition;

                orb.onclick = function() {
                    if (!SpeechAPI) {
                        status.innerText = 'يرجى استخدام متصفح Google Chrome للتحدث الصوتي.';
                        return;
                    }

                    if (isRec) {
                        if (recog) recog.stop();
                        return;
                    }

                    recog = new SpeechAPI();
                    recog.lang = 'ar-SA';
                    recog.interimResults = false;

                    recog.onstart = function() {
                        isRec = true;
                        orb.classList.add('listening-active');
                        ring.style.animation = 'ripple 1.5s linear infinite';
                        status.innerText = 'سيدرا تستمع إليك... تفضل بالحديث';
                        status.style.color = '#38bdf8';
                    };

                    recog.onresult = function(e) {
                        const spoken = e.results[0][0].transcript;
                        status.innerText = '⚡ فهمت صوتك: "' + spoken + '" - جاري تجهيز الرد...';
                        orb.classList.remove('listening-active');
                        ring.style.animation = 'none';

                        const url = new URL(win.location.href);
                        url.searchParams.set('sedra_voice_q', spoken);
                        win.location.href = url.toString();
                    };

                    recog.onerror = function() {
                        isRec = false;
                        orb.classList.remove('listening-active');
                        ring.style.animation = 'none';
                        status.innerText = 'تأكد من السماح بالمايكروفون ثم اضغط وتكلم مجدداً.';
                        status.style.color = '#f87171';
                    };

                    recog.onend = function() {
                        isRec = false;
                        orb.classList.remove('listening-active');
                        ring.style.animation = 'none';
                    };

                    recog.start();
                };
            </script>
            """
        st.components.v1.html(chatgpt_orb_html, height=225)

        typed = st.chat_input("أو اكتب أمرك هنا...", key="sedra_text_input")
        if typed:
            if len(curr["messages"]) <= 1:
                curr["title"] = " ".join(typed.split()[:5])

            curr["messages"].append({"role": "user", "content": typed})

            with st.spinner("سيدرا تفكر وتجهز الرد..."):
                answer = sedra_handle_command(typed, rag, admin_user)

            curr["messages"].append({"role": "assistant", "content": answer})
            save_sedra_sessions(data)

            audio_data = generate_speech_audio(answer)
            if audio_data:
                st.session_state["sedra_audio_play"] = audio_data

            st.rerun()


# ==============================================================================
# إشعارات المهام
# ==============================================================================
def check_new_task_notifications_for_employee(username, my_tasks_all):
    lr = load_last_read()
    seen_ids = set(lr.get(username, {}).get("seen_task_ids", []))
    current_ids = set(t["id"] for t in my_tasks_all)
    new_ids = current_ids - seen_ids
    for tid in new_ids:
        task = next(t for t in my_tasks_all if t["id"] == tid)
        st.toast(f"🔔 مهمة جديدة: {task.get('المهمة', '')}", icon="📌")
    lr.setdefault(username, {})
    lr[username]["seen_task_ids"] = list(current_ids)
    save_last_read(lr)


def check_new_task_completions_for_admin(admin_username):
    lr = load_last_read()
    seen_ids = set(lr.get(admin_username, {}).get("seen_completed_ids", []))
    tasks = load_json(TASKS_FILE, [])
    completed_tasks = [t for t in tasks if t.get("completed_at")]
    current_ids = set(t["id"] for t in completed_tasks)
    new_ids = current_ids - seen_ids
    users = load_json(USERS_FILE, default_users)
    for tid in new_ids:
        task = next(t for t in completed_tasks if t["id"] == tid)
        emp = next(
            (
                u
                for u in users
                if u["username"] == task["username"]
            ),
            None,
        )
        emp_name = emp["name"] if emp else task["username"]
        st.toast(f"✅ {emp_name} أنجز مهمة: {task.get('المهمة', '')}", icon="✅")
    lr.setdefault(admin_username, {})
    lr[admin_username]["seen_completed_ids"] = list(current_ids)
    save_last_read(lr)


def send_employee_email(
    to_email, employee_name, username, pin, job_title, action="update"
):
    cfg = load_json(EMAIL_CONFIG_FILE, {})
    sender = cfg.get("sender_email", "").strip()
    password = cfg.get("sender_password", "").strip()
    server = cfg.get("smtp_server", "smtp.gmail.com").strip()
    port = int(cfg.get("smtp_port", 587))

    if not sender or not password:
        return False, "⚠️ لم يتم ضبط بريد الإدارة في الإعدادات."
    if not to_email or "@" not in to_email:
        return False, "⚠️ البريد الإلكتروني للموظف غير صالح."

    try:
        msg = MIMEMultipart()
        msg["From"] = sender
        msg["To"] = to_email

        if action == "create":
            msg["Subject"] = "بيانات حسابك الجديد في النظام المركزي"
            body = f"""مرحباً {employee_name}،
تم إنشاء حساب عمل جديد لك في النظام المركزي:
- اسم المستخدم: {username}
- رمز الدخول (PIN): {pin}
- المسمى الوظيفي: {job_title}
مع تحيات الإدارة العامة."""
        else:
            msg["Subject"] = "تحديث بيانات حسابك في النظام المركزي"
            body = f"""مرحباً {employee_name}،
تم تحديث بيانات حسابك:
- اسم المستخدم: {username}
- رمز الدخول (PIN): {pin}
- المسمى الوظيفي: {job_title}"""

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

# التأكد من إنشاء الملفات الأولية إذا لم تكن موجودة
if not os.path.exists(USERS_FILE):
    save_json(USERS_FILE, default_users)
if not os.path.exists(TASKS_FILE):
    save_json(TASKS_FILE, [])
if not os.path.exists(CALLS_FILE):
    save_json(CALLS_FILE, [])
if not os.path.exists(CHATS_FILE):
    save_json(CHATS_FILE, {})

# تحميل أحدث البيانات من القرص دائماً لضمان التزامن الفوري عبر جميع الجلسات
st.session_state.users_db = load_json(USERS_FILE, default_users)
st.session_state.tasks_db = load_json(TASKS_FILE, [])
st.session_state.calls_db = load_json(CALLS_FILE, [])
st.session_state.chats_db = load_json(CHATS_FILE, {})
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

# تسجيل الدخول
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
                        and str(x["pin"]).strip() == p.strip()
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
        st.info("💡 ادخل بحسابك .")
    st.stop()

current_user = next(
    (u for u in st.session_state.users_db if u["username"] == st.session_state.logged_user),
    None
)
if not current_user:
    st.session_state.logged_user = None
    st.rerun()

# الشريط الجانبي
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
        st.subheader("⚡ الدخول لحساب الموظف:")
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
    st.title("🛡️ لوحة تحكم الإدارة العامة و المتابعة  ")

    check_new_task_completions_for_admin(current_user["username"])

    group_unread = count_unread_group(current_user["username"])
    private_unread_total = total_private_unread(current_user["username"])
    chats_badge = group_unread + private_unread_total
    chats_label = "💬 تواصل مع الموظفين " + (
        f" 🔴{chats_badge}" if chats_badge else ""
    )

    (
        tab_faq,
        tab_mychat,
        tab_chats,
        tab_sedra,
        tab_recordings,
        tab_mgmt,
        tab_email,
        tab_docs,
    ) = st.tabs([
        "💡 الأسئلة المتكرر",
        "💬 اسئلتي",
        chats_label,
        "🎙️ سيدرا",
        "🎧 سجل المكالمات ",
        "👥 إدارة الموظفين والمهام",
        "⚙️ إعدادات البريد الإلكتروني",
        "📁 ملفات الشركة ",
    ])

    with tab_faq:
        render_smart_faq(rag, allow_generate=True)

    with tab_mychat:
        render_chat_tab(rag, current_user["username"])

    with tab_chats:
        render_company_chats_tab(current_user)

    with tab_sedra:
        render_sedra_tab(rag, current_user)

    with tab_recordings:
        st.subheader("🎧 سجل المكالمات")
        calls = load_json(CALLS_FILE, [])
        if not calls:
            st.info("لا توجد مكالمات مسجلة بعد.")

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
                if audio_file and os.path.exists(audio_file):
                    with open(audio_file, "rb") as af:
                        st.audio(af.read(), format="audio/webm")

                if call.get("employee_note"):
                    st.info(f"📝 **رد الموظف:**\n{call.get('employee_note')}")

                if st.button("🗑️ حذف هذه المكالمة", key=f"del_c_{call_id}"):
                    calls = [c for c in calls if c.get("id") != call_id]
                    save_json(CALLS_FILE, calls)
                    st.session_state.calls_db = calls
                    st.success("تم الحذف.")
                    st.rerun()

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
                        current_users_list = load_json(USERS_FILE, default_users)
                        if any(
                            u["username"].lower() == clean_un
                            for u in current_users_list
                        ):
                            st.error("اسم المستخدم مسجل مسبقاً! اختر اسماً آخر.")
                        else:
                            new_id = (
                                max([u["id"] for u in current_users_list], default=0)
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
                            current_users_list.append(new_emp)
                            save_json(USERS_FILE, current_users_list)
                            st.session_state.users_db = current_users_list
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
                            st.rerun()

        st.markdown("---")
        st.subheader("سجلات وبطاقات الموظفين:")
        current_users_list = load_json(USERS_FILE, default_users)
        for emp in current_users_list:
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
                        if st.button("💾 حفظ البيانات", key=f"save_email_{emp_id}"):
                            emp["username"] = u_val.strip()
                            emp["pin"] = p_val.strip()
                            emp["email"] = e_val.strip()
                            emp["job_title"] = j_val.strip()
                            save_json(USERS_FILE, current_users_list)
                            st.session_state.users_db = current_users_list
                            st.success("تم الحفظ!")
                            st.rerun()

                    with btn_col2:
                        if st.button("📧 إرسال إيميل", key=f"send_only_{emp_id}"):
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

                    with btn_col3:
                        if st.button("🗑️ حذف الموظف", key=f"del_{emp_id}"):
                            current_users_list = [
                                u for u in current_users_list if u["id"] != emp_id
                            ]
                            save_json(USERS_FILE, current_users_list)
                            st.session_state.users_db = current_users_list
                            st.warning(f"تم حذف {emp_name}.")
                            st.rerun()

                    st.markdown("---")
                    task_c1, task_c2 = st.columns(2)
                    with task_c1:
                        task_text = st.text_input(
                            "إسناد مهمة جديدة:", key=f"task_in_{emp_id}"
                        )
                    with task_c2:
                        st.write("")
                        st.write("")
                        if st.button("➕ إرسال المهمة", key=f"btn_task_{emp_id}"):
                            if task_text.strip():
                                create_task_for_employee(
                                    emp, task_text.strip(), source="من لوحة الإدارة"
                                )
                                st.success("تم إرسال المهمة!")
                                st.rerun()

    with tab_email:
        st.subheader("⚙️ إعدادات البريد الإلكتروني (SMTP)")
        cfg = load_json(EMAIL_CONFIG_FILE, st.session_state.email_config)
        with st.form("smtp_config_form"):
            s_email = st.text_input(
                "بريدك الإلكتروني (Gmail):",
                value=cfg.get("sender_email", ""),
            )
            s_pass = st.text_input(
                "كلمة مرور التطبيقات (App Password):",
                type="password",
                value=cfg.get("sender_password", ""),
            )
            s_server = st.text_input(
                "خادم SMTP:",
                value=cfg.get("smtp_server", "smtp.gmail.com"),
            )
            s_port = st.number_input(
                "منفذ SMTP:",
                value=int(cfg.get("smtp_port", 587)),
            )
            if st.form_submit_button("💾 حفظ إعدادات البريد"):
                st.session_state.email_config = {
                    "sender_email": s_email.strip(),
                    "sender_password": s_pass.strip(),
                    "smtp_server": s_server.strip(),
                    "smtp_port": s_port,
                }
                save_json(EMAIL_CONFIG_FILE, st.session_state.email_config)
                st.success("تم الحفظ بنجاح!")

    with tab_docs:
        st.subheader("📁 ملفات الشركة")
        uploaded_files = st.file_uploader(
            "رفع ملفات جديدة:", type=["pdf", "txt", "docx"], accept_multiple_files=True
        )
        if uploaded_files:
            if st.button("🚀 بدء الحفظ وتحديث "):
                with st.spinner("جاري حفظ الملفات وتحديث..."):
                    for uf in uploaded_files:
                        target_path = os.path.join(DOCS_DIR, uf.name)
                        with open(target_path, "wb") as f_out:
                            f_out.write(uf.getbuffer())
                    sync_res = rag.sync_documents()
                    st.success("تم الحفظ وتحديث !")
                    st.rerun()

        st.markdown("---")
        st.write("#### 📑 ملفات الشركة المسجلة (يمكنك حذف أي ملف):")
        physical_files = get_physical_documents()
        if physical_files:
            for fname, finfo in list(physical_files.items()):
                col_f1, col_f2 = st.columns([3, 1])
                with col_f1:
                    st.write(f"- 📄 **{fname}** ({finfo['size']}) `[{finfo['folder']}]`")
                with col_f2:
                    safe_key = f"del_f_{re.sub(r'[^a-zA-Z0-9_]', '_', fname)}_{re.sub(r'[^a-zA-Z0-9_]', '_', finfo['folder'])}"
                    if st.button("🗑️ حذف الملف", key=safe_key):
                        try:
                            if os.path.exists(finfo["path"]):
                                os.remove(finfo["path"])
                            rag.sync_documents()
                            st.success(f"تم حذف {fname} وتحديث قاعدة المعرفة بنجاح!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"تعذر حذف الملف: {str(e)}")
        else:
            st.info("لا توجد ملفات.")

# ==============================================================================
#                      2. واجهة الموظف (EMPLOYEE DASHBOARD)
# ==============================================================================
else:
    st.title(f"💼 واجهة عمل الموظف: {current_user['name']}")

    tasks_current = load_json(TASKS_FILE, [])
    my_tasks_all = [
        t for t in tasks_current if t["username"] == current_user["username"]
    ]
    pending_count = sum(
        1 for t in my_tasks_all if t.get("الحالة", t.get("status")) != "تم"
    )
    check_new_task_notifications_for_employee(
        current_user["username"], my_tasks_all
    )

    group_unread = count_unread_group(current_user["username"])
    private_unread = total_private_unread(current_user["username"])
    chats_badge = group_unread + private_unread
    chats_label = "💬 قروب الشركة" + (
        f" 🔴{chats_badge}" if chats_badge else ""
    )
    tasks_label = "📌 مهامي وتنبيهات " + (
        f" 🔴{pending_count}" if pending_count else ""
    )

    t0, t_chats, t1, t2, t3 = st.tabs([
        "💡 الأسئلة الشائعة",
        chats_label,
        "📞 المكالمات المحولة إليّ ",
        tasks_label,
        "💬 الشات الذكي لك",
    ])

    with t0:
        render_smart_faq(rag, allow_generate=False)

    with t_chats:
        render_company_chats_tab(current_user)

    with t1:
        calls = load_json(CALLS_FILE, [])
        transferred = [
            c
            for c in calls
            if c.get("assigned_to") == current_user["username"]
            and c.get("status") == "محولة للموظف"
        ]
        if not transferred:
            st.info("🟢 لا توجد مكالمات محولة إليك.")
        for c in transferred:
            cid = c.get("id")
            st.error(f"🚨 مكالمة من: {c.get('customer_name')}")
            st.write(f"**سؤال العميل:** {c.get('inquiry')}")
            st.info(f"**الحل المستخرج:**\n{c.get('ai_initial_answer')}")

            if st.button("✅ تم الرد وحل المشكلة", key=f"done_c_{cid}"):
                c["status"] = "تم الحل بواسطة الموظف"
                save_json(CALLS_FILE, calls)
                st.session_state.calls_db = calls
                st.success("تم الحل!")
                st.rerun()

    with t2:
        if not my_tasks_all:
            st.info("لا توجد مهام مسندة.")
        for t in my_tasks_all:
            t_id = t["id"]
            task_name = t.get("المهمة", t.get("task", ""))
            task_status = t.get("الحالة", t.get("status", "قيد التنفيذ"))
            st.write(f"📌 **{task_name}** - الحالة: `{task_status}`")
            if task_status != "تم":
                if st.button("✅ تأكيد الإنجاز", key=f"finish_t_{t_id}"):
                    tasks_current = load_json(TASKS_FILE, [])
                    for tt in tasks_current:
                        if tt["id"] == t_id:
                            tt["الحالة"] = "تم"
                            tt["completed_at"] = jordan_now().strftime("%Y-%m-%d %H:%M")
                    save_json(TASKS_FILE, tasks_current)
                    st.session_state.tasks_db = tasks_current
                    st.success("تم التأكيد!")
                    st.rerun()

    with t3:
        render_chat_tab(rag, current_user["username"])
