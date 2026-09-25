import io
import json
import os
import time
from call_center import CallCenterEngine
from flask import Flask, jsonify, render_template_string, request, send_file, send_from_directory
from rag_engine import EnterpriseRAG

app = Flask(__name__)
rag = EnterpriseRAG()
rag.sync_documents()
call_center = CallCenterEngine(rag)

RECORDINGS_DIR = "recordings"
CALLS_FILE = "calls.json"
os.makedirs(RECORDINGS_DIR, exist_ok=True)

HTML_INTERFACE = """
<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>المكالمة الصوتية الذكية - ChatGPT Voice Mode</title>
    <link href="https://fonts.googleapis.com/css2?family=Alexandria:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * { box-sizing: border-box; font-family: 'Alexandria', sans-serif; margin: 0; padding: 0; }
        body { background: #000000; color: #ffffff; height: 100vh; display: flex; justify-content: center; align-items: center; overflow: hidden; }
        .voice-interface { width: 100vw; height: 100vh; max-width: 440px; display: flex; flex-direction: column; justify-content: space-between; align-items: center; padding: 35px 24px; position: relative; }
        .top-bar { display: flex; flex-direction: column; align-items: center; gap: 6px; z-index: 10; }
        .brand-title { font-size: 15px; font-weight: 600; color: #94a3b8; }
        .call-timer { font-size: 14px; color: #38bdf8; background: rgba(56, 189, 248, 0.1); padding: 2px 14px; border-radius: 20px; border: 1px solid rgba(56, 189, 248, 0.2); }
        .orb-wrapper { position: relative; width: 230px; height: 230px; display: flex; justify-content: center; align-items: center; }
        .orb { width: 180px; height: 180px; border-radius: 50%; background: radial-gradient(circle at 35% 35%, #38bdf8 0%, #2563eb 55%, #0f172a 100%); box-shadow: 0 0 70px rgba(56, 189, 248, 0.45); transition: all 0.5s ease; animation: morph 6s ease-in-out infinite, breathe 3s ease-in-out infinite; }
        .orb.listening { background: radial-gradient(circle at 35% 35%, #34d399 0%, #059669 55%, #064e3b 100%); box-shadow: 0 0 90px rgba(52, 211, 153, 0.7); animation: morph 3s ease-in-out infinite, pulse-listen 1.2s infinite; }
        .orb.thinking { background: radial-gradient(circle at 35% 35%, #c084fc 0%, #7c3aed 55%, #3b0764 100%); box-shadow: 0 0 90px rgba(192, 132, 252, 0.75); animation: spin-morph 2s linear infinite; }
        .orb.speaking { background: radial-gradient(circle at 35% 35%, #60a5fa 0%, #3b82f6 55%, #1d4ed8 100%); box-shadow: 0 0 100px rgba(96, 165, 250, 0.8); animation: morph 2s ease-in-out infinite, speak-wave 0.8s infinite alternate; }
        @keyframes morph { 0%, 100% { border-radius: 44% 56% 65% 35% / 48% 45% 55% 52%; } 50% { border-radius: 58% 42% 35% 65% / 55% 58% 42% 45%; } }
        @keyframes breathe { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.04); } }
        @keyframes pulse-listen { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.1); } }
        @keyframes speak-wave { 0% { transform: scale(1); } 100% { transform: scale(1.15); filter: brightness(1.2); } }
        @keyframes spin-morph { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }
        .status-pill { font-size: 13px; color: #cbd5e1; background: rgba(30, 41, 59, 0.6); padding: 6px 16px; border-radius: 30px; border: 1px solid rgba(51, 65, 85, 0.5); margin-top: 18px; text-align: center; }
        .live-caption { width: 100%; min-height: 65px; font-size: 13.5px; line-height: 1.6; color: #e2e8f0; text-align: center; padding: 10px 14px; background: rgba(15, 23, 42, 0.6); border-radius: 14px; border: 1px solid rgba(51, 65, 85, 0.4); margin-bottom: 20px; max-height: 90px; overflow-y: auto; }
        .btn-ctrl { width: 62px; height: 62px; border-radius: 50%; border: none; cursor: pointer; display: flex; justify-content: center; align-items: center; font-size: 24px; transition: all 0.2s; }
        .btn-end { background: #ef4444; color: white; box-shadow: 0 4px 20px rgba(239, 68, 68, 0.4); }
        .btn-end:hover { transform: scale(1.08); }
        .btn-start { width: 75px; height: 75px; font-size: 32px; background: #10b981; color: white; box-shadow: 0 6px 25px rgba(16, 185, 129, 0.45); }
        .pre-call-card { width: 100%; background: rgba(15, 23, 42, 0.8); border: 1px solid #1e293b; border-radius: 20px; padding: 20px; margin-bottom: 20px; text-align: right; }
        .pre-call-card input { width: 100%; background: #020617; border: 1px solid #334155; color: white; padding: 10px 12px; border-radius: 10px; margin-top: 6px; margin-bottom: 12px; font-size: 14px; }
    </style>
</head>
<body>

<div class="voice-interface">
    <div id="preCallView" style="width: 100%; text-align: center;">
        <div class="brand-title" style="font-size: 20px; color: #fff; margin-bottom: 6px;">المكالمة الصوتية التفاعلية المباشرة</div>
        <div style="font-size: 12.5px; color: #94a3b8; margin-bottom: 25px;">تسجيل صوتي كامل وموثق (صوت المتصل + صوت الذكاء الاصطناعي)</div>

        <div class="pre-call-card">
            <label style="font-size: 12px; color: #94a3b8;">اسم المتصل:</label>
            <input type="text" id="custName" value="أحمد محمد">
            <label style="font-size: 12px; color: #94a3b8;">رقم الهاتف:</label>
            <input type="text" id="custPhone" value="0791234567">
        </div>

        <button class="btn-ctrl btn-start" onclick="startCall()" style="margin: 0 auto;">📞</button>
        <div style="font-size: 13px; color: #94a3b8; margin-top: 12px;">اضغط لبدء المكالمة والتسجيل الصوتي</div>
    </div>

    <div id="activeCallView" style="width: 100%; display: none; flex-direction: column; justify-content: space-between; align-items: center; height: 100%;">
        <div class="top-bar">
            <div class="brand-title" id="callAgentTitle">المساعد الذكي (جاري التسجيل 🔴)</div>
            <div class="call-timer" id="callTimer">00:00</div>
        </div>

        <div class="orb-wrapper">
            <div class="orb" id="voiceOrb"></div>
        </div>

        <div style="width: 100%;">
            <div class="status-pill" id="statusPill">متصل الآن...</div>
            <div class="live-caption" id="captionBox">أهلاً بك، تفضل بطرح استفسارك بخصوص محتوى الملفات.</div>
        </div>

        <div class="controls">
            <button class="btn-ctrl btn-end" onclick="endCall()">✕</button>
        </div>
    </div>
</div>

<audio id="audioPlayer" crossorigin="anonymous"></audio>

<script>
let isCallActive = false;
let callStartTime = null;
let timerInterval = null;
let recognition = null;
let isAiSpeaking = false;
let currentCallId = null;

let audioCtx = null;
let mediaRecorder = null;
let recordedChunks = [];
let micStream = null;

if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    recognition = new SpeechRecognition();
    recognition.lang = 'ar-JO';
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = function() {
        if (!isAiSpeaking && isCallActive) {
            document.getElementById('voiceOrb').className = 'orb listening';
            document.getElementById('statusPill').innerText = '🎙️ أستمع لصوتك الآن... تفضل بالسؤال';
        }
    };

    recognition.onresult = function(event) {
        const userText = event.results[0][0].transcript;
        document.getElementById('captionBox').innerHTML = `<span style="color:#34d399;">أنت:</span> ${userText}`;
        sendToAi(userText);
    };

    recognition.onerror = function() {
        if (isCallActive && !isAiSpeaking) setTimeout(startListening, 1000);
    };

    recognition.onend = function() {
        if (isCallActive && !isAiSpeaking) startListening();
    };
}

function startListening() {
    if (isCallActive && !isAiSpeaking && recognition) {
        try { recognition.start(); } catch(e) {}
    }
}

function stopListening() {
    if (recognition) {
        try { recognition.stop(); } catch(e) {}
    }
}

async function startCall() {
    currentCallId = "call_" + Date.now();
    isCallActive = true;
    recordedChunks = [];

    try {
        micStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        
        const micSource = audioCtx.createMediaStreamSource(micStream);
        const mixedDest = audioCtx.createMediaStreamDestination();
        
        micSource.connect(mixedDest);

        const player = document.getElementById('audioPlayer');
        const playerSource = audioCtx.createMediaElementSource(player);
        playerSource.connect(mixedDest);
        playerSource.connect(audioCtx.destination);

        mediaRecorder = new MediaRecorder(mixedDest.stream);
        mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
        mediaRecorder.start(1000);
    } catch(err) {
        if (micStream) {
            mediaRecorder = new MediaRecorder(micStream);
            mediaRecorder.ondataavailable = (e) => { if (e.data.size > 0) recordedChunks.push(e.data); };
            mediaRecorder.start(1000);
        }
    }

    document.getElementById('preCallView').style.display = 'none';
    document.getElementById('activeCallView').style.display = 'flex';
    
    callStartTime = new Date();
    timerInterval = setInterval(updateTimer, 1000);

    speakText("أهلاً بك، معك المساعد الذكي المفهرس للملفات. تفضل بطرح سؤالك وسأجيبك فوراً.");
}

function updateTimer() {
    const now = new Date();
    const diff = Math.floor((now - callStartTime) / 1000);
    const m = String(Math.floor(diff / 60)).padStart(2, '0');
    const s = String(diff % 60).padStart(2, '0');
    document.getElementById('callTimer').innerText = `${m}:${s}`;
}

function sendToAi(userText) {
    stopListening();
    document.getElementById('voiceOrb').className = 'orb thinking';
    document.getElementById('statusPill').innerHTML = '<span style="color:#c084fc;">⏳ جاري استخراج الإجابة من الملفات...</span>';

    const name = document.getElementById('custName').value;
    const phone = document.getElementById('custPhone').value;

    fetch('/api/call_interaction', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ call_id: currentCallId, name: name, phone: phone, inquiry: userText })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('captionBox').innerHTML = `<span style="color:#60a5fa;">الذكاء الاصطناعي:</span> ${data.message}`;
        if (data.action === 'transferred') {
            document.getElementById('callAgentTitle').innerText = `تم التحويل إلى: ${data.assigned_agent}`;
            document.getElementById('statusPill').innerText = 'تم تحويل المكالمة للموظف البشري';
        }
        speakText(data.message);
    })
    .catch(() => {
        speakText("المعلومة واضحة في نصوص الملفات، تفضل بطرح استفسارك.");
    });
}

function speakText(text) {
    isAiSpeaking = true;
    stopListening();
    document.getElementById('voiceOrb').className = 'orb speaking';
    document.getElementById('statusPill').innerText = '🔊 يجيبك الذكاء الاصطناعي الآن...';

    const audio = document.getElementById('audioPlayer');
    audio.src = `/api/tts?text=${encodeURIComponent(text)}`;
    audio.play();

    audio.onended = function() {
        isAiSpeaking = false;
        document.getElementById('voiceOrb').className = 'orb';
        document.getElementById('statusPill').innerText = '🎙️ أستمع لك... تفضل بالسؤال';
        startListening();
    };
}

function endCall() {
    isCallActive = false;
    clearInterval(timerInterval);
    stopListening();
    document.getElementById('audioPlayer').pause();
    
    if (mediaRecorder && mediaRecorder.state !== 'inactive') {
        mediaRecorder.stop();
        setTimeout(uploadRecording, 500);
    }
    if (micStream) {
        micStream.getTracks().forEach(t => t.stop());
    }

    document.getElementById('activeCallView').style.display = 'none';
    document.getElementById('preCallView').style.display = 'block';
    alert("تم إنهاء المكالمة بنجاح، وحفظ التسجيل الصوتي الحقيقي في سجل المدير!");
}

function uploadRecording() {
    if (recordedChunks.length === 0) return;
    const blob = new Blob(recordedChunks, { type: 'audio/webm' });
    const formData = new FormData();
    formData.append('audio_data', blob, `${currentCallId}.webm`);
    formData.append('call_id', currentCallId);

    fetch('/api/save_call_audio', {
        method: 'POST',
        body: formData
    });
}
</script>

</body>
</html>
"""


@app.route("/")
def index():
  return render_template_string(HTML_INTERFACE)


@app.route("/api/call_interaction", methods=["POST"])
def call_interaction():
  data = request.get_json()
  call_id = data.get("call_id", f"call_{int(time.time())}")
  name = data.get("name", "عميل")
  phone = data.get("phone", "0790000000")
  inquiry = data.get("inquiry", "")

  answer = rag.query(inquiry)

  action = "answered"
  assigned_agent = None
  if any(
      kw in inquiry
      for kw in [
          "تحويل",
          "موظف",
          "إنسان",
          "مشكلة معقدة",
          "دعم فني",
          "أحمد",
      ]
  ):
    action = "transferred"
    assigned_agent = "أحمد علي (دعم فني)"

  calls = []
  if os.path.exists(CALLS_FILE):
    try:
      with open(CALLS_FILE, "r", encoding="utf-8") as f:
        calls = json.load(f)
    except Exception:
      calls = []

  audio_path = os.path.join(RECORDINGS_DIR, f"{call_id}.webm")
  call_record = next((c for c in calls if c.get("id") == call_id), None)

  if not call_record:
    new_call = {
        "id": call_id,
        "customer_name": name,
        "customer_phone": phone,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "inquiry": inquiry,
        "ai_initial_answer": answer,
        "handled_by": (
            "الذكاء الاصطناعي" if action == "answered" else "محولة للموظف"
        ),
        "status": (
            "مكتملة" if action == "answered" else "محولة للموظف"
        ),
        "assigned_to": "ahmad" if action == "transferred" else None,
        "audio_file": audio_path,
        "summary": (
            f"استفسار: {inquiry[:60]} - تم الشرح والإجابة المباشرة من نصوص"
            " الملفات."
        ),
    }
    calls.append(new_call)
  else:
    call_record["inquiry"] += f" | {inquiry}"
    call_record["ai_initial_answer"] += f"\n---\n{answer}"
    call_record["audio_file"] = audio_path
    if action == "transferred":
      call_record["status"] = "محولة للموظف"
      call_record["assigned_to"] = "ahmad"
      call_record["handled_by"] = "محولة للموظف"

  with open(CALLS_FILE, "w", encoding="utf-8") as f:
    json.dump(calls, f, ensure_ascii=False, indent=2)

  return jsonify({
      "action": action,
      "message": answer,
      "assigned_agent": assigned_agent,
  })


@app.route("/api/save_call_audio", methods=["POST"])
def save_call_audio():
  if "audio_data" not in request.files:
    return jsonify({"status": "no_file"}), 400

  file = request.files["audio_data"]
  call_id = request.form.get("call_id", f"call_{int(time.time())}")
  filepath = os.path.join(RECORDINGS_DIR, f"{call_id}.webm")
  file.save(filepath)

  if os.path.exists(CALLS_FILE):
    try:
      with open(CALLS_FILE, "r", encoding="utf-8") as f:
        calls = json.load(f)
      for c in calls:
        if c.get("id") == call_id:
          c["audio_file"] = filepath
      with open(CALLS_FILE, "w", encoding="utf-8") as f:
        json.dump(calls, f, ensure_ascii=False, indent=2)
    except Exception:
      pass

  return jsonify({"status": "saved", "path": filepath})


@app.route("/api/tts")
def tts_stream():
  text = request.args.get("text", "")
  audio_bytes = call_center.text_to_speech(text)
  if audio_bytes:
    return send_file(io.BytesIO(audio_bytes), mimetype="audio/mp3")
  return ("Error", 500)


if __name__ == "__main__":
  print("\n" + "=" * 60)
  print("🚀 خادم المكالمة الصوتية المسجلة بالكامل يعمل الآن!")
  print("👉 افتح الرابط في المتصفح: http://localhost:5001")
  print("=" * 60 + "\n")
  app.run(port=5001, debug=False)