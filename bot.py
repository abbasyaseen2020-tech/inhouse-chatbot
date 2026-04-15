# -*- coding: utf-8 -*-
"""
In-House Real Estate Marketing - AI Sales Agent
================================================
Facebook Messenger + Facebook Comments
Auto-detects OpenAI or Anthropic API
"""

import os
import re
import logging
import time
import random
import requests
from datetime import datetime
from collections import defaultdict
from flask import Flask, request, jsonify

# ============================================
# CONFIGURATION
# ============================================
app = Flask(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PAGE_ACCESS_TOKEN = os.getenv("PAGE_ACCESS_TOKEN", "")
VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "inhouse_bot_verify_2026")
FB_APP_ID = os.getenv("FB_APP_ID", "")
FB_APP_SECRET = os.getenv("FB_APP_SECRET", "")
PAGE_ID = os.getenv("PAGE_ID", "107119435824632")

# Telegram notification — group + admins
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_IDS = os.getenv("TELEGRAM_CHAT_IDS", "-5001305547").split(",")

# Google Sheets webhook (Apps Script URL) for persistent lead storage
SHEETS_WEBHOOK_URL = os.getenv("SHEETS_WEBHOOK_URL", "")

# ---- AI Provider Auto-Detection ----
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

if OPENAI_API_KEY:
    AI_PROVIDER = "openai"
    _DEFAULT_MODEL = "gpt-4.1-mini"
elif ANTHROPIC_API_KEY:
    AI_PROVIDER = "anthropic"
    _DEFAULT_MODEL = "claude-sonnet-4-20250514"
else:
    AI_PROVIDER = "fallback"
    _DEFAULT_MODEL = "none"

AI_MODEL = os.getenv("AI_MODEL", _DEFAULT_MODEL)

GRAPH_API_URL = "https://graph.facebook.com/v19.0"
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2025-01-01"

from knowledge_base import (
    get_system_prompt, COMMENT_KEYWORDS, COMPANY_INFO, WELCOME_DM,
    format_projects_for_search,
    EMOJI_POSITIVE, EMOJI_RESPONSES, THANK_WORDS
)

# ============================================
# DATA STORES (in-memory)
# ============================================
leads_db = []
conversation_history = defaultdict(list)
user_data = {}
phone_requested = {}
MAX_HISTORY = 20

# Track user activity for auto follow-up
user_last_activity = {}      # user_id -> timestamp of last message
followup_sent = {}            # user_id -> list of follow-up stages sent
FOLLOWUP_STAGE_1_HOURS = 24   # 1st follow-up after 24h of silence
FOLLOWUP_STAGE_2_HOURS = 72   # 2nd follow-up after 72h of silence

# ============================================
# ANALYTICS — lightweight in-memory counters
# ============================================
analytics = {
    "messages_received": 0,         # total incoming messages
    "messages_sent": 0,             # total outgoing messages (AI + manual)
    "comments_received": 0,         # incoming FB comments
    "comments_replied": 0,          # comments we replied to
    "leads_captured": 0,            # phone+name pairs captured
    "appointments_booked": 0,       # confirmed appointments
    "followups_sent": 0,            # auto follow-ups sent
    "broadcasts_sent": 0,           # broadcast messages delivered
    "new_conversations": 0,         # distinct new users started
    "daily": {},                    # date_str -> {counter_name: int}
    "started_at": datetime.now().isoformat(),
}


def track(metric, n=1):
    """Increment analytics counter — daily + lifetime."""
    try:
        analytics[metric] = analytics.get(metric, 0) + n
        today = datetime.now().strftime("%Y-%m-%d")
        if today not in analytics["daily"]:
            analytics["daily"][today] = {}
        analytics["daily"][today][metric] = analytics["daily"][today].get(metric, 0) + n
        # Keep last 60 days only
        if len(analytics["daily"]) > 60:
            old = sorted(analytics["daily"].keys())[:-60]
            for k in old:
                analytics["daily"].pop(k, None)
    except Exception as e:
        logger.warning(f"Track metric failed: {e}")

# ============================================
# DUPLICATE MESSAGE PREVENTION
# ============================================
_processed_messages = {}
_processed_comments = {}
DEDUP_TTL = 300


def is_duplicate_message(msg_id):
    now = time.time()
    for k in [k for k, v in _processed_messages.items() if now - v > DEDUP_TTL]:
        del _processed_messages[k]
    if msg_id in _processed_messages:
        return True
    _processed_messages[msg_id] = now
    return False


def is_duplicate_comment(comment_id):
    now = time.time()
    for k in [k for k, v in _processed_comments.items() if now - v > DEDUP_TTL]:
        del _processed_comments[k]
    if comment_id in _processed_comments:
        return True
    _processed_comments[comment_id] = now
    return False


# ============================================
# SANITIZE HISTORY
# ============================================
def sanitize_history(history):
    if not history:
        return []
    sanitized = []
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "").strip()
        if not content:
            continue
        if not sanitized:
            if role == "user":
                sanitized.append(msg)
            continue
        if role == sanitized[-1]["role"]:
            if role == "user":
                sanitized[-1] = {"role": "user", "content": sanitized[-1]["content"] + "\n" + content}
            else:
                sanitized[-1] = msg
        else:
            sanitized.append(msg)
    if sanitized and sanitized[-1]["role"] != "user":
        sanitized.pop()
    return sanitized


# ============================================
# AI API CALLS
# ============================================
def _call_openai(system_prompt, messages, max_tokens):
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": AI_MODEL,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "max_tokens": max_tokens, "temperature": 0.7,
    }
    resp = requests.post(OPENAI_API_URL, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def _call_anthropic(system_prompt, messages, max_tokens):
    headers = {
        "x-api-key": ANTHROPIC_API_KEY,
        "anthropic-version": ANTHROPIC_VERSION,
        "content-type": "application/json",
    }
    payload = {
        "model": AI_MODEL, "max_tokens": max_tokens,
        "temperature": 0.7, "system": system_prompt, "messages": messages,
    }
    resp = requests.post(ANTHROPIC_API_URL, headers=headers, json=payload, timeout=25)
    resp.raise_for_status()
    return resp.json()["content"][0]["text"]


def _call_ai(system_prompt, messages, max_tokens=150):
    if AI_PROVIDER == "openai":
        return _call_openai(system_prompt, messages, max_tokens)
    elif AI_PROVIDER == "anthropic":
        return _call_anthropic(system_prompt, messages, max_tokens)
    return None


# ============================================
# AI AGENT
# ============================================
def ask_ai(user_id, user_message, platform="messenger"):
    if AI_PROVIDER == "fallback":
        return fallback_response(user_message)

    conversation_history[user_id].append({"role": "user", "content": user_message})
    if len(conversation_history[user_id]) > MAX_HISTORY:
        conversation_history[user_id] = conversation_history[user_id][-MAX_HISTORY:]

    clean_history = sanitize_history(conversation_history[user_id])
    if not clean_history:
        clean_history = [{"role": "user", "content": user_message}]

    system_prompt = get_system_prompt()

    if user_id in user_data:
        data = user_data[user_id]
        if data.get("name"):
            ctx = f"\n[الاسم: {data['name']}"
            if data.get("phone"):
                ctx += f", التليفون: {data['phone']}"
            if data.get("interest"):
                ctx += f", مهتم بـ: {data['interest']}"
            ctx += "]"
            system_prompt += f"\n\n## العميل الحالي:{ctx}"

    msg_count = len(conversation_history[user_id])
    has_phone = bool(user_data.get(user_id, {}).get("phone"))
    asked_phone = phone_requested.get(user_id, False)
    sent_link = any("in-house-bnc.netlify.app" in m.get("content", "") for m in conversation_history.get(user_id, []) if m.get("role") == "assistant")

    # Conversation stage context
    if msg_count <= 1:
        system_prompt += "\n\n## أنت في مرحلة 1 (ترحيب): رحب جملة واحدة بسيطة واسأل العميل بيدور على إيه. ممنوع لينكات أو أرقام."
    elif msg_count <= 4:
        system_prompt += "\n\n## أنت في مرحلة 2 (اكتشاف): اسأل سؤال واحد بس عشان تفهم احتياج العميل (منطقة؟ مساحة؟ ميزانية؟). ممنوع لينكات أو عروض — ركّز على الأسئلة."
    elif not sent_link:
        system_prompt += "\n\n## أنت في مرحلة 3 (توجيه): لخّص اللي فهمته من العميل وابعت اللينك مرة واحدة: https://in-house-bnc.netlify.app"
    else:
        system_prompt += "\n\n## مرحلة 4 (متابعة): جاوب أسئلة العميل طبيعي. اللينك اتبعت قبل كده — ما تكررهوش."

    if has_phone:
        system_prompt += "\n## العميل ساب رقمه — ما تطلبش تاني."
    elif asked_phone:
        system_prompt += "\n## طلبت الرقم قبل كده — ما تكررش."

    try:
        ai_response = _call_ai(system_prompt, clean_history, max_tokens=300)
        if not ai_response:
            fb = fallback_response(user_message)
            conversation_history[user_id].append({"role": "assistant", "content": fb})
            return fb

        ai_response = _post_process(ai_response)
        conversation_history[user_id].append({"role": "assistant", "content": ai_response})
        extract_user_data(user_id, user_message, ai_response)

        phone_patterns = ["رقم حضرتك", "رقم تليفون", "رقم موبايل", "رقمك",
                          "ابعتلي رقم", "ابعتلنا رقم", "سيب رقمك"]
        if any(p in ai_response for p in phone_patterns):
            phone_requested[user_id] = True

        logger.info(f"[{AI_PROVIDER}] {user_id}: {ai_response[:100]}...")
        return ai_response

    except Exception as e:
        logger.error(f"{AI_PROVIDER} error: {e}")
        fb = fallback_response(user_message)
        conversation_history[user_id].append({"role": "assistant", "content": fb})
        return fb


def ask_ai_comment(comment_text, sender_name):
    if AI_PROVIDER == "fallback":
        return None
    system_prompt = get_system_prompt() + """

## رد على تعليق فيسبوك (عام)
- سطر واحد — 15 كلمة ماكس
- رحب بالعميل باسمه
- وجّهه يبعتلنا رسالة خاصة أو يدخل اللينك
- ممنوع أسعار أو أرقام تليفون في الرد العام
- رد كفريق إن هاوس
"""
    try:
        return _call_ai(system_prompt,
                        [{"role": "user", "content": f"العميل {sender_name} علّق: \"{comment_text}\"\nرد مختصر."}],
                        max_tokens=80)
    except Exception as e:
        logger.error(f"Comment AI error: {e}")
        return None


def _post_process(response):
    lines = response.strip().split('\n')
    has_details = any(c in l for l in lines for c in ['📐', '💰', '💵', '📅', '📍', '🏢'])
    max_lines = 15 if has_details else 10
    if len(lines) > max_lines:
        response = '\n'.join(lines[:max_lines])
    response = re.sub(r'\n{3,}', '\n\n', response).strip()
    return response


def extract_user_data(user_id, user_message, ai_response):
    if user_id not in user_data:
        user_data[user_id] = {}
    text = user_message.strip()

    phone = extract_phone(text)
    if phone:
        user_data[user_id]["phone"] = phone
        if user_data[user_id].get("name"):
            auto_save_lead(user_id)

    history = conversation_history.get(user_id, [])
    if len(history) >= 2:
        prev = history[-2].get("content", "") if history[-2]["role"] == "assistant" else ""
        if any(w in prev for w in ["اسمك", "اسم حضرتك", "نعرف اسمك"]):
            if len(text.split()) <= 4 and not text.startswith("0"):
                user_data[user_id]["name"] = text

    interests = []
    if any(w in text for w in ["شقة", "شقق", "سكني"]):
        interests.append("شقق سكنية")
    if any(w in text for w in ["إيجار", "أجار"]):
        interests.append("إيجار")
    if any(w in text for w in ["تمليك", "شراء"]):
        interests.append("تمليك")
    if interests:
        user_data[user_id]["interest"] = "، ".join(interests)


def extract_phone(text):
    clean = text.replace("-", "").replace(" ", "")
    patterns = [
        r'01[0125][0-9]{8}',      # Egyptian local: 01X + 8 digits
        r'\+201[0125][0-9]{8}',    # International +20
        r'00201[0125][0-9]{8}',    # International 0020
        r'\+[1-9][0-9]{9,14}',     # Any international +
        r'00[1-9][0-9]{9,14}',     # Any international 00
    ]
    for p in patterns:
        m = re.search(p, clean)
        if m:
            return m.group()
    return None


def auto_save_lead(user_id):
    data = user_data.get(user_id, {})
    if data.get("name") and data.get("phone"):
        if not any(l.get("phone") == data["phone"] for l in leads_db):
            lead = {
                "name": data["name"], "phone": data["phone"],
                "interest": data.get("interest", "غير محدد"),
                "platform": "messenger", "timestamp": datetime.now().isoformat(),
                "user_id": user_id,
            }
            leads_db.append(lead)
            logger.info(f"Lead saved: {data['name']} - {data['phone']}")
            track("leads_captured")
            notify_telegram_lead(lead)
            save_lead_to_sheet(lead)


def save_lead_to_sheet(lead):
    """Persist lead to Google Sheet via Apps Script webhook."""
    if not SHEETS_WEBHOOK_URL:
        return
    payload = {
        "name": lead.get("name", ""),
        "phone": lead.get("phone", ""),
        "source": "ماسنجر",
        "request_type": lead.get("interest", ""),
        "area": lead.get("area", ""),
        "budget": lead.get("budget", ""),
        "notes": lead.get("notes", ""),
        "status": "جديد",
    }
    try:
        requests.post(SHEETS_WEBHOOK_URL, json=payload, timeout=10)
    except Exception as e:
        logger.error(f"Sheet save failed: {e}")


def notify_telegram_lead(lead):
    """Send lead notification to all Telegram admins."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
        return
    text = (
        f"🏠 <b>ليد جديد من الشات بوت</b>\n\n"
        f"👤 <b>الاسم:</b> {lead['name']}\n"
        f"📞 <b>الرقم:</b> {lead['phone']}\n"
        f"🏷 <b>مهتم بـ:</b> {lead.get('interest', 'غير محدد')}\n"
        f"📱 <b>المنصة:</b> {lead.get('platform', 'messenger')}\n"
        f"🕐 <b>الوقت:</b> {lead['timestamp']}"
    )
    for chat_id in TELEGRAM_CHAT_IDS:
        chat_id = chat_id.strip()
        if not chat_id:
            continue
        try:
            requests.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={"chat_id": chat_id, "text": text, "parse_mode": "HTML"},
                timeout=10
            )
        except Exception as e:
            logger.error(f"Telegram notify failed for {chat_id}: {e}")


def fallback_response(message):
    text = message.lower().strip()
    if any(w in text for w in ["سلام", "هاي", "مرحبا", "صباح", "مساء", "اهلا", "أهلا", "هلو"]):
        return "أهلاً بيك! 🏠 حضرتك بتدور على شقة في بني سويف؟"
    if any(w in text for w in ["سعر", "كام", "تقسيط", "مقدم", "دفع", "قسط", "بكام"]):
        return "حضرتك مهتم بأنهي منطقة عشان أفيدك بالأسعار؟ 😊\nأو سيب طلبك هنا:\n👉 https://in-house-bnc.netlify.app"
    if any(w in text for w in ["شقة", "شقق", "سكن", "إيجار", "تمليك"]):
        return "عندنا شقق في كل بني سويف شرق وغرب 🏠\nقولي المنطقة والمساحة اللي تناسبك!"
    if any(w in text for w in ["رقم", "تواصل", "تليفون", "واتساب", "فون"]):
        return "📞 واتساب: +20 107 073 6979\n🌐 https://in-house-bnc.netlify.app\nتواصل معانا في أي وقت!"
    return "أهلاً بيك في إن هاوس! 🏠 إيه اللي تحب تعرفه؟"


# ============================================
# MESSAGE HANDLER
# ============================================
def handle_message(user_id, message_text, platform="messenger", message_id=None):
    text = message_text.strip()
    if not text:
        return
    if message_id and is_duplicate_message(message_id):
        return
    # Record activity for auto follow-up tracking
    is_new_user = user_id not in user_last_activity
    user_last_activity[user_id] = time.time()
    followup_sent.pop(user_id, None)  # reset if user re-engaged
    track("messages_received")
    if is_new_user:
        track("new_conversations")
    logger.info(f"[{platform}] {user_id}: {text[:100]}")
    ai_response = ask_ai(user_id, text, platform)

    # Attach quick replies at key conversation stages
    msg_count = len(conversation_history.get(user_id, []))
    data = user_data.get(user_id, {})

    quick_replies = None
    if msg_count == 2:
        # After first exchange, offer quick options
        quick_replies = [
            {"content_type": "text", "title": "🏠 شراء", "payload": "INTENT_BUY"},
            {"content_type": "text", "title": "💰 بيع", "payload": "INTENT_SELL"},
            {"content_type": "text", "title": "🔑 إيجار", "payload": "INTENT_RENT"},
        ]
    elif msg_count >= 8 and not data.get("phone"):
        quick_replies = [
            {"content_type": "text", "title": "📝 سجّل طلبك", "payload": "REGISTER_LEAD"},
            {"content_type": "text", "title": "📞 كلّم مسؤول", "payload": "ICE_CONTACT"},
        ]

    if quick_replies and len(ai_response) <= 2000:
        _send_messenger_raw(user_id, ai_response, quick_replies=quick_replies)
    else:
        send_messenger_message(user_id, ai_response)
    track("messages_sent")


# ============================================
# POSTBACK & WELCOME HANDLERS
# ============================================
def handle_postback(user_id, payload):
    """Handle postback buttons (Get Started, Ice Breakers, Persistent Menu)."""
    logger.info(f"[postback] {user_id}: {payload}")

    if payload == "GET_STARTED":
        send_welcome_message(user_id)
    elif payload in ("ICE_APARTMENT", "MENU_APARTMENTS", "INTENT_BUY"):
        handle_message(user_id, "عايز أشتري شقة في بني سويف", "messenger")
    elif payload in ("ICE_PRICES", "MENU_PRICES"):
        handle_message(user_id, "عايز أعرف الأسعار", "messenger")
    elif payload in ("ICE_CONTACT", "MENU_CONTACT"):
        send_contact_info(user_id)
    elif payload == "INTENT_SELL":
        handle_message(user_id, "عايز أبيع شقة", "messenger")
    elif payload == "INTENT_RENT":
        handle_message(user_id, "بدور على شقة إيجار", "messenger")
    else:
        handle_message(user_id, payload, "messenger")


def send_welcome_message(user_id):
    """Welcome message with quick reply buttons when user taps Get Started."""
    text = (
        "أهلاً بيك في إن هاوس! 🏠\n"
        "إحنا وسيط عقاري في بني سويف — بنساعدك تلاقي شقتك المناسبة.\n\n"
        "إيه اللي تحب تعرفه؟ 👇"
    )
    quick_replies = [
        {"content_type": "text", "title": "🏠 عايز شقة", "payload": "ICE_APARTMENT"},
        {"content_type": "text", "title": "💰 الأسعار", "payload": "ICE_PRICES"},
        {"content_type": "text", "title": "📞 كلّم مسؤول", "payload": "ICE_CONTACT"},
    ]
    _send_messenger_raw(user_id, text, quick_replies=quick_replies)


def send_contact_info(user_id):
    """Send contact details."""
    text = (
        "📞 تواصل مع فريق إن هاوس:\n\n"
        "💬 واتساب: +20 107 073 6979\n"
        "🌐 الموقع: https://in-house-bnc.netlify.app\n"
        "📱 فيسبوك: facebook.com/in.housebnc"
    )
    _send_messenger_raw(user_id, text)


# ============================================
# APPOINTMENT BOOKING (visit scheduling)
# ============================================
appointment_state = {}  # user_id -> {"stage": "...", "slot": "..."}

def start_appointment_booking(user_id):
    """Start appointment booking flow — show available slots."""
    from datetime import timedelta
    slots = []
    today = datetime.now()
    for i in range(1, 5):  # next 4 days
        d = today + timedelta(days=i)
        day_name = ["الاثنين", "الثلاثاء", "الأربعاء", "الخميس",
                    "الجمعة", "السبت", "الأحد"][d.weekday()]
        slots.append({
            "label": f"{day_name} {d.strftime('%d/%m')} - 11 ص",
            "payload": f"SLOT_{d.strftime('%Y-%m-%d')}_11",
        })
        slots.append({
            "label": f"{day_name} {d.strftime('%d/%m')} - 5 م",
            "payload": f"SLOT_{d.strftime('%Y-%m-%d')}_17",
        })

    appointment_state[user_id] = {"stage": "choosing_slot"}
    text = "📅 ممكن نحدد معاد المعاينة:\n\nاختر الوقت المناسب 👇"
    quick_replies = [{"content_type": "text", "title": s["label"], "payload": s["payload"]}
                     for s in slots[:8]]  # messenger limit ~13
    _send_messenger_raw(user_id, text, quick_replies=quick_replies)


def confirm_appointment(user_id, slot_payload):
    """Save appointment and notify admins."""
    # Format: SLOT_YYYY-MM-DD_HH
    try:
        _, date_str, hour = slot_payload.split("_")
        data = user_data.get(user_id, {})
        name = data.get("name", "عميل")
        phone = data.get("phone", "—")

        appointment_state[user_id] = {
            "stage": "confirmed",
            "slot": f"{date_str} {hour}:00",
        }

        # Confirm to user
        text = (
            f"✅ تم تسجيل معاد المعاينة:\n"
            f"📅 {date_str} الساعة {hour}:00\n\n"
            f"فريقنا هيتواصل معاك لتأكيد المكان والتفاصيل 📞"
        )
        if not phone or phone == "—":
            text += "\n\nممكن تبعتلي رقمك عشان نقدر نكلمك؟"
        _send_messenger_raw(user_id, text)

        # Notify Telegram group
        summary = (
            f"📅 <b>حجز معاينة جديد</b>\n\n"
            f"👤 الاسم: {name}\n"
            f"📞 الرقم: {phone}\n"
            f"🗓 المعاد: {date_str} الساعة {hour}:00\n"
            f"💬 User ID: <code>{user_id}</code>"
        )
        if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS:
            for cid in TELEGRAM_CHAT_IDS:
                try:
                    requests.post(
                        f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                        json={"chat_id": cid.strip(), "text": summary, "parse_mode": "HTML"},
                        timeout=10,
                    )
                except Exception:
                    pass

        # Save to sheet too
        save_lead_to_sheet({
            "name": name, "phone": phone,
            "interest": f"حجز معاينة {date_str} {hour}:00",
            "notes": f"معاد محجوز عبر البوت",
        })
        track("appointments_booked")

    except Exception as e:
        logger.error(f"Appointment confirm error: {e}")
        _send_messenger_raw(user_id, "حصل خطأ صغير، ممكن تبعتلنا واتساب: +20 107 073 6979")


# ============================================
# COMMENT HANDLER
# ============================================
def handle_comment(comment_data):
    comment_id = comment_data.get("comment_id")
    comment_text = comment_data.get("message", "")
    sender_name = comment_data.get("from", {}).get("name", "")
    sender_id = comment_data.get("from", {}).get("id", "")
    verb = comment_data.get("verb", "")
    post_id = comment_data.get("post_id", "")

    logger.info(f"Comment: id={comment_id}, from={sender_name}, verb={verb}, text='{comment_text[:50]}'")

    if verb != "add" or not comment_id:
        return
    if is_duplicate_comment(comment_id):
        return
    track("comments_received")

    page_id = post_id.split("_")[0] if post_id else ""
    if sender_id == page_id:
        return

    is_positive = (
        any(e in comment_text for e in EMOJI_POSITIVE) or
        any(w in comment_text for w in THANK_WORDS)
    )

    if is_positive and not any(k in comment_text for k in COMMENT_KEYWORDS):
        if any(e in comment_text for e in EMOJI_POSITIVE) and len(comment_text.strip()) <= 5:
            resp = random.choice(EMOJI_RESPONSES)
        else:
            resp = f"شكراً ليك يا {sender_name}! نورتنا 🙏❤️"
        reply_to_comment(comment_id, resp)
        send_private_reply(comment_id, WELCOME_DM)
        return

    if any(k in comment_text for k in COMMENT_KEYWORDS):
        ai_reply = ask_ai_comment(comment_text, sender_name)
        if ai_reply:
            reply_to_comment(comment_id, ai_reply)
        else:
            reply_to_comment(comment_id, f"أهلاً يا {sender_name}! 🏠 بعتنالك التفاصيل على الخاص 💬")
        send_private_reply(comment_id, WELCOME_DM)
        return

    reply_to_comment(comment_id, f"أهلاً يا {sender_name}! 🏠 لو محتاج معلومات ابعتلنا رسالة خاصة!")
    send_private_reply(comment_id, WELCOME_DM)


# ============================================
# SEND FUNCTIONS
# ============================================
def send_messenger_message(recipient_id, text):
    if len(text) > 2000:
        for chunk in split_message(text, 2000):
            _send_messenger_raw(recipient_id, chunk)
            time.sleep(0.5)
    else:
        _send_messenger_raw(recipient_id, text)


def _send_messenger_raw(recipient_id, text, quick_replies=None):
    url = f"{GRAPH_API_URL}/me/messages"
    message = {"text": text}
    if quick_replies:
        message["quick_replies"] = quick_replies
    payload = {"recipient": {"id": recipient_id}, "message": message, "messaging_type": "RESPONSE"}
    try:
        r = requests.post(url, json=payload, params={"access_token": PAGE_ACCESS_TOKEN}, timeout=10)
        if not r.ok:
            logger.error(f"Messenger send failed [{r.status_code}]: {r.text[:300]}")
        r.raise_for_status()
    except requests.exceptions.HTTPError:
        pass  # already logged above
    except Exception as e:
        logger.error(f"Messenger send exception: {e}")


def reply_to_comment(comment_id, text):
    url = f"{GRAPH_API_URL}/{comment_id}/comments"
    try:
        r = requests.post(url, data={"message": text}, params={"access_token": PAGE_ACCESS_TOKEN}, timeout=10)
        result = r.json()
        if "error" in result:
            logger.error(f"Comment reply ERROR: {result['error']}")
            return False
        logger.info(f"Comment reply SUCCESS")
        track("comments_replied")
        return True
    except Exception as e:
        logger.error(f"Comment reply EXCEPTION: {e}")
        return False


def send_private_reply(comment_id, text):
    url = f"{GRAPH_API_URL}/me/messages"
    payload = {"recipient": {"comment_id": comment_id}, "message": {"text": text}, "messaging_type": "RESPONSE"}
    try:
        r = requests.post(url, json=payload, params={"access_token": PAGE_ACCESS_TOKEN}, timeout=10)
        result = r.json()
        if "error" in result:
            logger.error(f"Private reply ERROR: {result['error']}")
    except Exception as e:
        logger.error(f"Private reply FAILED: {e}")


def split_message(text, max_length):
    chunks = []
    while len(text) > max_length:
        split_at = text.rfind('\n', 0, max_length)
        if split_at == -1:
            split_at = text.rfind(' ', 0, max_length)
        if split_at == -1:
            split_at = max_length
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
    if text:
        chunks.append(text)
    return chunks


# ============================================
# WEBHOOK ROUTES
# ============================================
@app.route("/webhook", methods=["GET"])
def verify_webhook():
    mode = request.args.get("hub.mode")
    token = request.args.get("hub.verify_token")
    challenge = request.args.get("hub.challenge")
    if mode == "subscribe" and token == VERIFY_TOKEN:
        logger.info("Webhook verified!")
        return challenge, 200
    return "Verification failed", 403


@app.route("/webhook", methods=["POST"])
def handle_webhook():
    data = request.get_json()
    if data.get("object") == "page":
        for entry in data.get("entry", []):
            for event in entry.get("messaging", []):
                sender_id = event["sender"]["id"]

                # Skip messages from the page itself
                if sender_id == PAGE_ID:
                    continue

                if "message" in event:
                    message = event["message"]

                    # Skip echo messages (bot's own replies)
                    if message.get("is_echo"):
                        logger.debug(f"Skipping echo message from {sender_id}")
                        continue

                    msg_id = message.get("mid")
                    qr_payload = message.get("quick_reply", {}).get("payload", "")
                    text = message.get("text", "")

                    # Handle special quick reply payloads
                    if qr_payload in ("ICE_CONTACT", "MENU_CONTACT"):
                        send_contact_info(sender_id)
                    elif qr_payload == "REGISTER_LEAD":
                        _send_messenger_raw(sender_id,
                            "📝 سجّل طلبك من هنا وهنتواصل معاك:\n👉 https://in-house-bnc.netlify.app\n\nأو ابعتلنا رقمك هنا وهنكلمك 😊")
                    elif qr_payload == "INTENT_BUY":
                        handle_message(sender_id, "عايز أشتري شقة", "messenger", msg_id)
                    elif qr_payload == "INTENT_SELL":
                        handle_message(sender_id, "عايز أبيع شقة", "messenger", msg_id)
                    elif qr_payload == "INTENT_RENT":
                        handle_message(sender_id, "بدور على شقة إيجار", "messenger", msg_id)
                    elif qr_payload.startswith("SLOT_"):
                        confirm_appointment(sender_id, qr_payload)
                    elif text and any(kw in text for kw in ["معاينة", "زيارة", "أعاين", "اشوف الشقة", "شوف الشقة"]):
                        start_appointment_booking(sender_id)
                    elif text:
                        handle_message(sender_id, text, "messenger", msg_id)
                elif "postback" in event:
                    handle_postback(sender_id, event["postback"]["payload"])
            for change in entry.get("changes", []):
                if change.get("field") == "feed" and change["value"].get("item") == "comment":
                    handle_comment(change["value"])
    return "OK", 200


# ============================================
# API ENDPOINTS
# ============================================
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "running",
        "bot": "In-House Real Estate",
        "ai_provider": AI_PROVIDER,
        "ai_model": AI_MODEL,
        "facebook": "configured" if PAGE_ACCESS_TOKEN else "off",
    })


@app.route("/api/leads", methods=["GET"])
def get_leads():
    return jsonify({"leads": leads_db, "total": len(leads_db)})


@app.route("/api/stats", methods=["GET"])
def get_stats():
    return jsonify({
        "total_leads": len(leads_db),
        "active_conversations": len(conversation_history),
        "users_with_data": len(user_data),
    })


# ============================================
# ANALYTICS ENDPOINTS
# ============================================
@app.route("/api/analytics", methods=["GET"])
def get_analytics():
    """Return full analytics snapshot — lifetime + daily breakdown."""
    today = datetime.now().strftime("%Y-%m-%d")
    today_stats = analytics["daily"].get(today, {})

    # Conversion rate = leads / distinct conversations
    convs = analytics.get("new_conversations", 0) or 1
    conv_rate = round((analytics.get("leads_captured", 0) / convs) * 100, 2)

    return jsonify({
        "lifetime": {
            "messages_received": analytics.get("messages_received", 0),
            "messages_sent": analytics.get("messages_sent", 0),
            "comments_received": analytics.get("comments_received", 0),
            "comments_replied": analytics.get("comments_replied", 0),
            "leads_captured": analytics.get("leads_captured", 0),
            "appointments_booked": analytics.get("appointments_booked", 0),
            "followups_sent": analytics.get("followups_sent", 0),
            "broadcasts_sent": analytics.get("broadcasts_sent", 0),
            "new_conversations": analytics.get("new_conversations", 0),
            "conversion_rate_pct": conv_rate,
        },
        "today": today_stats,
        "daily": analytics["daily"],
        "started_at": analytics.get("started_at"),
    })


def build_weekly_report():
    """Build a 7-day aggregated report string (HTML for Telegram)."""
    from datetime import timedelta
    today = datetime.now()
    last_7_days = [(today - timedelta(days=i)).strftime("%Y-%m-%d") for i in range(7)]

    agg = {
        "messages_received": 0,
        "messages_sent": 0,
        "comments_received": 0,
        "comments_replied": 0,
        "leads_captured": 0,
        "appointments_booked": 0,
        "followups_sent": 0,
        "new_conversations": 0,
    }
    for d in last_7_days:
        day = analytics["daily"].get(d, {})
        for k in agg:
            agg[k] += day.get(k, 0)

    # Pull sheet stats if available (authoritative lead count from the sheet)
    sheet_totals = {}
    if SHEETS_WEBHOOK_URL:
        try:
            r = requests.get(f"{SHEETS_WEBHOOK_URL}?action=stats", timeout=15)
            if r.ok:
                s = r.json()
                if s.get("totals"):
                    sheet_totals = s["totals"]
        except Exception as e:
            logger.warning(f"Weekly report — sheet stats fetch failed: {e}")

    convs = agg["new_conversations"] or 1
    conv_rate = round((agg["leads_captured"] / convs) * 100, 1)

    from_date = last_7_days[-1]
    to_date = last_7_days[0]

    lines = [
        "📊 <b>التقرير الأسبوعي — إن هاوس</b>",
        f"📅 من {from_date} إلى {to_date}",
        "",
        "<b>📨 الرسائل:</b>",
        f"• رسائل واردة: {agg['messages_received']}",
        f"• ردود مرسلة: {agg['messages_sent']}",
        "",
        "<b>💬 التعليقات:</b>",
        f"• تعليقات واردة: {agg['comments_received']}",
        f"• ردود تعليقات: {agg['comments_replied']}",
        "",
        "<b>🎯 النتائج:</b>",
        f"• محادثات جديدة: {agg['new_conversations']}",
        f"• ليدز محفوظة (من البوت): {agg['leads_captured']}",
        f"• حجوزات معاينة: {agg['appointments_booked']}",
        f"• رسائل متابعة: {agg['followups_sent']}",
        f"• نسبة التحويل: {conv_rate}%",
    ]

    if sheet_totals:
        lines.extend([
            "",
            "<b>📋 من الشيت (كل المصادر):</b>",
            f"• إجمالي الليدز: {sheet_totals.get('all', 0)}",
            f"• آخر أسبوع: {sheet_totals.get('this_week', 0)}",
            f"• آخر شهر: {sheet_totals.get('this_month', 0)}",
        ])

    lines.extend([
        "",
        "📈 الداشبورد: <a href=\"https://in-house-bnc.netlify.app/dashboard.html\">فتح</a>",
    ])

    return "\n".join(lines)


@app.route("/api/report/weekly", methods=["POST", "GET"])
def weekly_report():
    """Send weekly summary to Telegram group. Call via cron every Sunday 9am."""
    try:
        report = build_weekly_report()

        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
            return jsonify({"success": False, "error": "Telegram not configured", "preview": report})

        sent_to = []
        for cid in TELEGRAM_CHAT_IDS:
            cid = cid.strip()
            if not cid:
                continue
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={
                        "chat_id": cid,
                        "text": report,
                        "parse_mode": "HTML",
                        "disable_web_page_preview": True,
                    },
                    timeout=10,
                )
                if r.ok:
                    sent_to.append(cid)
            except Exception as e:
                logger.error(f"Weekly report send failed for {cid}: {e}")

        return jsonify({
            "success": len(sent_to) > 0,
            "sent_to": sent_to,
            "preview": report,
        })
    except Exception as e:
        logger.error(f"Weekly report error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/report/daily", methods=["POST", "GET"])
def daily_report():
    """Send a short daily summary to Telegram (call at 9pm daily)."""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        d = analytics["daily"].get(today, {})

        if not any(d.values()):
            # Nothing happened today — skip
            return jsonify({"success": True, "skipped": "no activity today"})

        lines = [
            f"🌙 <b>ملخص اليوم — {today}</b>",
            "",
            f"💬 رسائل: {d.get('messages_received', 0)} واردة / {d.get('messages_sent', 0)} ردود",
            f"👥 محادثات جديدة: {d.get('new_conversations', 0)}",
            f"📝 ليدز محفوظة: {d.get('leads_captured', 0)}",
            f"📅 حجوزات معاينة: {d.get('appointments_booked', 0)}",
            f"💬 تعليقات: {d.get('comments_received', 0)} ← رد على {d.get('comments_replied', 0)}",
        ]
        text = "\n".join(lines)

        if not TELEGRAM_BOT_TOKEN:
            return jsonify({"success": False, "preview": text})

        sent = 0
        for cid in TELEGRAM_CHAT_IDS:
            cid = cid.strip()
            if not cid:
                continue
            try:
                r = requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={"chat_id": cid, "text": text, "parse_mode": "HTML"},
                    timeout=10,
                )
                if r.ok:
                    sent += 1
            except Exception:
                pass

        return jsonify({"success": sent > 0, "sent_to_chats": sent, "preview": text})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ============================================
# BROADCAST — re-engagement message
# ============================================
BROADCAST_MESSAGE = """كل سنة وحضرتك بخير 🌟

فريق إن هاوس في بني سويف — لسه بنتابع تواصلنا معاك عشان نوفرلك أحسن العروض العقارية.

بتدور على إيه؟
🏠 شراء
💰 بيع
📈 استثمار

سجّل طلبك في ثواني:
👉 https://in-house-bnc.netlify.app

أو واتساب مباشر: +20 107 073 6979"""


def fetch_recent_conversations(limit=500):
    """Fetch conversations from the Page's inbox (last messaged users)."""
    if not PAGE_ACCESS_TOKEN:
        return []
    conversations = []
    url = f"{GRAPH_API_URL}/me/conversations"
    params = {
        "access_token": PAGE_ACCESS_TOKEN,
        "fields": "participants,updated_time",
        "limit": 100,
    }
    try:
        while url and len(conversations) < limit:
            r = requests.get(url, params=params, timeout=15)
            data = r.json()
            if "error" in data:
                logger.error(f"Conversations fetch error: {data['error']}")
                break
            for conv in data.get("data", []):
                for p in conv.get("participants", {}).get("data", []):
                    if p.get("id") and p["id"] != PAGE_ID:
                        conversations.append({
                            "user_id": p["id"],
                            "name": p.get("name", ""),
                            "updated_time": conv.get("updated_time", ""),
                        })
            paging = data.get("paging", {})
            url = paging.get("next")
            params = None
    except Exception as e:
        logger.error(f"Fetch conversations failed: {e}")
    return conversations


@app.route("/api/broadcast/preview", methods=["GET"])
def broadcast_preview():
    """Preview who will receive the broadcast (dry-run — no messages sent)."""
    convs = fetch_recent_conversations()
    return jsonify({
        "total_conversations": len(convs),
        "message": BROADCAST_MESSAGE,
        "sample_users": convs[:10],
        "note": "call POST /api/broadcast/send?confirm=YES to actually send",
    })


# ============================================
# AUTO FOLLOW-UP — silent conversation recovery
# ============================================
FOLLOWUP_STAGE_1 = """أهلاً 👋

فاكرنا؟ إحنا فريق إن هاوس — لسه بتدور على شقة في بني سويف؟

لو نفسك نساعدك، احكيلنا وهنرجعلك بأحسن عرض 🏠"""

FOLLOWUP_STAGE_2 = """مرحباً تاني 🌟

لو لسه مهتم بشقة (شراء/بيع/إيجار) في بني سويف:
👉 https://in-house-bnc.netlify.app

أو واتساب: +20 107 073 6979

في خدمتك أي وقت 🙏"""


@app.route("/api/followup/run", methods=["POST", "GET"])
def followup_run():
    """Check for silent conversations and send follow-up messages.
    Call this endpoint every 1-6 hours via external cron (cron-job.org, UptimeRobot, etc.)
    """
    now = time.time()
    stage1_sent = 0
    stage2_sent = 0
    errors = []

    for user_id, last_ts in list(user_last_activity.items()):
        hours_silent = (now - last_ts) / 3600
        sent_stages = followup_sent.get(user_id, [])

        # Skip if user already has phone — they're a lead, not abandoned
        if user_data.get(user_id, {}).get("phone"):
            continue

        text = None
        stage = None

        if hours_silent >= FOLLOWUP_STAGE_2_HOURS and 2 not in sent_stages:
            text = FOLLOWUP_STAGE_2
            stage = 2
        elif hours_silent >= FOLLOWUP_STAGE_1_HOURS and 1 not in sent_stages:
            text = FOLLOWUP_STAGE_1
            stage = 1

        if not text:
            continue

        try:
            # Use HUMAN_AGENT tag to allow sending outside 24h window
            payload = {
                "recipient": {"id": user_id},
                "message": {"text": text},
                "messaging_type": "MESSAGE_TAG",
                "tag": "HUMAN_AGENT",
            }
            r = requests.post(
                f"{GRAPH_API_URL}/me/messages",
                json=payload,
                params={"access_token": PAGE_ACCESS_TOKEN},
                timeout=10,
            )
            if r.ok:
                sent_stages.append(stage)
                followup_sent[user_id] = sent_stages
                track("followups_sent")
                if stage == 1: stage1_sent += 1
                else: stage2_sent += 1
            else:
                err = r.json().get("error", {}).get("message", "unknown")[:100]
                if len(errors) < 5:
                    errors.append({"user": user_id, "stage": stage, "error": err})
            time.sleep(0.2)
        except Exception as e:
            if len(errors) < 5:
                errors.append({"user": user_id, "stage": stage, "error": str(e)[:100]})

    return jsonify({
        "stage1_sent": stage1_sent,
        "stage2_sent": stage2_sent,
        "total_tracked": len(user_last_activity),
        "errors_sample": errors,
    })


@app.route("/api/broadcast/send", methods=["POST", "GET"])
def broadcast_send():
    """Send broadcast message to eligible users (24-hour window)."""
    if request.args.get("confirm") != "YES":
        return jsonify({
            "error": "Missing confirmation",
            "usage": "POST /api/broadcast/send?confirm=YES",
        }), 400

    convs = fetch_recent_conversations()
    sent = 0
    failed = 0
    errors_sample = []

    for conv in convs:
        user_id = conv["user_id"]
        try:
            # messaging_type=MESSAGE_TAG with HUMAN_AGENT allows 7-day window
            payload = {
                "recipient": {"id": user_id},
                "message": {"text": BROADCAST_MESSAGE},
                "messaging_type": "MESSAGE_TAG",
                "tag": "HUMAN_AGENT",
            }
            r = requests.post(
                f"{GRAPH_API_URL}/me/messages",
                json=payload,
                params={"access_token": PAGE_ACCESS_TOKEN},
                timeout=10,
            )
            if r.ok:
                sent += 1
                track("broadcasts_sent")
            else:
                failed += 1
                if len(errors_sample) < 5:
                    errors_sample.append({
                        "user": conv.get("name", user_id),
                        "error": r.json().get("error", {}).get("message", "unknown")[:120],
                    })
            time.sleep(0.2)  # rate limit safety
        except Exception as e:
            failed += 1
            if len(errors_sample) < 5:
                errors_sample.append({"user": user_id, "error": str(e)[:120]})

    result = {
        "total": len(convs),
        "sent": sent,
        "failed": failed,
        "errors_sample": errors_sample,
    }
    logger.info(f"Broadcast done: {result}")

    # Notify Telegram group with the result
    if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS:
        summary = (
            f"📣 <b>تم إرسال رسالة إعادة التفعيل</b>\n\n"
            f"📊 المحادثات الكلية: {len(convs)}\n"
            f"✅ اتبعتت بنجاح: {sent}\n"
            f"❌ فشلت: {failed}"
        )
        for cid in TELEGRAM_CHAT_IDS:
            try:
                requests.post(
                    f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                    json={"chat_id": cid.strip(), "text": summary, "parse_mode": "HTML"},
                    timeout=10,
                )
            except Exception:
                pass

    return jsonify(result)


@app.route("/api/debug-permissions", methods=["GET"])
def debug_permissions():
    if not PAGE_ACCESS_TOKEN:
        return jsonify({"error": "No PAGE_ACCESS_TOKEN"}), 400
    try:
        r = requests.get(f"{GRAPH_API_URL}/me/permissions", params={"access_token": PAGE_ACCESS_TOKEN}, timeout=10)
        r2 = requests.get(f"https://graph.facebook.com/debug_token",
                          params={"input_token": PAGE_ACCESS_TOKEN, "access_token": PAGE_ACCESS_TOKEN}, timeout=10)
        return jsonify({"permissions": r.json(), "token_info": r2.json()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/subscribe", methods=["GET", "POST"])
def api_subscribe():
    action = request.args.get("action", "")
    if request.method == "POST" or action == "subscribe":
        sub_result = subscribe_page_feed()
        status = check_subscription_status()
        return jsonify({"subscribe_result": sub_result, "current_status": status})
    status = check_subscription_status()
    return jsonify({"current_status": status})


@app.route("/api/exchange-token", methods=["GET"])
def exchange_token():
    global PAGE_ACCESS_TOKEN
    short_token = request.args.get("token")
    if not short_token:
        return jsonify({"error": "Missing 'token' param", "usage": "GET /api/exchange-token?token=SHORT_LIVED_TOKEN"}), 400
    if not FB_APP_SECRET:
        return jsonify({"error": "FB_APP_SECRET not set"}), 500
    try:
        # Step 1: short → long-lived
        r1 = requests.get(f"{GRAPH_API_URL}/oauth/access_token", params={
            "grant_type": "fb_exchange_token", "client_id": FB_APP_ID,
            "client_secret": FB_APP_SECRET, "fb_exchange_token": short_token
        }, timeout=15)
        d1 = r1.json()
        if "error" in d1:
            return jsonify({"step": 1, "error": d1["error"]}), 400
        long_token = d1["access_token"]

        # Step 2: long-lived → permanent page token
        r2 = requests.get(f"{GRAPH_API_URL}/me/accounts", params={"access_token": long_token}, timeout=15)
        d2 = r2.json()
        if "error" in d2:
            return jsonify({"step": 2, "error": d2["error"]}), 400

        pages = d2.get("data", [])
        page_token = None
        page_name = None
        for page in pages:
            if page.get("id") == PAGE_ID:
                page_token = page["access_token"]
                page_name = page["name"]
                break
        if not page_token and pages:
            page_token = pages[0]["access_token"]
            page_name = pages[0]["name"]
        if not page_token:
            return jsonify({"error": "No pages found", "pages": d2}), 400

        PAGE_ACCESS_TOKEN = page_token
        sub_result = subscribe_page_feed()

        return jsonify({
            "success": True, "page_name": page_name,
            "permanent_token": page_token,
            "token_preview": page_token[:20] + "...",
            "subscription": sub_result,
            "next_step": "Copy permanent_token and set PAGE_ACCESS_TOKEN on Railway!"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/setup-messenger", methods=["POST", "GET"])
def setup_messenger():
    """Configure Get Started, Ice Breakers, Persistent Menu, Greeting."""
    if not PAGE_ACCESS_TOKEN:
        return jsonify({"error": "No PAGE_ACCESS_TOKEN configured"}), 400

    results = {}
    profile_url = f"{GRAPH_API_URL}/me/messenger_profile"
    params = {"access_token": PAGE_ACCESS_TOKEN}

    # 1. Greeting text
    try:
        r = requests.post(profile_url, params=params, json={
            "greeting": [{
                "locale": "default",
                "text": "أهلاً بيك في إن هاوس للتسويق العقاري! 🏠\nابدأ دلوقتي عشان نساعدك تلاقي شقتك."
            }]
        }, timeout=10)
        results["greeting"] = r.json()
    except Exception as e:
        results["greeting"] = {"error": str(e)}

    # 2. Get Started button
    try:
        r = requests.post(profile_url, params=params, json={
            "get_started": {"payload": "GET_STARTED"}
        }, timeout=10)
        results["get_started"] = r.json()
    except Exception as e:
        results["get_started"] = {"error": str(e)}

    # 3. Ice Breakers
    try:
        r = requests.post(profile_url, params=params, json={
            "ice_breakers": [
                {"question": "عايز شقة في بني سويف 🏠", "payload": "ICE_APARTMENT"},
                {"question": "عايز أعرف الأسعار 💰", "payload": "ICE_PRICES"},
                {"question": "كلّم مسؤول المبيعات 📞", "payload": "ICE_CONTACT"},
            ]
        }, timeout=10)
        results["ice_breakers"] = r.json()
    except Exception as e:
        results["ice_breakers"] = {"error": str(e)}

    # 4. Persistent Menu
    try:
        r = requests.post(profile_url, params=params, json={
            "persistent_menu": [{
                "locale": "default",
                "composer_input_disabled": False,
                "call_to_actions": [
                    {"type": "postback", "title": "🏠 الشقق المتاحة", "payload": "MENU_APARTMENTS"},
                    {"type": "postback", "title": "💰 الأسعار", "payload": "MENU_PRICES"},
                    {"type": "postback", "title": "📞 تواصل معانا", "payload": "MENU_CONTACT"},
                    {"type": "web_url", "title": "🌐 الموقع", "url": "https://in-house-bnc.netlify.app"},
                ]
            }]
        }, timeout=10)
        results["persistent_menu"] = r.json()
    except Exception as e:
        results["persistent_menu"] = {"error": str(e)}

    # Also subscribe to required webhook fields
    sub_result = subscribe_page_feed()
    results["subscription"] = sub_result

    logger.info(f"Messenger profile setup: {results}")
    return jsonify(results)


def subscribe_page_feed():
    if not PAGE_ACCESS_TOKEN:
        return {"success": False, "error": "No token"}
    try:
        r = requests.post(f"{GRAPH_API_URL}/me/subscribed_apps",
                          params={"access_token": PAGE_ACCESS_TOKEN},
                          data={"subscribed_fields": "feed,messages,messaging_postbacks"}, timeout=10)
        result = r.json()
        logger.info(f"Subscribe result: {result}")
        return result
    except Exception as e:
        return {"error": str(e)}


def check_subscription_status():
    if not PAGE_ACCESS_TOKEN:
        return {"error": "No token"}
    try:
        r1 = requests.get(f"{GRAPH_API_URL}/me", params={"access_token": PAGE_ACCESS_TOKEN, "fields": "id,name"}, timeout=10)
        page = r1.json()
        r2 = requests.get(f"{GRAPH_API_URL}/{page.get('id', 'me')}/subscribed_apps",
                          params={"access_token": PAGE_ACCESS_TOKEN}, timeout=10)
        return {"page": page, "subscriptions": r2.json()}
    except Exception as e:
        return {"error": str(e)}


@app.route("/api/diagnose", methods=["GET"])
def diagnose():
    results = {"timestamp": datetime.now().isoformat(), "bot": "In-House"}
    results["ai"] = {"provider": AI_PROVIDER, "model": AI_MODEL}
    results["facebook_token"] = "Set" if PAGE_ACCESS_TOKEN else "Missing"
    if PAGE_ACCESS_TOKEN:
        try:
            r = requests.get(f"{GRAPH_API_URL}/me", params={"access_token": PAGE_ACCESS_TOKEN, "fields": "id,name"}, timeout=10)
            results["page"] = r.json()
        except:
            results["page"] = "Error"
    results["subscription"] = check_subscription_status()
    return jsonify(results)


@app.route("/", methods=["GET"])
def index():
    return jsonify({"bot": "In-House Real Estate Chatbot", "status": "running", "health": "/api/health"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
