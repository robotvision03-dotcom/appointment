# دفتر کارشناسی و خرید خودرو

منشی فارسی برای **خرید خودرو از فروشنده**. تماس شروع می‌شود با:

**سلام وقت بخیر. خودروی شما چه نوع است؟**

بعد حدود چهار سؤال کوتاه:

1. نوع / برند (مثلاً پژو پارس)
2. مدل یا سال ساخت
3. کارکرد کیلومتر
4. نام فروشنده

سپس یک **نوبت نیم‌ساعته** روی تقویم دفتر رزرو می‌شود. آدرس مراجعه:

**خیابان ایثار، کوچه خواجه پلاک ۲**

کارشناسی و تعیین قیمت **برای فروشنده رایگان** است.

تقویم سمت چپ مثل Airbnb است: ماه شمسی، روزهای باز، ساعت‌های نیم‌ساعتهٔ خالی. جمعه دفتر تعطیل است (۹ تا ۱۷).

## اجرا

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\Activate.ps1
pip install -r requirements.txt
cp .env.example .env
python -m src
```

باز کنید: **http://127.0.0.1:38471**

نمونه گفتگو:

1. پژو پارس  
2. ۱۳۹۹  
3. ۸۰ هزار  
4. علی رضایی  
5. بله  

یا از تقویم روز و ساعت را انتخاب کنید و فرم را بفرستید.

شنیدن: ابتدا نویز محیط سنجیده می‌شود، با شروع صدای کلام تشخیص آغاز می‌شود (حذف نویز + فیلتر بالاگذر). نام‌های ناقص مثل «سمن» یا «پرس» با دیکشنری خودروهای ایران (و در صورت نیاز Ollama) به سمند / پژو پارس اصلاح می‌شوند.

## شنیدن در برابر فهمیدن (مثل ChatGPT)

ChatGPT هنگام مکالمه دو کار جدا می‌کند؛ هیچ مدل ASR به‌تنهایی «مثل چت‌جی‌پی‌تی می‌فهمد»:

1. **شنیدن (ASR)** — **nezamisafa/whisper-persian-v4** (Whisper large-v3 fine-tuned on Persian, ~8.7% WER). Runtime is CTranslate2 int8 via faster-whisper so it fits this CPU. Shenava remains fallback; Gooya only if you have a vendor token.
2. **فهمیدن (NLU)** — متن → خودرو / سال / کیلومتر. واژه‌نامهٔ خودرو + اصلاح آوایی + در صورت نیاز Ollama.

## شنیدن گفتار

```bash
pip install -r requirements.txt   # faster-whisper لازم است
python -m src download-whisper    # ~1.6GB, nezamisafa/whisper-persian-v4
python -m src
```

اگر `faster_whisper` نصب نباشد، برنامه بی‌صدا روی Shenava می‌افتد؛ در `/health` مقدار
`stt.whisper.runtime_installed` را ببینید.

`STT_ENGINE=whisper` in `.env`. The checkpoint is forced to `language=fa`.
With that setting Whisper does **not** fall through to Shenava on an empty
result (that cascade was the 90-second waits). Leave `WHISPER_PERSIAN_ONLY=0`
and `OLLAMA_ENABLED=0` for the fast path.

اگر نویز اتاق باعث شود تشخیص بی‌خود شروع شود، `VAD_MIN_ENERGY` را بالا ببرید؛
اگر صدای آرام اصلاً شنیده نشود، پایین بیاورید (پیش‌فرض ۵۰۰).

Clips longer than `WHISPER_MAX_SECONDS` (default 5) are trimmed. Whisper
large-v3 on CPU is roughly realtime at 2–5s and much slower at 12s.

## سال مدل

`src/years.py` همهٔ شکل‌های گفتاری سال‌های **۱۳۷۰ تا ۱۴۱۰** را از قبل می‌سازد،
پس این‌ها همه یعنی ۱۳۸۸:

هزار و سیصد و هشتاد و هشت · یک هزار و سیصد و هشتاد و هشت · سیصد و هشتاد و هشت ·
هشتاد و هشت · هشت و هشت · یک سه هشت هشت · ۱۳۸۸ · ۳۸۸ · ۸۸

«نو» به‌جای «نه» هم پذیرفته می‌شود، پس «یک سه نو نو» = ۱۳۹۹. اگر جمله در جدول
نبود، جمع‌کنندهٔ عددی و بعد Ollama (در صورت فعال بودن) امتحان می‌شوند.

## تست

```bash
pytest tests/ -q
```
