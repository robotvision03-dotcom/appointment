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

1. **شنیدن (ASR)** — موج صدا → حروف. **Gooya v1.4** is preferred when `GOOYA_API_URL` and `GOOYA_API_TOKEN` are set (commercial API; no public Hugging Face weights). Otherwise **Shenava-Koochik CTC** runs on this machine.
2. **فهمیدن (NLU)** — متن → خودرو / سال / کیلومتر. واژه‌نامهٔ خودرو + اصلاح آوایی + در صورت نیاز Ollama.

## شنیدن گفتار

Gooya v1.4 cannot be downloaded. After you receive a vendor URL and bearer token:

```bash
# .env
STT_ENGINE=auto
GOOYA_API_URL=https://YOUR_HOST/transcribe
GOOYA_API_TOKEN=...
python -m src download-gooya
python -m src
```

On-device fallback (already installed):

```powershell
py -m src download-shenava
py -m src
```

## تست

```bash
pytest tests/test_dialogue.py -q
```
