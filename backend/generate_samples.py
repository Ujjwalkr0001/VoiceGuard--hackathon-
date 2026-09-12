"""
Generate audio files for VoiceGuard 1-Click Interactive Demos.
"""
from pathlib import Path
import win32com.client
import soundfile as sf

sample_dir = Path(__file__).parent / "app" / "static" / "sample_calls"
sample_dir.mkdir(parents=True, exist_ok=True)

samples = {
    "scam_bank_otp.wav": (
        "Attention customer, this is the security and fraud prevention department calling from your bank. "
        "An unauthorized international transaction of forty-eight thousand rupees has been attempted on your account. "
        "To cancel this charge immediately, you must share your six digit OTP and verify your debit card number. "
        "Enter your PIN right now or your bank account will be permanently suspended within ten minutes."
    ),
    "safe_call.wav": (
        "Hey there, hope you are having a wonderful day! Are we still on for lunch this Saturday? "
        "I was thinking we could check out that new cafe near the library around one o clock in the afternoon. "
        "Let me know what time works best for your schedule, talk to you later!"
    ),
    "ai_relative_emergency.wav": (
        "Uncle please help me, I have been detained by the police after an urgent car accident. "
        "They are demanding thirty thousand rupees right now to release me immediately without filing an FIR. "
        "Please transfer the money to this PhonePe number right now, do not tell mom please hurry!"
    ),
}

voice = win32com.client.Dispatch("SAPI.SpVoice")
for filename, text in samples.items():
    filepath = sample_dir / filename
    stream = win32com.client.Dispatch("SAPI.SpFileStream")
    stream.Open(str(filepath), 3)  # 3 = SSFMCreateForWrite
    voice.AudioOutputStream = stream
    voice.Speak(text)
    stream.Close()
    
    # Read back to print details
    data, sr = sf.read(str(filepath))
    duration = len(data) / sr
    print(f"Generated {filename}: {duration:.2f}s, {sr} Hz, {len(data)} samples")

print("All sample calls generated successfully!")
