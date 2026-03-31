"""Quick test: call Azure TTS and save the result as a WAV file."""
import os
import httpx
import struct

AZURE_TTS_KEY      = os.getenv("AZURE_TTS_KEY", "")
AZURE_TTS_ENDPOINT = os.getenv("AZURE_TTS_ENDPOINT", "https://francecentral.tts.speech.microsoft.com/cognitiveservices/v1")

ssml = (
    "<speak version='1.0' xml:lang='en-US'>"
    "<voice xml:lang='en-US' xml:gender='Female' name='en-US-AriaNeural'>"
    "Hello! This is a test of the Azure text to speech service."
    "</voice></speak>"
)

headers = {
    "Ocp-Apim-Subscription-Key": AZURE_TTS_KEY,
    "X-Microsoft-OutputFormat":  "raw-24khz-16bit-mono-pcm",
    "Content-Type":              "application/ssml+xml",
}

print("Sending request to Azure TTS...")
resp = httpx.post(AZURE_TTS_ENDPOINT, content=ssml, headers=headers, timeout=30.0)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    pcm = resp.content
    print(f"Got {len(pcm)} bytes of raw PCM")

    # Wrap in WAV header so you can play it
    sr, bits, channels = 24000, 16, 1
    data_size = len(pcm)
    wav_header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF", 36 + data_size, b"WAVE",
        b"fmt ", 16, 1, channels, sr, sr * channels * bits // 8, channels * bits // 8, bits,
        b"data", data_size,
    )
    out = "test_azure_output.wav"
    with open(out, "wb") as f:
        f.write(wav_header + pcm)
    print(f"Saved to {out} — open it to hear the audio")
else:
    print(f"Error: {resp.text}")
