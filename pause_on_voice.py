import time, sys, queue
import numpy as np
import sounddevice as sd
from scipy.signal import butter, lfilter
import keyboard

# ---------- Config ----------
SAMPLE_RATE = 16000
FRAME_MS = 30                     # 30 ms frames
BLOCK = int(SAMPLE_RATE * FRAME_MS / 1000)
PRESS_COOLDOWN = 2.5              # seconds between space taps
ONSET_FRAMES = 4                  # consecutive speech frames required (debounce)
RELEASE_FRAMES = 8                # frames of silence to reset
BOOST_DB = 12.0                   # how much above baseline to consider speech
ZCR_RANGE = (0.02, 0.20)          # human speech zero-crossing rate range
BP_LO, BP_HI = 100, 4000          # bandpass (Hz) to drop low rumbles & hiss
ADAPT_SECS = 2.0                  # baseline adaptation time at start

# ---------- State ----------
q = queue.Queue()
last_press = 0.0
speech_run = 0
silence_run = 0
have_baseline = False
baseline_db = -60.0
baseline_alpha = 0.995  # slow EWMA for baseline after warm-up

def butter_bandpass(lo, hi, fs, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lo/nyq, hi/nyq], btype='band')
    return b, a

B, A = butter_bandpass(BP_LO, BP_HI, SAMPLE_RATE)

def bandpass(x):
    return lfilter(B, A, x)

def frame_features(x):
    # x is int16 mono
    xf = x.astype(np.float32) / 32768.0
    xf = bandpass(xf)

    # Short-term energy (RMS -> dB)
    rms = np.sqrt(np.mean(xf * xf) + 1e-10)
    db = 20.0 * np.log10(rms + 1e-12)

    # Zero-crossing rate
    signs = np.sign(xf)
    zc = np.mean(signs[:-1] != signs[1:]) if len(signs) > 1 else 0.0
    return db, zc

def cb(indata, frames, time_info, status):
    if status:
        # you can print(status) for debugging
        pass
    q.put(indata.copy())

def main():
    global last_press, speech_run, silence_run, have_baseline, baseline_db

    print("Listening… (will tap SPACE on voice)")
    start_ts = time.time()

    with sd.InputStream(channels=1, samplerate=SAMPLE_RATE, dtype='int16',
                        blocksize=BLOCK, callback=cb):
        while True:
            buf = q.get()
            # process in exact frame sizes
            data = buf.reshape(-1)
            # if audio callback gives larger blocks, chunk them
            offset = 0
            while offset + BLOCK <= data.size:
                frame = data[offset:offset+BLOCK]
                offset += BLOCK

                db, zcr = frame_features(frame)

                # Warm-up baseline using first ADAPT_SECS
                if not have_baseline:
                    baseline_db = min(baseline_db, db)  # start conservative
                    if time.time() - start_ts >= ADAPT_SECS:
                        have_baseline = True
                    continue  # don’t trigger during warm-up

                # Update baseline slowly (follows room noise drift)
                baseline_db = baseline_alpha * baseline_db + (1 - baseline_alpha) * db

                # Speech when energy clearly above noise AND zcr in plausible speech range
                is_speech = (db >= baseline_db + BOOST_DB) and (ZCR_RANGE[0] <= zcr <= ZCR_RANGE[1])

                if is_speech:
                    speech_run += 1
                    silence_run = 0
                else:
                    silence_run += 1
                    speech_run = 0 if silence_run >= RELEASE_FRAMES else speech_run

                # Trigger on onset
                now = time.time()
                if speech_run >= ONSET_FRAMES and (now - last_press) > PRESS_COOLDOWN:
                    try:
                        keyboard.press_and_release("space")
                        last_press = now
                        speech_run = 0
                        print(">> Voice detected — tapped SPACE")
                    except Exception as e:
                        print(f"Keyboard error: {e}", file=sys.stderr)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")
