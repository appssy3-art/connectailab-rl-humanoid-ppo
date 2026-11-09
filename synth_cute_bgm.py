import numpy as np
import wave
import struct

def generate_cute_shorts_bgm(filename="shorts_final_bgm.wav", duration=16.0, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    # 8-bit / Chiptune 스타일의 귀여운 멜로디 (C장조 펜타토닉 / 발랄한 아르페지오)
    bpm = 135
    beat_dur = 60.0 / bpm
    notes = [
        261.63, 329.63, 392.00, 523.25, # C, E, G, C5
        293.66, 349.23, 440.00, 587.33, # D, F, A, D5
        329.63, 392.00, 493.88, 659.25, # E, G, B, E5
        392.00, 523.25, 659.25, 783.99, # G, C5, E5, G5
    ]
    
    audio = np.zeros_like(t)
    
    # 아르페지오 멜로디 (Square Wave)
    step_dur = beat_dur / 2.0  # 8분음표
    num_steps = int(duration / step_dur)
    
    for i in range(num_steps):
        t_start = i * step_dur
        t_end = min((i + 1) * step_dur, duration)
        idx = (t >= t_start) & (t < t_end)
        t_local = t[idx] - t_start
        
        freq = notes[i % len(notes)]
        # 펄스/사각파 (레트로 8비트 사운드)
        lead = 0.25 * np.sign(np.sin(2 * np.pi * freq * t_local))
        # 감쇠 엔벨로프
        env = np.exp(-t_local * 6.0)
        audio[idx] += lead * env
        
    # 베이스라인 (Triangle Wave)
    bass_notes = [130.81, 146.83, 164.81, 196.00] # C3, D3, E3, G3
    bass_dur = beat_dur * 2.0 # 2박자 베이스
    num_bass = int(duration / bass_dur)
    for i in range(num_bass):
        t_start = i * bass_dur
        t_end = min((i + 1) * bass_dur, duration)
        idx = (t >= t_start) & (t < t_end)
        t_local = t[idx] - t_start
        bfreq = bass_notes[i % len(bass_notes)]
        bass = 0.3 * np.arcsin(np.sin(2 * np.pi * bfreq * t_local)) / (np.pi / 2)
        audio[idx] += bass
        
    # 킥 & 스네어 리듬
    for i in range(int(duration / beat_dur)):
        t_start = i * beat_dur
        idx = (t >= t_start) & (t < t_start + 0.12)
        t_local = t[idx] - t_start
        if i % 2 == 0:
            # 킥 드럼 (피치 드롭 싸인파)
            kick_freq = 120 * np.exp(-t_local * 35.0)
            audio[idx] += 0.35 * np.sin(2 * np.pi * kick_freq * t_local)
        else:
            # 스네어 드럼 (화이트 노이즈 팝)
            audio[idx] += 0.2 * (np.random.rand(len(t_local)) * 2 - 1) * np.exp(-t_local * 25.0)
            
    # 전체 마스터링 & 노멀라이즈
    audio = audio / np.max(np.abs(audio) + 1e-6) * 0.75
    audio_int16 = (audio * 32767).astype(np.int16)
    
    with wave.open(filename, 'w') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_int16.tobytes())
        
    print(f"🎵 [귀여운 8비트 쇼츠 BGM 합성 완료]: {filename} ({duration}s)")

if __name__ == '__main__':
    generate_cute_shorts_bgm()
