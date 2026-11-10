# -*- coding: utf-8 -*-
"""
휴머노이드 0세대 ~ 250만세대 진화 쇼츠 - '현수 보이스' 나래이션 합성 자동화 스크립트
- Voice: ko-KR-HyunsuMultilingualNeural
- 5단계 클립별 싱크 맞춤 & BGM 볼륨 믹싱 (amix)
"""
import os, sys, asyncio, subprocess
import edge_tts

# 5단계 쇼츠 대본 ("3D 세상에" -> "걷는 법을 맡겨봤습니다"로 자연스럽게 전면 수정)
SCENES = [
    {
        'id': 'v1_gen0',
        'text': '아무것도 모르는 AI에게 걷는 법을 맡겨봤습니다. 시작하자마자 꽈당!',
        'video_src': 'humanoid_gen0_fail.mp4',
        'rate': '+12%',
        'pitch': '+0Hz'
    },
    {
        'id': 'v2_gen100',
        'text': '10만 번 넘어지더니, 팔을 벌려 필사적으로 중심을 잡기 시작합니다.',
        'video_src': 'humanoid_gen100_step.mp4',
        'rate': '+10%',
        'pitch': '+0Hz'
    },
    {
        'id': 'v3_gen500',
        'text': '50만 세대. 다리를 앞으로 뻗으며 드디어 첫 발을 내딛습니다.',
        'video_src': 'humanoid_gen500_run.mp4',
        'rate': '+10%',
        'pitch': '+1Hz'
    },
    {
        'id': 'v4_gen1000',
        'text': '100만 세대 돌파! 쓰러지지 않고 10초 연속 폭풍 질주를 시작합니다.',
        'video_src': 'humanoid_gen1000_sprint.mp4',
        'rate': '+10%',
        'pitch': '+1Hz'
    },
    {
        'id': 'v5_gen2500',
        'text': '그리고 250만 걸음 뒤, 시속 5.6km로 달리는 완벽한 마라토너가 되었습니다!',
        'video_src': 'humanoid_v5_marathon.mp4',
        'rate': '+10%',
        'pitch': '+2Hz'
    }
]

def get_duration(f):
    cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', f]
    return float(subprocess.run(cmd, capture_output=True, text=True).stdout.strip())

async def generate_voices():
    print("🎙️ [현수 보이스 나래이션 음성 합성 시작]...")
    for s in SCENES:
        mp3 = f"voice_{s['id']}.mp3"
        comm = edge_tts.Communicate(s['text'], voice='ko-KR-HyunsuMultilingualNeural', rate=s['rate'], pitch=s['pitch'])
        await comm.save(mp3)
        dur = get_duration(mp3)
        s['audio'] = mp3
        s['dur'] = dur
        print(f"  • {s['id']}: {dur:.2f}초 | \"{s['text']}\"")

def build_video():
    print("\n🎬 [영상 클립 음성 싱크 재단 및 합성]...")
    part_files = []
    
    for i, s in enumerate(SCENES):
        clip_dur = s['dur'] + 0.3
        part_mp4 = f"part_{i}_{s['id']}.mp4"
        
        cmd = [
            'ffmpeg', '-y',
            '-stream_loop', '2',
            '-i', s['video_src'],
            '-i', s['audio'],
            '-t', f"{clip_dur:.2f}",
            '-map', '0:v:0',
            '-map', '1:a:0',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-shortest',
            part_mp4
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        part_files.append(part_mp4)
        print(f"  ✅ 파트 {i+1} 완성 ({clip_dur:.2f}초): {part_mp4}")

    with open('concat_voice_list.txt', 'w', encoding='utf-8') as f:
        for p in part_files:
            f.write(f"file '{p}'\n")

    temp_voice_vid = "temp_voice_concat.mp4"
    subprocess.run([
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'concat_voice_list.txt',
        '-c', 'copy',
        temp_voice_vid
    ], check=True, capture_output=True)
    
    total_dur = get_duration(temp_voice_vid)
    print(f"\n📊 총 나래이션 비디오 길이: {total_dur:.2f}초")

    final_output = "휴머노이드_보이스_나래이션_쇼츠_최종본.mp4"
    filter_complex = "[1:a]volume=0.18[bgm];[0:a]volume=1.0[voice];[voice][bgm]amix=inputs=2:duration=first:dropout_transition=2[aout]"
    
    cmd_final = [
        'ffmpeg', '-y',
        '-i', temp_voice_vid,
        '-stream_loop', '-1',
        '-i', 'shorts_final_bgm.wav',
        '-filter_complex', filter_complex,
        '-map', '0:v:0',
        '-map', '[aout]',
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-c:a', 'aac',
        '-b:a', '192k',
        '-shortest',
        final_output
    ]
    subprocess.run(cmd_final, check=True)
    
    # 영문명 파일도 갱신
    import shutil
    shutil.copy(final_output, "humanoid_shorts_voice_final.mp4")
    print(f"\n🎉 [수정 완료]: {final_output} 및 humanoid_shorts_voice_final.mp4")

if __name__ == '__main__':
    asyncio.run(generate_voices())
    build_video()
