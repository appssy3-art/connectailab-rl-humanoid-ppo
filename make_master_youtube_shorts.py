import subprocess
import os

def build_grand_master_shorts():
    print("🎬 [유튜브 쇼츠 0세대 ~ 250만세대 완전체 마스터본 제작 시작]...")
    
    # 5단계 클립 트리밍 (총 16초)
    # 1. 0세대: 0~3.0초 (꽈당)
    # 2. 100세대: 0~3.0초 (걸음마)
    # 3. 500세대: 0~3.0초 (전진)
    # 4. 1000세대: 0~3.5초 (질주)
    # 5. 2500세대: 0~3.5초 (마라톤 조거)
    
    subprocess.run("ffmpeg -y -ss 0 -t 3.0 -i humanoid_gen0_fail.mp4 -c:v libx264 -pix_fmt yuv420p g1.mp4", shell=True, check=True)
    subprocess.run("ffmpeg -y -ss 0 -t 3.0 -i humanoid_gen100_step.mp4 -c:v libx264 -pix_fmt yuv420p g2.mp4", shell=True, check=True)
    subprocess.run("ffmpeg -y -ss 0 -t 3.0 -i humanoid_gen500_run.mp4 -c:v libx264 -pix_fmt yuv420p g3.mp4", shell=True, check=True)
    subprocess.run("ffmpeg -y -ss 0 -t 3.5 -i humanoid_gen1000_sprint.mp4 -c:v libx264 -pix_fmt yuv420p g4.mp4", shell=True, check=True)
    subprocess.run("ffmpeg -y -ss 0 -t 3.5 -i humanoid_v5_marathon.mp4 -c:v libx264 -pix_fmt yuv420p g5.mp4", shell=True, check=True)
    
    with open('grand_list.txt', 'w') as f:
        f.write("file 'g1.mp4'\n")
        f.write("file 'g2.mp4'\n")
        f.write("file 'g3.mp4'\n")
        f.write("file 'g4.mp4'\n")
        f.write("file 'g5.mp4'\n")
        
    out_mp4 = "휴머노이드_250만세대_진화_쇼츠_마스터본.mp4"
    cmd = (
        f"ffmpeg -y -f concat -safe 0 -i grand_list.txt -i shorts_final_bgm.wav "
        f"-c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest "
        f"{out_mp4}"
    )
    subprocess.run(cmd, shell=True, check=True)
    print(f"🎉 [유튜브 쇼츠 최종 마스터본 제작 완료]: {out_mp4} (16.0s)")

if __name__ == '__main__':
    build_grand_master_shorts()

