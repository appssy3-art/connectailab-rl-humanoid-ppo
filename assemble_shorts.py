import subprocess
import cv2

def assemble_ultimate_shorts():
    print("🎬 [유튜브 쇼츠 최종 완제품 어셈블리 시작]...")
    
    # 각 클립을 정확히 3.5초~4초씩 트리밍하여 15초 황금 쇼츠 제작
    # 1. Gen 0: 0 ~ 3.5초 (꽈당)
    # 2. Gen 100: 0 ~ 3.5초 (첫 걸음마)
    # 3. Gen 500: 0 ~ 3.5초 (전진 보행)
    # 4. Gen 1000: 0 ~ 4.5초 (100만 걸음 스프린트)
    
    subprocess.run("ffmpeg -y -ss 0 -t 3.5 -i humanoid_gen0_fail.mp4 -c copy clip1.mp4", shell=True, check=True)
    subprocess.run("ffmpeg -y -ss 0 -t 3.5 -i humanoid_gen100_step.mp4 -c copy clip2.mp4", shell=True, check=True)
    subprocess.run("ffmpeg -y -ss 0 -t 3.5 -i humanoid_gen500_run.mp4 -c copy clip3.mp4", shell=True, check=True)
    subprocess.run("ffmpeg -y -ss 0 -t 4.5 -i humanoid_gen1000_sprint.mp4 -c copy clip4.mp4", shell=True, check=True)
    
    with open('ultimate_list.txt', 'w') as f:
        f.write("file 'clip1.mp4'\n")
        f.write("file 'clip2.mp4'\n")
        f.write("file 'clip3.mp4'\n")
        f.write("file 'clip4.mp4'\n")
        
    cmd = (
        "ffmpeg -y -f concat -safe 0 -i ultimate_list.txt -i shorts_final_bgm.wav "
        "-c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 192k -shortest "
        "humanoid_shorts_final_15s.mp4"
    )
    subprocess.run(cmd, shell=True, check=True)
    print("🎉 [15초 유튜브 쇼츠 최종 마스터본 제작 완료]: humanoid_shorts_final_15s.mp4")

if __name__ == '__main__':
    assemble_ultimate_shorts()
