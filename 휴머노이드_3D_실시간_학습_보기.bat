@echo off
chcp 65001 > nul
cd /d "E:\ai랩\강화학습PPO"
echo =======================================================
echo 🤖 MuJoCo Humanoid-v5 실시간 3D GUI 학습 및 시연 시작
echo =======================================================
python train_humanoid_gui.py
pause

