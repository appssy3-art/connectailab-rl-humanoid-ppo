@echo off
chcp 949 >nul
title [Open Duck Mini v2] 학습된 3D 로봇 관찰기
cd /d "%~dp0"
echo ========================================================
echo   [Open Duck Mini v2] 강화학습 기립 & 보행 3D 뷰어
echo ========================================================
python eval_open_duck.py
pause
