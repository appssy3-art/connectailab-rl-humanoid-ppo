@echo off
chcp 949 >nul
title [Open Duck Mini v2] 실시간 이족보행 시뮬레이터
cd /d "%~dp0"
echo ========================================================
echo   [Open Duck Mini v2] 디즈니 BDX 오리 로봇 보행 시뮬레이터
echo ========================================================
python walk_open_duck.py
pause

