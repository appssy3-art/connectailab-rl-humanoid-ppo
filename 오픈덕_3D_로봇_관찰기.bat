@echo off
chcp 949 >nul
title [Open Duck Mini v2] 3D 로봇 모델 관찰기
cd /d "%~dp0"
echo ========================================================
echo   [Open Duck Mini v2] 디즈니 BDX 오리 로봇 3D 뷰어
echo ========================================================
python view_open_duck.py
pause

