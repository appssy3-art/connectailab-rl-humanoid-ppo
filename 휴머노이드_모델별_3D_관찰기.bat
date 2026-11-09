@echo off
chcp 65001 > nul
title MuJoCo 휴머노이드 3D 실시간 관찰기

:MENU
cls
echo ========================================================
echo   🤖 CONNECT AI LAB - MuJoCo 휴머노이드 3D 실시간 관찰기
echo ========================================================
echo.
echo   보고 싶은 모델 버전을 선택하세요:
echo.
echo     [1] v1 모델 : 10만 걸음 (제자리 서서 균형 잡기)
echo     [2] v2 모델 : 40만 걸음 (첫 걸음마 및 전진 시도)
echo     [3] v3 모델 : 100만 걸음 (10초 연속 고속 질주 대기록)
echo     [4] v4 모델 : 150만 걸음 (1.45m/s 자연스러운 인간 보행)
echo     [Q] 종료
echo.
echo ========================================================
set /p sel="번호를 입력하세요 (1~4): "

if "%sel%"=="1" goto RUN_V1
if "%sel%"=="2" goto RUN_V2
if "%sel%"=="3" goto RUN_V3
if "%sel%"=="4" goto RUN_V4
if /i "%sel%"=="q" exit
goto MENU

:RUN_V1
cls
python view_humanoid_models.py 1
pause
goto MENU

:RUN_V2
cls
python view_humanoid_models.py 2
pause
goto MENU

:RUN_V3
cls
python view_humanoid_models.py 3
pause
goto MENU

:RUN_V4
cls
python view_humanoid_models.py 4
pause
goto MENU
