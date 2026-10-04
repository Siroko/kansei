@echo off
rem Kimodo batch generation in Docker on a Windows PC with an NVIDIA GPU (Docker Desktop, WSL2).
rem
rem   generate.cmd setup                  clone Kimodo at the pinned commit, build the image
rem   generate.cmd <set> [batch options]  generate prompts\<set>.json into out\<set>
rem
rem Everything lives in %GENANIM% (default %USERPROFILE%\kansei-genanim): kimodo\ (the checkout),
rem genanim\ (this folder's parent: scripts and prompts, copied there), hf\ (Hugging Face cache:
rem the Kimodo weights and the Llama 3 text encoder) and out\ (the raw clips). Kimodo's text
rem encoder is gated: set HF_TOKEN in the environment to a Hugging Face token whose account
rem accepted Meta's Llama 3 licence; the container gets it from there (never write it to a
rem file). Delete the folder and the image to remove it all.
setlocal
if "%GENANIM%"=="" set GENANIM=%USERPROFILE%\kansei-genanim
set KIMODO_COMMIT=58e781898b3d7e328a676a75d3e338c45dce3ad9
set IMAGE=kansei-genanim-kimodo:58e7818

if "%1"=="setup" goto setup
if "%1"=="" goto usage
docker run --rm --gpus all --shm-size=8g -e KIMODO_COMMIT=%KIMODO_COMMIT% -e HF_TOKEN -e HF_HOME=/hf -e TEXT_ENCODER_MODE=local ^
  -v "%GENANIM%\hf:/hf" -v "%GENANIM%\genanim:/genanim:ro" -v "%GENANIM%\out:/out" ^
  %IMAGE% python /genanim/kimodo_batch.py /genanim/prompts/%1.json /out/%1 %2 %3 %4 %5 %6
goto :eof

:setup
if not exist "%GENANIM%\kimodo" git clone https://github.com/nv-tlabs/kimodo.git "%GENANIM%\kimodo" || exit /b 1
git -C "%GENANIM%\kimodo" checkout --quiet %KIMODO_COMMIT% || exit /b 1
copy /y "%GENANIM%\genanim\pc\Dockerfile" "%GENANIM%\kimodo\Dockerfile.kansei" >nul
docker build -f "%GENANIM%\kimodo\Dockerfile.kansei" -t %IMAGE% "%GENANIM%\kimodo" || exit /b 1
if not exist "%GENANIM%\hf" mkdir "%GENANIM%\hf"
if not exist "%GENANIM%\out" mkdir "%GENANIM%\out"
goto :eof

:usage
echo usage: generate.cmd setup ^| generate.cmd ^<set^> [--only PATTERN] [--samples N]
exit /b 2
