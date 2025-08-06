@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

set index=100

for %%f in (
    "data/“OLED”旋风__吹爆电子化学品行情_王文清_llm_correct.md"
    "data/“南京第壹有机光电OLED...块”科技成果鉴定会在京召开_llm_correct.md"
    "data/“胡锦鸟”平面起舞——Vi...75_TFT显示器测试手记_无威_llm_correct.md"
) do (
    set /a idx=index
    python raft.py ^
        --datapath "%%~f" ^
        --output outputs_test!idx! ^
        --output-format completion ^
        --distractors 3 ^
        --p 1.0 ^
        --doctype txt ^
        --chunk_size 512 ^
        --questions 2 ^
        --completion_model deepseek-r1-250120 ^
        --system-prompt-key deepseek-v2
    set /a index+=1
)

echo 所有任务已完成。
pause