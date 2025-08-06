$files = @(
    "data/“OLED”旋风__吹爆电子化学品行情_王文清_llm_correct.md",
    "data/“南京第壹有机光电OLED...块”科技成果鉴定会在京召开_llm_correct.md"
)
$index = 11

foreach ($file in $files) {
    $filename = [System.IO.Path]::GetFileNameWithoutExtension($file)
    $outputDir = "outputs_test$index"

    python raft.py `
        --datapath "$file" `
        --output $outputDir `
        --output-format completion `
        --distractors 3 `
        --p 1.0 `
        --doctype txt `
        --chunk_size 512 `
        --questions 2 `
        --completion_model deepseek-r1-250120 `
        --system-prompt-key deepseek-v2
    $index++
}

Write-Host "所有任务已完成。"