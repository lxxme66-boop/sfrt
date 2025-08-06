# 文本文件处理脚本使用说明

本项目提供了多个脚本来处理包含文件列表的文本文件。

## 脚本文件

### 1. `process_txt_files.py` - 通用文本文件处理器
这是核心的 Python 脚本，可以处理任何包含文件列表的文本文件。

**功能特性:**
- 读取文本文件中的文件列表
- 在指定目录中搜索这些文件
- 生成详细的处理报告
- 支持多种操作模式（列表、统计行数等）
- 完整的日志记录

**直接使用方法:**
```bash
# 处理默认文件 selected_papers.txt
python3 process_txt_files.py

# 处理指定文件
python3 process_txt_files.py filename.txt

# 统计文件行数
python3 process_txt_files.py filename.txt --operation count_lines

# 指定搜索目录
python3 process_txt_files.py filename.txt --search-dirs /path/to/dir1 /path/to/dir2

# 指定输出报告文件名
python3 process_txt_files.py filename.txt --output my_report.txt
```

### 2. `run.sh` - 简化的运行脚本
这是一个 Bash 脚本，提供了更简单的接口来运行文本处理器。

**使用方法:**
```bash
# 显示帮助信息
./run.sh --help

# 处理默认文件
./run.sh

# 处理指定文件
./run.sh filename.txt

# 处理文件并统计行数
./run.sh filename.txt --count
```

### 3. `run_papers_processor.py` - 专门的论文处理器
专门用于处理 `selected_papers.txt` 和 `selected_papers_2.txt` 文件的脚本。

**使用方法:**
```bash
python3 run_papers_processor.py
```

### 4. `run_papers.sh` - 论文处理运行脚本
用于运行论文处理器的 Bash 脚本。

**使用方法:**
```bash
./run_papers.sh
```

## 输出文件

脚本运行后会生成以下文件：

1. **处理报告** (`processing_summary.txt` 或 `papers_processing_report.txt`)
   - 包含完整的处理统计信息
   - 列出找到的文件和缺失的文件
   - 提供处理完整性百分比

2. **日志文件** (`txt_processing.log` 或 `papers_processing.log`)
   - 详细的执行日志
   - 错误信息和调试信息

## 使用示例

### 处理论文文件列表
```bash
# 使用简化脚本
./run.sh selected_papers.txt

# 或使用专门的论文处理器
./run_papers.sh
```

### 处理 requirements.txt
```bash
./run.sh requirements.txt
```

### 处理任意文本文件
```bash
# 假设你有一个名为 file_list.txt 的文件
./run.sh file_list.txt

# 如果需要统计行数
./run.sh file_list.txt --count
```

## 配置选项

### 搜索目录
默认搜索目录为：
- `/workspace/data`
- `/workspace/data_emb`
- `/workspace`

可以通过 `--search-dirs` 参数修改搜索目录。

### 操作模式
- `list`: 仅列出文件信息（默认）
- `count_lines`: 统计每个文件的行数

## 故障排除

### 常见问题

1. **权限错误**
   ```bash
   chmod +x run.sh
   chmod +x run_papers.sh
   ```

2. **Python 模块缺失**
   ```bash
   pip install -r requirements.txt
   ```

3. **文件编码问题**
   脚本使用 UTF-8 编码，确保输入文件也是 UTF-8 编码。

### 日志查看
如果遇到问题，可以查看日志文件获取详细信息：
```bash
cat txt_processing.log
# 或
cat papers_processing.log
```

## 扩展功能

如需添加新的处理功能，可以修改 `process_txt_files.py` 中的 `process_files` 函数，添加新的操作模式。