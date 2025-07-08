# ArXiv论文追踪与分析器

一个基于 GitHub Actions 的自动化工具，每天早上自动追踪和分析 arXiv 最新论文，并通过邮件发送分析报告。该工具使用 ChatGPT 进行论文分析和总结。

## 功能特点

- 每天早上 8 点自动运行（UTC+8）
- 自动追踪最近发布的 AI、机器学习和 NLP 类别的论文
- 使用 ChatGPT 进行论文分析和总结
- 通过邮件发送分析报告
- 自动保存分析结果到 conclusion.md
- 自动清理下载的 PDF 文件以节省空间

## 安装与配置

1. Fork 或克隆仓库：
```bash
git clone https://github.com/你的用户名/arxiv_paper_tracker.git
cd arxiv_paper_tracker
```

2. 在 GitHub 仓库设置中配置 Secrets（Settings > Secrets and variables > Actions）：

需要添加以下 Secrets：
- `OPENAI_API_KEY`: OpenAI API 密钥
- `SMTP_SERVER`: 邮件服务器地址（如：smtp.qq.com）
- `SMTP_PORT`: 邮件服务器端口（如：587）
- `SMTP_USERNAME`: 邮箱账号
- `SMTP_PASSWORD`: 邮箱授权码
- `EMAIL_FROM`: 发件人邮箱
- `EMAIL_TO`: 收件人邮箱

3. 安装依赖（本地测试时需要）：
```bash
pip install -r requirements.txt
```

## 使用方法

### 自动运行
- 工作流会在每天早上 8 点（北京时间）自动运行
- 运行结果会：
  1. 发送到配置的邮箱
  2. 保存在 conclusion.md 文件中
  3. 自动提交到仓库

### 手动触发
1. 在仓库的 Actions 页面
2. 选择 "Daily Paper Analysis" 工作流
3. 点击 "Run workflow"
4. 选择 "Run workflow" 确认运行

## 配置说明

### 论文类别
默认追踪以下类别：
- cs.AI（人工智能）
- cs.LG（机器学习）
- cs.CL（计算语言学）

可以在 `src/main.py` 中修改 `CATEGORIES` 变量来调整追踪的论文类别。

### 邮件配置
支持主流邮箱服务：
- QQ 邮箱：需要在邮箱设置中开启 SMTP 服务并获取授权码
- Gmail：需要开启两步验证并生成应用专用密码
- 其他邮箱：需要确保支持 SMTP 服务

## 注意事项

- 确保 OpenAI API 密钥有效
- 邮箱配置正确（特别是授权码/应用专用密码）
- GitHub Actions 每月有 2000 分钟的免费额度，足够日常使用
- 如需修改运行时间，可以在 `.github/workflows/daily_paper_analysis.yml` 中调整 cron 表达式

## 许可证

MIT License 