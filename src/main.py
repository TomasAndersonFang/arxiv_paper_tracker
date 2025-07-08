#!/usr/bin/env python3
# ArXiv论文追踪与分析器

import os
import arxiv
import datetime
from pathlib import Path
import openai
import time
import logging
import sys
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from jinja2 import Template

# 加载环境变量
load_dotenv()

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                   handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# 配置
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SMTP_SERVER = os.getenv("SMTP_SERVER")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD")
EMAIL_FROM = os.getenv("EMAIL_FROM")
# 支持多个收件人邮箱，用逗号分隔
EMAIL_TO = [email.strip() for email in os.getenv("EMAIL_TO", "").split(",") if email.strip()]

PAPERS_DIR = Path("./papers")
CONCLUSION_FILE = Path("./conclusion.md")
CATEGORIES = ["cs.SE"]
MAX_PAPERS_SEARCH = 30  # 每次搜索的论文数量
MAX_PAPERS_ANALYZE = 5  # 每次分析的论文数量

# 配置OpenAI API
openai.api_key = OPENAI_API_KEY

# 如果不存在论文目录则创建
PAPERS_DIR.mkdir(exist_ok=True)
logger.info(f"论文将保存在: {PAPERS_DIR.absolute()}")
logger.info(f"分析结果将写入: {CONCLUSION_FILE.absolute()}")

def get_recent_papers(categories, max_results=MAX_PAPERS_SEARCH):
    """获取最近发布的指定类别的论文"""
    try:
        # 简化查询，只按类别搜索，避免复杂的日期范围查询
        category_query = " OR ".join([f"cat:{cat}" for cat in categories])
        
        logger.info(f"正在搜索论文，查询条件: {category_query}")
        
        # 使用新的Client API
        client = arxiv.Client()
        search = arxiv.Search(
            query=category_query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending
        )
        
        results = list(client.results(search))
        logger.info(f"找到{len(results)}篇符合条件的论文")
        
        # 过滤最近几天的论文
        today = datetime.datetime.now()
        recent_cutoff = today - datetime.timedelta(days=7)  # 扩大到7天
        
        recent_papers = []
        for paper in results:
            # 确保published字段是datetime对象
            if hasattr(paper.published, 'replace'):
                # 移除时区信息进行比较
                paper_date = paper.published.replace(tzinfo=None)
                if paper_date >= recent_cutoff:
                    recent_papers.append(paper)
        
        logger.info(f"最近7天内发布的论文: {len(recent_papers)}篇")
        return recent_papers
        
    except Exception as e:
        logger.error(f"搜索论文时发生错误: {str(e)}")
        # 如果搜索失败，尝试更简单的查询
        try:
            logger.info("尝试简化查询...")
            client = arxiv.Client()
            search = arxiv.Search(
                query=categories[0],  # 只搜索第一个类别
                max_results=max_results,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending
            )
            results = list(client.results(search))
            logger.info(f"简化查询找到{len(results)}篇论文")
            return results[:max_results]  # 返回限定数量的结果
        except Exception as e2:
            logger.error(f"简化查询也失败: {str(e2)}")
            return []

def download_paper(paper, output_dir):
    """将论文PDF下载到指定目录"""
    pdf_path = output_dir / f"{paper.get_short_id().replace('/', '_')}.pdf"
    
    # 如果已下载则跳过
    if pdf_path.exists():
        logger.info(f"论文已下载: {pdf_path}")
        return pdf_path
    
    try:
        logger.info(f"正在下载: {paper.title}")
        paper.download_pdf(filename=str(pdf_path))
        logger.info(f"已下载到 {pdf_path}")
        return pdf_path
    except Exception as e:
        logger.error(f"下载论文失败 {paper.title}: {str(e)}")
        return None

def analyze_paper_with_chatgpt(pdf_path, paper):
    """使用ChatGPT API分析论文（使用OpenAI 0.28.0兼容格式）"""
    try:
        # 从Author对象中提取作者名
        author_names = [author.name for author in paper.authors]
        
        prompt = f"""
        Paper Title: {paper.title}
        Authors: {', '.join(author_names)}
        Categories: {', '.join(paper.categories)}
        Published: {paper.published}
        
        Please analyze this research paper and provide a CONCISE review in the following structured format. Keep each section brief and focused:

        #### Executive Summary
        Write 2-3 sentences summarizing the core problem, approach, and main result.

        ### Key Contributions
        - List 2-3 main contributions (one line each)
        - Focus on what's genuinely novel

        ### Method & Results
        - Core methodology in 1-2 bullet points
        - Key datasets/tools used (if any)
        - Main experimental results (quantitative when possible)
        - Performance compared to baselines (if reported)

        ### Impact & Limitations
        - Practical significance (1-2 sentences)
        - Main limitations or future work directions (1-2 points)

        Keep the entire analysis under 200 words. Be precise and avoid redundancy. Use bullet points for clarity.
        """
        
        logger.info(f"正在分析论文: {paper.title}")
        response = openai.ChatCompletion.create(
            model="gpt-4.1-mini",
            messages=[
                {"role": "system", "content": "You are a research assistant specialized in summarizing and analyzing academic papers. Please provide structured, comprehensive analysis in English."},
                {"role": "user", "content": prompt},
            ]
        )
        
        analysis = response.choices[0].message.content
        logger.info(f"论文分析完成: {paper.title}")
        return analysis
    except Exception as e:
        logger.error(f"分析论文失败 {paper.title}: {str(e)}")
        return f"**Paper Analysis Error**: {str(e)}"

def clean_duplicate_entries():
    """清理conclusion.md中的重复条目"""
    if not CONCLUSION_FILE.exists():
        logger.info("conclusion.md文件不存在，无需清理")
        return
    
    try:
        with open(CONCLUSION_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分割为不同的论文条目
        import re
        sections = re.split(r'\n### (.+?)\n', content)
        
        # 第一个section是文件头部，保留
        cleaned_content = sections[0] if sections else ""
        
        seen_papers = set()
        
        # 处理每个论文条目
        for i in range(1, len(sections), 2):
            if i + 1 < len(sections):
                paper_title = sections[i].strip()
                paper_content = sections[i + 1]
                
                # 提取论文ID
                arxiv_match = re.search(r'https?://arxiv\.org/abs/([^)\s\n]+)', paper_content)
                if arxiv_match:
                    paper_id = arxiv_match.group(1)
                    paper_id_no_version = re.sub(r'v\d+$', '', paper_id)
                    
                    # 检查是否已处理过
                    if paper_id not in seen_papers and paper_id_no_version not in seen_papers:
                        seen_papers.add(paper_id)
                        seen_papers.add(paper_id_no_version)
                        cleaned_content += f"\n### {paper_title}\n{paper_content}"
                    else:
                        logger.info(f"发现重复论文，已移除: {paper_title}")
                else:
                    # 如果没有找到arxiv链接，保留条目
                    cleaned_content += f"\n### {paper_title}\n{paper_content}"
        
        # 写回文件
        with open(CONCLUSION_FILE, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
            
        logger.info("重复条目清理完成")
        
    except Exception as e:
        logger.error(f"清理重复条目时出错: {str(e)}")

def get_analyzed_papers():
    """获取已分析过的论文ID列表"""
    if not CONCLUSION_FILE.exists():
        return set()
    
    analyzed_papers = set()
    try:
        with open(CONCLUSION_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
            import re
            
            # 匹配所有arxiv链接格式 (http和https都支持)
            arxiv_links = re.findall(r'https?://arxiv\.org/abs/([^)\s\n]+)', content)
            analyzed_papers.update(arxiv_links)
            
            # 也匹配直接的paper ID格式
            paper_ids = re.findall(r'arxiv:([^)\s\n]+)', content)
            analyzed_papers.update(paper_ids)
            
            # 去除版本号后缀，统一格式（如 2507.05245v1 -> 2507.05245）
            normalized_ids = set()
            for paper_id in analyzed_papers:
                # 移除版本号后缀 (如 v1, v2 等)
                normalized_id = re.sub(r'v\d+$', '', paper_id)
                normalized_ids.add(normalized_id)
                # 同时保留原始ID
                normalized_ids.add(paper_id)
            
            analyzed_papers = normalized_ids
            
    except Exception as e:
        logger.error(f"读取已分析论文列表时出错: {str(e)}")
        return set()
    
    logger.info(f"已分析过的论文数量: {len(analyzed_papers)}")
    if analyzed_papers:
        logger.info(f"已分析论文ID示例: {list(analyzed_papers)[:3]}")
    return analyzed_papers

def write_to_conclusion(papers_analyses):
    """将分析结果写入conclusion.md"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 创建或追加到结果文件
    with open(CONCLUSION_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n\n## ArXiv论文 - 最近7天 (截至 {today})\n\n")
        
        for paper, analysis in papers_analyses:
            # 从Author对象中提取作者名
            author_names = [author.name for author in paper.authors]
            
            f.write(f"### {paper.title}\n")
            f.write(f"**作者**: {', '.join(author_names)}\n")
            f.write(f"**类别**: {', '.join(paper.categories)}\n")
            f.write(f"**发布日期**: {paper.published.strftime('%Y-%m-%d')}\n")
            f.write(f"**链接**: {paper.entry_id}\n\n")
            f.write(f"{analysis}\n\n")
            f.write("---\n\n")
    
    logger.info(f"分析结果已写入 {CONCLUSION_FILE}")

def format_email_content(papers_analyses):
    """格式化邮件内容，只包含当天分析的论文"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    content = f"# ArXiv Paper Analysis Report ({today})\n\n"
    
    for paper, analysis in papers_analyses:
        # 从Author对象中提取作者名
        author_names = [author.name for author in paper.authors]
        
        content += f"## {paper.title}\n\n"
        content += f"**Authors**: {', '.join(author_names)}\n"
        content += f"**Categories**: {', '.join(paper.categories)}\n"
        content += f"**Published**: {paper.published.strftime('%Y-%m-%d')}\n"
        content += f"**ArXiv Link**: {paper.entry_id}\n\n"
        content += f"{analysis}\n\n"
        content += "---\n\n"
    
    return content

def delete_pdf(pdf_path):
    """删除PDF文件"""
    try:
        if pdf_path.exists():
            pdf_path.unlink()
            logger.info(f"已删除PDF文件: {pdf_path}")
        else:
            logger.info(f"PDF文件不存在，无需删除: {pdf_path}")
    except Exception as e:
        logger.error(f"删除PDF文件失败 {pdf_path}: {str(e)}")

def send_email(content):
    """发送邮件，支持多个收件人"""
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USERNAME, SMTP_PASSWORD, EMAIL_FROM]) or not EMAIL_TO:
        logger.error("邮件配置不完整，跳过发送邮件")
        return
    
    # 确保所有必需的配置都不为None
    if not SMTP_SERVER or not SMTP_USERNAME or not SMTP_PASSWORD or not EMAIL_FROM:
        logger.error("邮件配置存在空值，跳过发送邮件")
        return

    try:
        msg = MIMEMultipart()
        msg['From'] = EMAIL_FROM
        msg['To'] = ", ".join(EMAIL_TO)
        msg['Subject'] = f"ArXiv Paper Analysis Report - {datetime.datetime.now().strftime('%Y-%m-%d')}"

        # 改进的HTML模板，规范化标题样式
        html_template = """
        <html>
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1000px;
                    margin: 0 auto;
                    padding: 20px;
                    background-color: #f5f5f5;
                    color: #333;
                }
                .container {
                    background-color: white;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                }
                /* 规范化标题样式 - 确保一致的字体大小 */
                h1 {
                    color: #2c3e50;
                    font-size: 24px;
                    border-bottom: 2px solid #3498db;
                    padding-bottom: 10px;
                    margin: 30px 0 20px 0;
                }
                h2 {
                    color: #34495e;
                    font-size: 20px;
                    margin: 25px 0 15px 0;
                    padding-bottom: 8px;
                    border-bottom: 1px solid #eee;
                }
                h3 {
                    color: #2980b9;
                    font-size: 18px;
                    margin: 20px 0 10px 0;
                }
                h4 {
                    color: #2c3e50;
                    font-size: 16px;
                    margin: 15px 0 10px 0;
                    font-weight: 600;
                }
                .paper-header {
                    background-color: #f8f9fa;
                    padding: 15px;
                    border-left: 4px solid #3498db;
                    margin: 20px 0;
                    border-radius: 4px;
                }
                .paper-info {
                    margin: 5px 0;
                    font-size: 14px;
                }
                .paper-info strong {
                    color: #2c3e50;
                }
                a {
                    color: #3498db;
                    text-decoration: none;
                }
                a:hover {
                    text-decoration: underline;
                }
                hr {
                    border: none;
                    border-top: 1px solid #eee;
                    margin: 30px 0;
                }
                ul, ol {
                    margin: 10px 0;
                    padding-left: 20px;
                }
                li {
                    margin: 5px 0;
                }
                .section-content {
                    margin-bottom: 20px;
                }
                strong {
                    color: #2c3e50;
                    font-weight: 600;
                }
            </style>
        </head>
        <body>
            <div class="container">
                {{ content }}
            </div>
        </body>
        </html>
        """
        
        # 改进的Markdown到HTML转换逻辑
        content_html = content
        
        # 按顺序处理标题（从最长到最短，避免重复替换）
        content_html = content_html.replace("#### ", "<h4>")
        content_html = content_html.replace("### ", "<h3>")
        content_html = content_html.replace("## ", "<h2>")
        content_html = content_html.replace("# ", "<h1>")
        
        # 处理加粗文本
        import re
        content_html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', content_html)
        
        # 处理列表项
        content_html = re.sub(r'^- (.+)$', r'<li>\1</li>', content_html, flags=re.MULTILINE)
        content_html = re.sub(r'(<li>.*</li>)', r'<ul>\1</ul>', content_html, flags=re.DOTALL)
        
        # 处理段落分隔
        content_html = content_html.replace("\n\n", "</p><p>")
        content_html = "<p>" + content_html + "</p>"
        
        # 处理分隔线
        content_html = content_html.replace("---", "<hr>")
        
        # 修复一些HTML标签问题
        content_html = content_html.replace("<p><h", "<h")
        content_html = content_html.replace("</h1></p>", "</h1>")
        content_html = content_html.replace("</h2></p>", "</h2>")
        content_html = content_html.replace("</h3></p>", "</h3>")
        content_html = content_html.replace("</h4></p>", "</h4>")
        content_html = content_html.replace("<p><hr>", "<hr>")
        content_html = content_html.replace("<hr></p>", "<hr>")
        content_html = content_html.replace("<p><ul>", "<ul>")
        content_html = content_html.replace("</ul></p>", "</ul>")
        
        template = Template(html_template)
        html_content = template.render(content=content_html)
        
        msg.attach(MIMEText(html_content, 'html'))

        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.send_message(msg)

        logger.info(f"邮件发送成功，收件人: {', '.join(EMAIL_TO)}")
    except Exception as e:
        logger.error(f"发送邮件失败: {str(e)}")

def main():
    logger.info("开始ArXiv论文跟踪")
    
    # 清理重复条目
    clean_duplicate_entries()
    
    # 获取已分析过的论文ID列表
    analyzed_papers = get_analyzed_papers()
    
    # 获取最近7天的论文
    papers = get_recent_papers(CATEGORIES, MAX_PAPERS_SEARCH)
    logger.info(f"从最近7天找到{len(papers)}篇论文")
    logger.info(f"搜索配置: 搜索{MAX_PAPERS_SEARCH}篇，每次分析{MAX_PAPERS_ANALYZE}篇")
    
    if not papers:
        logger.info("所选时间段没有找到论文。退出。")
        return
    
    # 过滤出未分析过的论文，按发布时间排序（最新的在前）
    new_papers = []
    for paper in papers:
        paper_id = paper.get_short_id()
        
        # 检查多种格式的匹配
        import re
        paper_id_no_version = re.sub(r'v\d+$', '', paper_id)  # 移除版本号
        
        # 检查是否已分析过（检查原始ID、无版本号ID）
        is_analyzed = (paper_id in analyzed_papers or 
                      paper_id_no_version in analyzed_papers)
        
        if not is_analyzed:
            new_papers.append(paper)
            logger.info(f"新论文待分析: {paper.title} ({paper_id})")
        else:
            logger.info(f"论文已分析过，跳过: {paper.title} ({paper_id})")
    
    # 按发布时间排序新论文（最新的在前）
    new_papers.sort(key=lambda p: p.published, reverse=True)
    
    logger.info(f"发现{len(new_papers)}篇新论文")
    
    # 只分析指定数量的论文
    papers_to_analyze = new_papers[:MAX_PAPERS_ANALYZE]
    logger.info(f"本次将分析{len(papers_to_analyze)}篇论文")
    
    if len(new_papers) > MAX_PAPERS_ANALYZE:
        logger.info(f"还有{len(new_papers) - MAX_PAPERS_ANALYZE}篇论文待下次分析")
    
    if not papers_to_analyze:
        logger.info("没有新论文需要分析。退出。")
        return
    
    # 处理每篇新论文
    papers_analyses = []
    for i, paper in enumerate(papers_to_analyze, 1):
        logger.info(f"正在处理论文 {i}/{len(papers_to_analyze)}: {paper.title}")
        # 下载论文
        pdf_path = download_paper(paper, PAPERS_DIR)
        if pdf_path:
            # 休眠以避免达到API速率限制
            time.sleep(2)
            
            # 分析论文
            analysis = analyze_paper_with_chatgpt(pdf_path, paper)
            papers_analyses.append((paper, analysis))
            
            # 分析完成后删除PDF文件
            delete_pdf(pdf_path)
    
    # 将分析结果写入conclusion.md（包含所有历史记录）
    if papers_analyses:
        write_to_conclusion(papers_analyses)
        
        # 发送邮件（只包含当天分析的论文）
        email_content = format_email_content(papers_analyses)
        send_email(email_content)
        
        logger.info("ArXiv论文追踪和分析完成")
        logger.info(f"结果已保存至 {CONCLUSION_FILE.absolute()}")
        
        # 显示待分析论文队列状态
        remaining_papers = new_papers[MAX_PAPERS_ANALYZE:]
        if remaining_papers:
            logger.info(f"待分析论文队列 ({len(remaining_papers)}篇):")
            for i, paper in enumerate(remaining_papers[:5], 1):  # 只显示前5篇
                logger.info(f"  {i}. {paper.title} ({paper.get_short_id()})")
            if len(remaining_papers) > 5:
                logger.info(f"  ...还有{len(remaining_papers) - 5}篇")
                
        # 运行总结
        logger.info("=== 运行总结 ===")
        logger.info(f"本次搜索: {len(papers)}篇论文")
        logger.info(f"已分析总数: {len(analyzed_papers)}篇")
        logger.info(f"发现新论文: {len(new_papers)}篇")
        logger.info(f"本次分析: {len(papers_analyses)}篇")
        logger.info(f"队列待分析: {len(new_papers) - len(papers_analyses)}篇")
        logger.info("===============")
    else:
        logger.info("没有成功分析的论文。")

if __name__ == "__main__":
    main()
