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
# 配置不同领域的类别
CATEGORY_CONFIGS = {
    "软件工程": {
        "categories": ["cs.SE"],
        "max_search": 30,
        "max_analyze": 5
    },
    "安全领域": {
        "categories": ["cs.CR"],  # cs.CR: Cryptography and Security, cs.CY: Computers and Society
        "max_search": 30,
        "max_analyze": 5
    }
}

# 保持向后兼容的全局配置（从第一个配置中取值）
CATEGORIES = CATEGORY_CONFIGS["软件工程"]["categories"]  
MAX_PAPERS_SEARCH = 30  # 每个领域搜索的论文数量
MAX_PAPERS_ANALYZE = 5  # 每个领域分析的论文数量

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

def write_to_conclusion_with_domains(papers_analyses):
    """将分析结果按领域写入conclusion.md"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    # 按领域组织论文
    domain_papers = {}
    for paper, analysis, domain in papers_analyses:
        if domain not in domain_papers:
            domain_papers[domain] = []
        domain_papers[domain].append((paper, analysis))
    
    # 创建或追加到结果文件
    with open(CONCLUSION_FILE, 'a', encoding='utf-8') as f:
        f.write(f"\n\n## ArXiv论文 - 最近7天 (截至 {today})\n\n")
        
        for domain, papers in domain_papers.items():
            f.write(f"### {domain} 领域\n\n")
            
            for paper, analysis in papers:
                # 从Author对象中提取作者名
                author_names = [author.name for author in paper.authors]
                
                f.write(f"#### {paper.title}\n")
                f.write(f"**作者**: {', '.join(author_names)}\n")
                f.write(f"**类别**: {', '.join(paper.categories)}\n")
                f.write(f"**发布日期**: {paper.published.strftime('%Y-%m-%d')}\n")
                f.write(f"**链接**: {paper.entry_id}\n\n")
                f.write(f"{analysis}\n\n")
                f.write("---\n\n")
    
    logger.info(f"分析结果已写入 {CONCLUSION_FILE}")

def format_email_content_with_domains(papers_analyses):
    """按领域格式化邮件内容"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    
    content = f"# ArXiv Paper Analysis Report ({today})\n\n"
    
    # 按领域组织论文
    domain_papers = {}
    for paper, analysis, domain in papers_analyses:
        if domain not in domain_papers:
            domain_papers[domain] = []
        domain_papers[domain].append((paper, analysis))
    
    for domain, papers in domain_papers.items():
        content += f"## {domain} 领域\n\n"
        
        for paper, analysis in papers:
            # 从Author对象中提取作者名
            author_names = [author.name for author in paper.authors]
            
            content += f"### {paper.title}\n\n"
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
    
    all_papers_analyses = []  # 存储所有领域的分析结果
    
    # 分别处理每个领域
    for domain_name, config in CATEGORY_CONFIGS.items():
        logger.info(f"\n=== 开始处理 {domain_name} 领域 ===")
        categories = config["categories"]
        max_search = config["max_search"] 
        max_analyze = config["max_analyze"]
        
        logger.info(f"类别: {', '.join(categories)}")
        logger.info(f"搜索配置: 搜索{max_search}篇，分析{max_analyze}篇")
        
        # 获取该领域最近7天的论文
        papers = get_recent_papers(categories, max_search)
        logger.info(f"{domain_name}领域从最近7天找到{len(papers)}篇论文")
        
        if not papers:
            logger.info(f"{domain_name}领域在所选时间段没有找到论文，跳过该领域")
            continue
        
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
                logger.info(f"{domain_name}新论文待分析: {paper.title} ({paper_id})")
            else:
                logger.info(f"{domain_name}论文已分析过，跳过: {paper.title} ({paper_id})")
        
        # 按发布时间排序新论文（最新的在前）
        new_papers.sort(key=lambda p: p.published, reverse=True)
        
        logger.info(f"{domain_name}领域发现{len(new_papers)}篇新论文")
        
        # 只分析指定数量的论文
        papers_to_analyze = new_papers[:max_analyze]
        logger.info(f"{domain_name}领域本次将分析{len(papers_to_analyze)}篇论文")
        
        if len(new_papers) > max_analyze:
            logger.info(f"{domain_name}领域还有{len(new_papers) - max_analyze}篇论文待下次分析")
        
        if not papers_to_analyze:
            logger.info(f"{domain_name}领域没有新论文需要分析")
            continue
        
        # 处理该领域的每篇新论文
        domain_analyses = []
        for i, paper in enumerate(papers_to_analyze, 1):
            logger.info(f"正在处理{domain_name}论文 {i}/{len(papers_to_analyze)}: {paper.title}")
            # 下载论文
            pdf_path = download_paper(paper, PAPERS_DIR)
            if pdf_path:
                # 休眠以避免达到API速率限制
                time.sleep(2)
                
                # 分析论文
                analysis = analyze_paper_with_chatgpt(pdf_path, paper)
                domain_analyses.append((paper, analysis, domain_name))  # 添加领域标识
                
                # 分析完成后删除PDF文件
                delete_pdf(pdf_path)
        
        all_papers_analyses.extend(domain_analyses)
        
        # 显示该领域待分析论文队列状态
        remaining_papers = new_papers[max_analyze:]
        if remaining_papers:
            logger.info(f"{domain_name}领域待分析论文队列 ({len(remaining_papers)}篇):")
            for i, paper in enumerate(remaining_papers[:3], 1):  # 只显示前3篇
                logger.info(f"  {i}. {paper.title} ({paper.get_short_id()})")
            if len(remaining_papers) > 3:
                logger.info(f"  ...还有{len(remaining_papers) - 3}篇")
        
        logger.info(f"=== {domain_name} 领域处理完成 ===\n")
    
    # 将所有领域的分析结果写入conclusion.md
    if all_papers_analyses:
        write_to_conclusion_with_domains(all_papers_analyses)
        
        # 发送邮件（包含所有领域当天分析的论文）
        email_content = format_email_content_with_domains(all_papers_analyses)
        send_email(email_content)
        
        logger.info("ArXiv论文追踪和分析完成")
        logger.info(f"结果已保存至 {CONCLUSION_FILE.absolute()}")
                
        # 运行总结
        logger.info("=== 总体运行总结 ===")
        total_analyzed = len(all_papers_analyses)
        logger.info(f"本次总共分析: {total_analyzed}篇论文")
        
        # 按领域统计
        domain_stats = {}
        for _, _, domain in all_papers_analyses:
            domain_stats[domain] = domain_stats.get(domain, 0) + 1
        
        for domain, count in domain_stats.items():
            logger.info(f"  {domain}: {count}篇")
            
        logger.info(f"已分析历史总数: {len(analyzed_papers)}篇")
        logger.info("====================")
    else:
        logger.info("没有成功分析的论文。")

def test_configuration_and_functions():
    """测试函数 - 验证配置和核心功能"""
    print("=== 开始测试 ArXiv 论文追踪系统 ===\n")
    
    # 1. 测试配置
    print("1. 测试配置:")
    print(f"   领域数量: {len(CATEGORY_CONFIGS)}")
    for domain_name, config in CATEGORY_CONFIGS.items():
        print(f"   {domain_name}:")
        print(f"     类别: {config['categories']}")
        print(f"     搜索数量: {config['max_search']}")
        print(f"     分析数量: {config['max_analyze']}")
    
    # 2. 测试向后兼容性
    print(f"\n2. 向后兼容性:")
    print(f"   CATEGORIES: {CATEGORIES}")
    print(f"   MAX_PAPERS_SEARCH: {MAX_PAPERS_SEARCH}")
    print(f"   MAX_PAPERS_ANALYZE: {MAX_PAPERS_ANALYZE}")
    
    # 3. 测试模拟论文数据结构
    print(f"\n3. 测试数据结构:")
    
    # 创建模拟的论文对象
    class MockAuthor:
        def __init__(self, name):
            self.name = name
    
    class MockPaper:
        def __init__(self, title, authors, categories, published, entry_id, short_id):
            self.title = title
            self.authors = [MockAuthor(author) for author in authors]
            self.categories = categories
            self.published = published
            self.entry_id = entry_id
            self._short_id = short_id
        
        def get_short_id(self):
            return self._short_id
    
    # 创建测试数据
    test_papers_data = [
        ("Software Engineering Paper 1", ["Alice Smith", "Bob Johnson"], ["cs.SE"], 
         datetime.datetime.now() - datetime.timedelta(days=1), 
         "https://arxiv.org/abs/2024.01001", "2024.01001"),
        ("Security Paper 1", ["Charlie Brown", "Diana Prince"], ["cs.CR"], 
         datetime.datetime.now() - datetime.timedelta(days=2), 
         "https://arxiv.org/abs/2024.01002", "2024.01002"),
        ("Cybersecurity Paper 1", ["Eve Wilson"], ["cs.CY"], 
         datetime.datetime.now() - datetime.timedelta(days=3), 
         "https://arxiv.org/abs/2024.01003", "2024.01003"),
    ]
    
    mock_papers = []
    for title, authors, categories, published, entry_id, short_id in test_papers_data:
        mock_papers.append(MockPaper(title, authors, categories, published, entry_id, short_id))
    
    # 创建模拟分析结果
    test_analyses = [
        (mock_papers[0], "### Executive Summary\nThis is a test analysis for software engineering paper.\n\n### Key Contributions\n- Test contribution 1\n- Test contribution 2", "软件工程"),
        (mock_papers[1], "### Executive Summary\nThis is a test analysis for security paper.\n\n### Key Contributions\n- Security contribution 1\n- Security contribution 2", "安全领域"),
        (mock_papers[2], "### Executive Summary\nThis is a test analysis for cybersecurity paper.\n\n### Key Contributions\n- Cyber contribution 1\n- Cyber contribution 2", "安全领域"),
    ]
    
    print(f"   创建了 {len(mock_papers)} 个模拟论文对象")
    print(f"   创建了 {len(test_analyses)} 个模拟分析结果")
    
    # 4. 测试邮件内容格式化
    print(f"\n4. 测试邮件内容格式化:")
    try:
        email_content = format_email_content_with_domains(test_analyses)
        print("   ✓ 邮件格式化成功")
        print(f"   邮件内容长度: {len(email_content)} 字符")
        
        # 检查是否包含预期的领域标题
        if "软件工程 领域" in email_content and "安全领域 领域" in email_content:
            print("   ✓ 领域分类正确")
        else:
            print("   ✗ 领域分类有问题")
            
    except Exception as e:
        print(f"   ✗ 邮件格式化失败: {e}")
    
    # 5. 测试按领域统计
    print(f"\n5. 测试按领域统计:")
    domain_stats = {}
    for _, _, domain in test_analyses:
        domain_stats[domain] = domain_stats.get(domain, 0) + 1
    
    print("   领域统计结果:")
    for domain, count in domain_stats.items():
        print(f"     {domain}: {count}篇")
    
    # 6. 测试已分析论文检查逻辑
    print(f"\n6. 测试论文ID处理:")
    test_paper_ids = ["2024.01001", "2024.01002v1", "2024.01003"]
    analyzed_papers = {"2024.01001", "2024.01002"}  # 模拟已分析的论文
    
    import re
    for paper_id in test_paper_ids:
        paper_id_no_version = re.sub(r'v\d+$', '', paper_id)
        is_analyzed = (paper_id in analyzed_papers or paper_id_no_version in analyzed_papers)
        status = "已分析" if is_analyzed else "待分析"
        print(f"   论文 {paper_id} -> {paper_id_no_version}: {status}")
    
    # 7. 测试环境变量检查
    print(f"\n7. 测试环境变量:")
    required_vars = ["OPENAI_API_KEY", "SMTP_SERVER", "SMTP_USERNAME", "SMTP_PASSWORD", "EMAIL_FROM", "EMAIL_TO"]
    for var in required_vars:
        value = os.getenv(var)
        status = "✓ 已设置" if value else "✗ 未设置"
        print(f"   {var}: {status}")
    
    print(f"\n=== 测试完成 ===")
    print("如果所有测试都通过，说明配置正确，可以运行主程序。")
    print("注意：实际运行时需要确保环境变量已正确配置。")

if __name__ == "__main__":
    # 检查命令行参数
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_configuration_and_functions()
    else:
        main()
