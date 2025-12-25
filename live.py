# rss_visible_sentiment.py
"""
Investing.com RSS Sentiment Scanner - VISIBLE MODE
- Opens RSS feed in visible browser
- Shows sentiment score for each event
- Displays exact release time
- Updates every 2 minutes
"""

import time
import re
import sys
import json
import feedparser
from datetime import datetime, timedelta
from typing import List, Dict, Set
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
import os

# Browser automation for visible mode
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

# Sentiment analysis
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np

# Color printing
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

# ============================================================================
# 1. RSS FEED MONITOR
# ============================================================================
class RSSFeedMonitor:
    """Monitor RSS feed for new articles"""
    
    def __init__(self, rss_url):
        self.rss_url = rss_url
        self.processed_guids = set()
        self.articles_history = []
        self.last_fetch = None
        
    def fetch_articles(self):
        """Fetch and parse RSS feed"""
        try:
            print(f"{Colors.BLUE}📡 Fetching RSS feed...{Colors.END}")
            
            feed = feedparser.parse(self.rss_url)
            
            if feed.bozo:
                print(f"{Colors.RED}❌ RSS parse error: {feed.bozo_exception}{Colors.END}")
                return []
            
            articles = []
            for entry in feed.entries:
                # Generate unique ID
                guid = entry.get('id', entry.get('link', ''))
                
                # Parse date
                pub_date = self.parse_entry_date(entry)
                
                article = {
                    'guid': guid,
                    'title': entry.get('title', 'No Title'),
                    'link': entry.get('link', ''),
                    'published': pub_date,
                    'published_str': pub_date.strftime("%Y-%m-%d %H:%M:%S"),
                    'author': entry.get('author', 'Unknown'),
                    'summary': entry.get('summary', ''),
                    'image': self.get_image_url(entry)
                }
                
                articles.append(article)
            
            self.last_fetch = datetime.now()
            print(f"{Colors.GREEN}✅ Found {len(articles)} articles{Colors.END}")
            return articles
            
        except Exception as e:
            print(f"{Colors.RED}❌ Error fetching RSS: {e}{Colors.END}")
            return []
    
    def parse_entry_date(self, entry):
        """Parse publication date from RSS entry"""
        try:
            if 'published_parsed' in entry:
                return datetime(*entry.published_parsed[:6])
            elif 'updated_parsed' in entry:
                return datetime(*entry.updated_parsed[:6])
            else:
                # Try to parse string date
                date_str = entry.get('published', entry.get('updated', ''))
                if date_str:
                    # Parse common RSS date formats
                    for fmt in ['%Y-%m-%d %H:%M:%S', '%a, %d %b %Y %H:%M:%S %z', '%Y-%m-%dT%H:%M:%SZ']:
                        try:
                            return datetime.strptime(date_str, fmt)
                        except:
                            continue
        except:
            pass
        
        return datetime.now()
    
    def get_image_url(self, entry):
        """Extract image URL from entry"""
        try:
            # Check for enclosure
            if 'links' in entry:
                for link in entry.links:
                    if link.get('type', '').startswith('image/'):
                        return link.get('href', '')
            
            # Check for media content
            if 'media_content' in entry:
                for media in entry.media_content:
                    if media.get('type', '').startswith('image/'):
                        return media.get('url', '')
        except:
            pass
        
        return ''
    
    def get_new_articles(self, articles):
        """Get articles not seen before"""
        new_articles = []
        
        for article in articles:
            if article['guid'] not in self.processed_guids:
                self.processed_guids.add(article['guid'])
                new_articles.append(article)
        
        return new_articles

# ============================================================================
# 2. SENTIMENT ANALYZER
# ============================================================================
class FinBertSentimentAnalyzer:
    """Financial sentiment analyzer using FinBERT"""
    
    def __init__(self):
        print(f"{Colors.BLUE}🤖 Loading FinBERT model...{Colors.END}")
        
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"{Colors.CYAN}📊 Using device: {self.device}{Colors.END}")
            
            model_name = "ProsusAI/finbert"
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print(f"{Colors.GREEN}✅ FinBERT loaded successfully!{Colors.END}")
            self.model_loaded = True
            
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to load FinBERT: {e}{Colors.END}")
            print(f"{Colors.YELLOW}⚠️ Using rule-based sentiment{Colors.END}")
            self.model_loaded = False
    
    def analyze(self, text):
        """Analyze sentiment of text"""
        if self.model_loaded:
            return self._analyze_with_finbert(text)
        else:
            return self._analyze_with_rules(text)
    
    def _analyze_with_finbert(self, text):
        """Use FinBERT for sentiment analysis"""
        try:
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            probs = predictions[0].cpu().numpy()
            
            # FinBERT labels: 0=negative, 1=neutral, 2=positive
            labels = ["BEARISH", "NEUTRAL", "BULLISH"]
            idx = np.argmax(probs)
            
            # Calculate sentiment score (-1 to +1)
            score = probs[2] - probs[0]
            
            return {
                'sentiment': labels[idx],
                'score': float(score),
                'confidence': float(probs[idx]),
                'is_bullish': idx == 2,
                'is_bearish': idx == 0,
                'is_neutral': idx == 1,
                'probabilities': {
                    'bearish': float(probs[0]),
                    'neutral': float(probs[1]),
                    'bullish': float(probs[2])
                }
            }
            
        except Exception as e:
            print(f"{Colors.RED}❌ FinBERT error: {e}{Colors.END}")
            return self._analyze_with_rules(text)
    
    def _analyze_with_rules(self, text):
        """Rule-based sentiment analysis"""
        text_lower = text.lower()
        
        bullish_terms = {
            'gain': 1.5, 'rise': 1.5, 'higher': 1.0, 'up': 1.0, 'strong': 1.0,
            'positive': 1.5, 'profit': 1.5, 'beat': 1.5, 'surge': 2.0, 'rally': 2.0,
            'increase': 1.0, 'growth': 1.5, 'optimistic': 1.5, 'soar': 2.0,
            'jump': 1.5, 'record high': 2.5, 'all-time high': 2.5, 'bullish': 2.0,
            'exceeds': 1.5, 'outperform': 1.5, 'recovery': 1.0, 'boost': 1.0,
            'strengthen': 1.0, 'advance': 1.0, 'climb': 1.0
        }
        
        bearish_terms = {
            'drop': 1.5, 'fall': 1.5, 'lower': 1.0, 'down': 1.0, 'weak': 1.0,
            'negative': 1.5, 'loss': 1.5, 'miss': 1.5, 'plunge': 2.0, 'decline': 1.5,
            'decrease': 1.0, 'slump': 2.0, 'pessimistic': 1.5, 'crash': 2.5,
            'tumble': 1.5, 'slide': 1.0, 'bearish': 2.0, 'risk': 1.0,
            'uncertainty': 1.0, 'volatile': 1.0, 'pressure': 1.0, 'concern': 1.0,
            'warning': 1.5, 'cut': 1.0, 'reduce': 1.0, 'downgrade': 1.5
        }
        
        bullish_score = sum(score for term, score in bullish_terms.items() 
                          if term in text_lower)
        bearish_score = sum(score for term, score in bearish_terms.items() 
                          if term in text_lower)
        
        total_score = bullish_score - bearish_score
        
        if total_score > 1.0:
            sentiment = "BULLISH"
            confidence = min(0.9, 0.5 + total_score * 0.1)
        elif total_score < -1.0:
            sentiment = "BEARISH"
            confidence = min(0.9, 0.5 + abs(total_score) * 0.1)
        else:
            sentiment = "NEUTRAL"
            confidence = 0.6
        
        normalized_score = max(-1.0, min(1.0, total_score / 5.0))
        
        return {
            'sentiment': sentiment,
            'score': normalized_score,
            'confidence': confidence,
            'is_bullish': sentiment == "BULLISH",
            'is_bearish': sentiment == "BEARISH",
            'is_neutral': sentiment == "NEUTRAL",
            'probabilities': {
                'bearish': 0.33 if sentiment != "BEARISH" else 0.6,
                'neutral': 0.34,
                'bullish': 0.33 if sentiment != "BULLISH" else 0.6
            }
        }

# ============================================================================
# 3. VISIBLE BROWSER DISPLAY
# ============================================================================
class VisibleRSSDisplay:
    """Display RSS feed and sentiment in visible browser"""
    
    def __init__(self):
        self.driver = None
        self.html_content = ""
        self.setup_browser()
    
    def setup_browser(self):
        """Setup visible Chrome browser"""
        print(f"{Colors.BLUE}🚀 Launching Chrome browser...{Colors.END}")
        
        options = Options()
        
        # IMPORTANT: VISIBLE MODE (no headless)
        # options.add_argument('--headless')  # COMMENT THIS LINE!
        
        options.add_argument('--start-maximized')
        options.add_argument('--window-size=1400,900')
        options.add_argument('--disable-infobars')
        
        # Anti-detection
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # User agent
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        options.add_argument(f'user-agent={user_agent}')
        
        try:
            self.driver = webdriver.Chrome(options=options)
            print(f"{Colors.GREEN}✅ Chrome launched successfully!{Colors.END}")
            print(f"{Colors.CYAN}👀 Look for the Chrome window on your screen...{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to launch Chrome: {e}{Colors.END}")
            print(f"\n{Colors.YELLOW}Troubleshooting:{Colors.END}")
            print("1. Download Chrome: https://www.google.com/chrome/")
            print("2. Download ChromeDriver: https://chromedriver.chromium.org/")
            print("3. Extract chromedriver.exe to this folder")
            raise
    
    def update_display(self, analyzed_articles, summary_stats):
        """Update the browser display with new data"""
        try:
            html = self.generate_html(analyzed_articles, summary_stats)
            self.html_content = html
            
            # Save HTML to file
            with open('rss_dashboard.html', 'w', encoding='utf-8') as f:
                f.write(html)
            
            # Load in browser
            file_path = f'file:///{os.path.abspath("rss_dashboard.html")}'
            
            if self.driver.current_url == file_path:
                # Refresh if already on the page
                self.driver.refresh()
            else:
                # Navigate to the page
                self.driver.get(file_path)
            
            print(f"{Colors.GREEN}📊 Browser display updated{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}❌ Error updating display: {e}{Colors.END}")
    
    def generate_html(self, analyzed_articles, summary_stats):
        """Generate HTML dashboard"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate time differences
        for article in analyzed_articles:
            pub_time = datetime.strptime(article['published_str'], "%Y-%m-%d %H:%M:%S")
            now = datetime.now()
            time_diff = now - pub_time
            
            if time_diff.days > 0:
                time_ago = f"{time_diff.days}d ago"
            elif time_diff.seconds >= 3600:
                hours = time_diff.seconds // 3600
                time_ago = f"{hours}h ago"
            elif time_diff.seconds >= 60:
                minutes = time_diff.seconds // 60
                time_ago = f"{minutes}m ago"
            else:
                time_ago = "Just now"
            
            article['time_ago'] = time_ago
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Investing.com RSS Sentiment Dashboard</title>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    font-family: 'Segoe UI', Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    min-height: 100vh;
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                
                .header {{
                    background: white;
                    padding: 25px;
                    border-radius: 15px;
                    margin-bottom: 25px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }}
                
                .header h1 {{
                    color: #333;
                    margin: 0;
                    font-size: 28px;
                }}
                
                .header .timestamp {{
                    color: #666;
                    margin-top: 10px;
                    font-size: 14px;
                }}
                
                .stats-container {{
                    display: flex;
                    gap: 20px;
                    margin-bottom: 25px;
                }}
                
                .stat-box {{
                    flex: 1;
                    background: white;
                    padding: 25px;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                
                .stat-box.bullish {{
                    border-top: 5px solid #10b981;
                }}
                
                .stat-box.bearish {{
                    border-top: 5px solid #ef4444;
                }}
                
                .stat-box.neutral {{
                    border-top: 5px solid #f59e0b;
                }}
                
                .stat-value {{
                    font-size: 42px;
                    font-weight: bold;
                    margin: 10px 0;
                }}
                
                .stat-label {{
                    color: #666;
                    font-size: 16px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }}
                
                .articles-container {{
                    background: white;
                    border-radius: 15px;
                    padding: 25px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.1);
                }}
                
                .article {{
                    padding: 20px;
                    margin: 15px 0;
                    border-radius: 10px;
                    border-left: 5px solid #ccc;
                    transition: transform 0.2s;
                }}
                
                .article:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                }}
                
                .article.bullish {{
                    border-left-color: #10b981;
                    background: linear-gradient(90deg, rgba(16,185,129,0.05) 0%, white 100%);
                }}
                
                .article.bearish {{
                    border-left-color: #ef4444;
                    background: linear-gradient(90deg, rgba(239,68,68,0.05) 0%, white 100%);
                }}
                
                .article.neutral {{
                    border-left-color: #f59e0b;
                    background: linear-gradient(90deg, rgba(245,158,11,0.05) 0%, white 100%);
                }}
                
                .article-header {{
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    margin-bottom: 10px;
                }}
                
                .sentiment-badge {{
                    padding: 5px 15px;
                    border-radius: 20px;
                    font-weight: bold;
                    font-size: 14px;
                }}
                
                .badge-bullish {{
                    background: #10b981;
                    color: white;
                }}
                
                .badge-bearish {{
                    background: #ef4444;
                    color: white;
                }}
                
                .badge-neutral {{
                    background: #f59e0b;
                    color: white;
                }}
                
                .article-title {{
                    font-size: 18px;
                    color: #333;
                    margin-bottom: 10px;
                    line-height: 1.4;
                }}
                
                .article-meta {{
                    display: flex;
                    gap: 20px;
                    color: #666;
                    font-size: 14px;
                }}
                
                .score-container {{
                    display: flex;
                    align-items: center;
                    gap: 10px;
                    margin-top: 15px;
                }}
                
                .score-bar {{
                    flex: 1;
                    height: 10px;
                    background: #e5e7eb;
                    border-radius: 5px;
                    overflow: hidden;
                }}
                
                .score-fill {{
                    height: 100%;
                    border-radius: 5px;
                }}
                
                .score-fill.bullish {{
                    background: linear-gradient(90deg, #10b981, #34d399);
                }}
                
                .score-fill.bearish {{
                    background: linear-gradient(90deg, #ef4444, #f87171);
                }}
                
                .score-fill.neutral {{
                    background: linear-gradient(90deg, #f59e0b, #fbbf24);
                }}
                
                .score-text {{
                    font-weight: bold;
                    min-width: 60px;
                }}
                
                .time-badge {{
                    background: #3b82f6;
                    color: white;
                    padding: 3px 10px;
                    border-radius: 12px;
                    font-size: 12px;
                }}
                
                .footer {{
                    text-align: center;
                    margin-top: 30px;
                    color: white;
                    font-size: 14px;
                }}
                
                .last-updated {{
                    background: rgba(255,255,255,0.2);
                    padding: 10px;
                    border-radius: 10px;
                    display: inline-block;
                }}
                
                @media (max-width: 768px) {{
                    .stats-container {{
                        flex-direction: column;
                    }}
                    .article-header {{
                        flex-direction: column;
                        align-items: flex-start;
                    }}
                    .article-meta {{
                        flex-direction: column;
                        gap: 5px;
                    }}
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📈 Investing.com RSS Sentiment Dashboard</h1>
                    <div class="timestamp">Last updated: {timestamp}</div>
                </div>
                
                <div class="stats-container">
                    <div class="stat-box bullish">
                        <div class="stat-value">{summary_stats['bullish']}</div>
                        <div class="stat-label">🟢 Bullish</div>
                    </div>
                    <div class="stat-box bearish">
                        <div class="stat-value">{summary_stats['bearish']}</div>
                        <div class="stat-label">🔴 Bearish</div>
                    </div>
                    <div class="stat-box neutral">
                        <div class="stat-value">{summary_stats['neutral']}</div>
                        <div class="stat-label">⚪ Neutral</div>
                    </div>
                </div>
                
                <div class="articles-container">
                    <h2 style="color: #333; margin-top: 0;">📰 Latest Articles ({len(analyzed_articles)} total)</h2>
        """
        
        # Add articles
        for article in analyzed_articles:
            sentiment_class = article['sentiment'].lower()
            badge_class = f"badge-{sentiment_class}"
            
            # Calculate score percentage for bar
            score_percent = (article['score'] + 1) / 2 * 100  # Convert -1 to +1 into 0 to 100
            
            html += f"""
                    <div class="article {sentiment_class}">
                        <div class="article-header">
                            <div class="sentiment-badge {badge_class}">
                                {article['sentiment']} (Confidence: {article['confidence']:.2f})
                            </div>
                            <div class="time-badge">
                                ⏰ {article['published_str']} ({article['time_ago']})
                            </div>
                        </div>
                        
                        <div class="article-title">
                            {article['title']}
                        </div>
                        
                        <div class="article-meta">
                            <div>👤 {article['author']}</div>
                            <div>🔗 <a href="{article['link']}" target="_blank">Read full article</a></div>
                        </div>
                        
                        <div class="score-container">
                            <div class="score-text">
                                Score: {article['score']:.3f}
                            </div>
                            <div class="score-bar">
                                <div class="score-fill {sentiment_class}" style="width: {score_percent}%"></div>
                            </div>
                            <div style="color: #666; font-size: 12px;">
                                -1 (Bearish) ↔ +1 (Bullish)
                            </div>
                        </div>
                    </div>
            """
        
        html += """
                </div>
                
                <div class="footer">
                    <div class="last-updated">
                        🔄 Auto-updates every 2 minutes | Total analyzed: {total_analyzed}
                    </div>
                </div>
            </div>
            
            <script>
                // Auto-refresh every 60 seconds
                setTimeout(function() {{
                    location.reload();
                }}, 60000);
                
                // Update timestamp every second
                function updateTimestamp() {{
                    const now = new Date();
                    const timestamp = now.toLocaleTimeString();
                    document.querySelector('.timestamp').textContent = 'Last updated: ' + timestamp;
                }}
                
                setInterval(updateTimestamp, 1000);
            </script>
        </body>
        </html>
        """.format(total_analyzed=summary_stats['total'])
        
        return html
    
    def close(self):
        """Close the browser"""
        try:
            if self.driver:
                print(f"{Colors.YELLOW}🛑 Closing Chrome browser...{Colors.END}")
                self.driver.quit()
        except:
            pass

# ============================================================================
# 4. MAIN MONITORING SYSTEM
# ============================================================================
class RSSLiveSentimentMonitor:
    """Main monitoring system"""
    
    def __init__(self, rss_url):
        self.rss_monitor = RSSFeedMonitor(rss_url)
        self.sentiment_analyzer = FinBertSentimentAnalyzer()
        self.browser_display = VisibleRSSDisplay()
        
        self.analyzed_articles = []
        self.stats = {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0,
            'total': 0
        }
        
        print(f"\n{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}📡 INVESTING.COM RSS SENTIMENT MONITOR{Colors.END}")
        print(f"{Colors.GREEN}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}📊 RSS Feed: {rss_url}{Colors.END}")
        print(f"{Colors.YELLOW}🔄 Refresh interval: Every 2 minutes{Colors.END}")
        print(f"{Colors.PURPLE}👁️  Browser: Visible mode (Chrome window){Colors.END}")
        print(f"{Colors.CYAN}⏰ Display: Scores + exact release times{Colors.END}")
        print(f"{Colors.GREEN}{'='*70}{Colors.END}")
    
    def run_scan_cycle(self):
        """Run one scan cycle"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        print(f"\n{Colors.BLUE}⏰ [{timestamp}] Scanning for new articles...{Colors.END}")
        
        # Fetch articles from RSS
        all_articles = self.rss_monitor.fetch_articles()
        
        if not all_articles:
            print(f"{Colors.YELLOW}📭 No articles found{Colors.END}")
            return
        
        # Get new articles
        new_articles = self.rss_monitor.get_new_articles(all_articles)
        
        if not new_articles:
            print(f"{Colors.YELLOW}📭 No NEW articles found{Colors.END}")
        else:
            print(f"{Colors.GREEN}📰 Found {len(new_articles)} NEW article(s){Colors.END}")
            
            # Analyze each new article
            for article in new_articles:
                self.analyze_article(article)
        
        # Update statistics
        self.update_stats()
        
        # Update browser display
        self.browser_display.update_display(self.analyzed_articles, self.stats)
        
        # Show terminal summary
        self.show_terminal_summary()
    
    def analyze_article(self, article):
        """Analyze a single article"""
        print(f"\n{Colors.WHITE}🔍 Analyzing: {article['title'][:60]}...{Colors.END}")
        
        # Get sentiment analysis
        sentiment = self.sentiment_analyzer.analyze(article['title'])
        
        # Combine article data with sentiment
        analyzed_article = {
            **article,
            **sentiment
        }
        
        self.analyzed_articles.append(analyzed_article)
        
        # Display in terminal
        self.show_article_terminal(analyzed_article)
    
    def show_article_terminal(self, article):
        """Show article analysis in terminal"""
        if article['is_bullish']:
            color = Colors.GREEN
            icon = "🟢"
        elif article['is_bearish']:
            color = Colors.RED
            icon = "🔴"
        else:
            color = Colors.YELLOW
            icon = "⚪"
        
        print(f"\n{color}{'━'*50}{Colors.END}")
        print(f"{color}{icon} {article['sentiment']} | Score: {article['score']:.3f} | Conf: {article['confidence']:.2f}{Colors.END}")
        print(f"{Colors.WHITE}{article['title']}{Colors.END}")
        print(f"{Colors.CYAN}⏰ Released: {article['published_str']}{Colors.END}")
        print(f"{Colors.BLUE}👤 Author: {article['author']}{Colors.END}")
    
    def update_stats(self):
        """Update statistics"""
        self.stats['bullish'] = sum(1 for a in self.analyzed_articles if a['is_bullish'])
        self.stats['bearish'] = sum(1 for a in self.analyzed_articles if a['is_bearish'])
        self.stats['neutral'] = sum(1 for a in self.analyzed_articles if a['is_neutral'])
        self.stats['total'] = len(self.analyzed_articles)
    
    def show_terminal_summary(self):
        """Show summary in terminal"""
        print(f"\n{Colors.CYAN}{'━'*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.WHITE}📈 LIVE SENTIMENT SUMMARY{Colors.END}")
        print(f"{Colors.CYAN}{'━'*70}{Colors.END}")
        
        if self.stats['total'] > 0:
            bullish_pct = (self.stats['bullish'] / self.stats['total']) * 100
            bearish_pct = (self.stats['bearish'] / self.stats['total']) * 100
            neutral_pct = (self.stats['neutral'] / self.stats['total']) * 100
            
            print(f"{Colors.GREEN}🟢 Bullish: {self.stats['bullish']} ({bullish_pct:.1f}%){Colors.END}")
            print(f"{Colors.RED}🔴 Bearish: {self.stats['bearish']} ({bearish_pct:.1f}%){Colors.END}")
            print(f"{Colors.YELLOW}⚪ Neutral: {self.stats['neutral']} ({neutral_pct:.1f}%){Colors.END}")
            
            # Calculate average score
            avg_score = sum(a['score'] for a in self.analyzed_articles) / self.stats['total']
            print(f"{Colors.CYAN}🎯 Average Score: {avg_score:.3f}{Colors.END}")
            
            # Overall sentiment
            if avg_score > 0.2:
                print(f"{Colors.BOLD}{Colors.GREEN}📈 OVERALL MARKET SENTIMENT: BULLISH{Colors.END}")
            elif avg_score < -0.2:
                print(f"{Colors.BOLD}{Colors.RED}📉 OVERALL MARKET SENTIMENT: BEARISH{Colors.END}")
            else:
                print(f"{Colors.BOLD}{Colors.YELLOW}⚖️  OVERALL MARKET SENTIMENT: NEUTRAL{Colors.END}")
        
        print(f"{Colors.CYAN}{'━'*70}{Colors.END}")
    
    def run_continuous(self, interval_minutes=2):
        """Run continuous monitoring"""
        print(f"\n{Colors.GREEN}🚀 Starting continuous monitoring...{Colors.END}")
        print(f"{Colors.YELLOW}📡 Scanning RSS feed every {interval_minutes} minutes{Colors.END}")
        print(f"{Colors.CYAN}👀 Chrome browser window is open - visible display!{Colors.END}")
        print(f"{Colors.PURPLE}🛑 Press Ctrl+C to stop{Colors.END}")
        
        try:
            # Initial scan
            self.run_scan_cycle()
            
            # Continuous scanning
            while True:
                try:
                    # Wait for next scan
                    wait_seconds = interval_minutes * 60
                    
                    for remaining in range(wait_seconds, 0, -1):
                        mins, secs = divmod(remaining, 60)
                        time_str = f"{mins:02d}:{secs:02d}"
                        
                        sys.stdout.write(f"\r⏳ Next scan in: {time_str} | Total analyzed: {self.stats['total']}")
                        sys.stdout.flush()
                        
                        time.sleep(1)
                    
                    print("\n")
                    self.run_scan_cycle()
                    
                except KeyboardInterrupt:
                    print(f"\n\n{Colors.YELLOW}🛑 Stopping monitor...{Colors.END}")
                    break
                    
        except Exception as e:
            print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print(f"\n{Colors.YELLOW}🧹 Cleaning up...{Colors.END}")
        self.browser_display.close()
        self.save_results()
        
        print(f"\n{Colors.GREEN}✅ Monitor stopped.{Colors.END}")
        print(f"{Colors.CYAN}📊 Final Stats: {self.stats['total']} articles analyzed{Colors.END}")
    
    def save_results(self):
        """Save results to JSON file"""
        if not self.analyzed_articles:
            return
        
        filename = f"rss_sentiment_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump({
                    'total_articles': self.stats['total'],
                    'bullish_count': self.stats['bullish'],
                    'bearish_count': self.stats['bearish'],
                    'neutral_count': self.stats['neutral'],
                    'articles': self.analyzed_articles,
                    'scan_end_time': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
            
            print(f"{Colors.GREEN}💾 Results saved to: {filename}{Colors.END}")
            
        except Exception as e:
            print(f"{Colors.RED}❌ Could not save results: {e}{Colors.END}")

# ============================================================================
# 5. MAIN EXECUTION
# ============================================================================
def main():
    """Main function"""
    
    print(f"""
{Colors.BOLD}{Colors.CYAN}
╔══════════════════════════════════════════════════════════╗
║    📡 INVESTING.COM RSS SENTIMENT MONITOR               ║
║    Visible Browser + Real-time Analysis                  ║
║    Scores + Exact Release Times                          ║
╚══════════════════════════════════════════════════════════╝{Colors.END}
    """)
    
    # RSS feed URLs
    RSS_FEEDS = {
        "1": {
            "name": "Commodities News",
            "url": "https://za.investing.com/rss/news_11.rss",
            "desc": "Gold, oil, metals, energy prices"
        },
        "2": {
            "name": "Stock Market News",
            "url": "https://za.investing.com/rss/news_25.rss", 
            "desc": "Global stock markets and equities"
        },
        "3": {
            "name": "Forex & Currencies",
            "url": "https://za.investing.com/rss/news_301.rss",
            "desc": "Currency exchange rates and forex"
        },
        "4": {
            "name": "Economic Indicators",
            "url": "https://za.investing.com/rss/news_291.rss",
            "desc": "GDP, inflation, employment data"
        },
        "5": {
            "name": "Cryptocurrency News",
            "url": "https://za.investing.com/rss/news_20.rss",
            "desc": "Bitcoin, Ethereum, crypto markets"
        }
    }
    
    # Display feed selection
    print(f"\n{Colors.BOLD}{Colors.CYAN}📊 AVAILABLE RSS FEEDS:{Colors.END}")
    print(f"{Colors.CYAN}{'='*70}{Colors.END}")
    
    for key, feed in RSS_FEEDS.items():
        print(f"{Colors.YELLOW}{key}. {feed['name']}{Colors.END}")
        print(f"   {Colors.WHITE}{feed['desc']}{Colors.END}")
        print(f"   {Colors.BLUE}{feed['url']}{Colors.END}\n")
    
    # Get user selection
    while True:
        choice = input(f"{Colors.BOLD}Select feed number (1-5) or Enter for Commodities: {Colors.END}").strip()
        
        if not choice:
            choice = "1"
        
        if choice in RSS_FEEDS:
            selected_feed = RSS_FEEDS[choice]
            break
        else:
            print(f"{Colors.RED}❌ Invalid choice. Please enter 1-5.{Colors.END}")
    
    print(f"\n{Colors.GREEN}✅ Selected: {selected_feed['name']}{Colors.END}")
    print(f"{Colors.CYAN}📡 Feed URL: {selected_feed['url']}{Colors.END}")
    
    try:
        # Start monitor
        monitor = RSSLiveSentimentMonitor(selected_feed['url'])
        
        # Run continuous monitoring
        monitor.run_continuous(interval_minutes=2)
        
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}👋 Exiting...{Colors.END}")
    except Exception as e:
        print(f"\n{Colors.RED}❌ Error: {e}{Colors.END}")
        import traceback
        traceback.print_exc()

# ============================================================================
# 6. INSTALLATION CHECK
# ============================================================================
def check_installation():
    """Check if required packages are installed"""
    required_packages = [
        ('selenium', 'selenium'),
        ('feedparser', 'feedparser'),
        ('transformers', 'transformers'),
        ('torch', 'torch'),
    ]
    
    missing = []
    
    print(f"{Colors.YELLOW}🔍 Checking required packages...{Colors.END}")
    
    for import_name, package_name in required_packages:
        try:
            __import__(import_name)
            print(f"{Colors.GREEN}✅ {package_name}{Colors.END}")
        except ImportError:
            print(f"{Colors.RED}❌ {package_name}{Colors.END}")
            missing.append(package_name)
    
    if missing:
        print(f"\n{Colors.RED}❌ Missing packages!{Colors.END}")
        print(f"{Colors.YELLOW}Run this command to install:{Colors.END}")
        print(f"{Colors.WHITE}pip install {' '.join(missing)}{Colors.END}")
        return False
    
    print(f"{Colors.GREEN}✅ All packages are installed!{Colors.END}")
    return True

if __name__ == "__main__":
    # Check installation
    if check_installation():
        # Run main program
        main()
    else:
        input("\nPress Enter to exit...")
