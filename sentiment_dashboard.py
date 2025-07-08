# Social Media Sentiment Dashboard - COMPLETE WORKING VERSION
# With Enhanced Dashboard Visualization

!pip install tweepy textblob pandas matplotlib wordcloud ipywidgets
import tweepy
from textblob import TextBlob
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import re
from datetime import datetime, timedelta
import random
from IPython.display import display
import ipywidgets as widgets

# ================= CONFIGURATION =================
class Config:
    # Twitter API credentials (replace if you have them)
    TWITTER_API_KEY = 'your_api_key'
    TWITTER_API_SECRET = 'your_api_secret'
    TWITTER_ACCESS_TOKEN = 'your_access_token'
    TWITTER_ACCESS_SECRET = 'your_access_secret'
    
    # Search parameters
    SEARCH_QUERY = "#healthcare OR #covid OR #vaccine -filter:retweets"
    TWEET_LIMIT = 100  # Reduced for demo purposes
    USE_MOCK_DATA = True  # Set to False to use Twitter API

# ================= DATA COLLECTION =================
class DataCollector:
    def __init__(self):
        self.api = self._init_twitter_api()
        
    def _init_twitter_api(self):
        if Config.USE_MOCK_DATA:
            return None
            
        try:
            auth = tweepy.OAuth1UserHandler(
                Config.TWITTER_API_KEY,
                Config.TWITTER_API_SECRET,
                Config.TWITTER_ACCESS_TOKEN,
                Config.TWITTER_ACCESS_SECRET
            )
            return tweepy.API(auth, wait_on_rate_limit=True)
        except Exception as e:
            print(f"Twitter API initialization failed: {e}")
            return None

    def clean_text(self, text):
        return ' '.join(re.sub(r"(@\w+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", text).split())

    def get_tweets(self):
        if Config.USE_MOCK_DATA or not self.api:
            return self._get_mock_data()
            
        try:
            tweets = tweepy.Cursor(self.api.search_tweets,
                                 q=Config.SEARCH_QUERY,
                                 lang="en",
                                 tweet_mode='extended').items(Config.TWEET_LIMIT)
            
            return pd.DataFrame([{
                'text': self.clean_text(tweet.full_text),
                'created_at': tweet.created_at,
                'user': tweet.user.screen_name
            } for tweet in tweets])
            
        except Exception as e:
            print(f"Twitter API failed, using mock data: {e}")
            return self._get_mock_data()

    def _get_mock_data(self):
        print("Using mock dataset (sample healthcare tweets)")
        mock_tweets = [
            "New vaccine shows promising results in clinical trials #healthcare",
            "Nurses look bad",
            "Nurses deserve better pay and working conditions",
            "Breakthrough in cancer research gives hope to millions",
            "Mental health services need more funding #healthcare",
            "The pandemic exposed weaknesses in our healthcare system",
            "Telemedicine is revolutionizing patient care",
            "I had a terrible experience at the hospital yesterday",
            "Grateful for the doctors who saved my life last week",
            "Healthcare costs are rising unsustainably",
            "HEalth is Wealth",
        ]
        return pd.DataFrame([{
            'text': self.clean_text(tweet),
            'created_at': datetime.now() - timedelta(days=random.randint(0,6)),
            'user': f"user_{i+1}"
        } for i, tweet in enumerate(mock_tweets)])

# ================= SENTIMENT ANALYSIS =================
class SentimentAnalyzer:
    def analyze(self, df):
        if df.empty:
            return df
            
        # Get sentiment scores
        df['polarity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.polarity)
        df['subjectivity'] = df['text'].apply(lambda x: TextBlob(x).sentiment.subjectivity)
        
        # Categorize sentiment
        df['sentiment'] = df['polarity'].apply(
            lambda x: 'positive' if x > 0.1 else 'negative' if x < -0.1 else 'neutral'
        )
        return df

# ================= ENHANCED DASHBOARD VISUALIZATION =================
class Dashboard:
    @staticmethod
    def create_dashboard(df):
        if df.empty:
            print("No data available for visualization")
            return
            
        plt.figure(figsize=(15, 12))
        plt.suptitle('Healthcare Social Media Sentiment Dashboard', fontsize=16, y=1.02)
        
        # Grid layout
        grid = plt.GridSpec(3, 2, hspace=0.5, wspace=0.3)
        
        # Pie Chart
        plt.subplot(grid[0, 0])
        sentiment_counts = df['sentiment'].value_counts()
        plt.pie(sentiment_counts, 
                labels=sentiment_counts.index, 
                autopct='%1.1f%%',
                colors=['green', 'red', 'blue'],
                startangle=90)
        plt.title('Sentiment Distribution')
        
        # Trend Chart
        plt.subplot(grid[0, 1])
        df['date'] = df['created_at'].dt.date
        trend = df.groupby(['date', 'sentiment']).size().unstack().fillna(0)
        trend.plot(kind='line', marker='o', ax=plt.gca())
        plt.title('Daily Sentiment Trend')
        plt.ylabel('Number of Tweets')
        plt.grid(True)
        
        # Word Clouds
        plt.subplot(grid[1, 0])
        positive_text = ' '.join(df[df['sentiment']=='positive']['text'])
        if positive_text:
            wordcloud = WordCloud(width=600, height=300, background_color='white').generate(positive_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('Positive Terms')
            plt.axis('off')
        
        plt.subplot(grid[1, 1])
        negative_text = ' '.join(df[df['sentiment']=='negative']['text'])
        if negative_text:
            wordcloud = WordCloud(width=600, height=300, background_color='white').generate(negative_text)
            plt.imshow(wordcloud, interpolation='bilinear')
            plt.title('Negative Terms')
            plt.axis('off')
        
        # Insights
        plt.subplot(grid[2, :])
        plt.axis('off')
        insights = [
            f"• Total Tweets Analyzed: {len(df)}",
            f"• Dominant Sentiment: {sentiment_counts.idxmax()} ({sentiment_counts.max()/len(df):.1%})",
            f"• Most Active Date: {trend.sum(axis=1).idxmax()} ({trend.sum(axis=1).max()} tweets)",
            f"• Average Polarity Score: {df['polarity'].mean():.2f}",
            "• Common Positive Terms: " + ', '.join(list(set(positive_text.lower().split()))[:5]),
            "• Common Negative Terms: " + ', '.join(list(set(negative_text.lower().split()))[:5])
        ]
        plt.text(0.1, 0.8, "\n".join(insights), fontsize=12, va='top')
        
        plt.tight_layout()
        plt.show()

    @staticmethod
    def create_interactive_filter(df):
        sentiment_filter = widgets.Dropdown(
            options=['all','positive','negative','neutral'],
            description='Filter:',
            style={'description_width': '50px'}
        )
        search_term = widgets.Text(
            placeholder='Search keywords...',
            description='Keyword:',
            style={'description_width': '70px'}
        )

        def show_tweets(sentiment, keyword):
            subset = df.copy()
            if sentiment != 'all':
                subset = subset[subset['sentiment']==sentiment]
            if keyword:
                subset = subset[subset['text'].str.contains(keyword, case=False)]
            display(subset[['text','sentiment','polarity','date']].head(10))

        return widgets.interactive(show_tweets, sentiment=sentiment_filter, keyword=search_term)

# ================= MAIN EXECUTION =================
if __name__ == "__main__":
    print("""
    Healthcare Sentiment Dashboard
    ============================
    Note: Using mock data by default. To use Twitter API:
    1. Set Config.USE_MOCK_DATA = False
    2. Add valid Twitter API credentials
    """)
    
    # Run pipeline
    collector = DataCollector()
    analyzer = SentimentAnalyzer()
    dashboard = Dashboard()
    
    # Get and analyze data
    tweets_df = collector.get_tweets()
    analyzed_df = analyzer.analyze(tweets_df)
    
    # Show results
    if not analyzed_df.empty:
        print(f"\nAnalyzed {len(analyzed_df)} tweets:")
        display(analyzed_df[['text','sentiment','polarity']].head(3))
        
        print("\nDashboard Visualizations:")
        dashboard.create_dashboard(analyzed_df)
        
        print("\nInteractive Tweet Explorer:")
        display(dashboard.create_interactive_filter(analyzed_df))
    else:
        print("No data available for analysis")
