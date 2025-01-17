import pandas as pd
import streamlit as st
import plotly.express as px
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from textblob import TextBlob

# Load the dataset
@st.cache_data
def load_data():
    return pd.read_csv("synthetic_social_media_dataset.csv")

data = load_data()

# Function to perform sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    if polarity > 0:
        return "Positive"
    elif polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Add sentiment analysis to the dataset
data['Computed_Sentiment'] = data['Post_Content'].apply(analyze_sentiment)

# Streamlit App Title
st.title("Social Media Sentiment Analysis Dashboard")

# Sidebar Options
st.sidebar.header("Filter Options")
topics = data['Post_Content'].str.extract(r"#(\w+)")[0].unique()
selected_topic = st.sidebar.selectbox("Select Topic", options=["All"] + list(topics))

# Filter data by topic
if selected_topic != "All":
    data = data[data['Post_Content'].str.contains(f"#{selected_topic}", na=False)]

# Sentiment Distribution
st.subheader("Sentiment Distribution")
sentiment_counts = data['Computed_Sentiment'].value_counts()
fig_sentiment = px.pie(
    names=sentiment_counts.index,
    values=sentiment_counts.values,
    title="Computed Sentiment Distribution",
    color_discrete_sequence=px.colors.qualitative.Set2
)
st.plotly_chart(fig_sentiment)

# Engagement Metrics
st.subheader("Engagement Metrics by Computed Sentiment")
avg_engagement = data.groupby('Computed_Sentiment')[['Likes', 'Shares', 'Comments']].mean().reset_index()
fig_engagement = px.bar(
    avg_engagement,
    x='Computed_Sentiment',
    y=['Likes', 'Shares', 'Comments'],
    barmode='group',
    title="Average Engagement by Computed Sentiment",
    labels={"value": "Average Engagement", "variable": "Metric"},
    color_discrete_sequence=px.colors.qualitative.Pastel
)
st.plotly_chart(fig_engagement)


# Time-Series Analysis
st.subheader("Sentiment Trend Over Time")
data['Post_Date'] = pd.to_datetime(data['Post_Date'])
time_series = data.groupby([data['Post_Date'].dt.date, 'Computed_Sentiment']).size().reset_index(name='Count')
fig_time_series = px.line(
    time_series,
    x='Post_Date',
    y='Count',
    color='Computed_Sentiment',
    title="Sentiment Trend Over Time",
    labels={"Post_Date": "Date", "Count": "Number of Posts"},
    color_discrete_sequence=px.colors.qualitative.Set1
)
st.plotly_chart(fig_time_series)

# User-Level Insights (Simulated Data)
st.subheader("User-Level Insights")
data['User_ID'] = [f"user_{i}" for i in range(len(data))]  # Simulated user IDs
most_active_users = data['User_ID'].value_counts().head(5).reset_index()
most_active_users.columns = ['User_ID', 'Post_Count']
st.write("Top 5 Most Active Users:")
st.write(most_active_users)

# Sentiment by Topic
st.subheader("Sentiment Distribution by Topic")
topic_sentiment = data.groupby([data['Post_Content'].str.extract(r"#(\w+)")[0], 'Computed_Sentiment']).size().reset_index(name='Count')
fig_topic_sentiment = px.bar(
    topic_sentiment,
    x=0,
    y='Count',
    color='Computed_Sentiment',
    title="Sentiment Distribution by Topic",
    labels={0: "Topic", "Count": "Number of Posts"},
    barmode='group',
    color_discrete_sequence=px.colors.qualitative.Dark2
)
st.plotly_chart(fig_topic_sentiment)

# Custom Keyword Analysis
st.subheader("Custom Keyword Analysis")
keyword = st.text_input("Enter a keyword or hashtag")
if keyword:
    keyword_data = data[data['Post_Content'].str.contains(keyword, case=False, na=False)]
    st.write(f"Total posts containing '{keyword}': {len(keyword_data)}")
    if len(keyword_data) > 0:
        keyword_sentiment_counts = keyword_data['Computed_Sentiment'].value_counts()
        fig_keyword_sentiment = px.pie(
            names=keyword_sentiment_counts.index,
            values=keyword_sentiment_counts.values,
            title=f"Sentiment Distribution for '{keyword}'",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_keyword_sentiment)

# Engagement Heatmap
st.subheader("Engagement Heatmap")
heatmap_data = data[['Likes', 'Shares', 'Comments']]
fig_heatmap = px.imshow(
    heatmap_data.corr(),
    title="Engagement Correlation Heatmap",
    text_auto=True,
    color_continuous_scale='Viridis'
)
st.plotly_chart(fig_heatmap)

# Outlier Detection
st.subheader("Outlier Detection")
likes_threshold = data['Likes'].quantile(0.95)
outlier_posts = data[data['Likes'] > likes_threshold]
st.write(f"Posts with Likes above the 95th percentile ({likes_threshold} likes):")
st.write(outlier_posts[['Post_Content', 'Likes']])

# Insights Section
st.subheader("Actionable Insights")
positive_posts = data[data['Computed_Sentiment'] == "Positive"]
negative_posts = data[data['Computed_Sentiment'] == "Negative"]

st.write(f"Total Positive Posts: {len(positive_posts)}")
st.write(f"Total Negative Posts: {len(negative_posts)}")

most_liked_positive = positive_posts.sort_values(by='Likes', ascending=False).head(1)
st.write("Most Liked Positive Post:")
st.write(most_liked_positive[['Post_Content', 'Likes']])

most_shared_negative = negative_posts.sort_values(by='Shares', ascending=False).head(1)
st.write("Most Shared Negative Post:")
st.write(most_shared_negative[['Post_Content', 'Shares']])

# Raw Data
st.subheader("View Raw Data")
if st.checkbox("Show raw data"):
    st.write(data)

# Advanced Filtering
st.sidebar.subheader("Advanced Filtering")
selected_sentiments = st.sidebar.multiselect("Filter by Sentiment", options=data['Computed_Sentiment'].unique(), default=data['Computed_Sentiment'].unique())
start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime(data['Post_Date']).min())
end_date = st.sidebar.date_input("End Date", value=pd.to_datetime(data['Post_Date']).max())

# Apply Advanced Filtering
start_date = pd.to_datetime(start_date)  # Ensure the start_date is in datetime format
end_date = pd.to_datetime(end_date)      # Ensure the end_date is in datetime format

filtered_data = data[
    (data['Computed_Sentiment'].isin(selected_sentiments)) &
    (data['Post_Date'] >= start_date) &  # Ensure comparison is done with datetime values
    (data['Post_Date'] <= end_date)
]
st.write(f"Filtered Data: {len(filtered_data)} posts")

# N-Gram Analysis
st.subheader("N-Gram Analysis")
n_gram_range = st.slider("Select N-Gram Range", min_value=1, max_value=3, value=2)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(ngram_range=(n_gram_range, n_gram_range), stop_words='english')
ngrams = vectorizer.fit_transform(filtered_data['Post_Content'].dropna())
ngram_counts = pd.DataFrame(ngrams.sum(axis=0).T, index=vectorizer.get_feature_names_out(), columns=['Count']).sort_values(by='Count', ascending=False)
st.write(f"Top {n_gram_range}-Grams:")
st.write(ngram_counts.head(10))

# Machine Learning Sentiment Comparison
st.subheader("Sentiment Analysis Comparison")
st.write("Coming soon: Compare TextBlob sentiment analysis with a pre-trained model using Hugging Face.")

# Word Cloud for Post Content
show_wordcloud = st.sidebar.checkbox("Show Word Cloud", value=True)
if show_wordcloud:
    st.subheader("Word Cloud of Post Content")
    wordcloud_text = " ".join(filtered_data['Post_Content'].dropna())
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(wordcloud_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)
