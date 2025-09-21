# YouTube Disaster Monitoring System - Documentation

## Overview
This enhanced YouTube monitoring system collects real-time data about coastal disasters and weather events in the Indian subregion. It's designed to feed structured data into NLP engines for analysis of location, severity, public sentiment, and emergency response needs.

## Key Features

### 🎯 **Smart Content Discovery**
- **Targeted Keywords**: 50+ disaster-related keywords specific to Indian coastal regions
- **Geographic Filtering**: Focus on Indian content with region-specific searches
- **Time-based Filtering**: Prioritizes recent content (last 6-24 hours)
- **Live Content Detection**: Identifies live streams and breaking news

### 📊 **Comprehensive Data Collection**
- **Video Metadata**: Title, description, duration, views, likes, comments count
- **Channel Information**: Channel name, subscriber count, verification status
- **Comments & Replies**: All comments with engagement metrics and user info
- **Location Data**: GPS coordinates when available, location mentions in text
- **Temporal Data**: Upload time, comment timestamps, trending indicators

### 🧠 **NLP-Ready Data Processing**
- **Severity Classification**: Extracts severity indicators (extreme, high, medium, low)
- **Location Extraction**: Identifies mentioned cities, states, coastal areas
- **Urgency Assessment**: Rates content urgency based on keywords and recency
- **Sentiment Analysis**: Basic sentiment indicators from comments
- **Personal Experience Detection**: Identifies first-hand accounts vs news reports
- **Credibility Scoring**: Assesses content reliability based on multiple factors

### ⚡ **Real-time Monitoring**
- **Continuous Polling**: Configurable check intervals (15-60 minutes)
- **Smart Rate Limiting**: Respects YouTube API quotas and limits
- **Error Recovery**: Robust error handling with exponential backoff
- **Data Streaming**: Outputs to JSONL format for real-time NLP processing

## Installation & Setup

### Prerequisites
```bash
pip install google-api-python-client
```

### Configuration
1. **Get YouTube API Key**:
   - Go to Google Cloud Console
   - Create a new project or select existing
   - Enable YouTube Data API v3
   - Create credentials (API Key)
   - Replace the API key in `youtube_script.py`

2. **Configure Monitoring**:
   - Edit `youtube_config.py` for your specific needs
   - Set check intervals, keyword priorities
   - Configure output formats and file paths

## Usage

### Quick Demo
```python
python youtube_script.py
# Choose option 1 for demo mode
```

### Continuous Monitoring
```python
python youtube_script.py
# Choose option 2 for continuous monitoring
# Press Ctrl+C to stop
```

### Programmatic Usage
```python
from youtube_script import DisasterYouTubeMonitor, search_recent_disaster_videos

# Create monitor instance
monitor = DisasterYouTubeMonitor(output_file="my_data.jsonl")

# Collect recent data once
data = monitor.collect_recent_data()

# Or start continuous monitoring
monitor.monitor_continuously(check_interval_minutes=30)
```

## Output Data Structure

### Video Data Format
```json
{
  "data_type": "youtube_disaster_content",
  "collected_at": "2025-09-20T10:30:00",
  "search_keyword": "Mumbai flood live",
  "video": {
    "videoId": "abc123",
    "title": "Mumbai Floods Live: Heavy Rain Causes Waterlogging",
    "description": "Live coverage of flooding in Mumbai...",
    "publishedAt": "2025-09-20T08:00:00Z",
    "channelTitle": "News Channel",
    "viewCount": 15000,
    "likeCount": 200,
    "commentCount": 45,
    "duration": "PT15M30S",
    "url": "https://youtube.com/watch?v=abc123",
    "urgency_level": "critical",
    "severity_indicators": {
      "extreme": ["devastating", "unprecedented"],
      "high": ["heavy", "severe"]
    },
    "location_mentions": ["mumbai", "maharashtra", "india"],
    "liveBroadcastContent": "live"
  },
  "comments": [
    {
      "type": "main_comment",
      "author": "LocalResident123",
      "text": "Water level rising in our area, very scary situation",
      "publishedAt": "2025-09-20T09:15:00Z",
      "likeCount": 5,
      "personal_experience": true,
      "location_mentions": ["our area"],
      "severity_indicators": {"high": ["very"]},
      "sentiment_indicators": {"negative": ["scary"]}
    }
  ],
  "aggregated_insights": {
    "unique_locations": ["mumbai", "maharashtra"],
    "disaster_type": "flood",
    "estimated_severity": "high",
    "public_sentiment": "negative",
    "credibility_score": 8,
    "personal_experience_count": 12,
    "is_live_content": true
  }
}
```

## NLP Integration Guide

### Key Fields for NLP Processing
1. **Location Analysis**: `video.location_mentions`, `aggregated_insights.unique_locations`
2. **Severity Assessment**: `video.severity_indicators`, `aggregated_insights.estimated_severity`
3. **Urgency Detection**: `video.urgency_level`, `aggregated_insights.is_live_content`
4. **Sentiment Analysis**: `comments[].sentiment_indicators`, `aggregated_insights.public_sentiment`
5. **Credibility Scoring**: `aggregated_insights.credibility_score`
6. **Personal Accounts**: `comments[].personal_experience`

### Sample NLP Pipeline
```python
import json

# Read collected data
with open('youtube_disaster_data.jsonl', 'r') as f:
    for line in f:
        data = json.loads(line)
        
        # Extract for location NER
        locations = data['aggregated_insights']['unique_locations']
        
        # Extract for severity classification
        severity = data['aggregated_insights']['estimated_severity']
        
        # Extract personal accounts for impact assessment
        personal_accounts = [
            comment['text'] for comment in data['comments']
            if comment.get('personal_experience', False)
        ]
        
        # Process with your NLP models
        process_disaster_event(data)
```

## API Quota Management

### YouTube API Limits
- **Daily Quota**: 10,000 units per day
- **Search Operation**: 100 units per request
- **Video Details**: 1 unit per video
- **Comments**: 1 unit per request (100 comments)

### Optimization Strategies
1. **Smart Keyword Selection**: Use priority keywords during peak hours
2. **Batch Processing**: Group API calls efficiently
3. **Caching**: Store recent results to avoid duplicate calls
4. **Rate Limiting**: Built-in delays between requests

### Quota Usage Example
- 10 keywords × 20 videos × 1 search = 200 units
- 200 videos × 1 video detail = 200 units  
- 200 videos × 2 comment requests = 400 units
- **Total**: ~800 units per monitoring cycle

## Monitoring Dashboard Data

The system provides key metrics for dashboard creation:

### Real-time Metrics
- **Active Disasters**: Count of high-urgency events
- **Geographic Hotspots**: Areas with most mentions
- **Engagement Trends**: Public interest over time
- **Credibility Scores**: Content reliability indicators

### Historical Analysis
- **Disaster Timeline**: Chronological event tracking
- **Public Sentiment Trends**: How sentiment changes during events
- **Location-based Patterns**: Regional disaster frequency
- **Response Effectiveness**: Community vs official responses

## Troubleshooting

### Common Issues
1. **API Quota Exceeded**: Reduce check frequency or keyword count
2. **No Data Collected**: Check API key, verify keywords are returning results
3. **High Memory Usage**: Reduce max_comments_per_video setting
4. **Rate Limiting**: Increase delays between requests

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization
- **Reduce Keywords**: Focus on most effective search terms
- **Limit Comments**: Reduce max comments per video for faster processing
- **Filter Results**: Use credibility scores to filter low-quality content
- **Batch Processing**: Process collected data separately from collection

## Security & Privacy

### API Key Security
- Never commit API keys to version control
- Use environment variables: `os.getenv('YOUTUBE_API_KEY')`
- Rotate keys regularly

### Data Privacy
- Respect user privacy in comment collection
- Consider anonymizing personal information
- Follow YouTube's Terms of Service
- Implement data retention policies

## Integration Examples

### With Disaster Management Systems
```python
# Real-time alert system
def check_for_emergencies(data):
    if data['aggregated_insights']['estimated_severity'] == 'extreme':
        send_emergency_alert(data)
        
    if data['aggregated_insights']['personal_experience_count'] > 20:
        escalate_to_authorities(data)
```

### With Social Media Dashboard
```python
# Dashboard data feed
def format_for_dashboard(data):
    return {
        'event_type': data['aggregated_insights']['disaster_type'],
        'location': data['aggregated_insights']['primary_location'],
        'severity': data['aggregated_insights']['estimated_severity'],
        'engagement': data['video']['viewCount'],
        'timestamp': data['collected_at']
    }
```

## Future Enhancements

### Planned Features
1. **Multi-language Support**: Hindi, Tamil, Bengali, Gujarati
2. **Image Analysis**: Thumbnail and video frame analysis
3. **Cross-platform Integration**: Twitter, Instagram, Facebook
4. **Machine Learning**: Automated severity and credibility scoring
5. **Geospatial Mapping**: GPS coordinate extraction and mapping
6. **Predictive Analytics**: Early warning based on trending patterns

### Contributing
To contribute to this system:
1. Fork the repository
2. Add your enhancements
3. Test with real disaster scenarios
4. Submit pull request with documentation

## Support
For issues and questions:
- Check logs first: `youtube_disaster_monitor.log`
- Verify API quotas and limits
- Test with simple keywords first
- Contact system administrator with log details