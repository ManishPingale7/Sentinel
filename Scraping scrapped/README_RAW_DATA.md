# YouTube Raw Data Collection System

## Overview
This enhanced YouTube monitoring system collects **maximum raw data** from YouTube videos and comments related to disaster incidents in the Indian coastal regions. The system is designed to feed comprehensive, unprocessed data into your NLP pipeline for analysis.

## Key Features

### 🎯 **Comprehensive Data Collection**
- **Complete Video Metadata**: Title, description, channel info, statistics, content details, recording info
- **Full Comment Threads**: All comments with author info, engagement metrics, reply threads
- **Channel Intelligence**: Complete channel history, subscriber info, creation date, statistics
- **Content Analysis**: Hashtags, mentions, location references, timestamps
- **Search Context**: Keywords used, search parameters, collection metadata

### 🌊 **Disaster-Focused Monitoring**
- **50+ Targeted Keywords**: Flood, cyclone, tsunami, coastal erosion, heavy rain terms
- **Indian Geographic Focus**: Mumbai, Chennai, Kolkata, Visakhapatnam, Kochi, and coastal regions
- **Recent Content Priority**: Configurable time windows (6-24 hours back)
- **Multi-language Support**: Hindi, Tamil, Telugu, Bengali disaster terms

### 📊 **Raw Data Structure**
Each collected item contains:

```json
{
  "search_metadata": {
    "keyword": "flood Mumbai",
    "search_timestamp": "2024-01-20T10:30:00Z",
    "location_focus": "indian_subregion"
  },
  "video_metadata": {
    "id": "video_id",
    "title": "Complete video title",
    "description": "Full description text...",
    "channel_info": {
      "id": "channel_id",
      "title": "Channel Name",
      "description": "Channel description...",
      "statistics": {
        "subscriberCount": "1234567",
        "videoCount": "890",
        "viewCount": "12345678"
      },
      "creation_date": "2015-03-15T00:00:00Z"
    },
    "statistics": {
      "viewCount": "12345",
      "likeCount": "567",
      "commentCount": "89",
      "favoriteCount": "0"
    },
    "content_details": {
      "duration": "PT5M30S",
      "definition": "hd",
      "caption": "false",
      "live_broadcast_content": "none"
    },
    "recording_details": {
      "recording_date": "2024-01-20T08:00:00Z",
      "location_description": "Mumbai, India"
    },
    "live_streaming_details": {
      "actual_start_time": "2024-01-20T08:00:00Z",
      "scheduled_start_time": "2024-01-20T08:00:00Z"
    },
    "topic_details": {
      "topic_categories": [
        "https://en.wikipedia.org/wiki/News"
      ]
    }
  },
  "all_comments": [
    {
      "id": "comment_id",
      "text": "Complete comment text...",
      "author": {
        "display_name": "Author Name",
        "channel_id": "author_channel_id",
        "profile_image_url": "https://..."
      },
      "metadata": {
        "published_at": "2024-01-20T09:15:00Z",
        "updated_at": "2024-01-20T09:15:00Z",
        "parent_id": null,
        "can_rate": true,
        "total_reply_count": 5,
        "is_reply": false
      },
      "engagement": {
        "like_count": 12,
        "reply_count": 5,
        "is_public": true
      },
      "replies": [
        {
          "id": "reply_id",
          "text": "Reply text...",
          "author": {...},
          "metadata": {...}
        }
      ]
    }
  ],
  "content_analysis": {
    "hashtags": ["#MumbaiFloods", "#Rain", "#Maharashtra"],
    "mentions": ["@MumbaiPolice", "@weatherindia"],
    "location_mentions": ["Mumbai", "Bandra", "Andheri", "Maharashtra"],
    "total_text_length": 15678,
    "language_detected": "en"
  },
  "collection_timestamp": "2024-01-20T10:30:00Z"
}
```

## Usage

### 🚀 **Quick Start**
```python
from youtube_script import DisasterYouTubeMonitor

# Initialize monitor
monitor = DisasterYouTubeMonitor()

# Start continuous monitoring (collects every 30 minutes)
monitor.monitor_continuously(check_interval_minutes=30)

# Or collect data once
raw_data = monitor.collect_recent_data()
```

### ⚙️ **Configuration**
Edit `youtube_config.py`:
```python
# API Configuration
YOUTUBE_API_KEY = "your_api_key_here"

# Search Parameters
MAX_RESULTS_PER_KEYWORD = 20
SEARCH_HOURS_BACK = 6
MAX_COMMENTS_PER_VIDEO = 200

# Output
OUTPUT_FILE = "youtube_disaster_data.jsonl"
```

### 🔧 **Testing**
```bash
# Test raw data collection
python test_raw_data_collection.py

# Test API connectivity
python simple_youtube_test.py

# Validate complete system
python test_youtube_system.py
```

## File Structure

### 📁 **Core Files**
- `youtube_script.py` - Main data collection system
- `youtube_config.py` - Configuration and keywords
- `youtube_disaster_data.jsonl` - Output file (JSONL format)

### 🧪 **Testing Files**
- `test_raw_data_collection.py` - Test comprehensive data collection
- `simple_youtube_test.py` - Test API connectivity
- `test_youtube_system.py` - Validate complete system

### 📚 **Documentation**
- `README_FIX.md` - This file
- `README_YouTube_Monitor.md` - Original documentation

## NLP Integration

### 📊 **Data Processing Pipeline**
The collected raw data is perfect for feeding into your NLP pipeline:

1. **Text Extraction**: Video titles, descriptions, comment text
2. **Metadata Analysis**: Engagement metrics, temporal patterns, channel authority
3. **Geographic Processing**: Location mentions, regional language content
4. **Sentiment Analysis**: Comment sentiment, public reaction patterns
5. **Trend Detection**: Keyword frequency, viral content identification
6. **Credibility Assessment**: Channel history, engagement patterns, content consistency

### 🔄 **Real-time Processing**
- Data saved in JSONL format for streaming processing
- Each line is a complete, self-contained data object
- Timestamps enable temporal analysis
- Incremental processing supported

### 🎯 **Use Cases**
- **Disaster Early Warning**: Real-time content monitoring
- **Public Sentiment Analysis**: Community reaction to disasters
- **Information Verification**: Cross-reference multiple sources
- **Trend Analysis**: Emerging disaster patterns
- **Geographic Intelligence**: Location-specific disaster insights

## Rate Limiting & Quotas

### ⚡ **YouTube API Limits**
- **Daily Quota**: 10,000 units/day (default)
- **Search Cost**: 100 units per request
- **Video Details**: 1 unit per video
- **Comments**: 1 unit per 100 comments

### 🔄 **Optimization Strategies**
- **Keyword Rotation**: Cycles through disaster terms
- **Time-based Filtering**: Recent content focus (6-24 hours)
- **Batch Processing**: Multiple videos per search
- **Smart Caching**: Avoid duplicate API calls

### ⚠️ **Monitoring**
- Rate limit detection and automatic backoff
- Quota usage logging
- Error handling with exponential retry
- Graceful degradation on quota exceeded

## Troubleshooting

### 🔧 **Common Issues**

**API Key Problems**:
```bash
python simple_youtube_test.py  # Test API connectivity
```

**No Data Collected**:
- Check if keywords match recent content
- Verify time window (hours_back parameter)
- Ensure Indian geographic regions have recent disasters

**Quota Exceeded**:
- Reduce MAX_RESULTS_PER_KEYWORD in config
- Increase check_interval_minutes
- Use fewer disaster keywords

**Missing Comments**:
- Some videos have comments disabled
- Private/restricted content not accessible
- API may limit comment access

### 📝 **Logging**
All operations logged to `youtube_disaster_monitor.log`:
- API calls and responses
- Data collection statistics
- Error details and recovery
- Performance metrics

## Advanced Features

### 🌐 **Multi-language Support**
- Hindi disaster terms: बाढ़, तूफान, सुनामी
- Regional keywords for Bengali, Tamil, Telugu
- Automatic language detection in content

### 📈 **Analytics Ready**
- Structured data for time-series analysis
- Engagement metrics for virality detection
- Geographic data for spatial analysis
- Channel authority for credibility scoring

### 🔗 **Integration Points**
- **NLP Libraries**: spaCy, NLTK, transformers
- **Analytics**: pandas, matplotlib, seaborn
- **Databases**: MongoDB, PostgreSQL, Elasticsearch
- **Real-time**: Apache Kafka, Redis Streams

## Next Steps

1. **Test the system**: Run `test_raw_data_collection.py`
2. **Configure keywords**: Edit disaster terms in `youtube_config.py`
3. **Start monitoring**: Run continuous collection
4. **Feed to NLP**: Process JSONL output with your pipeline
5. **Analyze patterns**: Build insights from comprehensive raw data

The system now provides **maximum raw data extraction** without any preprocessing, giving you complete control over NLP analysis and feature extraction for your disaster monitoring pipeline.