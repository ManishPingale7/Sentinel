from googleapiclient.discovery import build
import json
import time
import logging
import os
from datetime import datetime, timedelta
import re

import os
from dotenv import load_dotenv

load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('youtube_disaster_monitor.log'),
        logging.StreamHandler()
    ]
)

# [KEY] Replace with your API key
API_KEY = os.getenv("YOUTUBE_API_KEY_SCRIPT")

# Validate API key
def validate_api_key():
    """Validate the YouTube API key"""
    if not API_KEY or API_KEY == "YOUR_API_KEY_HERE":
        logging.error("[ERROR] Invalid API key. Please set a valid YouTube Data API v3 key.")
        return False
    
    if len(API_KEY) != 39 or not API_KEY.startswith("AIza"):
        logging.error("[ERROR] API key format appears invalid. Should be 39 characters starting with 'AIza'")
        return False
    
    return True

# Initialize YouTube API with validation
def initialize_youtube_api():
    """Initialize YouTube API with proper error handling"""
    try:
        if not validate_api_key():
            return None
        
        youtube = build("youtube", "v3", developerKey=API_KEY)
        
        # Test the API key with a simple request
        test_request = youtube.search().list(
            q="test",
            part="id",
            maxResults=1,
            type="video"
        )
        test_response = test_request.execute()
        logging.info("[SUCCESS] YouTube API initialized successfully")
        return youtube
        
    except Exception as e:
        logging.error(f"[ERROR] Failed to initialize YouTube API: {str(e)}")
        if "invalid" in str(e).lower() or "forbidden" in str(e).lower():
            logging.error("   This appears to be an API key issue. Please check your key.")
        return None

youtube = initialize_youtube_api()

# Coastal disaster keywords for Indian subregion (prioritized for coastal hazards)
DISASTER_KEYWORDS = [
    # PRIMARY: Coastal and ocean hazards (highest priority)
    "tsunami India", "tsunami Indian Ocean", "tidal wave India", "storm surge India",
    "coastal flooding India", "sea level rise India", "coastal erosion India",
    "high tide flooding India", "king tide India", "saltwater intrusion India",
    
    # Cyclones and hurricanes (coastal focused)
    "cyclone India", "cyclone Indian Ocean", "hurricane India", "typhoon India",
    "tropical cyclone India", "severe cyclonic storm India", "super cyclone India",
    "cyclone landfall India", "cyclone warning India",
    
    # Coastal cities and regions (specific)
    "Mumbai coastal flood", "Chennai storm surge", "Kolkata cyclone", "Odisha cyclone",
    "Kerala coastal erosion", "Gujarat cyclone", "Tamil Nadu tsunami", "West Bengal cyclone",
    "Andhra Pradesh storm surge", "Karnataka coastal flood", "Goa coastal flooding",
    
    # Major coastal ports and vulnerable areas
    "Visakhapatnam cyclone", "Kochi storm surge", "Mangalore coastal flood", 
    "Puducherry tsunami", "Paradip cyclone", "Haldia tidal surge", "Diamond Harbour storm",
    "Sundarbans cyclone", "Rann of Kutch flood", "Konkan coast flood",
    
    # SECONDARY: General flooding (lower priority)
    "flood India coastal", "monsoon flooding coastal India", "urban coastal flooding India",
    "flash flood coastal India", "riverine flood India", "delta flooding India",
    
    # Ocean and marine specific
    "Arabian Sea cyclone", "Bay of Bengal cyclone", "Indian Ocean tsunami",
    "marine weather India", "sea storm India", "oceanic disturbance India",
    
    # Weather systems affecting coasts
    "low pressure Arabian Sea", "low pressure Bay of Bengal", "depression India",
    "monsoon depression India", "westerly disturbance India",
    
    # Recent/live coastal focus
    "live coastal India", "breaking tsunami India", "urgent cyclone India",
    "coastal disaster today India", "marine emergency India", "port closure India"
]

# Indian coastal regions and states
INDIAN_REGIONS = [
    "India", "Mumbai", "Chennai", "Kolkata", "Kerala", "Gujarat", "Tamil Nadu",
    "West Bengal", "Odisha", "Andhra Pradesh", "Karnataka", "Goa", "Maharashtra",
    "Visakhapatnam", "Kochi", "Mangalore", "Puducherry", "IMD", "NDMA"
]

# --- Duplicate Prevention System ---
class VideoDuplicateTracker:
    """
    Track processed video IDs to prevent duplicate data collection
    """
    def __init__(self, max_cache_size=10000):
        self.processed_videos = set()
        self.max_cache_size = max_cache_size
        self.cache_file = "processed_videos_cache.txt"
        self.load_cache()
    
    def load_cache(self):
        """Load previously processed video IDs from file"""
        try:
            if os.path.exists(self.cache_file):
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    lines = f.read().strip().split('\n')
                    self.processed_videos = set(line.strip() for line in lines if line.strip())
                logging.info(f"Loaded {len(self.processed_videos)} processed video IDs from cache")
        except Exception as e:
            logging.warning(f"Could not load video cache: {str(e)}")
            self.processed_videos = set()
    
    def save_cache(self):
        """Save processed video IDs to file"""
        try:
            # Limit cache size to prevent infinite growth
            if len(self.processed_videos) > self.max_cache_size:
                # Keep only the most recent entries (simple approach)
                self.processed_videos = set(list(self.processed_videos)[-self.max_cache_size:])
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                for video_id in self.processed_videos:
                    f.write(f"{video_id}\n")
            logging.info(f"Saved {len(self.processed_videos)} video IDs to cache")
        except Exception as e:
            logging.warning(f"Could not save video cache: {str(e)}")
    
    def is_duplicate(self, video_id):
        """Check if video ID has been processed before"""
        return video_id in self.processed_videos
    
    def mark_processed(self, video_id):
        """Mark video ID as processed"""
        self.processed_videos.add(video_id)
    
    def get_stats(self):
        """Get duplicate prevention statistics"""
        return {
            "total_processed": len(self.processed_videos),
            "cache_file": self.cache_file,
            "max_cache_size": self.max_cache_size
        }

# Global duplicate tracker instance
video_tracker = VideoDuplicateTracker()

# --- Enhanced Function 1: Search videos by keyword with time filters ---
def search_recent_disaster_videos(query, max_results=50, hours_back=24):
    """
    Search for recent disaster-related videos with comprehensive metadata
    """
    if not youtube:
        logging.error("YouTube API not initialized")
        return []
    
    try:
        # Calculate time filter for recent videos
        published_after = (datetime.now() - timedelta(hours=hours_back)).isoformat() + 'Z'
        
        # Build search parameters carefully
        search_params = {
            'q': query,
            'part': 'id,snippet',
            'type': 'video',
            'maxResults': min(max_results, 50),  # YouTube API limit
            'publishedAfter': published_after,
            'order': 'date',
            'safeSearch': 'none'
        }
        
        # Add optional parameters only if they work
        try:
            search_params['regionCode'] = 'IN'
        except:
            pass
            
        try:
            search_params['relevanceLanguage'] = 'en'  # Use only English for now
        except:
            pass
        
        logging.info(f"Searching for: {query} with params: {search_params}")
        
        request = youtube.search().list(**search_params)
        response = request.execute()
        
        videos = []
        for item in response.get("items", []):
            try:
                video_data = extract_video_metadata(item)
                if video_data:  # Only add if extraction was successful
                    videos.append(video_data)
            except Exception as e:
                logging.warning(f"Error extracting metadata for video item: {str(e)}")
                continue
            
        logging.info(f"Found {len(videos)} videos for query: {query}")
        return videos
        
    except Exception as e:
        logging.error(f"Error searching videos for query '{query}': {str(e)}")
        
        # Try with simpler parameters if the full search fails
        try:
            logging.info("Retrying with simplified parameters...")
            simple_request = youtube.search().list(
                q=query,
                part='id,snippet',
                type='video',
                maxResults=min(max_results, 10),
                order='date'
            )
            simple_response = simple_request.execute()
            
            videos = []
            for item in simple_response.get("items", []):
                video_data = extract_video_metadata(item)
                videos.append(video_data)
            
            logging.info(f"Simple search found {len(videos)} videos")
            return videos
            
        except Exception as e2:
            logging.error(f"Simple search also failed: {str(e2)}")
            return []

def extract_video_metadata(item):
    """
    Extract ALL possible metadata from video item for NLP processing
    Fixed to handle different API response structures
    """
    # Handle different video ID structures from YouTube API
    try:
        if isinstance(item.get("id"), dict):
            video_id = item["id"]["videoId"]
        else:
            video_id = item["id"]
    except (KeyError, TypeError):
        logging.error(f"Could not extract video ID from item: {item}")
        return None
    
    # Check for snippet data
    if "snippet" not in item:
        logging.error(f"No snippet data for video {video_id}")
        return None
    
    snippet = item["snippet"]
    
    # RECENCY CHECK: Only process videos from last 48 hours
    published_at = snippet.get("publishedAt")
    if published_at:
        try:
            published_date = datetime.fromisoformat(published_at.replace('Z', '+00:00'))
            current_time = datetime.now(published_date.tzinfo)
            age_hours = (current_time - published_date).total_seconds() / 3600
            
            if age_hours > 48:  # Reject videos older than 48 hours
                logging.debug(f"Skipping old video ({age_hours:.1f} hours old): {snippet.get('title', 'Unknown')}")
                return None
        except Exception as e:
            logging.warning(f"Error checking video age: {str(e)}")
    
    # Get additional video details with error handling
    video_details = get_video_details(video_id)
    
    # Get channel details with error handling
    channel_id = snippet.get("channelId")
    if channel_id:
        channel_details = get_channel_details(channel_id)
    else:
        channel_details = {}
    
    # Get comprehensive tags from title, description, and YouTube tags
    title = snippet.get("title", "")
    description = snippet.get("description", "")
    youtube_tags = snippet.get("tags", [])
    
    try:
        comprehensive_tags = extract_comprehensive_tags(title, description, youtube_tags)
    except Exception as e:
        logging.warning(f"Error extracting tags for video {video_id}: {str(e)}")
        comprehensive_tags = {"detailed_tags": {}, "unified_tags": [], "total_tag_count": 0}
    
    # Get video transcription/captions information with error handling
    try:
        transcription_info = extract_video_transcription_info(video_id)
    except Exception as e:
        logging.warning(f"Error getting transcription for video {video_id}: {str(e)}")
        transcription_info = {"has_captions": False, "extraction_method": "error"}
    
    return {
        "videoId": video_id,
        "url": f"https://www.youtube.com/watch?v={video_id}",
        "collected_at": datetime.now().isoformat(),
        
        # Basic video information
        "title": title,
        "description": description,
        "publishedAt": snippet.get("publishedAt"),
        "defaultLanguage": snippet.get("defaultLanguage"),
        "defaultAudioLanguage": snippet.get("defaultAudioLanguage"),
        "categoryId": snippet.get("categoryId"),
        "liveBroadcastContent": snippet.get("liveBroadcastContent", "none"),
        
        # Comprehensive tags (NEW - combines all tag types)
        "tags": comprehensive_tags,
        
        # Channel information
        "channel": {
            "channelId": channel_id,
            "channelTitle": snippet.get("channelTitle", ""),
            "channelDetails": channel_details
        },
        
        # Video statistics and details
        "statistics": {
            "viewCount": video_details.get("viewCount", 0),
            "likeCount": video_details.get("likeCount", 0),
            "dislikeCount": video_details.get("dislikeCount", 0),
            "favoriteCount": video_details.get("favoriteCount", 0),
            "commentCount": video_details.get("commentCount", 0)
        },
        
        # Content details
        "contentDetails": {
            "duration": video_details.get("duration"),
            "dimension": video_details.get("dimension"),
            "definition": video_details.get("definition"),
            "caption": video_details.get("caption"),
            "licensedContent": video_details.get("licensedContent"),
            "projection": video_details.get("projection")
        },
        
        # Transcription/Captions information (NEW)
        "transcription": transcription_info,
        
        # Location data if available
        "location": video_details.get("location"),
        
        # Live streaming details if applicable
        "liveStreamingDetails": video_details.get("liveStreamingDetails"),
        
        # Legacy fields for compatibility (will be removed in next version)
        "hashtags": comprehensive_tags.get("detailed_tags", {}).get("hashtags", []),
        "mentions": extract_mentions(title + " " + description)
    }

def get_video_details(video_id):
    """
    Get ALL possible video statistics and details
    """
    if not youtube:
        return {}
        
    try:
        request = youtube.videos().list(
            part="statistics,contentDetails,recordingDetails,liveStreamingDetails,localizations,player,processingDetails,status,suggestions,topicDetails",
            id=video_id
        )
        response = request.execute()
        
        if response["items"]:
            item = response["items"][0]
            details = {}
            
            # Statistics
            if "statistics" in item:
                stats = item["statistics"]
                details.update({
                    "viewCount": int(stats.get("viewCount", 0)),
                    "likeCount": int(stats.get("likeCount", 0)),
                    "dislikeCount": int(stats.get("dislikeCount", 0)),
                    "favoriteCount": int(stats.get("favoriteCount", 0)),
                    "commentCount": int(stats.get("commentCount", 0))
                })
            
            # Content details
            if "contentDetails" in item:
                content = item["contentDetails"]
                details.update({
                    "duration": content.get("duration"),
                    "dimension": content.get("dimension"),
                    "definition": content.get("definition"),
                    "caption": content.get("caption"),
                    "licensedContent": content.get("licensedContent"),
                    "projection": content.get("projection"),
                    "regionRestriction": content.get("regionRestriction"),
                    "contentRating": content.get("contentRating")
                })
            
            # Recording details (location if available)
            if "recordingDetails" in item:
                recording = item["recordingDetails"]
                if "location" in recording:
                    details["location"] = {
                        "latitude": recording["location"].get("latitude"),
                        "longitude": recording["location"].get("longitude"),
                        "altitude": recording["location"].get("altitude")
                    }
                if "recordingDate" in recording:
                    details["recordingDate"] = recording["recordingDate"]
            
            # Live streaming details
            if "liveStreamingDetails" in item:
                details["liveStreamingDetails"] = item["liveStreamingDetails"]
            
            # Status information
            if "status" in item:
                details["status"] = item["status"]
            
            # Topic details (categories)
            if "topicDetails" in item:
                details["topicDetails"] = item["topicDetails"]
            
            return details
            
    except Exception as e:
        logging.warning(f"Could not get video details for {video_id}: {str(e)}")
        # Try with minimal parts if full request fails
        try:
            simple_request = youtube.videos().list(
                part="statistics,contentDetails",
                id=video_id
            )
            simple_response = simple_request.execute()
            if simple_response["items"]:
                item = simple_response["items"][0]
                simple_details = {}
                
                if "statistics" in item:
                    stats = item["statistics"]
                    simple_details.update({
                        "viewCount": int(stats.get("viewCount", 0)),
                        "likeCount": int(stats.get("likeCount", 0)),
                        "commentCount": int(stats.get("commentCount", 0))
                    })
                
                if "contentDetails" in item:
                    simple_details["duration"] = item["contentDetails"].get("duration")
                
                return simple_details
        except:
            pass
        
        return {}
    
    return {}

def get_channel_details(channel_id):
    """
    Get comprehensive channel information
    """
    if not youtube:
        return {}
    
    try:
        request = youtube.channels().list(
            part="snippet,statistics,status,topicDetails,brandingSettings,contentDetails",
            id=channel_id
        )
        response = request.execute()
        
        if response["items"]:
            item = response["items"][0]
            
            channel_info = {
                "channelId": channel_id,
                "title": item["snippet"].get("title"),
                "description": item["snippet"].get("description"),
                "customUrl": item["snippet"].get("customUrl"),
                "publishedAt": item["snippet"].get("publishedAt"),
                "defaultLanguage": item["snippet"].get("defaultLanguage"),
                "country": item["snippet"].get("country")
            }
            
            # Channel statistics
            if "statistics" in item:
                stats = item["statistics"]
                channel_info["statistics"] = {
                    "viewCount": int(stats.get("viewCount", 0)),
                    "subscriberCount": int(stats.get("subscriberCount", 0)),
                    "videoCount": int(stats.get("videoCount", 0)),
                    "hiddenSubscriberCount": stats.get("hiddenSubscriberCount", False)
                }
            
            # Channel status
            if "status" in item:
                channel_info["status"] = item["status"]
            
            # Topic details
            if "topicDetails" in item:
                channel_info["topicDetails"] = item["topicDetails"]
            
            # Branding settings
            if "brandingSettings" in item:
                channel_info["brandingSettings"] = item["brandingSettings"]
            
            # Content details
            if "contentDetails" in item:
                channel_info["contentDetails"] = item["contentDetails"]
            
            return channel_info
            
    except Exception as e:
        logging.warning(f"Could not get channel details for {channel_id}: {str(e)}")
        return {"channelId": channel_id, "error": str(e)}
    
    return {}

def extract_comprehensive_tags(title, description, youtube_tags=None):
    """
    Extract comprehensive tags from video title, description, and YouTube tags
    Combines hashtags, keywords, and YouTube-provided tags
    """
    import re
    
    # Combine title and description text
    full_text = f"{title} {description}"
    
    # 1. Extract hashtags
    hashtag_pattern = r'#\w+'
    hashtags = re.findall(hashtag_pattern, full_text, re.IGNORECASE)
    hashtags = [tag.lower() for tag in hashtags]
    
    # 2. Extract disaster-related keywords
    disaster_keywords = [
        'flood', 'flooding', 'tsunami', 'cyclone', 'hurricane', 'storm', 'rain', 'drought',
        'earthquake', 'landslide', 'coastal', 'erosion', 'weather', 'disaster', 'emergency',
        'rescue', 'evacuation', 'alert', 'warning', 'damage', 'destruction', 'relief',
        'mumbai', 'chennai', 'kolkata', 'kerala', 'gujarat', 'odisha', 'bengal', 'tamil',
        'monsoon', 'heavy', 'extreme', 'severe', 'urgent', 'breaking', 'live'
    ]
    
    # Find disaster keywords in text (case insensitive)
    found_keywords = []
    for keyword in disaster_keywords:
        if re.search(r'\b' + keyword + r'\b', full_text, re.IGNORECASE):
            found_keywords.append(keyword)
    
    # 3. Extract location mentions (Indian regions/cities)
    location_pattern = r'\b(?:mumbai|chennai|kolkata|delhi|bengaluru|hyderabad|ahmedabad|pune|surat|jaipur|lucknow|kanpur|nagpur|visakhapatnam|indore|thane|bhopal|pimpri|patna|vadodara|ghaziabad|ludhiana|agra|nashik|faridabad|meerut|rajkot|kalyan|vasai|varanasi|srinagar|aurangabad|dhanbad|amritsar|navi mumbai|allahabad|ranchi|howrah|coimbatore|jabalpur|gwalior|vijayawada|jodhpur|madurai|raipur|kota|guwahati|chandigarh|solapur|hubballi|tiruchirappalli|salem|mira bhayandar|thiruvananthapuram|bhiwandi|saharanpur|guntur|bikaner|amravati|noida|jamshedpur|bhilai|warangal|cuttack|firozabad|kochi|bhavnagar|dehradun|durgapur|asansol|rourkela|nanded|kolhapur|ajmer|akola|gulbarga|jamnagar|ujjain|loni|siliguri|jhansi|ulhasnagar|jammu|sangli miraj kupwad|mangalore|erode|belgaum|ambattur|tirunelveli|malegaon|gaya|jalgaon|udaipur|maheshtala|kerala|karnataka|tamil nadu|andhra pradesh|telangana|maharashtra|gujarat|rajasthan|west bengal|odisha|bihar|uttar pradesh|madhya pradesh|punjab|haryana|himachal pradesh|uttarakhand|jharkhand|chhattisgarh|assam|meghalaya|manipur|tripura|mizoram|arunachal pradesh|nagaland|sikkim|goa)\b'
    
    locations = re.findall(location_pattern, full_text, re.IGNORECASE)
    locations = [loc.lower() for loc in locations]
    
    # 4. Include YouTube provided tags
    youtube_provided_tags = youtube_tags if youtube_tags else []
    youtube_provided_tags = [tag.lower() for tag in youtube_provided_tags]
    
    # 5. Extract additional meaningful words (nouns, adjectives)
    meaningful_words = []
    # Simple keyword extraction - words longer than 3 characters, excluding common words
    common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'may', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'with', 'have', 'this', 'will', 'your', 'from', 'they', 'know', 'want', 'been', 'good', 'much', 'some', 'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make', 'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were'}
    
    words = re.findall(r'\b[a-zA-Z]{4,}\b', full_text.lower())
    meaningful_words = [word for word in words if word not in common_words]
    
    # Combine all tags and remove duplicates
    all_tags = {
        'hashtags': list(set(hashtags)),
        'disaster_keywords': list(set(found_keywords)),
        'location_mentions': list(set(locations)),
        'youtube_tags': list(set(youtube_provided_tags)),
        'extracted_keywords': list(set(meaningful_words[:20]))  # Limit to top 20
    }
    
    # Create a unified tag list for easy searching
    unified_tags = list(set(hashtags + found_keywords + locations + youtube_provided_tags + meaningful_words[:10]))
    
    return {
        'detailed_tags': all_tags,
        'unified_tags': unified_tags,
        'total_tag_count': len(unified_tags)
    }

def extract_hashtags(text):
    """
    Extract hashtags from text (legacy function for compatibility)
    """
    import re
    hashtag_pattern = r'#\w+'
    hashtags = re.findall(hashtag_pattern, text)
    return list(set(hashtags))  # Remove duplicates

def extract_mentions(text):
    """
    Extract @mentions from text
    """
    import re
    mention_pattern = r'@\w+'
    mentions = re.findall(mention_pattern, text)
    return list(set(mentions))  # Remove duplicates

def validate_content_seriousness(video_data, comments=None, transcription=None):
    """
    Validate if content is serious disaster-related content or humor/comedy/parody
    Returns: (is_serious: bool, confidence_score: float, reasons: list)
    """
    import re
    
    # Keywords that indicate non-serious content
    humor_keywords = [
        # Comedy/humor indicators
        'comedy', 'funny', 'humor', 'humour', 'joke', 'jokes', 'joking', 'hilarious',
        'laugh', 'laughing', 'lol', 'haha', 'rofl', 'lmao', 'amusing', 'entertaining',
        
        # Parody/mimicry indicators
        'parody', 'spoof', 'satire', 'satirical', 'mimicry', 'mimic', 'imitation',
        'mock', 'mocking', 'mockery', 'fake', 'spoof', 'trolling', 'troll',
        
        # Sarcasm indicators
        'sarcasm', 'sarcastic', 'irony', 'ironic', 'cynical', 'witty', 'clever',
        
        # Prank indicators
        'prank', 'pranks', 'pranking', 'prankster', 'trick', 'hoax', 'fake news',
        'clickbait', 'staged', 'acting', 'drama', 'theatrical',
        
        # Entertainment indicators
        'entertainment', 'show', 'episode', 'series', 'vlog', 'blogger', 'youtuber',
        'reaction', 'review', 'unboxing', 'challenge', 'trend', 'viral',
        
        # Meme culture
        'meme', 'memes', 'trending', 'viral video', 'for fun', 'just kidding',
        'not serious', 'dont take seriously', "don't take seriously"
    ]
    
    # Keywords that indicate serious disaster content
    serious_keywords = [
        # Emergency indicators
        'emergency', 'urgent', 'critical', 'alert', 'warning', 'evacuation',
        'rescue', 'relief', 'disaster', 'catastrophe', 'crisis', 'calamity',
        
        # News indicators
        'breaking news', 'news update', 'live coverage', 'report', 'reporting',
        'journalist', 'correspondent', 'official', 'government', 'authorities',
        
        # Damage indicators
        'damage', 'destruction', 'devastation', 'casualties', 'victims', 'injured',
        'death', 'fatalities', 'missing', 'trapped', 'stranded',
        
        # Response indicators
        'ndrf', 'army', 'navy', 'coast guard', 'police', 'fire brigade',
        'medical', 'hospital', 'ambulance', 'shelter', 'relief camp'
    ]
    
    # Coastal/ocean specific serious keywords
    coastal_serious_keywords = [
        'tsunami', 'tidal wave', 'storm surge', 'coastal flooding', 'sea level',
        'high tide', 'low pressure', 'cyclone', 'hurricane', 'typhoon',
        'coastal erosion', 'sea wall', 'embankment', 'saltwater intrusion'
    ]
    
    confidence_score = 0.0
    reasons = []
    
    # Combine all text sources for analysis
    all_text_sources = []
    
    # Add video title and description
    if video_data:
        all_text_sources.append(('title', video_data.get('title', '')))
        all_text_sources.append(('description', video_data.get('description', '')))
        
        # Add channel information
        channel_info = video_data.get('channel', {}).get('channelDetails', {})
        if channel_info:
            all_text_sources.append(('channel_title', channel_info.get('title', '')))
            all_text_sources.append(('channel_description', channel_info.get('description', '')))
    
    # Add comments
    if comments:
        for i, comment in enumerate(comments[:10]):  # Check first 10 comments
            all_text_sources.append((f'comment_{i}', comment.get('text', '')))
    
    # Add transcription
    if transcription and isinstance(transcription, str):
        all_text_sources.append(('transcription', transcription))
    elif transcription and isinstance(transcription, dict):
        transcript_text = transcription.get('transcription', '') or transcription.get('text', '')
        if transcript_text:
            all_text_sources.append(('transcription', transcript_text))
    
    # Analyze each text source
    humor_score = 0
    serious_score = 0
    coastal_score = 0
    
    for source_type, text in all_text_sources:
        if not text:
            continue
            
        text_lower = text.lower()
        
        # Check for humor keywords
        humor_matches = []
        for keyword in humor_keywords:
            if re.search(r'\b' + keyword + r'\b', text_lower):
                humor_matches.append(keyword)
                humor_score += 1
        
        if humor_matches:
            reasons.append(f"Humor indicators in {source_type}: {', '.join(humor_matches[:3])}")
        
        # Check for serious keywords
        serious_matches = []
        for keyword in serious_keywords:
            if re.search(r'\b' + keyword + r'\b', text_lower):
                serious_matches.append(keyword)
                serious_score += 1
        
        # Check for coastal serious keywords (higher weight)
        coastal_matches = []
        for keyword in coastal_serious_keywords:
            if re.search(r'\b' + keyword + r'\b', text_lower):
                coastal_matches.append(keyword)
                coastal_score += 2  # Higher weight for coastal keywords
                serious_score += 2
        
        if serious_matches:
            reasons.append(f"Serious indicators in {source_type}: {', '.join(serious_matches[:3])}")
        
        if coastal_matches:
            reasons.append(f"Coastal hazard indicators in {source_type}: {', '.join(coastal_matches[:3])}")
    
    # Calculate confidence score
    total_score = humor_score + serious_score + coastal_score
    
    if total_score == 0:
        confidence_score = 0.5  # Neutral
        is_serious = True  # Default to serious if uncertain
        reasons.append("No clear humor or serious indicators found - defaulting to serious")
    else:
        # Weight serious and coastal content more heavily
        weighted_serious_score = serious_score + (coastal_score * 1.5)
        confidence_score = weighted_serious_score / (humor_score + weighted_serious_score)
        
        # Content is serious if confidence > 0.6 and coastal score > 0
        is_serious = (confidence_score > 0.6) or (coastal_score > 0 and humor_score < serious_score)
    
    # Special case: If humor score is very high, mark as non-serious regardless
    if humor_score > serious_score + coastal_score + 2:
        is_serious = False
        confidence_score = 1.0 - confidence_score
        reasons.append("High humor content detected - marked as non-serious")
    
    return is_serious, confidence_score, reasons

def extract_mentions(text):
    """
    Extract @mentions from text
    """
    import re
    mention_pattern = r'@\w+'
    mentions = re.findall(mention_pattern, text)
    return list(set(mentions))  # Remove duplicates

# --- Enhanced Function 2: Fetch ALL comments with complete metadata ---
def fetch_all_comments(video_id, max_comments=500):
    """
    Fetch ALL possible comment data for comprehensive NLP analysis
    """
    if not youtube:
        logging.warning("YouTube API not initialized - cannot fetch comments")
        return []
        
    comments = []
    try:
        request = youtube.commentThreads().list(
            part="snippet,replies",  # Get both main comments and replies
            videoId=video_id,
            maxResults=100,
            textFormat="plainText",
            order="relevance"
        )
        
        while request and len(comments) < max_comments:
            response = request.execute()
            
            for item in response.get("items", []):
                # Main comment with ALL metadata
                main_comment = item["snippet"]["topLevelComment"]["snippet"]
                comment_data = {
                    "commentId": item["snippet"]["topLevelComment"]["id"],
                    "type": "main_comment",
                    "videoId": video_id,
                    
                    # Author information
                    "author": {
                        "displayName": main_comment["authorDisplayName"],
                        "profileImageUrl": main_comment.get("authorProfileImageUrl"),
                        "channelUrl": main_comment.get("authorChannelUrl"),
                        "channelId": main_comment.get("authorChannelId", {}).get("value") if main_comment.get("authorChannelId") else None
                    },
                    
                    # Comment content
                    "text": main_comment["textDisplay"],
                    "textOriginal": main_comment.get("textOriginal", main_comment["textDisplay"]),
                    
                    # Metadata
                    "publishedAt": main_comment["publishedAt"],
                    "updatedAt": main_comment.get("updatedAt"),
                    "likeCount": main_comment.get("likeCount", 0),
                    "moderationStatus": main_comment.get("moderationStatus"),
                    "canRate": main_comment.get("canRate", True),
                    "viewerRating": main_comment.get("viewerRating", "none"),
                    
                    # Thread information
                    "totalReplyCount": item["snippet"].get("totalReplyCount", 0),
                    "canReply": item["snippet"].get("canReply", True),
                    "isPublic": item["snippet"].get("isPublic", True),
                    
                    # Extracted elements
                    "hashtags": extract_hashtags(main_comment["textDisplay"]),
                    "mentions": extract_mentions(main_comment["textDisplay"]),
                    "urls": extract_urls(main_comment["textDisplay"]),
                    
                    # Raw data for NLP
                    "raw_data": {
                        "full_snippet": main_comment,
                        "thread_snippet": item["snippet"]
                    }
                }
                comments.append(comment_data)
                
                # Get all replies if they exist
                if "replies" in item:
                    for reply in item["replies"]["comments"]:
                        reply_snippet = reply["snippet"]
                        reply_data = {
                            "commentId": reply["id"],
                            "type": "reply",
                            "videoId": video_id,
                            "parentCommentId": item["snippet"]["topLevelComment"]["id"],
                            
                            # Author information
                            "author": {
                                "displayName": reply_snippet["authorDisplayName"],
                                "profileImageUrl": reply_snippet.get("authorProfileImageUrl"),
                                "channelUrl": reply_snippet.get("authorChannelUrl"),
                                "channelId": reply_snippet.get("authorChannelId", {}).get("value") if reply_snippet.get("authorChannelId") else None
                            },
                            
                            # Comment content
                            "text": reply_snippet["textDisplay"],
                            "textOriginal": reply_snippet.get("textOriginal", reply_snippet["textDisplay"]),
                            
                            # Metadata
                            "publishedAt": reply_snippet["publishedAt"],
                            "updatedAt": reply_snippet.get("updatedAt"),
                            "likeCount": reply_snippet.get("likeCount", 0),
                            "moderationStatus": reply_snippet.get("moderationStatus"),
                            "canRate": reply_snippet.get("canRate", True),
                            "viewerRating": reply_snippet.get("viewerRating", "none"),
                            
                            # Parent context
                            "parentCommentText": main_comment["textDisplay"][:100] + "..." if len(main_comment["textDisplay"]) > 100 else main_comment["textDisplay"],
                            
                            # Extracted elements
                            "hashtags": extract_hashtags(reply_snippet["textDisplay"]),
                            "mentions": extract_mentions(reply_snippet["textDisplay"]),
                            "urls": extract_urls(reply_snippet["textDisplay"]),
                            
                            # Raw data for NLP
                            "raw_data": {
                                "full_snippet": reply_snippet
                            }
                        }
                        comments.append(reply_data)
            
            # Get next page of comments
            try:
                request = youtube.commentThreads().list_next(request, response)
            except:
                break
                
        logging.info(f"Collected {len(comments)} comments for video {video_id}")
        return comments
        
    except Exception as e:
        logging.error(f"Error fetching comments for video {video_id}: {str(e)}")
        if "disabled" in str(e).lower():
            logging.info("Comments appear to be disabled for this video")
        return []

def extract_urls(text):
    """
    Extract URLs from text
    """
    import re
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_pattern, text)
    return urls

def get_video_captions(video_id):
    """
    Enhanced video captions/transcription extraction for NLP analysis and seriousness validation
    """
    if not youtube:
        return {"has_captions": False, "transcription": "", "caption_tracks": [], "extraction_method": "api_unavailable"}
    
    try:
        # First, check if captions are available
        captions_request = youtube.captions().list(
            part="snippet",
            videoId=video_id
        )
        captions_response = captions_request.execute()
        
        caption_data = {
            "has_captions": False,
            "transcription": "",
            "caption_tracks": [],
            "auto_generated": False,
            "languages": [],
            "extraction_method": "none",
            "text_for_analysis": ""
        }
        
        if not captions_response.get("items"):
            return caption_data
        
        caption_data["has_captions"] = True
        
        # Process available caption tracks
        available_tracks = []
        for caption_track in captions_response["items"]:
            track_info = {
                "id": caption_track["id"],
                "name": caption_track["snippet"].get("name", ""),
                "language": caption_track["snippet"].get("language", ""),
                "track_kind": caption_track["snippet"].get("trackKind", ""),
                "is_auto_generated": caption_track["snippet"].get("trackKind") == "asr",
                "is_draft": caption_track["snippet"].get("isDraft", False),
                "failure_reason": caption_track["snippet"].get("failureReason", "")
            }
            available_tracks.append(track_info)
            caption_data["caption_tracks"].append(track_info)
            caption_data["languages"].append(track_info["language"])
            
            # Mark if any track is auto-generated
            if track_info["is_auto_generated"]:
                caption_data["auto_generated"] = True
        
        # Try to extract text content using alternative method
        # Since direct download requires special authentication, we'll extract from video details
        try:
            # Get video snippet which sometimes includes caption-like text
            video_request = youtube.videos().list(
                part="snippet,contentDetails",
                id=video_id
            )
            video_response = video_request.execute()
            
            if video_response.get("items"):
                video_item = video_response["items"][0]
                snippet = video_item.get("snippet", {})
                
                # Use description as proxy for content analysis if it's detailed
                description = snippet.get("description", "")
                
                # If description is substantial, use it for seriousness analysis
                if len(description) > 100:
                    caption_data["text_for_analysis"] = description
                    caption_data["extraction_method"] = "description_proxy"
                    caption_data["transcription"] = f"Description-based content: {description[:500]}..."
                
                # Check if captions are confirmed available
                content_details = video_item.get("contentDetails", {})
                caption_status = content_details.get("caption", "false")
                caption_data["confirmed_available"] = caption_status == "true"
        
        except Exception as detail_error:
            logging.warning(f"Could not extract video details for captions: {str(detail_error)}")
        
        # For now, mark that captions are available for future processing
        if caption_data["caption_tracks"]:
            preferred_track = None
            
            # Prefer English captions first
            for track in caption_data["caption_tracks"]:
                if track["language"] in ["en", "en-US", "en-GB"]:
                    preferred_track = track
                    break
            
            # If no English, prefer manual captions over auto-generated
            if not preferred_track:
                manual_tracks = [t for t in caption_data["caption_tracks"] if not t["is_auto_generated"]]
                if manual_tracks:
                    preferred_track = manual_tracks[0]
                else:
                    preferred_track = caption_data["caption_tracks"][0]
            
            caption_data["selected_track"] = preferred_track
            caption_data["extraction_method"] = "track_available"
            
            # Note about limitations
            caption_data["note"] = "Caption tracks detected but content extraction requires additional authentication"
            
            logging.info(f"Captions available for video {video_id} - Language: {preferred_track['language']}, Auto-generated: {preferred_track['is_auto_generated']}")
        
        return caption_data
        
    except Exception as e:
        logging.warning(f"Error checking captions for video {video_id}: {str(e)}")
        return {
            "has_captions": False,
            "transcription": "",
            "caption_tracks": [],
            "error": str(e),
            "extraction_method": "error"
        }

def extract_video_transcription_info(video_id):
    """
    Extract transcription information and caption availability
    Alternative approach using video details to check caption availability
    """
    try:
        # Check if captions are enabled in video details
        video_request = youtube.videos().list(
            part="contentDetails",
            id=video_id
        )
        video_response = video_request.execute()
        
        transcription_info = {
            "video_id": video_id,
            "captions_available": False,
            "caption_status": "unknown",
            "manual_captions": False,
            "auto_captions": False,
            "transcription_note": "Caption content requires special authentication to download"
        }
        
        if video_response.get("items"):
            content_details = video_response["items"][0].get("contentDetails", {})
            caption_status = content_details.get("caption", "false")
            
            transcription_info["captions_available"] = caption_status == "true"
            transcription_info["caption_status"] = caption_status
            
            if transcription_info["captions_available"]:
                # Get detailed caption information
                caption_details = get_video_captions(video_id)
                transcription_info.update(caption_details)
        
        return transcription_info
        
    except Exception as e:
        logging.warning(f"Error getting transcription info for {video_id}: {str(e)}")
        return {
            "video_id": video_id,
            "captions_available": False,
            "error": str(e)
        }

def get_channel_recent_videos(channel_id, max_videos=10):
    """
    Get recent videos from a channel for context
    """
    if not youtube:
        return []
    
    try:
        # Get uploads playlist ID
        channel_request = youtube.channels().list(
            part="contentDetails",
            id=channel_id
        )
        channel_response = channel_request.execute()
        
        if not channel_response["items"]:
            return []
        
        uploads_playlist_id = channel_response["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
        
        # Get recent videos from uploads playlist
        playlist_request = youtube.playlistItems().list(
            part="snippet",
            playlistId=uploads_playlist_id,
            maxResults=max_videos
        )
        playlist_response = playlist_request.execute()
        
        recent_videos = []
        for item in playlist_response.get("items", []):
            recent_videos.append({
                "videoId": item["snippet"]["resourceId"]["videoId"],
                "title": item["snippet"]["title"],
                "publishedAt": item["snippet"]["publishedAt"]
            })
        
        return recent_videos
        
    except Exception as e:
        logging.warning(f"Could not get recent videos for channel {channel_id}: {str(e)}")
        return []

# --- Data Aggregation (No NLP Processing) ---
def aggregate_raw_data(video, comments, search_keyword):
    """
    Aggregate all raw data for NLP processing - NO preprocessing
    """
    # Get channel's recent videos for context
    recent_videos = get_channel_recent_videos(video["channel"]["channelId"], max_videos=5)
    
    return {
        "data_type": "youtube_raw_content",
        "collected_at": datetime.now().isoformat(),
        "search_keyword": search_keyword,
        
        # Complete video data
        "video": video,
        
        # All comments with complete metadata
        "comments": comments,
        "comment_statistics": {
            "total_comments": len(comments),
            "main_comments": len([c for c in comments if c["type"] == "main_comment"]),
            "replies": len([c for c in comments if c["type"] == "reply"]),
            "total_likes": sum(c.get("likeCount", 0) for c in comments),
            "unique_authors": len(set(c["author"]["displayName"] for c in comments))
        },
        
        # Channel context
        "channel_context": {
            "recent_videos": recent_videos,
            "channel_details": video["channel"]["channelDetails"]
        },
        
        # All text content for NLP processing
        "text_content": {
            "video_title": video["title"],
            "video_description": video["description"],
            "channel_description": video["channel"]["channelDetails"].get("description", ""),
            "all_comment_text": [c["text"] for c in comments],
            "hashtags_found": list(set(video["hashtags"] + [tag for c in comments for tag in c.get("hashtags", [])])),
            "mentions_found": list(set(video["mentions"] + [mention for c in comments for mention in c.get("mentions", [])])),
            "urls_found": [url for c in comments for url in c.get("urls", [])]
        },
        
        # Metadata for analysis
        "engagement_metrics": {
            "view_count": video["statistics"]["viewCount"],
            "like_count": video["statistics"]["likeCount"],
            "comment_count": video["statistics"]["commentCount"],
            "channel_subscriber_count": video["channel"]["channelDetails"].get("statistics", {}).get("subscriberCount", 0),
            "channel_video_count": video["channel"]["channelDetails"].get("statistics", {}).get("videoCount", 0)
        }
    }


# --- Real-time Monitoring System ---
class DisasterYouTubeMonitor:
    """
    Real-time YouTube disaster monitoring system
    """
    
    def __init__(self, output_file="youtube_disaster_data.jsonl"):
        self.output_file = output_file
        self.last_check = datetime.now()
        
    def monitor_continuously(self, check_interval_hours=24):
        """
        Continuously monitor YouTube for new disaster content with 24-hour comprehensive cycles
        Optimized for thorough research of all videos uploaded in the last 24 hours
        """
        logging.info(f"Starting 24-hour comprehensive YouTube disaster monitoring...")
        logging.info(f"Collection cycle: Every {check_interval_hours} hours")
        
        cycle_count = 0
        
        while True:
            try:
                cycle_count += 1
                logging.info(f"[CYCLE] Starting collection cycle #{cycle_count}")
                
                # Comprehensive recent data collection (48 hours max for recency)
                all_raw_data = self.collect_recent_data(hours_back=48)
                
                if all_raw_data:
                    logging.info(f"[STATS] Cycle #{cycle_count} Summary:")
                    logging.info(f"  [SUCCESS] Total videos collected: {len(all_raw_data)}")
                    
                    # Analysis of collected data
                    coastal_count = sum(1 for item in all_raw_data 
                                      if item.get('validation', {}).get('coastal_focused', False))
                    high_confidence = sum(1 for item in all_raw_data 
                                        if item.get('validation', {}).get('seriousness_confidence', 0) > 0.8)
                    
                    logging.info(f"  [COASTAL] Coastal-focused content: {coastal_count}")
                    logging.info(f"  [CONFIDENCE] High-confidence serious content: {high_confidence}")
                else:
                    logging.info(f"[EMPTY] Cycle #{cycle_count}: No new disaster content found")
                
                # Wait for next 24-hour cycle
                wait_seconds = check_interval_hours * 3600
                wait_hours = wait_seconds / 3600
                
                logging.info(f"[WAIT] Waiting {wait_hours} hours until next comprehensive collection cycle...")
                logging.info(f"   Next cycle will start at: {(datetime.now() + timedelta(hours=check_interval_hours)).strftime('%Y-%m-%d %H:%M:%S')}")
                
                time.sleep(wait_seconds)
                
            except KeyboardInterrupt:
                logging.info("[STOP] Monitoring stopped by user")
                break
            except Exception as e:
                logging.error(f"[CYCLE ERROR] Error in monitoring cycle #{cycle_count}: {str(e)}")
                logging.info("[RETRY] Waiting 1 hour before retrying...")
                time.sleep(3600)  # Wait 1 hour on error
    
    def collect_recent_data(self, hours_back=48):
        """
        Collect ONLY recent raw data from YouTube for NLP processing with enhanced validations
        Optimized for recent content (max 48 hours) to ensure fresh disaster monitoring
        """
        all_data = []
        processed_count = 0
        skipped_duplicates = 0
        skipped_non_serious = 0
        api_errors = 0
        
        logging.info(f"Starting 24-hour comprehensive data collection (last {hours_back} hours)")
        
        # Use ALL keywords for thorough 24-hour research
        total_keywords = len(DISASTER_KEYWORDS)
        for i, keyword in enumerate(DISASTER_KEYWORDS, 1):
            try:
                logging.info(f"Processing keyword {i}/{total_keywords}: {keyword}")
                
                # Get more videos for thorough 24-hour research
                videos = search_recent_disaster_videos(keyword, max_results=50, hours_back=hours_back)
                
                if not videos:
                    logging.info(f"No videos found for keyword: {keyword}")
                    continue
                
                for video in videos:
                    if not video:  # Skip None results from failed extractions
                        continue
                        
                    video_id = video.get("videoId")
                    if not video_id:
                        logging.warning("Video missing videoId, skipping")
                        continue
                    
                    # 1. Check for duplicates
                    if video_tracker.is_duplicate(video_id):
                        skipped_duplicates += 1
                        logging.debug(f"Skipping duplicate video: {video_id}")
                        continue
                    
                    # 2. Get ALL comments for validation (more for 24-hour research)
                    try:
                        comments = fetch_all_comments(video_id, max_comments=200)
                    except Exception as e:
                        logging.warning(f"Error fetching comments for {video_id}: {str(e)}")
                        comments = []
                    
                    # 3. Get transcription for content validation
                    transcription_data = video.get('transcription', {})
                    transcription_text = transcription_data.get('text_for_analysis', '')
                    
                    # 4. Validate content seriousness
                    try:
                        is_serious, confidence_score, validation_reasons = validate_content_seriousness(
                            video, 
                            comments, 
                            transcription_text
                        )
                    except Exception as e:
                        logging.warning(f"Error in seriousness validation for {video_id}: {str(e)}")
                        # Default to serious if validation fails
                        is_serious, confidence_score, validation_reasons = True, 0.5, ["Validation error - defaulted to serious"]
                    
                    # 5. Filter out non-serious content
                    if not is_serious:
                        skipped_non_serious += 1
                        video_title = video.get('title', 'Unknown')[:50]
                        logging.info(f"Skipping non-serious content: {video_title}... (confidence: {confidence_score:.2f})")
                        logging.debug(f"Validation reasons: {validation_reasons}")
                        continue
                    
                    # 6. Mark as processed to prevent future duplicates
                    video_tracker.mark_processed(video_id)
                    
                    # 7. Aggregate all validated raw data
                    try:
                        complete_data = aggregate_raw_data(video, comments, keyword)
                    except Exception as e:
                        logging.error(f"Error aggregating data for {video_id}: {str(e)}")
                        continue
                    
                    # 8. Add validation metadata
                    complete_data['validation'] = {
                        'is_serious_content': is_serious,
                        'seriousness_confidence': confidence_score,
                        'validation_reasons': validation_reasons,
                        'coastal_focused': any(coastal_term in keyword.lower() for coastal_term in 
                                             ['coastal', 'tsunami', 'storm surge', 'tidal', 'ocean', 'sea']),
                        'transcription_available': transcription_data.get('has_captions', False),
                        'collection_cycle': '24hour_thorough',
                        'search_keyword': keyword,
                        'coastal_focused': any(coastal_term in keyword.lower() for coastal_term in 
                                             ['coastal', 'tsunami', 'storm surge', 'tidal', 'ocean', 'sea']),
                        'transcription_available': transcription_data.get('has_captions', False)
                    }
                    
                    all_data.append(complete_data)
                    processed_count += 1
                    
                    video_title = video.get('title', 'Unknown')[:60]
                    logging.info(f"[COLLECTED] Collected serious disaster content: {video_title}...")
                
                # Rate limiting for API stability (longer for thorough research)
                time.sleep(5)  # 5 seconds between keywords for stability
                
            except Exception as e:
                logging.error(f"Error processing keyword '{keyword}': {str(e)}")
                api_errors += 1
                continue
        
        # Save cache of processed videos after each collection cycle
        video_tracker.save_cache()
        
        # Save collected validated data
        if all_data:
            self.save_data(all_data)
        
        # Comprehensive logging for 24-hour cycle
        logging.info(f"24-Hour comprehensive data collection completed:")
        logging.info(f"  [STATS] Keywords processed: {total_keywords}")
        logging.info(f"  [SUCCESS] Serious videos collected: {processed_count}")
        logging.info(f"  [DUPLICATES] Duplicates skipped: {skipped_duplicates}")
        logging.info(f"  [FILTERED] Non-serious content filtered: {skipped_non_serious}")
        logging.info(f"  [ERRORS] API errors: {api_errors}")
        logging.info(f"  💾 Total videos in cache: {video_tracker.get_stats()['total_processed']}")
        
        # Calculate efficiency metrics
        total_found = processed_count + skipped_duplicates + skipped_non_serious
        if total_found > 0:
            serious_ratio = processed_count / total_found
            duplicate_ratio = skipped_duplicates / total_found
            logging.info(f"  📈 Collection efficiency:")
            logging.info(f"     - Serious content ratio: {serious_ratio:.1%}")
            logging.info(f"     - Duplicate prevention: {duplicate_ratio:.1%}")
        
        return all_data
    
    def save_data(self, data):
        """
        Save collected raw data to JSONL file for NLP processing
        """
        try:
            with open(self.output_file, "a", encoding="utf-8") as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
            logging.info(f"Saved {len(data)} complete data items to {self.output_file}")
        except Exception as e:
            logging.error(f"Error saving data: {str(e)}")


# --- Demo and Testing ---
def run_demo():
    """
    Demo function to test the enhanced system
    """
    print("🌊 YouTube Coastal Disaster Monitoring System - Demo")
    print("=" * 60)
    
    if not youtube:
        print("\n⚠️  YouTube API not available - showing example data structure")
        run_offline_demo()
        return
    
    # Test with recent flood/cyclone keywords
    test_keywords = ["Mumbai flood today", "cyclone India recent", "Chennai rain live"]
    
    for keyword in test_keywords:
        print(f"\n🔍 Searching for: {keyword}")
        videos = search_recent_disaster_videos(keyword, max_results=3, hours_back=72)
        
        if videos:
            for video in videos:
                print(f"\n📺 {video['title']}")
                print(f"   📅 Published: {video['publishedAt']}")
                print(f"   👀 Views: {video['viewCount']}, 💬 Comments: {video['commentCount']}")
                print(f"   🎯 Urgency: {video['urgency_level']}")
                print(f"   📍 Locations: {video['location_mentions']}")
                
                # Get some comments
                comments = fetch_all_comments(video["videoId"], max_comments=5)
                print(f"   💬 Sample comments:")
                for comment in comments[:2]:
                    print(f"      - {comment['author']}: {comment['text'][:100]}...")
                    if comment.get('personal_experience'):
                        print("        ⚡ Contains personal experience")
        else:
            print("   No recent videos found for this keyword")

def run_offline_demo():
    """
    Show example data structure when API is not available
    """
    print("\n📊 Example Data Structure for NLP Processing:")
    print("-" * 50)
    
    example_data = {
        "data_type": "youtube_disaster_content",
        "collected_at": "2025-09-20T13:56:00",
        "search_keyword": "Mumbai flood live",
        "video": {
            "videoId": "abc123xyz",
            "title": "Mumbai Heavy Rain: Severe Waterlogging in Multiple Areas",
            "description": "Live coverage of heavy monsoon rain causing severe flooding...",
            "publishedAt": "2025-09-20T10:30:00Z",
            "channelTitle": "News24 Live",
            "viewCount": 25000,
            "likeCount": 450,
            "commentCount": 89,
            "duration": "PT12M30S",
            "urgency_level": "critical",
            "severity_indicators": {
                "extreme": ["severe"],
                "high": ["heavy"]
            },
            "location_mentions": ["Mumbai", "Maharashtra", "India"],
            "liveBroadcastContent": "live"
        },
        "comments": [
            {
                "type": "main_comment",
                "author": "MumbaiResident92",
                "text": "Water level rising rapidly in our society in Andheri East. Very scary situation here.",
                "publishedAt": "2025-09-20T11:15:00Z",
                "likeCount": 12,
                "personal_experience": True,
                "location_mentions": ["Andheri East"],
                "severity_indicators": {"high": ["rapidly"]},
                "sentiment_indicators": {"negative": ["scary"]}
            },
            {
                "type": "main_comment", 
                "author": "WeatherWatcher",
                "text": "IMD has issued red alert for Mumbai and surrounding areas",
                "publishedAt": "2025-09-20T11:20:00Z",
                "likeCount": 8,
                "personal_experience": False,
                "location_mentions": ["Mumbai"],
                "severity_indicators": {"extreme": ["red alert"]},
                "sentiment_indicators": {"neutral": ["alert"]}
            }
        ],
        "aggregated_insights": {
            "unique_locations": ["Mumbai", "Maharashtra", "Andheri East"],
            "disaster_type": "flood",
            "estimated_severity": "extreme",
            "public_sentiment": "negative", 
            "credibility_score": 8,
            "personal_experience_count": 1,
            "is_live_content": True,
            "content_summary": {
                "primary_location": "Mumbai",
                "disaster_type": "flood",
                "estimated_severity": "extreme",
                "public_sentiment": "negative",
                "credibility_score": 8
            }
        }
    }
    
    print("🎯 Video Information:")
    video = example_data["video"]
    print(f"   Title: {video['title']}")
    print(f"   Urgency: {video['urgency_level']}")
    print(f"   Severity: {video['severity_indicators']}")
    print(f"   Locations: {video['location_mentions']}")
    print(f"   Engagement: {video['viewCount']} views, {video['commentCount']} comments")
    
    print("\n💬 Comments Analysis:")
    for comment in example_data["comments"]:
        print(f"   - {comment['author']}: {comment['text'][:60]}...")
        print(f"     Personal Experience: {comment['personal_experience']}")
        print(f"     Locations: {comment['location_mentions']}")
        print(f"     Sentiment: {list(comment['sentiment_indicators'].keys())}")
    
    print("\n📈 Aggregated Insights for NLP:")
    insights = example_data["aggregated_insights"]
    print(f"   Primary Location: {insights['content_summary']['primary_location']}")
    print(f"   Disaster Type: {insights['content_summary']['disaster_type']}")
    print(f"   Estimated Severity: {insights['content_summary']['estimated_severity']}")
    print(f"   Public Sentiment: {insights['content_summary']['public_sentiment']}")
    print(f"   Credibility Score: {insights['content_summary']['credibility_score']}/10")
    print(f"   Personal Experiences: {insights['personal_experience_count']}")
    print(f"   Live Content: {insights['is_live_content']}")
    
    print("\n💡 This data structure is ready for:")
    print("   ✅ Location extraction and geocoding")
    print("   ✅ Severity classification")
    print("   ✅ Sentiment analysis")
    print("   ✅ Personal account identification")
    print("   ✅ Credibility assessment")
    print("   ✅ Real-time event tracking")
    
    print(f"\n📝 To fix API issues:")
    print("   1. Check your YouTube Data API v3 key")
    print("   2. Ensure API is enabled in Google Cloud Console") 
    print("   3. Verify billing is set up (required for API access)")
    print("   4. Test with: python test_youtube_system.py")


if __name__ == "__main__":
    # Choose demo or continuous monitoring
    mode = input("Choose mode: (1) Demo (2) Continuous Monitoring: ")
    
    if mode == "1":
        run_demo()
    else:
        monitor = DisasterYouTubeMonitor()
        print("Starting continuous monitoring... Press Ctrl+C to stop")
        monitor.monitor_continuously()
