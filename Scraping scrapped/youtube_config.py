import os
from dotenv import load_dotenv

load_dotenv()

# YouTube Disaster Monitoring Configuration

# API Configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY_CONFIG")  # Replace with your key
API_QUOTA_LIMIT = 10000  # Daily quota limit

# Monitoring Configuration
CHECK_INTERVAL_MINUTES = 15  # How often to check for new content
MAX_VIDEOS_PER_KEYWORD = 20  # Limit videos per search to manage quota
MAX_COMMENTS_PER_VIDEO = 100  # Limit comments per video
HOURS_BACK_FOR_RECENT = 6  # How many hours back to search for "recent" content

# Output Configuration
OUTPUT_FILE = "youtube_disaster_data.jsonl"
LOG_FILE = "youtube_disaster_monitor.log"

# Geographic Focus
TARGET_REGION = "IN"  # India
TARGET_LANGUAGES = ["hi", "en"]  # Hindi and English

# Priority Keywords (these will be checked more frequently)
PRIORITY_KEYWORDS = [
    "cyclone India live",
    "flood India breaking",
    "tsunami India alert",
    "storm India emergency",
    "Mumbai flood live",
    "Chennai cyclone now",
    "Kolkata storm today"
]

# Extended Keywords for comprehensive monitoring
EXTENDED_KEYWORDS = [
    # Coastal disasters
    "cyclone India", "tsunami India", "coastal flooding India", "storm surge India",
    "hurricane India", "typhoon India", "coastal erosion India", "sea level rise India",
    
    # Flooding variants
    "flood India", "flooding India", "heavy rain India", "monsoon flooding India",
    "urban flooding India", "flash flood India", "waterlogging India",
    
    # Weather emergencies
    "severe weather India", "storm India", "heavy rainfall India", "weather alert India",
    "IMD warning", "meteorological warning India", "extreme weather India",
    
    # City-specific
    "Mumbai flood", "Chennai flood", "Kolkata cyclone", "Odisha cyclone",
    "Kerala flood", "Gujarat cyclone", "Tamil Nadu cyclone", "West Bengal cyclone",
    "Andhra Pradesh cyclone", "Karnataka flood", "Goa flood",
    
    # Coastal cities
    "Visakhapatnam storm", "Kochi flood", "Mangalore flood", "Puducherry cyclone",
    "Paradip cyclone", "Haldia flood", "Diamond Harbour cyclone",
    
    # Live/urgent content
    "India weather live", "India disaster breaking", "India emergency alert",
    "India storm warning", "India flood rescue"
]

# NLP Processing Configuration
SEVERITY_KEYWORDS = {
    "extreme": ["catastrophic", "devastating", "extreme", "severe", "massive", "unprecedented", "historic"],
    "high": ["heavy", "intense", "major", "serious", "significant", "dangerous", "critical"],
    "medium": ["moderate", "considerable", "notable", "substantial", "strong"],
    "low": ["light", "minor", "slight", "small", "weak"]
}

URGENCY_KEYWORDS = {
    "critical": ["breaking", "urgent", "emergency", "alert", "immediate", "crisis"],
    "high": ["warning", "live", "now", "today", "current", "developing"],
    "medium": ["recent", "latest", "update", "ongoing"],
    "low": ["report", "news", "information"]
}

PERSONAL_EXPERIENCE_INDICATORS = [
    "i am", "we are", "my house", "our area", "i saw", "i experienced", 
    "happening here", "in my city", "my family", "our village", "i live",
    "from my window", "outside my house", "in our locality"
]

# Indian Geographic Regions
INDIAN_REGIONS = [
    # States
    "andhra pradesh", "arunachal pradesh", "assam", "bihar", "chhattisgarh",
    "goa", "gujarat", "haryana", "himachal pradesh", "jharkhand", "karnataka",
    "kerala", "madhya pradesh", "maharashtra", "manipur", "meghalaya",
    "mizoram", "nagaland", "odisha", "punjab", "rajasthan", "sikkim",
    "tamil nadu", "telangana", "tripura", "uttar pradesh", "uttarakhand",
    "west bengal",
    
    # Union Territories
    "delhi", "jammu and kashmir", "ladakh", "puducherry", "chandigarh",
    "dadra and nagar haveli", "daman and diu", "lakshadweep",
    "andaman and nicobar",
    
    # Major Cities
    "mumbai", "delhi", "bangalore", "hyderabad", "ahmedabad", "chennai",
    "kolkata", "surat", "pune", "jaipur", "lucknow", "kanpur", "nagpur",
    "visakhapatnam", "indore", "thane", "bhopal", "pimpri-chinchwad",
    "patna", "vadodara", "ghaziabad", "ludhiana", "agra", "nashik",
    "faridabad", "meerut", "rajkot", "kalyan-dombivali", "vasai-virar",
    
    # Coastal Cities (Priority)
    "mumbai", "chennai", "kolkata", "kochi", "visakhapatnam", "mangalore",
    "puducherry", "goa", "surat", "vadodara", "thiruvananthapuram",
    "kozhikode", "thrissur", "alappuzha", "kannur", "kasaragod",
    
    # Common abbreviations
    "india", "bharat", "hindustan", "imd", "ndma", "ndrf"
]

# Rate Limiting Configuration
REQUESTS_PER_MINUTE = 100  # YouTube API allows 100 requests per 100 seconds
DELAY_BETWEEN_REQUESTS = 1  # Seconds between API calls
RETRY_DELAY = 5  # Seconds to wait before retrying failed requests
MAX_RETRIES = 3  # Maximum number of retries for failed requests

# Content Filtering
MIN_VIDEO_LENGTH = 30  # Minimum video length in seconds
MAX_VIDEO_AGE_HOURS = 168  # Maximum video age in hours (7 days)
EXCLUDE_CHANNELS = [
    # Add channel IDs to exclude (spam/irrelevant channels)
]

# Data Quality Configuration
MIN_CREDIBILITY_SCORE = 3  # Minimum credibility score to include data
MIN_ENGAGEMENT_THRESHOLD = 10  # Minimum views/likes/comments for inclusion
REQUIRE_LOCATION_MATCH = True  # Only include content with location mentions