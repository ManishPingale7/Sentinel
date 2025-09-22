#!/usr/bin/env python3
"""
Social Media Disaster Simulator
===============================

Generates synthetic social media posts for disaster-related analytics.
Simulates Twitter, Facebook, Instagram, News, and YouTube-like posts.

Author: Disaster Analytics Team
Date: September 2025
"""

import json
import random
import uuid
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
import os
import glob

import simulator_data as sd

try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    print("Warning: Redis not installed. Install with: pip install redis")


class DisasterSocialMediaSimulator:
    """
    Comprehensive simulator for generating realistic social media posts
    related to disaster events and normal content.
    """
    
    def __init__(self, media_folder: str = "../Media"):
        """Initialize the simulator with media folder path."""
        self.media_folder = media_folder
        self.media_base_url = "http://localhost/media"
        
        # Redis configuration
        self.redis_client = None
        self.redis_stream_name = "disaster_posts"
        
        # Hazard categories
        self.hazard_categories = sd.HAZARD_CATEGORIES
        
        # Supported languages
        self.languages = sd.LANGUAGES
        
        # Platforms
        self.platforms = sd.PLATFORMS
        
        # Post type distribution
        self.post_type_weights = sd.POST_TYPE_WEIGHTS
        
        # Initialize media categorization
        self.media_files = self._categorize_media_files()
        
        # Initialize content templates by assigning from the data module
        self.locations = sd.LOCATIONS
        self.hazard_templates = sd.HAZARD_TEMPLATES
        self.false_alarm_templates = sd.FALSE_ALARM_TEMPLATES
        self.noise_templates = sd.NOISE_TEMPLATES
        self.youtube_templates = sd.YOUTUBE_TEMPLATES
        self.youtube_comments = sd.YOUTUBE_COMMENTS
        self.news_templates = sd.NEWS_TEMPLATES
        self.twitter_short_templates = sd.TWITTER_SHORT_TEMPLATES
        self.social_casual_templates = sd.SOCIAL_CASUAL_TEMPLATES
        self.code_switching_templates = sd.CODE_SWITCHING_TEMPLATES
        self.author_names = sd.AUTHOR_NAMES
        self.news_orgs = sd.NEWS_ORGS
        self.hashtags = sd.HASHTAGS
        self.emojis = sd.EMOJIS
    
    def setup_redis(self, host: str = "localhost", port: int = 6379, db: int = 0) -> bool:
        """Setup Redis connection for streaming."""
        if not REDIS_AVAILABLE:
            print("Error: Redis library not installed. Please install with: pip install redis")
            return False
        
        try:
            self.redis_client = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            # Test connection
            self.redis_client.ping()
            print(f"✓ Connected to Redis at {host}:{port}")
            return True
        except redis.ConnectionError:
            print(f"✗ Failed to connect to Redis at {host}:{port}")
            print("Make sure Redis server is running.")
            return False
        except Exception as e:
            print(f"✗ Redis setup error: {e}")
            return False
    
    def stream_to_redis(self, post: Dict) -> bool:
        """Stream a post to Redis."""
        if not self.redis_client:
            print("Error: Redis not connected. Call setup_redis() first.")
            return False
        
        try:
            # Add post to Redis stream
            post_id = self.redis_client.xadd(
                self.redis_stream_name,
                {
                    "post_data": json.dumps(post, ensure_ascii=False),
                    "timestamp": post.get("timestamp", datetime.now().isoformat()),
                    "platform": post.get("platform", "unknown"),
                    "post_type": post.get("post_type", "unknown")
                }
            )
            return True
        except Exception as e:
            print(f"Error streaming to Redis: {e}")
            return False
    
    def _get_random_timestamp(self, days_back_max: int = 30) -> str:
        """Generate random timestamp, sometimes old for historical detection."""
        now = datetime.now(timezone.utc)
        
        # 20% chance of old content (more than 30 days old)
        if random.random() < 0.2:
            days_back = random.randint(31, 365)  # 1 month to 1 year old
        else:
            days_back = random.randint(0, days_back_max)  # Recent content
        
        random_time = now - timedelta(
            days=days_back,
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        
        return random_time.isoformat()
    
    def _categorize_media_files(self) -> Dict[str, List[str]]:
        """Categorize media files by hazard type based on filename prefixes."""
        media_categories = {category.lower(): [] for category in self.hazard_categories}
        
        # Handle special mapping for typhoon -> cyclone
        media_categories['typhoon'] = []
        
        try:
            media_path = os.path.join(os.path.dirname(__file__), self.media_folder)
            if not os.path.exists(media_path):
                media_path = self.media_folder
                
            for file_pattern in ['*.jpg', '*.jpeg', '*.png', '*.jfif']:
                for file_path in glob.glob(os.path.join(media_path, file_pattern)):
                    filename = os.path.basename(file_path).lower()
                    
                    # Categorize based on filename prefix
                    for category in media_categories.keys():
                        if filename.startswith(category):
                            media_categories[category].append(os.path.basename(file_path))
                            break
                    else:
                        # If no category matches, check for coastal_flooding
                        if filename.startswith('coastal'):
                            media_categories['coastal_flooding'].append(os.path.basename(file_path))
                        
        except Exception as e:
            print(f"Warning: Could not load media files: {e}")
            # Fallback with sample files
            media_categories = {
                'coastal_flooding': ['Coastal_flooding1.jpg', 'Coastal_flooding2.jpg'],
                'cyclone': ['cyclone1.png', 'cyclone2.jpg'],
                'rain': ['rain1.png', 'rain2.png'],
                'tsunami': ['tsunami1.jpg', 'tsunami2.jpg'],
                'erosion': ['erosion1.jpg'],
                'non': ['non1.png', 'non2.jpg'],
                'typhoon': ['typhoon1.jpg', 'typhoon2.png']
            }
        
        return media_categories

    def _generate_realistic_channel_name(self, language: str = 'English') -> str:
        """Generate realistic YouTube channel names."""
        
        channels = sd.YOUTUBE_CHANNEL_PATTERNS.get(language, sd.YOUTUBE_CHANNEL_PATTERNS['English'])
        base_name = random.choice(channels)
        
        # Sometimes add numbers or year
        if random.random() > 0.7:
            base_name += f" {random.randint(1, 99)}"
        
        return f"@{base_name.replace(' ', '')}"

    def _select_random_media(self, hazard_category: str, force_mismatch: bool = False) -> Optional[str]:
        """Select random media file, optionally forcing a mismatch."""
        if not self.media_files:
            return None
            
        if force_mismatch:
            # Deliberately pick wrong category
            wrong_categories = [cat for cat in self.media_files.keys() 
                              if cat != hazard_category.lower() and self.media_files[cat]]
            if wrong_categories:
                category = random.choice(wrong_categories)
                filename = random.choice(self.media_files[category])
                return f"{self.media_base_url}/{filename}"
        
        # Normal selection
        category_key = hazard_category.lower()
        if category_key == 'coastal_flooding':
            category_key = 'coastal_flooding'
        elif category_key not in self.media_files:
            return None
            
        if self.media_files[category_key]:
            filename = random.choice(self.media_files[category_key])
            return f"{self.media_base_url}/{filename}"
        
        return None
    
    def _generate_realistic_username(self, language: str = 'English') -> str:
        """Generate realistic Indian usernames based on language/region."""
        
        names = sd.USER_NAMES_BY_LANGUAGE.get(language, sd.USER_NAMES_BY_LANGUAGE['English'])
        first_name = random.choice(names['first'])
        last_name = random.choice(names['last'])
        
        pattern = random.choice(sd.USERNAME_PATTERNS)
        username = pattern.format(
            first=first_name,
            last=last_name,
            year=random.choice(sd.BIRTH_YEARS),
            num=random.randint(1, 999),
            city=random.choice(sd.CITIES)
        )
        
        import re
        if re.search(r'[^\x00-\x7F]', username):
            english_names = sd.USER_NAMES_BY_LANGUAGE['English']
            first_name = random.choice(english_names['first'])
            last_name = random.choice(english_names['last'])
            username = pattern.format(
                first=first_name,
                last=last_name,
                year=random.choice(sd.BIRTH_YEARS),
                num=random.randint(1, 999),
                city=random.choice(sd.CITIES)
            )
        
        username = username.lower().replace(' ', '_')
        
        return f"@{username}"
    
    def _generate_user_context(self, language: str = 'English', post_type: str = 'hazard') -> Dict:
        """Generate user context including follower count, verified status, bio, profile pic."""
        
        is_official = random.random() < 0.15
        
        if is_official:
            account_type = random.choice(['news', 'government', 'organization', 'journalist'])
            
            if account_type == 'news':
                follower_count = random.randint(10000, 2000000)
                verified = random.random() < 0.8
                bio_templates = sd.BIO_TEMPLATES['news']
            elif account_type == 'government':
                follower_count = random.randint(50000, 1000000)
                verified = random.random() < 0.9
                bio_templates = sd.BIO_TEMPLATES['government']
            elif account_type == 'journalist':
                follower_count = random.randint(5000, 500000)
                verified = random.random() < 0.4
                bio_templates = sd.BIO_TEMPLATES['journalist']
            else:  # organization
                follower_count = random.randint(1000, 100000)
                verified = random.random() < 0.6
                bio_templates = sd.BIO_TEMPLATES['organization']
        else:
            follower_count = random.randint(50, 5000)
            verified = random.random() < 0.01
            bio_templates = sd.BIO_TEMPLATES['personal']
            account_type = 'personal'

        available_bios = bio_templates.get(language, bio_templates['English'])
        bio = random.choice(available_bios)
        
        profile_pic = f"https://api.dicebear.com/7.x/avataaars/svg?seed={random.randint(1000, 9999)}"
        
        return {
            'follower_count': follower_count,
            'verified': verified,
            'bio': bio,
            'profile_pic': profile_pic,
            'account_type': 'official' if is_official else 'personal'
        }
    
    def _calculate_engagement_metrics(self, post_type: str, user_context: Dict, platform: str) -> Dict:
        """Calculate realistic engagement metrics based on user context and post type."""
        follower_count = user_context['follower_count']
        is_verified = user_context['verified']
        account_type = user_context['account_type']
        
        rates = sd.BASE_ENGAGEMENT_RATES.get(platform, sd.BASE_ENGAGEMENT_RATES['twitter'])
        
        total_multiplier = (
            sd.TYPE_MULTIPLIERS.get(post_type, 1.0) * 
            sd.ACCOUNT_MULTIPLIERS.get(account_type, 1.0) * 
            (1.8 if is_verified else 1.0)
        )
        
        randomness = random.uniform(0.5, 1.5)
        total_multiplier *= randomness
        
        engagement = {}
        for metric, rate in rates.items():
            base_count = int(follower_count * rate * total_multiplier)
            
            if metric in ['likes', 'views']:
                engagement[metric] = max(random.randint(base_count // 2, base_count * 2), 1)
            else:
                engagement[metric] = max(random.randint(base_count // 3, base_count), 0)
        
        if platform == 'twitter':
            if 'retweets' in engagement and 'likes' in engagement:
                engagement['retweets'] = min(engagement['retweets'], engagement['likes'] // 2)
                
        elif platform == 'facebook':
            total_likes = engagement.get('likes', 0)
            engagement['reactions'] = {
                'like': random.randint(total_likes // 2, total_likes),
                'love': random.randint(0, total_likes // 5),
                'care': random.randint(0, total_likes // 8) if post_type == 'hazard' else 0,
                'angry': random.randint(0, total_likes // 10) if post_type == 'false_alarm' else 0,
                'sad': random.randint(0, total_likes // 4) if post_type == 'hazard' else 0,
                'wow': random.randint(0, total_likes // 6) if post_type in ['hazard', 'false_alarm'] else 0
            }
            
        elif platform == 'youtube':
            if 'likes' in engagement and 'dislikes' in engagement:
                max_dislikes = engagement['likes'] // 10
                engagement['dislikes'] = min(engagement['dislikes'], max_dislikes)
                
            view_multiplier = random.randint(20, 200)
            engagement['views'] = engagement.get('likes', 10) * view_multiplier
        
        return engagement
    
    def _generate_hashtags(self, hazard_category: str, platform: str = 'twitter', count: int = None) -> List[str]:
        """Generate hashtags for a post based on platform-specific patterns."""
        if count is None:
            min_count, max_count = sd.PLATFORM_HASHTAG_COUNTS.get(platform, (1, 3))
            count = random.randint(min_count, max_count)
        
        available_tags = self.hashtags.get(hazard_category, self.hashtags['Non']).copy()
        
        if platform == 'instagram':
            available_tags.extend(sd.INSTAGRAM_EXTRA_HASHTAGS)
        elif platform == 'twitter':
            available_tags.extend(sd.TWITTER_EXTRA_HASHTAGS)
        elif platform == 'youtube':
            available_tags.extend(sd.YOUTUBE_EXTRA_HASHTAGS)
        
        count = min(count, len(available_tags))
        return random.sample(available_tags, count) if count > 0 else []
    
    def _add_emojis(self, text: str, hazard_category: str) -> str:
        """Add random emojis to text."""
        if random.random() > 0.7:
            available_emojis = self.emojis.get(hazard_category, self.emojis['Non'])
            emoji = random.choice(available_emojis)
            if random.random() > 0.5:
                return f"{emoji} {text}"
            else:
                return f"{text} {emoji}"
        return text
    
    def _get_code_switching_content(self, post_type: str, hazard_category: str = None) -> Tuple[str, str]:
        """Generate code-switching content with mixed languages."""
        if post_type == 'noise' or random.random() > 0.3:
            return None, None
            
        lang_pairs = list(self.code_switching_templates.keys())
        chosen_pair = random.choice(lang_pairs)
        
        if post_type in ['hazard', 'false_alarm']:
            if hazard_category is None:
                hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
            
            templates = self.code_switching_templates[chosen_pair]
            template = random.choice(templates)
            
            if 'Hindi' in chosen_pair:
                location = random.choice(self.locations.get('Hindi', self.locations['English']))
            elif 'Marathi' in chosen_pair:
                location = random.choice(self.locations.get('Marathi', self.locations['English']))
            elif 'Tamil' in chosen_pair:
                location = random.choice(self.locations.get('Tamil', self.locations['English']))
            elif 'Telugu' in chosen_pair:
                location = random.choice(self.locations.get('Telugu', self.locations['English']))
            else:
                location = random.choice(self.locations['English'])
            
            hazard_name = hazard_category.lower().replace('_', ' ')
            text = template.format(location=location, hazard=hazard_name)
            
            return text, chosen_pair
        
        return None, None
    
    def generate_hazard_post(self, hazard_category: str) -> Dict:
        """Generate a hazard-type post (legacy method for compatibility)."""
        platform = random.choice(self.platforms)
        return self._generate_platform_specific_post(platform, 'hazard', hazard_category)
    
    def generate_false_alarm_post(self) -> Dict:
        """Generate a false alarm post (legacy method for compatibility)."""
        platform = random.choice(self.platforms)
        hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
        return self._generate_platform_specific_post(platform, 'false_alarm', hazard_category)
    
    def generate_noise_post(self) -> Dict:
        """Generate a noise/joke post (legacy method for compatibility)."""
        platform = random.choice(self.platforms)
        return self._generate_platform_specific_post(platform, 'noise', 'Non')
    
    def _generate_random_date(self, old_content: bool = False) -> str:
        """Generate random timestamp, sometimes in the past for old content detection."""
        if old_content:
            days_ago = random.randint(1, 365)
            timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago)
        else:
            days_ago = random.randint(0, 7)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            timestamp = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago, minutes=minutes_ago)
        
        return timestamp.isoformat()

    def generate_twitter_post(self, post_type: str = None, hazard_category: str = None) -> Dict:
        """Generate a Twitter-style post with short text and optional image."""
        if post_type is None:
            post_type = random.choices(
                list(self.post_type_weights.keys()),
                weights=list(self.post_type_weights.values())
            )[0]
        
        code_switching_text, lang_pair = self._get_code_switching_content(post_type, hazard_category)
        
        if code_switching_text:
            text = code_switching_text
            language = lang_pair
        else:
            language = random.choice(self.languages)
            
            if post_type == 'hazard':
                if hazard_category is None:
                    hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
                
                if random.random() < 0.4:
                    templates = self.twitter_short_templates.get(language, self.twitter_short_templates['English'])
                    template = random.choice(templates)
                    location = random.choice(self.locations.get(language, self.locations['English']))
                    hazard_name = hazard_category.lower().replace('_', ' ')
                    text_template = random.choice(templates)
                    text = text_template.replace('{location}', location).replace('{hazard}', hazard_name)
                else:
                    templates = self.hazard_templates.get(hazard_category, {}).get(language, 
                              self.hazard_templates.get(hazard_category, {}).get('English', []))
                    template = random.choice(templates) if templates else f"Alert: {hazard_category} warning!"
                    location = random.choice(self.locations.get(language, self.locations['English']))
                    text = template.replace('{location}', location)
                    
            elif post_type == 'false_alarm':
                templates = self.false_alarm_templates.get(language, self.false_alarm_templates['English'])
                template = random.choice(templates)
                location = random.choice(self.locations.get(language, self.locations['English']))
                hazard_name = hazard_category.lower().replace('_', ' ') if hazard_category else 'disaster'
                text = template.replace('{location}', location).replace('{hazard_type}', hazard_name)
            else:
                templates = self.noise_templates.get(language, self.noise_templates['English'])
                text = random.choice(templates)
        
        # Keep tweets short (under 280 chars)
        if len(text) > 250:
            text = text[:247] + "..."
        
        # Add emojis only if not already present in code-switching
        if not code_switching_text:
            text = self._add_emojis(text, hazard_category or 'Non')
        
        # Generate user context
        user_context = self._generate_user_context(language if not code_switching_text else 'English', post_type)
        
        # Calculate realistic engagement metrics
        engagement = self._calculate_engagement_metrics(post_type, user_context, 'twitter')
        
        # Determine if old content
        is_old = random.random() < 0.1  # 10% chance of old content
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': self._generate_random_date(is_old),
            'platform': 'twitter',
            'user': self._generate_realistic_username(language if not code_switching_text else 'English'),
            'lang': language,
            'text': text,
            'hashtags': self._generate_hashtags(hazard_category or 'Non', 'twitter'),
            'image_url': self._select_random_media(hazard_category or 'Non', post_type == 'false_alarm'),
            'retweets': engagement.get('retweets', 0),
            'likes': engagement.get('likes', 1),
            'replies': engagement.get('replies', 0),
            'verified': user_context['verified'],
            'follower_count': user_context['follower_count'],
            'bio': user_context['bio'],
            'profile_pic': user_context['profile_pic'],
            'account_type': user_context['account_type']
        }

    def generate_youtube_post(self, post_type: str = None, hazard_category: str = None) -> Dict:
        """Generate a YouTube video post with comments and video-specific metadata."""
        if post_type is None:
            post_type = random.choices(
                list(self.post_type_weights.keys()),
                weights=list(self.post_type_weights.values())
            )[0]
        
        language = random.choice(self.languages)
        
        if post_type == 'hazard':
            if hazard_category is None:
                hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
            templates = self.hazard_templates.get(hazard_category, {}).get(language, 
                      self.hazard_templates.get(hazard_category, {}).get('English', []))
            title_template = random.choice(templates) if templates else f"Alert: {hazard_category} warning!"
            location = random.choice(self.locations.get(language, self.locations['English']))
            title = title_template.replace('{location}', location)
            
            descriptions = sd.YOUTUBE_DESCRIPTIONS['hazard']
            description = descriptions.get(language, descriptions['English']).replace('{hazard_category}', hazard_category.lower()).replace('{location}', location)
        elif post_type == 'false_alarm':
            location = random.choice(self.locations.get(language, self.locations['English']))
            titles = sd.YOUTUBE_TITLES['false_alarm']
            title_template = random.choice(titles.get(language, titles['English']))
            title = title_template.replace('{hazard_category}', hazard_category or 'disaster').replace('{location}', location)
            description = "Debunking false information. Always verify from official sources."
        else:
            titles = sd.YOUTUBE_TITLES['noise']
            title_list = titles.get(language, titles['English'])
            title = random.choice(title_list)
            description = "Comedy content for entertainment purposes only. Like and subscribe!"
        
        comments = []
        comment_count = random.randint(5, 50)
        available_comments = sd.YOUTUBE_COMMENTS_TEMPLATES.get(language, sd.YOUTUBE_COMMENTS_TEMPLATES['English'])
        
        for _ in range(min(comment_count, 10)):
            comments.append({
                'user': self._generate_realistic_username(language),
                'text': random.choice(available_comments),
                'likes': random.randint(0, 100),
                'timestamp': self._generate_random_date()
            })
        
        channel_context = self._generate_user_context(language, post_type)
        engagement = self._calculate_engagement_metrics(post_type, channel_context, 'youtube')
        is_old = random.random() < 0.15
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': self._generate_random_date(is_old),
            'platform': 'youtube',
            'channel': self._generate_realistic_channel_name(language),
            'lang': language,
            'title': title,
            'description': description,
            'hashtags': self._generate_hashtags(hazard_category or 'Non', 'youtube'),
            'thumbnail_url': self._select_random_media(hazard_category or 'Non', post_type == 'false_alarm'),
            'views': engagement.get('views', 100),
            'likes': engagement.get('likes', 1),
            'dislikes': engagement.get('dislikes', 0),
            'comments': comments,
            'duration': f"{random.randint(1, 10)}:{random.randint(10, 59)}",
            'subscribers': channel_context['follower_count'],
            'verified': channel_context['verified'],
            'channel_description': channel_context['bio'],
            'channel_type': channel_context['account_type']
        }

    def generate_news_post(self, post_type: str = None, hazard_category: str = None) -> Dict:
        """Generate a news article with full content, author, and publication details."""
        if post_type is None:
            post_type = random.choices(
                list(self.post_type_weights.keys()),
                weights=list(self.post_type_weights.values())
            )[0]
        
        language = random.choice(self.languages)
        location = random.choice(self.locations.get(language, self.locations['English']))
        
        if post_type == 'hazard':
            if hazard_category is None:
                hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
            
            headlines = sd.NEWS_HEADLINE_TEMPLATES['hazard'].get(language, sd.NEWS_HEADLINE_TEMPLATES['hazard']['English'])
            headline_template = random.choice(headlines)
            headline = headline_template.replace('{hazard_category}', hazard_category.replace('_', ' ').title()).replace('{location}', location)
            
            article_content = sd.NEWS_ARTICLE_CONTENT['hazard']
            content = article_content.get(language, article_content['English']).replace(
                '{location}', location 
            ).replace(
                '{hazard_category}', hazard_category.lower().replace('_', ' ')
            )
            
        elif post_type == 'false_alarm':
            headlines = sd.NEWS_HEADLINE_TEMPLATES['false_alarm']
            headline_template = random.choice(headlines.get(language, headlines['English']))
            headline = headline_template.replace('{hazard_category}', hazard_category or 'Disaster').replace('{location}', location)
            content = f"Officials clarify that recent social media reports about {hazard_category or 'disaster'} threat to {location} are unfounded. Citizens are advised to rely only on official sources."
            
        else:
            headlines = sd.NEWS_HEADLINE_TEMPLATES['noise']
            headline_template = random.choice(headlines.get(language, headlines['English']))
            headline = headline_template.replace('{location}', location)
            content = "A local resident's humorous social media post about weather conditions has gone viral, bringing some light moments during serious times."
        
        is_old = random.random() < 0.2
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': self._generate_random_date(is_old),
            'platform': 'news',
            'outlet': random.choice(sd.NEWS_OUTLETS.get(language, sd.NEWS_OUTLETS['English'])),
            'author': random.choice(sd.NEWS_AUTHORS.get(language, sd.NEWS_AUTHORS['English'])),
            'reporter': random.choice(sd.NEWS_AUTHORS.get(language, sd.NEWS_AUTHORS['English'])),
            'lang': language,
            'headline': headline,
            'article_content': content,
            'location': location,
            'hashtags': self._generate_hashtags(hazard_category or 'Non', 'news'),
            'images': [self._select_random_media(hazard_category or 'Non', post_type == 'false_alarm')] if random.random() > 0.3 else [],
            'category': 'breaking' if post_type == 'hazard' else 'general',
            'reading_time': f"{random.randint(2, 8)} min read",
            'shares': random.randint(10, 1000)
        }

    def generate_instagram_post(self, post_type: str = None, hazard_category: str = None) -> Dict:
        """Generate an Instagram post with casual content and multiple images."""
        if post_type is None:
            post_type = random.choices(
                list(self.post_type_weights.keys()),
                weights=list(self.post_type_weights.values())
            )[0]
        
        language = random.choice(self.languages)
        location = random.choice(self.locations.get(language, self.locations['English']))
        
        if post_type == 'hazard':
            if hazard_category is None:
                hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
            
            captions = sd.INSTAGRAM_CAPTIONS['hazard']
            caption_template = random.choice(captions.get(language, captions['English']))
            caption = caption_template.replace(
                '{hazard_category}', hazard_category.lower().replace('_', ' ')
            ).replace('{location}', location)
            
        elif post_type == 'false_alarm':
            captions = sd.INSTAGRAM_CAPTIONS['false_alarm']
            caption_template = random.choice(captions.get(language, captions['English']))
            caption = caption_template.replace(
                '{hazard_category}', hazard_category or 'disaster'
            ).replace('{location}', location)
            
        else:
            captions = sd.INSTAGRAM_CAPTIONS['noise']
            caption_templates = captions.get(language, captions['English'])
            caption = random.choice(caption_templates).replace('{location}', location)
        
        images = []
        image_count = random.randint(1, 3)
        for _ in range(image_count):
            img = self._select_random_media(hazard_category or 'Non', post_type == 'false_alarm')
            if img:
                images.append(img)
        
        user_context = self._generate_user_context(language, post_type)
        engagement = self._calculate_engagement_metrics(post_type, user_context, 'instagram')
        is_old = random.random() < 0.12
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': self._generate_random_date(is_old),
            'platform': 'instagram',
            'user': self._generate_realistic_username(language),
            'lang': language,
            'caption': caption,
            'hashtags': self._generate_hashtags(hazard_category or 'Non', 'instagram'),
            'images': images,
            'likes': engagement.get('likes', 1),
            'comments': engagement.get('comments', 0),
            'saves': engagement.get('saves', 0),
            'story_highlights': random.random() > 0.8,
            'verified': user_context['verified'],
            'follower_count': user_context['follower_count'],
            'bio': user_context['bio'],
            'profile_pic': user_context['profile_pic'],
            'account_type': user_context['account_type']
        }

    def generate_facebook_post(self, post_type: str = None, hazard_category: str = None) -> Dict:
        """Generate a Facebook post with longer content and community feel."""
        if post_type is None:
            post_type = random.choices(
                list(self.post_type_weights.keys()),
                weights=list(self.post_type_weights.values())
            )[0]
        
        language = random.choice(self.languages)
        location = random.choice(self.locations.get(language, self.locations['English']))
        
        if post_type == 'hazard':
            if hazard_category is None:
                hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
            
            posts = sd.FACEBOOK_POSTS['hazard']
            content_template = random.choice(posts.get(language, posts['English']))
            content = content_template.replace('{hazard_category}', hazard_category.replace('_', ' ').title()).replace('{location}', location)
            
        elif post_type == 'false_alarm':
            posts = sd.FACEBOOK_POSTS['false_alarm']
            content_template = random.choice(posts.get(language, posts['English']))
            content = content_template.replace('{hazard_category}', hazard_category or 'disaster').replace('{location}', location)
            
        else:
            posts = sd.FACEBOOK_POSTS['noise']
            content_template = random.choice(posts.get(language, posts['English']))
            content = content_template.replace('{location}', location)
        
        images = []
        if random.random() > 0.4:
            image_count = random.randint(1, 4)
            for _ in range(image_count):
                img = self._select_random_media(hazard_category or 'Non', post_type == 'false_alarm')
                if img:
                    images.append(img)
        
        user_context = self._generate_user_context(language, post_type)
        engagement = self._calculate_engagement_metrics(post_type, user_context, 'facebook')
        is_old = random.random() < 0.08
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': self._generate_random_date(is_old),
            'platform': 'facebook',
            'user': self._generate_realistic_username(language),
            'lang': language,
            'content': content,
            'hashtags': self._generate_hashtags(hazard_category or 'Non', 'facebook'),
            'images': images,
            'likes': engagement.get('likes', 1),
            'comments': engagement.get('comments', 0),
            'shares': engagement.get('shares', 0),
            'reactions': engagement.get('reactions', {'like': 1}),
            'verified': user_context['verified'],
            'follower_count': user_context['follower_count'],
            'bio': user_context['bio'],
            'profile_pic': user_context['profile_pic'],
            'account_type': user_context['account_type']
        }

    def generate_post(self) -> Dict:
        """Generate a single social media post based on platform and type distribution."""
        platform = random.choice(self.platforms)
        
        post_type = random.choices(
            list(self.post_type_weights.keys()),
            weights=list(self.post_type_weights.values())
        )[0]
        
        hazard_category = None
        if post_type in ['hazard', 'false_alarm']:
            hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
        
        if platform == 'twitter':
            return self.generate_twitter_post(post_type, hazard_category)
        elif platform == 'youtube':
            return self.generate_youtube_post(post_type, hazard_category)
        elif platform == 'news':
            return self.generate_news_post(post_type, hazard_category)
        elif platform == 'instagram':
            return self.generate_instagram_post(post_type, hazard_category)
        elif platform == 'facebook':
            return self.generate_facebook_post(post_type, hazard_category)
        else:
            return self.generate_twitter_post(post_type, hazard_category)
    
    def _generate_platform_specific_post(self, platform: str, post_type: str, hazard_category: str) -> Dict:
        """Generate platform-specific content."""
        language = random.choice(self.languages)
        timestamp = self._get_random_timestamp()
        
        if platform == 'youtube':
            return self._generate_youtube_post_legacy(post_type, hazard_category, language, timestamp)
        elif platform == 'news':
            return self._generate_news_post_legacy(post_type, hazard_category, language, timestamp)
        elif platform == 'twitter':
            return self._generate_twitter_post_legacy(post_type, hazard_category, language, timestamp)
        elif platform in ['facebook', 'instagram']:
            return self._generate_social_post_legacy(platform, post_type, hazard_category, language, timestamp)
        else:
            return self.generate_hazard_post(hazard_category)
    
    def _generate_youtube_post_legacy(self, post_type: str, hazard_category: str, language: str, timestamp: str) -> Dict:
        """Generate YouTube video post with comments."""
        location = random.choice(self.locations.get(language, self.locations['English']))
        
        if post_type == 'hazard':
            templates = self.youtube_templates['hazard'].get(language, self.youtube_templates['hazard']['English'])
            title = random.choice(templates['titles']).format(hazard=hazard_category, location=location)
            description = random.choice(templates['descriptions']).format(hazard=hazard_category, location=location)
        else:
            templates = self.youtube_templates['noise']['English']
            title = random.choice(templates['titles']).format(location=location)
            description = random.choice(templates['descriptions'])
        
        comment_lang = language if language in self.youtube_comments else 'English'
        num_comments = random.randint(3, 8)
        comments = random.sample(self.youtube_comments[comment_lang], 
                                min(num_comments, len(self.youtube_comments[comment_lang])))
        
        image_url = None
        if random.random() > 0.2:
            if post_type == 'false_alarm':
                image_url = self._select_random_media(hazard_category, force_mismatch=True)
            else:
                image_url = self._select_random_media(hazard_category)
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': timestamp,
            'platform': 'youtube',
            'user': self._generate_realistic_username(language),
            'lang': language,
            'title': title,
            'description': description,
            'text': f"{title}\n\n{description}",
            'hashtags': self._generate_hashtags(hazard_category),
            'image_url': image_url,
            'video_url': f"http://localhost/videos/video_{random.randint(1000, 9999)}.mp4",
            'comments': comments,
            'views': random.randint(100, 50000),
            'likes': random.randint(10, 5000),
            'channel': f"Channel{random.randint(100, 999)}",
            'post_type': post_type
        }
    
    def _generate_news_post_legacy(self, post_type: str, hazard_category: str, language: str, timestamp: str) -> Dict:
        """Generate comprehensive news article."""
        location = random.choice(self.locations.get(language, self.locations['English']))
        author = random.choice(self.author_names.get(language, self.author_names['English']))
        organization = random.choice(self.news_orgs)
        
        if post_type == 'hazard':
            templates = self.news_templates['hazard'].get(language, self.news_templates['hazard']['English'])
            headline = random.choice(templates['headlines']).format(hazard=hazard_category, location=location)
            
            article_template = random.choice(templates['articles'])
            article = article_template.format(
                hazard=hazard_category,
                location=location,
                date=timestamp[:10],
                number=random.randint(500, 5000),
                official_name=random.choice(self.author_names.get(language, self.author_names['English']))
            )
        else:
            headline = f"Clarification: No {hazard_category} threat in {location}, authorities confirm"
            article = f"{location} - Local authorities have clarified that recent social media reports about {hazard_category} in the region are unfounded. District officials urge residents not to panic and rely only on official sources for emergency information."
        
        images = []
        num_images = random.randint(1, 3)
        for _ in range(num_images):
            if post_type == 'false_alarm' and random.random() > 0.5:
                img = self._select_random_media(hazard_category, force_mismatch=True)
            else:
                img = self._select_random_media(hazard_category)
            if img:
                images.append(img)
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': timestamp,
            'platform': 'news',
            'user': author,
            'lang': language,
            'headline': headline,
            'text': article,
            'author': author,
            'reporter': author,
            'organization': organization,
            'location': location,
            'images': images,
            'image_url': images[0] if images else None,
            'hashtags': self._generate_hashtags(hazard_category),
            'word_count': len(article.split()),
            'post_type': post_type
        }
    
    def _generate_twitter_post_legacy(self, post_type: str, hazard_category: str, language: str, timestamp: str) -> Dict:
        """Generate short Twitter post."""
        location = random.choice(self.locations.get(language, self.locations['English']))
        
        if post_type == 'hazard':
            templates = self.twitter_short_templates.get(language, self.twitter_short_templates['English'])
            text = random.choice(templates).format(hazard=hazard_category, location=location)
        elif post_type == 'false_alarm':
            templates = self.false_alarm_templates.get(language, self.false_alarm_templates['English'])
            text = random.choice(templates).format(location=location)
        else:
            templates = self.noise_templates.get(language, self.noise_templates['English'])
            text = random.choice(templates)
        
        text = text[:280]
        
        image_url = None
        if random.random() > 0.6:
            if post_type == 'false_alarm':
                image_url = self._select_random_media(hazard_category, force_mismatch=True)
            else:
                image_url = self._select_random_media(hazard_category)
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': timestamp,
            'platform': 'twitter',
            'user': self._generate_realistic_username(language),
            'lang': language,
            'text': text,
            'hashtags': self._generate_hashtags(hazard_category),
            'image_url': image_url,
            'retweets': random.randint(0, 1000),
            'likes': random.randint(0, 5000),
            'replies': random.randint(0, 100),
            'post_type': post_type
        }
    
    def _generate_social_post_legacy(self, platform: str, post_type: str, hazard_category: str, language: str, timestamp: str) -> Dict:
        """Generate Facebook/Instagram casual post."""
        location = random.choice(self.locations.get(language, self.locations['English']))
        
        if post_type == 'hazard':
            templates = self.social_casual_templates.get(language, self.social_casual_templates['English'])
            text = random.choice(templates).format(hazard=hazard_category, location=location)
        elif post_type == 'false_alarm':
            text = f"Guys turns out that {hazard_category} news about {location} was fake 😅 my aunt shared it without checking... always verify before sharing! #FactCheck"
        else:
            templates = self.noise_templates.get(language, self.noise_templates['English'])
            text = random.choice(templates)
        
        images = []
        num_images = random.randint(0, 3)
        for _ in range(num_images):
            if post_type == 'false_alarm' and random.random() > 0.5:
                img = self._select_random_media(hazard_category, force_mismatch=True)
            else:
                img = self._select_random_media(hazard_category)
            if img:
                images.append(img)
        
        return {
            'id': str(uuid.uuid4()),
            'timestamp': timestamp,
            'platform': platform,
            'user': self._generate_realistic_username(language),
            'lang': language,
            'text': text,
            'hashtags': self._generate_hashtags(hazard_category),
            'images': images,
            'image_url': images[0] if images else None,
            'likes': random.randint(0, 500),
            'comments': random.randint(0, 50),
            'shares': random.randint(0, 100),
            'post_type': post_type
        }
    
    def generate_stream(self, count: int = 10, delay: float = 1.0) -> None:
        """Generate a stream of posts with delays."""
        print("Starting social media disaster simulation...")
        print(f"Generating {count} posts with {delay}s delay between posts")
        print("-" * 60)
        
        for i in range(count):
            post = self.generate_post()
            print(f"Post {i+1}/{count}:")
            print(json.dumps(post, indent=2, ensure_ascii=False))
            print("-" * 60)
            
            if i < count - 1:
                time.sleep(delay)
    
    def generate_stream_periodic(self, schedule: Dict[str, float]):
        """Generate a stream of posts based on a periodic schedule."""
        print("Starting periodic social media disaster simulation...")
        print("Schedule (seconds):", schedule)
        print("Press Ctrl+C to stop.")
        print("-" * 60)
        
        last_post_time = {platform: time.time() - interval for platform, interval in schedule.items()}
        
        try:
            while True:
                now = time.time()
                for platform, interval in schedule.items():
                    if now - last_post_time.get(platform, 0) >= interval:
                        print(f"Generating '{platform}' post...")
                        
                        post_type = random.choices(
                            list(self.post_type_weights.keys()),
                            weights=list(self.post_type_weights.values())
                        )[0]
                        
                        hazard_category = None
                        if post_type in ['hazard', 'false_alarm']:
                            hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
                        
                        if platform == 'twitter':
                            post = self.generate_twitter_post(post_type, hazard_category)
                        elif platform == 'youtube':
                            post = self.generate_youtube_post(post_type, hazard_category)
                        elif platform == 'news':
                            post = self.generate_news_post(post_type, hazard_category)
                        elif platform == 'instagram':
                            post = self.generate_instagram_post(post_type, hazard_category)
                        elif platform == 'facebook':
                            post = self.generate_facebook_post(post_type, hazard_category)
                        else:
                            # Fallback to twitter post if platform is unknown
                            post = self.generate_twitter_post(post_type, hazard_category)

                        print(json.dumps(post, indent=2, ensure_ascii=False))
                        print("-" * 60)
                        
                        last_post_time[platform] = now
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print("\nSimulation stopped by user.")
    
    def generate_stream_periodic_redis(self, schedule: Dict[str, float], redis_host: str = "localhost", redis_port: int = 6379):
        """Generate a stream of posts based on a periodic schedule and stream to Redis."""
        if not self.setup_redis(redis_host, redis_port):
            print("Failed to setup Redis. Exiting...")
            return
        
        print("Starting periodic social media disaster simulation with Redis streaming...")
        print(f"Redis stream: {self.redis_stream_name}")
        print("Schedule (seconds):", schedule)
        print("Press Ctrl+C to stop.")
        print("-" * 60)
        
        last_post_time = {platform: time.time() - interval for platform, interval in schedule.items()}
        posts_streamed = 0
        
        try:
            while True:
                now = time.time()
                for platform, interval in schedule.items():
                    if now - last_post_time.get(platform, 0) >= interval:
                        print(f"Generating '{platform}' post...")
                        
                        post_type = random.choices(
                            list(self.post_type_weights.keys()),
                            weights=list(self.post_type_weights.values())
                        )[0]
                        
                        hazard_category = None
                        if post_type in ['hazard', 'false_alarm']:
                            hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
                        
                        if platform == 'twitter':
                            post = self.generate_twitter_post(post_type, hazard_category)
                        elif platform == 'youtube':
                            post = self.generate_youtube_post(post_type, hazard_category)
                        elif platform == 'news':
                            post = self.generate_news_post(post_type, hazard_category)
                        elif platform == 'instagram':
                            post = self.generate_instagram_post(post_type, hazard_category)
                        elif platform == 'facebook':
                            post = self.generate_facebook_post(post_type, hazard_category)
                        else:
                            # Fallback to twitter post if platform is unknown
                            post = self.generate_twitter_post(post_type, hazard_category)

                        # Add post_type to the post data for better tracking
                        post['post_type'] = post_type
                        
                        # Stream to Redis
                        if self.stream_to_redis(post):
                            posts_streamed += 1
                            print(f"✓ Streamed {platform} post to Redis (Total: {posts_streamed})")
                        else:
                            print(f"✗ Failed to stream {platform} post to Redis")
                        
                        print("-" * 60)
                        
                        last_post_time[platform] = now
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\nSimulation stopped by user. Total posts streamed: {posts_streamed}")
            if self.redis_client:
                self.redis_client.close()

    def generate_stream_periodic_redis_timed(self, schedule: Dict[str, float], duration: int, redis_host: str = "localhost", redis_port: int = 6379):
        """Generate posts for a specific duration and stream to Redis."""
        if not self.setup_redis(redis_host, redis_port):
            print("Failed to setup Redis. Exiting...")
            return
        
        print(f"Starting timed social media simulation for {duration} seconds...")
        print(f"Redis stream: {self.redis_stream_name}")
        print("Schedule (seconds):", schedule)
        print("-" * 60)
        
        start_time = time.time()
        end_time = start_time + duration
        last_post_time = {platform: time.time() - interval for platform, interval in schedule.items()}
        posts_streamed = 0
        
        try:
            while time.time() < end_time:
                now = time.time()
                remaining = int(end_time - now)
                
                for platform, interval in schedule.items():
                    if now - last_post_time.get(platform, 0) >= interval:
                        print(f"[{remaining}s remaining] Generating '{platform}' post...")
                        
                        post_type = random.choices(
                            list(self.post_type_weights.keys()),
                            weights=list(self.post_type_weights.values())
                        )[0]
                        
                        hazard_category = None
                        if post_type in ['hazard', 'false_alarm']:
                            hazard_category = random.choice([h for h in self.hazard_categories if h != 'Non'])
                        
                        if platform == 'twitter':
                            post = self.generate_twitter_post(post_type, hazard_category)
                        elif platform == 'youtube':
                            post = self.generate_youtube_post(post_type, hazard_category)
                        elif platform == 'news':
                            post = self.generate_news_post(post_type, hazard_category)
                        elif platform == 'instagram':
                            post = self.generate_instagram_post(post_type, hazard_category)
                        elif platform == 'facebook':
                            post = self.generate_facebook_post(post_type, hazard_category)
                        else:
                            post = self.generate_twitter_post(post_type, hazard_category)

                        post['post_type'] = post_type
                        
                        if self.stream_to_redis(post):
                            posts_streamed += 1
                            print(f"✓ Streamed {platform} post to Redis (Total: {posts_streamed})")
                        else:
                            print(f"✗ Failed to stream {platform} post to Redis")
                        
                        print("-" * 60)
                        last_post_time[platform] = now
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            print(f"\nSimulation stopped by user.")
        
        print(f"\nSimulation completed! Total posts streamed: {posts_streamed}")
        if self.redis_client:
            self.redis_client.close()

    def generate_batch(self, count: int = 100) -> List[Dict]:
        """Generate a batch of posts without delays."""
        return [self.generate_post() for _ in range(count)]
    
    def save_to_file(self, posts: List[Dict], filename: str = "synthetic_posts.jsonl") -> None:
        """Save posts to a JSONL file."""
        with open(filename, 'w', encoding='utf-8') as f:
            for post in posts:
                f.write(json.dumps(post, ensure_ascii=False) + '\n')
        print(f"Saved {len(posts)} posts to {filename}")


def _parse_time_input(time_str: str) -> Optional[float]:
    """Parse time input like '2h', '30m', '1.5h' into seconds."""
    time_str = time_str.strip().lower()
    if not time_str:
        return None
    
    try:
        if time_str.endswith('h'):
            return float(time_str[:-1]) * 3600
        elif time_str.endswith('m'):
            return float(time_str[:-1]) * 60
        else:
            # Assume hours if no unit
            return float(time_str) * 3600
    except ValueError:
        return None

def get_periodic_settings(platforms: List[str]) -> Dict[str, float]:
    """Get periodic time settings from the user for each platform."""
    print("\n--- Configure Periodic Post Generation ---")
    print("Enter the time interval for each platform (e.g., '2h', '30m', '0.5h').")
    
    schedule = {}
    for platform in platforms:
        while True:
            time_input = input(f"  - Interval for '{platform}' posts: ")
            interval_seconds = _parse_time_input(time_input)
            if interval_seconds is not None and interval_seconds > 0:
                schedule[platform] = interval_seconds
                break
            else:
                print("  Invalid input. Please use a format like '2h', '30m', or a number for hours.")
    
    print("----------------------------------------\n")
    return schedule


def main():
    """Main function to run the simulator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Social Media Disaster Simulator')
    parser.add_argument('--count', '-c', type=int, default=10, 
                       help='Number of posts to generate (default: 10)')
    parser.add_argument('--delay', '-d', type=float, default=2.0, 
                       help='Delay between posts in seconds (default: 2.0)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file to save posts (JSONL format)')
    parser.add_argument('--batch', '-b', action='store_true',
                       help='Generate all posts at once without delays')
    parser.add_argument('--media', '-m', type=str, default="../Media",
                       help='Path to media folder (default: ../Media)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='Run in interactive mode to set periodic post times')
    parser.add_argument('--redis', '-r', action='store_true',
                        help='Enable Redis streaming mode (requires --interactive)')
    parser.add_argument('--redis-host', type=str, default='localhost',
                        help='Redis host (default: localhost)')
    parser.add_argument('--redis-port', type=int, default=6379,
                        help='Redis port (default: 6379)')
    parser.add_argument('--duration', type=int, default=None,
                        help='Duration to run in seconds (for Redis mode, overrides interactive schedule)')
    
    args = parser.parse_args()
    
    simulator = DisasterSocialMediaSimulator(media_folder=args.media)
    
    if args.duration and args.redis:
        # Duration-based Redis streaming mode
        print(f"🚀 Starting Redis streaming for {args.duration} seconds...")
        simulator.setup_redis(args.redis_host, args.redis_port)
        
        # Create a simple schedule with fast generation
        simple_schedule = {
            'twitter': 3,      # Every 3 seconds
            'facebook': 5,     # Every 5 seconds  
            'instagram': 7,    # Every 7 seconds
            'news': 10,        # Every 10 seconds
            'youtube': 15      # Every 15 seconds
        }
        
        simulator.generate_stream_periodic_redis_timed(simple_schedule, args.duration, args.redis_host, args.redis_port)
        
    elif args.interactive:
        platforms = simulator.platforms
        schedule = get_periodic_settings(platforms)
        
        if args.redis:
            simulator.generate_stream_periodic_redis(schedule, args.redis_host, args.redis_port)
        else:
            simulator.generate_stream_periodic(schedule)

    elif args.batch:
        posts = simulator.generate_batch(args.count)
        
        languages = {}
        platforms = {}
        
        for post in posts:
            languages[post['lang']] = languages.get(post['lang'], 0) + 1
            platforms[post['platform']] = platforms.get(post['platform'], 0) + 1
        
        print(f"Generated {len(posts)} posts:")
        print(f"Languages: {languages}")
        print(f"Platforms: {platforms}")
        
        if args.output:
            simulator.save_to_file(posts, args.output)
        else:
            print("\nExample posts:")
            for i, post in enumerate(posts[:3]):
                print(f"\nPost {i+1}:")
                print(json.dumps(post, indent=2, ensure_ascii=False))
    else:
        simulator.generate_stream(args.count, args.delay)


if __name__ == "__main__":
    main()