#!/usr/bin/env python3
"""
NLP Engine for Disaster Post Analysis
=====================================

Processes social media posts using Gemini API for:
1. Verification (real report vs false alarm/prank)
2. Information extraction (location, details, translation)
3. Structured output for Firebase storage

Author: Disaster Analytics Team
Date: September 2025
"""

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import re

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("Warning: Google Generative AI not installed. Install with: pip install google-generativeai")

try:
    import firebase_admin
    from firebase_admin import credentials, firestore
    FIREBASE_AVAILABLE = True
except ImportError:
    FIREBASE_AVAILABLE = False
    print("Warning: Firebase Admin SDK not installed. Install with: pip install firebase-admin")


class DisasterNLPEngine:
    """
    NLP Engine for processing disaster-related social media posts.
    """
    
    def __init__(self, gemini_api_key: str, firebase_cred_path: Optional[str] = None):
        """Initialize the NLP engine."""
        self.gemini_api_key = gemini_api_key
        self.firebase_cred_path = firebase_cred_path
        
        # Setup logging first
        logging.basicConfig(
            level=logging.DEBUG,  # Enable debug logging to see raw responses
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Initialize Gemini
        self.setup_gemini()
        
        # Initialize Firebase
        self.db = None
        if firebase_cred_path:
            self.setup_firebase()
        
        # Statistics
        self.processed_count = 0
        self.verified_count = 0
        self.discarded_count = 0
        self.stored_count = 0
    
    def setup_gemini(self):
        """Setup Gemini API."""
        if not GEMINI_AVAILABLE:
            raise ImportError("Google Generative AI library not installed")
        
        genai.configure(api_key=self.gemini_api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')  # Updated model name
        self.logger.info("✓ Gemini API configured successfully")
    
    def setup_firebase(self):
        """Setup Firebase connection."""
        if not FIREBASE_AVAILABLE:
            self.logger.warning("Firebase Admin SDK not available")
            return False
        
        try:
            if not firebase_admin._apps:
                cred = credentials.Certificate(self.firebase_cred_path)
                firebase_admin.initialize_app(cred)
            
            self.db = firestore.client()
            self.logger.info("✓ Firebase connected successfully")
            return True
        except Exception as e:
            self.logger.error(f"Firebase setup failed: {e}")
            return False
    
    def create_verification_prompt(self, post_data: Dict) -> str:
        """Create few-shot prompt for verification."""
        
        # Extract relevant content based on platform
        platform = post_data.get('platform', 'unknown')
        content = ""
        
        if platform == 'twitter':
            content = post_data.get('text', '')
        elif platform == 'news':
            content = f"Headline: {post_data.get('headline', '')}\nContent: {post_data.get('article_content', '')}"
        elif platform == 'youtube':
            content = f"Title: {post_data.get('title', '')}\nDescription: {post_data.get('description', '')}"
        elif platform in ['facebook', 'instagram']:
            content = post_data.get('content', '') or post_data.get('caption', '')
        
        prompt = f"""
You are an expert disaster verification analyst. Analyze social media posts to determine if they report genuine disaster events or are false alarms/pranks.

EXAMPLES:

INPUT: "🌊 TSUNAMI WARNING for coastal areas near Mumbai! Evacuate immediately! Water levels rising rapidly. #TsunamiAlert #Mumbai"
OUTPUT: {{
    "is_genuine": true,
    "verification_score": 0.85,
    "reasoning": "Contains specific location (Mumbai), urgent evacuation advice, and disaster-specific details. Language suggests official warning.",
    "confidence": "high"
}}

INPUT: "Haha guys look at this 'flood' in my backyard 😂 Mom forgot to turn off the sprinkler system. #fakealert #justwater"
OUTPUT: {{
    "is_genuine": false,
    "verification_score": 0.05,
    "reasoning": "Clearly states it's fake, mentions backyard sprinkler, uses laughing emoji and sarcastic tone.",
    "confidence": "high"
}}

INPUT: "Cyclone approaching Bengal region. High winds expected. Stay safe everyone! 🌪️"
OUTPUT: {{
    "is_genuine": true,
    "verification_score": 0.75,
    "reasoning": "Mentions specific region, describes weather phenomenon, includes safety advice. Consistent with cyclone reporting.",
    "confidence": "medium"
}}

Now analyze this post:

PLATFORM: {platform}
LANGUAGE: {post_data.get('lang', 'unknown')}
USER: {post_data.get('user', 'unknown')}
VERIFIED_ACCOUNT: {post_data.get('verified', False)}
CONTENT: {content}

Respond with ONLY a JSON object following the exact format shown in examples above.
"""
        return prompt
    
    def create_extraction_prompt(self, post_data: Dict) -> str:
        """Create few-shot prompt for information extraction."""
        
        # Extract relevant content based on platform
        platform = post_data.get('platform', 'unknown')
        content = ""
        
        if platform == 'twitter':
            content = post_data.get('text', '')
        elif platform == 'news':
            content = f"Headline: {post_data.get('headline', '')}\nContent: {post_data.get('article_content', '')}"
        elif platform == 'youtube':
            content = f"Title: {post_data.get('title', '')}\nDescription: {post_data.get('description', '')}"
        elif platform in ['facebook', 'instagram']:
            content = post_data.get('content', '') or post_data.get('caption', '')
        
        prompt = f"""
You are an expert disaster information extraction analyst. Extract structured information from verified disaster reports.

EXAMPLES:

INPUT: "🌊 TSUNAMI WARNING for coastal areas near Mumbai! Water levels rising 3 meters. Evacuate Worli, Bandra, Juhu immediately!"
OUTPUT: {{
    "disaster_type": "tsunami",
    "locations": [
        {{"name": "Mumbai", "type": "city", "specificity": "general"}},
        {{"name": "Worli", "type": "neighborhood", "specificity": "specific"}},
        {{"name": "Bandra", "type": "neighborhood", "specificity": "specific"}},
        {{"name": "Juhu", "type": "neighborhood", "specificity": "specific"}}
    ],
    "severity_indicators": ["water levels rising 3 meters", "evacuate immediately"],
    "impact_scale": "high",
    "urgency_level": "immediate",
    "affected_population": "coastal residents",
    "infrastructure_mentioned": [],
    "casualties_mentioned": false,
    "official_source": false,
    "translated_content": "TSUNAMI WARNING for coastal areas near Mumbai! Water levels rising 3 meters. Evacuate Worli, Bandra, Juhu immediately!",
    "key_details": [
        "Tsunami warning issued",
        "3-meter water level rise",
        "Multiple Mumbai coastal areas affected",
        "Immediate evacuation required"
    ]
}}

INPUT: "Cyclone Biparjoy hitting Gujarat coast. Winds 150 kmph. Surat port closed. 50,000 people moved to shelters."
OUTPUT: {{
    "disaster_type": "cyclone",
    "locations": [
        {{"name": "Gujarat", "type": "state", "specificity": "general"}},
        {{"name": "Surat", "type": "city", "specificity": "specific"}}
    ],
    "severity_indicators": ["winds 150 kmph", "port closed", "50,000 people moved to shelters"],
    "impact_scale": "high",
    "urgency_level": "critical",
    "affected_population": "50,000 people",
    "infrastructure_mentioned": ["Surat port"],
    "casualties_mentioned": false,
    "official_source": true,
    "translated_content": "Cyclone Biparjoy hitting Gujarat coast. Winds 150 kmph. Surat port closed. 50,000 people moved to shelters.",
    "key_details": [
        "Cyclone Biparjoy identified",
        "150 kmph wind speed",
        "Port operations suspended",
        "Large-scale evacuation (50,000 people)",
        "Emergency shelters activated"
    ]
}}

Now extract information from this post:

PLATFORM: {platform}
LANGUAGE: {post_data.get('lang', 'unknown')}
USER: {post_data.get('user', 'unknown')}
VERIFIED_ACCOUNT: {post_data.get('verified', False)}
TIMESTAMP: {post_data.get('timestamp', '')}
CONTENT: {content}

If the language is not English, translate the content to English in the 'translated_content' field.
Respond with ONLY a JSON object following the exact format shown in examples above.
"""
        return prompt
    
    def verify_post(self, post_data: Dict) -> Tuple[bool, Dict]:
        """Verify if a post is a genuine disaster report."""
        try:
            prompt = self.create_verification_prompt(post_data)
            response = self.model.generate_content(prompt)
            
            # Debug: Show raw response
            raw_response = response.text.strip()
            self.logger.debug(f"Raw API response: {raw_response[:200]}...")
            
            # Try to extract JSON from response (handle cases where model adds extra text)
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = raw_response[json_start:json_end]
                verification_result = json.loads(json_text)
            else:
                # If no JSON found, create a default false result
                self.logger.warning("No valid JSON found in response, marking as false alarm")
                verification_result = {
                    "is_genuine": False,
                    "verification_score": 0.0,
                    "reasoning": "Failed to parse response",
                    "error": "Invalid response format"
                }
            
            is_genuine = verification_result.get('is_genuine', False)
            
            self.logger.info(f"Verification: {'GENUINE' if is_genuine else 'FALSE ALARM'} "
                           f"(Score: {verification_result.get('verification_score', 0):.2f})")
            
            return is_genuine, verification_result
            
        except Exception as e:
            self.logger.error(f"Verification failed: {e}")
            return False, {"error": str(e)}
    
    def extract_information(self, post_data: Dict) -> Optional[Dict]:
        """Extract structured information from verified post."""
        try:
            prompt = self.create_extraction_prompt(post_data)
            response = self.model.generate_content(prompt)
            
            # Debug: Show raw response
            raw_response = response.text.strip()
            self.logger.debug(f"Raw extraction response: {raw_response[:200]}...")
            
            # Try to extract JSON from response
            json_start = raw_response.find('{')
            json_end = raw_response.rfind('}') + 1
            
            if json_start >= 0 and json_end > json_start:
                json_text = raw_response[json_start:json_end]
                extracted_info = json.loads(json_text)
            else:
                self.logger.warning("No valid JSON found in extraction response")
                return None
            
            # Add metadata
            extracted_info['source_platform'] = post_data.get('platform', 'unknown')
            extracted_info['source_user'] = post_data.get('user', 'unknown')
            extracted_info['source_verified'] = post_data.get('verified', False)
            extracted_info['original_timestamp'] = post_data.get('timestamp', '')
            extracted_info['processed_timestamp'] = datetime.now().isoformat()
            extracted_info['original_language'] = post_data.get('lang', 'unknown')
            extracted_info['engagement_metrics'] = {
                'likes': post_data.get('likes', 0),
                'shares': post_data.get('shares', 0) or post_data.get('retweets', 0),
                'comments': post_data.get('comments', 0) or post_data.get('replies', 0)
            }
            
            return extracted_info
            
        except Exception as e:
            self.logger.error(f"Information extraction failed: {e}")
            return None
    
    def store_to_firebase(self, extracted_data: Dict) -> bool:
        """Store extracted data to Firebase Firestore."""
        if not self.db:
            self.logger.warning("Firebase not configured. Skipping storage.")
            return False
        
        try:
            # Generate document ID
            doc_id = f"{extracted_data['source_platform']}_{int(time.time())}_{self.stored_count}"
            
            # Store in 'disaster_reports' collection
            doc_ref = self.db.collection('disaster_reports').document(doc_id)
            doc_ref.set(extracted_data)
            
            self.stored_count += 1
            self.logger.info(f"✓ Stored to Firebase: {doc_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Firebase storage failed: {e}")
            return False
    
    def process_post(self, post_data: Dict) -> Dict:
        """Process a single post through the complete pipeline."""
        self.processed_count += 1
        
        result = {
            'processed': True,
            'genuine': False,
            'stored': False,
            'verification_result': None,
            'extracted_data': None,
            'error': None
        }
        
        try:
            self.logger.info(f"Processing post #{self.processed_count} from {post_data.get('platform', 'unknown')}")
            
            # Step 1: Verify the post
            is_genuine, verification_result = self.verify_post(post_data)
            result['verification_result'] = verification_result
            
            if not is_genuine:
                self.discarded_count += 1
                self.logger.info("❌ Post discarded as false alarm/prank")
                return result
            
            # Step 2: Extract information
            self.verified_count += 1
            result['genuine'] = True
            
            extracted_data = self.extract_information(post_data)
            if not extracted_data:
                result['error'] = "Information extraction failed"
                return result
            
            result['extracted_data'] = extracted_data
            
            # Step 3: Store to Firebase
            if self.store_to_firebase(extracted_data):
                result['stored'] = True
            
            self.logger.info("✅ Post processed and stored successfully")
            
        except Exception as e:
            result['error'] = str(e)
            self.logger.error(f"Error processing post: {e}")
        
        return result
    
    def get_statistics(self) -> Dict:
        """Get processing statistics."""
        return {
            'processed_count': self.processed_count,
            'verified_count': self.verified_count,
            'discarded_count': self.discarded_count,
            'stored_count': self.stored_count,
            'verification_rate': self.verified_count / self.processed_count if self.processed_count > 0 else 0,
            'storage_success_rate': self.stored_count / self.verified_count if self.verified_count > 0 else 0
        }
    
    def print_statistics(self):
        """Print current statistics."""
        stats = self.get_statistics()
        print("\n" + "="*60)
        print("🤖 NLP ENGINE STATISTICS")
        print("="*60)
        print(f"📊 Total processed: {stats['processed_count']}")
        print(f"✅ Verified genuine: {stats['verified_count']}")
        print(f"❌ Discarded (false): {stats['discarded_count']}")
        print(f"💾 Stored to Firebase: {stats['stored_count']}")
        print(f"📈 Verification rate: {stats['verification_rate']:.1%}")
        print(f"💽 Storage success rate: {stats['storage_success_rate']:.1%}")
        print("="*60)


def main():
    """Test the NLP engine."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Disaster NLP Engine')
    parser.add_argument('--gemini-key', required=True, help='Gemini API key')
    parser.add_argument('--firebase-cred', help='Firebase credentials JSON path')
    parser.add_argument('--test', action='store_true', help='Run test with sample data')
    
    args = parser.parse_args()
    
    # Initialize engine
    engine = DisasterNLPEngine(
        gemini_api_key=args.gemini_key,
        firebase_cred_path=args.firebase_cred
    )
    
    if args.test:
        # Test with sample data
        sample_post = {
            "platform": "twitter",
            "user": "@emergencyalert",
            "lang": "English",
            "text": "🌊 TSUNAMI WARNING for coastal areas near Mumbai! Water levels rising rapidly. Evacuate Worli, Bandra immediately! #TsunamiAlert",
            "verified": True,
            "timestamp": "2025-09-22T10:30:00Z",
            "likes": 1500,
            "retweets": 800,
            "replies": 200
        }
        
        print("Testing NLP Engine with sample post...")
        result = engine.process_post(sample_post)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        engine.print_statistics()


if __name__ == "__main__":
    main()