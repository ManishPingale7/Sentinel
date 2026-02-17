#!/usr/bin/env python3
"""
Sentinel — Social Media Feed + Disaster Map API
=================================================

Two-page API for the Next.js frontend:

PAGE 1 — Social Media Feed (real-time stream)
  GET   /api/feed                     → paginated social media posts
  GET   /api/feed/platforms           → available platforms + counts
  GET   /api/feed/trending            → trending hashtags / topics
  GET   /api/feed/{post_id}           → single post detail
  WS    /ws/feed                      → real-time new-post stream

PAGE 2 — Disaster Map (verified reports pinned on India map)
  GET   /api/map/reports              → all verified report pins (lat, lng, summary)
  GET   /api/map/reports/{report_id}  → full report card detail
  GET   /api/map/clusters             → clustered pins for zoom levels
  GET   /api/map/heatmap              → density grid for heat-map layer
  GET   /api/map/states               → per-state disaster summary
  POST  /api/map/reports              → (admin) create/update a verified report

SHARED
  GET   /api/health                   → health-check
  GET   /api/stats                    → global statistics

Run:
    cd G:\\Sentinel
    uvicorn backend.feed_and_map_api:app --reload --port 8000

Swagger: http://localhost:8000/docs
"""

import os
import sys
import json
import uuid
import math
import asyncio
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set
from collections import defaultdict
from enum import Enum

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("SentinelAPI")

# ═══════════════════════════════════════════════════════════════════════════
# CONSTANTS — Indian city coordinates for realistic map pins
# ═══════════════════════════════════════════════════════════════════════════

INDIA_LOCATIONS: Dict[str, Dict] = {
    # ── Coastal cities (flood / cyclone / tsunami prone) ──
    "Mumbai": {"lat": 19.0760, "lng": 72.8777, "state": "Maharashtra"},
    "Chennai": {"lat": 13.0827, "lng": 80.2707, "state": "Tamil Nadu"},
    "Kolkata": {"lat": 22.5726, "lng": 88.3639, "state": "West Bengal"},
    "Kochi": {"lat": 9.9312, "lng": 76.2673, "state": "Kerala"},
    "Visakhapatnam": {"lat": 17.6868, "lng": 83.2185, "state": "Andhra Pradesh"},
    "Paradip": {"lat": 20.3164, "lng": 86.6085, "state": "Odisha"},
    "Mangalore": {"lat": 12.9141, "lng": 74.8560, "state": "Karnataka"},
    "Goa": {"lat": 15.2993, "lng": 74.1240, "state": "Goa"},
    "Puri": {"lat": 19.8135, "lng": 85.8312, "state": "Odisha"},
    "Surat": {"lat": 21.1702, "lng": 72.8311, "state": "Gujarat"},
    "Ratnagiri": {"lat": 16.9902, "lng": 73.3120, "state": "Maharashtra"},
    "Tuticorin": {"lat": 8.7642, "lng": 78.1348, "state": "Tamil Nadu"},
    "Kakinada": {"lat": 16.9891, "lng": 82.2475, "state": "Andhra Pradesh"},
    "Bhubaneswar": {"lat": 20.2961, "lng": 85.8245, "state": "Odisha"},
    "Thiruvananthapuram": {"lat": 8.5241, "lng": 76.9366, "state": "Kerala"},
    # ── Inland cities ──
    "Delhi": {"lat": 28.7041, "lng": 77.1025, "state": "Delhi"},
    "Bengaluru": {"lat": 12.9716, "lng": 77.5946, "state": "Karnataka"},
    "Hyderabad": {"lat": 17.3850, "lng": 78.4867, "state": "Telangana"},
    "Ahmedabad": {"lat": 23.0225, "lng": 72.5714, "state": "Gujarat"},
    "Pune": {"lat": 18.5204, "lng": 73.8567, "state": "Maharashtra"},
    "Jaipur": {"lat": 26.9124, "lng": 75.7873, "state": "Rajasthan"},
    "Lucknow": {"lat": 26.8467, "lng": 80.9462, "state": "Uttar Pradesh"},
    "Patna": {"lat": 25.6093, "lng": 85.1376, "state": "Bihar"},
    "Nagpur": {"lat": 21.1458, "lng": 79.0882, "state": "Maharashtra"},
    "Indore": {"lat": 22.7196, "lng": 75.8577, "state": "Madhya Pradesh"},
    "Bhopal": {"lat": 23.2599, "lng": 77.4126, "state": "Madhya Pradesh"},
    "Shimla": {"lat": 31.1048, "lng": 77.1734, "state": "Himachal Pradesh"},
    "Dehradun": {"lat": 30.3165, "lng": 78.0322, "state": "Uttarakhand"},
    "Guwahati": {"lat": 26.1445, "lng": 91.7362, "state": "Assam"},
    "Imphal": {"lat": 24.8170, "lng": 93.9368, "state": "Manipur"},
    "Gangtok": {"lat": 27.3389, "lng": 88.6065, "state": "Sikkim"},
    "Ranchi": {"lat": 23.3441, "lng": 85.3096, "state": "Jharkhand"},
    "Raipur": {"lat": 21.2514, "lng": 81.6296, "state": "Chhattisgarh"},
    "Wayanad": {"lat": 11.6854, "lng": 76.1320, "state": "Kerala"},
    "Uttarkashi": {"lat": 30.7268, "lng": 78.4354, "state": "Uttarakhand"},
}

DISASTER_TYPES = [
    "flood", "cyclone", "earthquake", "landslide",
    "tsunami", "drought", "fire", "heavy_rain",
]

SEVERITY_LEVELS = ["critical", "high", "medium", "low"]

PLATFORMS = ["twitter", "facebook", "youtube", "news", "instagram"]


# ═══════════════════════════════════════════════════════════════════════════
# PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════

# ── Feed models ───────────────────────────────────────────────────────────

class Author(BaseModel):
    name: str
    handle: str
    avatar_url: Optional[str] = None
    verified: bool = False
    followers: int = 0
    account_type: str = "personal"  # personal | journalist | government | ngo | news_outlet

class Engagement(BaseModel):
    likes: int = 0
    shares: int = 0
    comments: int = 0
    views: int = 0

class PostLocation(BaseModel):
    name: str
    lat: float
    lng: float
    state: Optional[str] = None

class Analysis(BaseModel):
    is_disaster: bool = False
    disaster_type: Optional[str] = None
    confidence: float = 0.0
    urgency: str = "low"          # critical | high | medium | low
    sentiment: str = "neutral"    # negative | neutral | positive
    credibility_score: float = 0.0

class FeedPost(BaseModel):
    id: str
    platform: str
    content: Dict[str, Any]       # text, headline, title, media_urls, hashtags
    author: Author
    engagement: Engagement
    analysis: Analysis
    location: Optional[PostLocation] = None
    timestamp: str
    language: str = "english"
    raw_data: Optional[Dict] = None   # original platform-specific fields

# ── Map models ────────────────────────────────────────────────────────────

class ReportLocation(BaseModel):
    name: str
    state: str
    lat: float
    lng: float

class Verification(BaseModel):
    status: str = "verified"       # verified | unverified | disputed
    confidence: float = 0.0
    source_count: int = 0
    platforms: List[str] = []
    verified_by: str = "system"    # system | manual | cross_reference

class Casualties(BaseModel):
    deaths: int = 0
    injured: int = 0
    missing: int = 0
    rescued: int = 0
    displaced: int = 0

class Impact(BaseModel):
    affected_area_km2: float = 0.0
    estimated_affected_people: int = 0
    casualties: Casualties = Casualties()
    infrastructure_damage: List[str] = []
    economic_loss_crore: float = 0.0

class SourceSnippet(BaseModel):
    platform: str
    post_id: str
    snippet: str
    author: str
    timestamp: str

class TimelineEvent(BaseModel):
    timestamp: str
    event: str

class MapReport(BaseModel):
    id: str
    title: str
    description: str
    disaster_type: str
    severity: str                  # critical | high | medium | low
    location: ReportLocation
    verification: Verification
    impact: Impact
    sources: List[SourceSnippet] = []
    timeline: List[TimelineEvent] = []
    response_actions: List[str] = []
    created_at: str
    updated_at: str

class MapPin(BaseModel):
    """Lightweight model for rendering map markers (no full detail)."""
    id: str
    lat: float
    lng: float
    disaster_type: str
    severity: str
    title: str
    location_name: str
    source_count: int
    created_at: str

class ClusterPoint(BaseModel):
    lat: float
    lng: float
    count: int
    dominant_type: str
    severity: str
    report_ids: List[str]

class StateStats(BaseModel):
    state: str
    total_reports: int
    active_reports: int
    disaster_types: Dict[str, int]
    severity_breakdown: Dict[str, int]
    latest_report_at: Optional[str] = None

# ── Request / Create models ──────────────────────────────────────────────

class CreateReportRequest(BaseModel):
    title: str
    description: str
    disaster_type: str
    severity: str = "medium"
    location_name: str             # key into INDIA_LOCATIONS or custom
    lat: Optional[float] = None
    lng: Optional[float] = None
    state: Optional[str] = None
    affected_area_km2: float = 0.0
    estimated_affected_people: int = 0
    casualties: Optional[Casualties] = None
    infrastructure_damage: List[str] = []
    source_post_ids: List[str] = []
    response_actions: List[str] = []


# ═══════════════════════════════════════════════════════════════════════════
# IN-MEMORY STORES
# ═══════════════════════════════════════════════════════════════════════════

class FeedStore:
    """Stores social-media posts for the feed page."""

    def __init__(self, max_size: int = 10_000):
        self._posts: List[Dict] = []
        self._index: Dict[str, int] = {}
        self._max = max_size

    # ── mutate ──

    def add(self, post: Dict):
        pid = post.get("id", "")
        if pid in self._index:
            return
        self._posts.append(post)
        self._index[pid] = len(self._posts) - 1
        if len(self._posts) > self._max:
            old = self._posts.pop(0)
            self._index.pop(old.get("id", ""), None)
            self._index = {p["id"]: i for i, p in enumerate(self._posts)}

    def add_many(self, posts: List[Dict]):
        for p in posts:
            self.add(p)

    # ── read ──

    def get(self, post_id: str) -> Optional[Dict]:
        idx = self._index.get(post_id)
        return self._posts[idx] if idx is not None else None

    def query(
        self,
        platform: Optional[str] = None,
        disaster_type: Optional[str] = None,
        urgency: Optional[str] = None,
        is_disaster: Optional[bool] = None,
        language: Optional[str] = None,
        search: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> List[Dict]:
        out = self._posts
        if platform:
            out = [p for p in out if p.get("platform") == platform]
        if disaster_type:
            out = [p for p in out if p.get("analysis", {}).get("disaster_type") == disaster_type]
        if urgency:
            out = [p for p in out if p.get("analysis", {}).get("urgency") == urgency]
        if is_disaster is not None:
            out = [p for p in out if p.get("analysis", {}).get("is_disaster") == is_disaster]
        if language:
            out = [p for p in out if p.get("language", "").lower() == language.lower()]
        if search:
            q = search.lower()
            out = [p for p in out if q in json.dumps(p.get("content", {})).lower()]
        out = sorted(out, key=lambda p: p.get("timestamp", ""), reverse=True)
        return out[offset: offset + limit]

    @property
    def count(self) -> int:
        return len(self._posts)

    def platform_counts(self) -> Dict[str, int]:
        c: Dict[str, int] = defaultdict(int)
        for p in self._posts:
            c[p.get("platform", "unknown")] += 1
        return dict(c)

    def trending_hashtags(self, top_n: int = 15) -> List[Dict]:
        counts: Dict[str, int] = defaultdict(int)
        for p in self._posts:
            for tag in p.get("content", {}).get("hashtags", []):
                counts[tag.lower()] += 1
        ranked = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:top_n]
        return [{"hashtag": h, "count": c} for h, c in ranked]


class ReportStore:
    """Stores verified disaster reports for the map page."""

    def __init__(self):
        self._reports: Dict[str, Dict] = {}

    def add(self, report: Dict):
        self._reports[report["id"]] = report

    def get(self, report_id: str) -> Optional[Dict]:
        return self._reports.get(report_id)

    def all(self) -> List[Dict]:
        return list(self._reports.values())

    def query(
        self,
        disaster_type: Optional[str] = None,
        severity: Optional[str] = None,
        state: Optional[str] = None,
        verified_only: bool = True,
    ) -> List[Dict]:
        out = list(self._reports.values())
        if disaster_type:
            out = [r for r in out if r.get("disaster_type") == disaster_type]
        if severity:
            out = [r for r in out if r.get("severity") == severity]
        if state:
            out = [r for r in out if r.get("location", {}).get("state", "").lower() == state.lower()]
        if verified_only:
            out = [r for r in out if r.get("verification", {}).get("status") == "verified"]
        return sorted(out, key=lambda r: r.get("created_at", ""), reverse=True)

    @property
    def count(self) -> int:
        return len(self._reports)


class WSManager:
    """WebSocket connection manager for real-time feed."""

    def __init__(self):
        self.active: Set[WebSocket] = set()

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.add(ws)

    def disconnect(self, ws: WebSocket):
        self.active.discard(ws)

    async def broadcast(self, payload: Dict):
        dead: Set[WebSocket] = set()
        for ws in self.active:
            try:
                await ws.send_json(payload)
            except Exception:
                dead.add(ws)
        self.active -= dead


# ═══════════════════════════════════════════════════════════════════════════
# SEED DATA GENERATOR — creates realistic demo data
# ═══════════════════════════════════════════════════════════════════════════

def _ts(minutes_ago: int = 0) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _seed_feed_posts() -> List[Dict]:
    """Generate demo social media posts for all platforms."""
    posts: List[Dict] = []

    # ── Templates per platform ──
    TWEETS = [
        {"text": "🌊 Severe coastal flooding in {loc}! Water levels rising fast. Stay safe everyone. #FloodAlert #DisasterResponse", "disaster": True, "dtype": "flood", "urgency": "critical"},
        {"text": "🌀 Cyclone warning upgraded for {loc}. IMD advises immediate evacuation of low-lying areas. #CycloneAlert", "disaster": True, "dtype": "cyclone", "urgency": "critical"},
        {"text": "🌧️ Non-stop heavy rain in {loc} for 48 hours. Rivers overflowing, roads waterlogged. #HeavyRain", "disaster": True, "dtype": "heavy_rain", "urgency": "high"},
        {"text": "⚠️ Earthquake felt across {loc} region. Magnitude ~5.2 reported by seismological dept. #Earthquake", "disaster": True, "dtype": "earthquake", "urgency": "high"},
        {"text": "🏔️ Landslide blocks highway near {loc}. Rescue teams deployed. Multiple vehicles trapped. #Landslide", "disaster": True, "dtype": "landslide", "urgency": "high"},
        {"text": "NDRF teams deployed in {loc} after flooding. Several families rescued from rooftops. #Rescue #NDRF", "disaster": True, "dtype": "flood", "urgency": "critical"},
        {"text": "Beautiful sunrise in {loc} today 🌅 What a peaceful morning!", "disaster": False, "dtype": None, "urgency": "low"},
        {"text": "Just had amazing dosa at a street stall in {loc}! 🍽️ Highly recommend.", "disaster": False, "dtype": None, "urgency": "low"},
        {"text": "Traffic jam near {loc} flyover as usual. When will the metro construction finish? 😤", "disaster": False, "dtype": None, "urgency": "low"},
        {"text": "IMD issues red alert for {loc}: extremely heavy rainfall likely in next 24 hours. All schools closed.", "disaster": True, "dtype": "heavy_rain", "urgency": "critical"},
        {"text": "🔥 Major fire breaks out in industrial area near {loc}. Fire brigades on site. #FireAlert", "disaster": True, "dtype": "fire", "urgency": "high"},
        {"text": "Drought conditions worsen in {loc} district. Farmers demand government action. #WaterCrisis #Drought", "disaster": True, "dtype": "drought", "urgency": "medium"},
    ]

    FB_POSTS = [
        {"text": "URGENT: Flooding in {loc} neighbourhood. Our house is surrounded by water. Can someone send help? Sharing live photos.", "disaster": True, "dtype": "flood", "urgency": "critical"},
        {"text": "Cyclone heading towards {loc}. State government announced holiday for all offices and schools tomorrow. Stay indoors.", "disaster": True, "dtype": "cyclone", "urgency": "high"},
        {"text": "Enjoyed wonderful family picnic at {loc} park! The weather was perfect. 🌞", "disaster": False, "dtype": None, "urgency": "low"},
        {"text": "Community alert: Heavy waterlogging reported in {loc} west. Avoid traveling through that area. Many cars stalled.", "disaster": True, "dtype": "flood", "urgency": "medium"},
        {"text": "Our NGO is collecting relief materials for {loc} flood victims. Drop-off points in the comments below. 🙏", "disaster": True, "dtype": "flood", "urgency": "medium"},
    ]

    YT_POSTS = [
        {"title": "LIVE: Cyclone {loc} - Real-time tracking and ground report", "desc": "Live coverage of cyclone approaching {loc}. Ground reporters showing wind speed and rainfall intensity.", "disaster": True, "dtype": "cyclone", "urgency": "critical"},
        {"title": "Massive Flooding in {loc} | Drone Footage 2026", "desc": "Aerial drone footage showing extent of flooding in {loc} after 3 days of continuous rain.", "disaster": True, "dtype": "flood", "urgency": "high"},
        {"title": "{loc} Street Food Tour | Best Places to Eat", "desc": "Exploring the amazing street food scene in {loc}!", "disaster": False, "dtype": None, "urgency": "low"},
        {"title": "Earthquake Aftermath in {loc} - Buildings Damaged", "desc": "Footage from {loc} following the 5.2 magnitude earthquake. Several structures show cracks.", "disaster": True, "dtype": "earthquake", "urgency": "high"},
    ]

    NEWS_POSTS = [
        {"headline": "{loc} floods: Death toll rises to {n}, thousands displaced", "body": "The situation in {loc} remains grim as rescue operations continue for the fourth day. NDRF boats are being used to evacuate stranded families.", "disaster": True, "dtype": "flood", "urgency": "critical"},
        {"headline": "Cyclone {name} to make landfall near {loc} by tonight: IMD", "body": "India Meteorological Department has issued a red alert for coastal districts of {state}. Wind speeds expected to reach 150 kmph.", "disaster": True, "dtype": "cyclone", "urgency": "critical"},
        {"headline": "{loc} receives highest single-day rainfall in 50 years", "body": "A new record was set as {loc} recorded 280 mm of rainfall in 24 hours, surpassing the previous 1976 record.", "disaster": True, "dtype": "heavy_rain", "urgency": "high"},
        {"headline": "Infrastructure push: New metro line connecting {loc} suburbs approved", "body": "The cabinet approved ₹12,000 crore for the new metro corridor.", "disaster": False, "dtype": None, "urgency": "low"},
        {"headline": "Landslide in {loc} hills blocks national highway, 200 tourists stranded", "body": "Rescue operations underway after heavy rain triggered a massive landslide on NH-58 near {loc}.", "disaster": True, "dtype": "landslide", "urgency": "high"},
    ]

    INSTA_POSTS = [
        {"caption": "Praying for {loc} 🙏 The floods are devastating. Please donate to relief funds. #StandWith{loc_tag} #FloodRelief", "disaster": True, "dtype": "flood", "urgency": "medium"},
        {"caption": "The aftermath of cyclone in {loc}. Trees uprooted, power lines down. Volunteers needed! 💪 #CycloneRelief", "disaster": True, "dtype": "cyclone", "urgency": "medium"},
        {"caption": "Golden hour at Marine Drive, {loc} ✨ #Sunset #CityLove #{loc_tag}", "disaster": False, "dtype": None, "urgency": "low"},
    ]

    locations = list(INDIA_LOCATIONS.keys())
    cyclone_names = ["Biparjoy", "Tauktae", "Michaung", "Remal", "Dana", "Fengal"]
    news_outlets = ["Times of India", "NDTV", "The Hindu", "India Today", "Hindustan Times", "The Wire", "Scroll.in", "Indian Express"]

    def _author(platform: str, verified_chance: float = 0.15):
        names = [
            "Rahul Sharma", "Priya Patel", "Amit Kumar", "Sneha Reddy",
            "Vikash Singh", "Kavita Nair", "Arjun Das", "Meera Iyer",
            "Rajesh Gupta", "Ananya Dutta", "Kiran Rao", "Deepak Menon",
            "Sanya Malhotra", "Nikhil Verma", "Pooja Hegde", "Rohan Joshi",
        ]
        gov_names = ["NDRF India", "IMD Official", "SDMA Official", "CM Office", "District Collector"]
        news_handles = ["@TimesNow", "@ndaborea", "@the_hindu", "@CNNnews18"]

        is_gov = random.random() < 0.08
        is_news = random.random() < 0.12
        name = random.choice(gov_names) if is_gov else (random.choice(news_outlets) if is_news else random.choice(names))
        handle = f"@{name.lower().replace(' ', '_').replace('.', '')}"
        if is_news:
            handle = random.choice(news_handles)
        return {
            "name": name,
            "handle": handle,
            "avatar_url": f"https://ui-avatars.com/api/?name={name.replace(' ', '+')}&background=random",
            "verified": is_gov or is_news or random.random() < verified_chance,
            "followers": random.randint(500, 500_000) if is_gov or is_news else random.randint(50, 50_000),
            "account_type": "government" if is_gov else ("news_outlet" if is_news else random.choice(["personal", "journalist", "ngo"])),
        }

    for i in range(120):
        loc = random.choice(locations)
        loc_info = INDIA_LOCATIONS[loc]
        loc_tag = loc.replace(" ", "")
        minutes_ago = random.randint(1, 1440)  # up to 24 hours ago

        # Twitter
        if i < 35:
            t = random.choice(TWEETS)
            text = t["text"].format(loc=loc)
            hashtags = [w for w in text.split() if w.startswith("#")]
            posts.append({
                "id": f"tw_{uuid.uuid4().hex[:12]}",
                "platform": "twitter",
                "content": {"text": text, "hashtags": hashtags, "media_urls": []},
                "author": _author("twitter"),
                "engagement": {"likes": random.randint(5, 8000), "shares": random.randint(0, 3000), "comments": random.randint(0, 500), "views": random.randint(100, 200000)},
                "analysis": {"is_disaster": t["disaster"], "disaster_type": t["dtype"], "confidence": round(random.uniform(0.7, 0.98), 2) if t["disaster"] else round(random.uniform(0.05, 0.3), 2), "urgency": t["urgency"], "sentiment": "negative" if t["disaster"] else "neutral", "credibility_score": round(random.uniform(0.6, 0.95), 2)},
                "location": {"name": loc, "lat": loc_info["lat"], "lng": loc_info["lng"], "state": loc_info["state"]} if random.random() > 0.2 else None,
                "timestamp": _ts(minutes_ago),
                "language": random.choice(["english", "hindi"]),
            })

        # Facebook
        elif i < 55:
            t = random.choice(FB_POSTS)
            text = t["text"].format(loc=loc)
            posts.append({
                "id": f"fb_{uuid.uuid4().hex[:12]}",
                "platform": "facebook",
                "content": {"text": text, "hashtags": [w for w in text.split() if w.startswith("#")], "media_urls": [f"https://placeholder.pics/svg/600x400/DEDEDE/555555/{loc}%20flood"] if t["disaster"] else []},
                "author": _author("facebook"),
                "engagement": {"likes": random.randint(10, 5000), "shares": random.randint(5, 2000), "comments": random.randint(2, 800), "views": random.randint(500, 100000)},
                "analysis": {"is_disaster": t["disaster"], "disaster_type": t["dtype"], "confidence": round(random.uniform(0.65, 0.95), 2) if t["disaster"] else round(random.uniform(0.05, 0.25), 2), "urgency": t["urgency"], "sentiment": "negative" if t["disaster"] else "positive", "credibility_score": round(random.uniform(0.55, 0.90), 2)},
                "location": {"name": loc, "lat": loc_info["lat"], "lng": loc_info["lng"], "state": loc_info["state"]},
                "timestamp": _ts(minutes_ago),
                "language": random.choice(["english", "hindi", "marathi"]),
            })

        # YouTube
        elif i < 70:
            t = random.choice(YT_POSTS)
            title = t["title"].format(loc=loc)
            desc = t["desc"].format(loc=loc)
            posts.append({
                "id": f"yt_{uuid.uuid4().hex[:12]}",
                "platform": "youtube",
                "content": {"title": title, "text": desc, "hashtags": [f"#{loc_tag}", "#DisasterAlert"], "media_urls": [f"https://img.youtube.com/vi/placeholder/hqdefault.jpg"]},
                "author": _author("youtube", verified_chance=0.25),
                "engagement": {"likes": random.randint(50, 20000), "shares": random.randint(10, 5000), "comments": random.randint(20, 3000), "views": random.randint(1000, 500000)},
                "analysis": {"is_disaster": t["disaster"], "disaster_type": t["dtype"], "confidence": round(random.uniform(0.7, 0.97), 2) if t["disaster"] else round(random.uniform(0.05, 0.2), 2), "urgency": t["urgency"], "sentiment": "negative" if t["disaster"] else "positive", "credibility_score": round(random.uniform(0.6, 0.92), 2)},
                "location": {"name": loc, "lat": loc_info["lat"], "lng": loc_info["lng"], "state": loc_info["state"]},
                "timestamp": _ts(minutes_ago),
                "language": "english",
            })

        # News
        elif i < 95:
            t = random.choice(NEWS_POSTS)
            headline = t["headline"].format(loc=loc, n=random.randint(3, 45), name=random.choice(cyclone_names), state=loc_info["state"])
            body = t["body"].format(loc=loc, state=loc_info["state"])
            posts.append({
                "id": f"nw_{uuid.uuid4().hex[:12]}",
                "platform": "news",
                "content": {"headline": headline, "text": body, "hashtags": [f"#{loc_tag}"], "media_urls": []},
                "author": {"name": random.choice(news_outlets), "handle": f"@{random.choice(news_outlets).lower().replace(' ', '')}", "avatar_url": None, "verified": True, "followers": random.randint(100000, 5000000), "account_type": "news_outlet"},
                "engagement": {"likes": random.randint(100, 15000), "shares": random.randint(50, 8000), "comments": random.randint(30, 2000), "views": random.randint(5000, 1000000)},
                "analysis": {"is_disaster": t["disaster"], "disaster_type": t["dtype"], "confidence": round(random.uniform(0.8, 0.99), 2) if t["disaster"] else round(random.uniform(0.02, 0.15), 2), "urgency": t["urgency"], "sentiment": "negative" if t["disaster"] else "neutral", "credibility_score": round(random.uniform(0.8, 0.98), 2)},
                "location": {"name": loc, "lat": loc_info["lat"], "lng": loc_info["lng"], "state": loc_info["state"]},
                "timestamp": _ts(minutes_ago),
                "language": "english",
            })

        # Instagram
        else:
            t = random.choice(INSTA_POSTS)
            caption = t["caption"].format(loc=loc, loc_tag=loc_tag)
            posts.append({
                "id": f"ig_{uuid.uuid4().hex[:12]}",
                "platform": "instagram",
                "content": {"text": caption, "hashtags": [w for w in caption.split() if w.startswith("#")], "media_urls": [f"https://placeholder.pics/svg/600x600/DEDEDE/555555/{loc}"]},
                "author": _author("instagram"),
                "engagement": {"likes": random.randint(20, 30000), "shares": random.randint(5, 2000), "comments": random.randint(5, 500), "views": random.randint(200, 100000)},
                "analysis": {"is_disaster": t["disaster"], "disaster_type": t["dtype"], "confidence": round(random.uniform(0.5, 0.90), 2) if t["disaster"] else round(random.uniform(0.05, 0.2), 2), "urgency": t["urgency"], "sentiment": "negative" if t["disaster"] else "positive", "credibility_score": round(random.uniform(0.4, 0.80), 2)},
                "location": {"name": loc, "lat": loc_info["lat"], "lng": loc_info["lng"], "state": loc_info["state"]},
                "timestamp": _ts(minutes_ago),
                "language": random.choice(["english", "hindi"]),
            })

    return posts


def _seed_map_reports(feed_posts: List[Dict]) -> List[Dict]:
    """Generate verified disaster reports from aggregated feed data."""
    reports: List[Dict] = []

    # Build reports around specific disaster incidents
    INCIDENTS = [
        {
            "title": "Severe Flooding in Mumbai - Red Alert",
            "description": "Multiple areas of Mumbai experiencing severe waterlogging and flooding following 72 hours of continuous heavy rainfall. Mithi River has breached its banks. Over 50,000 people affected across Dharavi, Sion, Kurla and Andheri areas. NDRF has deployed 12 teams. BMC has opened 87 relief camps. Local trains suspended on Central and Western lines.",
            "disaster_type": "flood", "severity": "critical",
            "location_name": "Mumbai",
            "impact": {"affected_area_km2": 45.5, "estimated_affected_people": 52000, "casualties": {"deaths": 8, "injured": 43, "missing": 5, "rescued": 1200, "displaced": 15000}, "infrastructure_damage": ["roads", "railway_tracks", "bridges", "power_lines", "water_supply"], "economic_loss_crore": 850.0},
            "response_actions": ["NDRF teams deployed", "Relief camps opened", "Army on standby", "Emergency helpline activated: 1916", "Schools closed for 3 days"],
        },
        {
            "title": "Cyclone Dana Approaches Odisha Coast",
            "description": "Cyclonic storm Dana, classified as a Severe Cyclonic Storm by IMD, is expected to make landfall between Paradip and Puri by tonight. Maximum sustained wind speed of 120 kmph. Coastal areas being evacuated. Odisha state government has cancelled all leaves and activated disaster response protocols.",
            "disaster_type": "cyclone", "severity": "critical",
            "location_name": "Paradip",
            "impact": {"affected_area_km2": 320.0, "estimated_affected_people": 180000, "casualties": {"deaths": 0, "injured": 12, "missing": 0, "rescued": 0, "displaced": 85000}, "infrastructure_damage": ["power_lines", "telecommunication", "fishing_boats", "coastal_structures"], "economic_loss_crore": 1200.0},
            "response_actions": ["Mass evacuation of coastal areas", "NDRF pre-positioned in 5 districts", "Navy ships on standby", "IMD issuing hourly bulletins", "Airport operations suspended"],
        },
        {
            "title": "Earthquake Tremors Felt Across Delhi-NCR",
            "description": "An earthquake of magnitude 5.2 on the Richter scale struck near Uttarkashi, Uttarakhand. Strong tremors felt across Delhi, Noida, Gurgaon and surrounding NCR region. Epicenter at 30.7°N, 78.4°E at a depth of 10 km. Several buildings reported minor cracks. No major structural damage reported so far.",
            "disaster_type": "earthquake", "severity": "high",
            "location_name": "Delhi",
            "impact": {"affected_area_km2": 150.0, "estimated_affected_people": 25000, "casualties": {"deaths": 0, "injured": 7, "missing": 0, "rescued": 0, "displaced": 500}, "infrastructure_damage": ["building_cracks", "road_cracks"], "economic_loss_crore": 50.0},
            "response_actions": ["NDMA advisory issued", "Building inspections ordered", "Emergency services on high alert", "School buildings being assessed", "Seismological monitoring intensified"],
        },
        {
            "title": "Landslide Blocks NH-58 Near Uttarkashi",
            "description": "A massive landslide triggered by heavy rain has blocked National Highway 58 near Uttarkashi, Uttarakhand. Approximately 200 tourists and locals are stranded on both sides. Rescue operations underway by SDRF and BRO teams. Alternate routes being assessed.",
            "disaster_type": "landslide", "severity": "high",
            "location_name": "Uttarkashi",
            "impact": {"affected_area_km2": 2.5, "estimated_affected_people": 800, "casualties": {"deaths": 2, "injured": 15, "missing": 3, "rescued": 45, "displaced": 200}, "infrastructure_damage": ["national_highway", "power_lines", "water_pipeline"], "economic_loss_crore": 25.0},
            "response_actions": ["SDRF teams deployed", "BRO clearing highway", "Helicopter rescue for critical cases", "Temporary shelters arranged", "Food and water being airlifted"],
        },
        {
            "title": "Wayanad Landslide - Emergency Operations",
            "description": "Multiple landslides hit Wayanad district in Kerala following unprecedented rainfall. Towns of Mundakkai and Chooralmala severely affected. Several houses buried under debris. Indian Army and Kerala Fire Force conducting rescue operations.",
            "disaster_type": "landslide", "severity": "critical",
            "location_name": "Wayanad",
            "impact": {"affected_area_km2": 12.0, "estimated_affected_people": 5000, "casualties": {"deaths": 45, "injured": 120, "missing": 78, "rescued": 350, "displaced": 4500}, "infrastructure_damage": ["houses", "roads", "bridges", "school_buildings", "hospitals"], "economic_loss_crore": 500.0},
            "response_actions": ["Army rescue operations ongoing", "NDRF 6 teams deployed", "Air Force helicopters for evacuation", "PM announces relief package", "Donation drive launched"],
        },
        {
            "title": "Heavy Rainfall and Flash Floods in Chennai",
            "description": "Chennai and surrounding districts experiencing torrential rainfall. Adyar river water level rising dangerously. Low-lying areas of Velachery, T. Nagar and Mylapore submerged. Chennai airport operations halted. TN government declares holiday.",
            "disaster_type": "flood", "severity": "high",
            "location_name": "Chennai",
            "impact": {"affected_area_km2": 30.0, "estimated_affected_people": 35000, "casualties": {"deaths": 3, "injured": 22, "missing": 0, "rescued": 600, "displaced": 8000}, "infrastructure_damage": ["roads", "metro", "airport", "power_supply"], "economic_loss_crore": 400.0},
            "response_actions": ["State emergency declared", "NDRF deployed", "Army boats for rescue", "Schools and offices closed", "Relief camps in 45 locations"],
        },
        {
            "title": "Drought Conditions Worsen in Marathwada",
            "description": "Severe drought conditions persist across Marathwada region of Maharashtra. Water reservoirs at less than 15% capacity. Crop failure reported in 8 districts. Tanker water supply being arranged for over 1200 villages.",
            "disaster_type": "drought", "severity": "medium",
            "location_name": "Nagpur",
            "impact": {"affected_area_km2": 8500.0, "estimated_affected_people": 250000, "casualties": {"deaths": 0, "injured": 0, "missing": 0, "rescued": 0, "displaced": 0}, "infrastructure_damage": ["water_reservoirs", "agricultural_land", "livestock_loss"], "economic_loss_crore": 2000.0},
            "response_actions": ["Water tanker supply to 1200 villages", "State drought relief fund activated", "Crop insurance claims fast-tracked", "Bore-well drilling authorized", "Cattle camps opened"],
        },
        {
            "title": "Kolkata Waterlogging After Nor'wester",
            "description": "Severe waterlogging across Kolkata after a powerful nor'wester brought 120mm of rain in 3 hours. Multiple underpasses flooded. Traffic gridlock across the city. KMC pumps working at full capacity.",
            "disaster_type": "heavy_rain", "severity": "medium",
            "location_name": "Kolkata",
            "impact": {"affected_area_km2": 18.0, "estimated_affected_people": 15000, "casualties": {"deaths": 1, "injured": 8, "missing": 0, "rescued": 50, "displaced": 2000}, "infrastructure_damage": ["roads", "underpasses", "power_lines", "drainage_system"], "economic_loss_crore": 75.0},
            "response_actions": ["KMC pumps deployed", "Traffic diversions in place", "Emergency helpline: 1800-345-3637", "Electricity restoration underway"],
        },
        {
            "title": "Industrial Fire in Surat Chemical Factory",
            "description": "A major fire broke out at a chemical factory in Sachin GIDC, Surat. Fire spread to two adjacent units. Thick black smoke visible from several kilometres. 15 fire tenders on site. Workers evacuated.",
            "disaster_type": "fire", "severity": "high",
            "location_name": "Surat",
            "impact": {"affected_area_km2": 0.5, "estimated_affected_people": 500, "casualties": {"deaths": 2, "injured": 18, "missing": 0, "rescued": 35, "displaced": 0}, "infrastructure_damage": ["factory_building", "storage_units", "vehicles"], "economic_loss_crore": 120.0},
            "response_actions": ["15 fire tenders deployed", "Nearby factories evacuated", "Air quality monitoring activated", "Hospital trauma centres alerted", "GPCB investigating cause"],
        },
        {
            "title": "Guwahati Flood from Brahmaputra Rise",
            "description": "Brahmaputra river water level crosses danger mark in Guwahati. Several areas including Fancy Bazaar, Panbazaar and Uzan Bazaar inundated. Assam State Disaster Management Authority has issued a high alert for Kamrup Metropolitan district.",
            "disaster_type": "flood", "severity": "high",
            "location_name": "Guwahati",
            "impact": {"affected_area_km2": 22.0, "estimated_affected_people": 28000, "casualties": {"deaths": 4, "injured": 15, "missing": 2, "rescued": 450, "displaced": 12000}, "infrastructure_damage": ["roads", "markets", "residential_areas", "power_supply"], "economic_loss_crore": 200.0},
            "response_actions": ["SDRF boats deployed", "Relief camps opened in 30 locations", "ASDMA monitoring 24/7", "Central team for assessment", "Gratuitous relief distribution started"],
        },
    ]

    # Collect matching feed posts for each incident
    for inc in INCIDENTS:
        loc_name = inc["location_name"]
        loc_info = INDIA_LOCATIONS.get(loc_name, {"lat": 20.5, "lng": 78.9, "state": "Unknown"})

        # Find related feed posts
        sources: List[Dict] = []
        for p in feed_posts:
            p_loc = p.get("location", {})
            if p_loc and p_loc.get("name") == loc_name and p.get("analysis", {}).get("is_disaster"):
                sources.append({
                    "platform": p["platform"],
                    "post_id": p["id"],
                    "snippet": (p.get("content", {}).get("text") or p.get("content", {}).get("headline") or "")[:120],
                    "author": p.get("author", {}).get("name", "Unknown"),
                    "timestamp": p["timestamp"],
                })
                if len(sources) >= 5:
                    break

        # If not enough real sources, add synthetic ones
        while len(sources) < 3:
            sources.append({
                "platform": random.choice(PLATFORMS),
                "post_id": f"syn_{uuid.uuid4().hex[:10]}",
                "snippet": f"Reports of {inc['disaster_type']} situation in {loc_name}",
                "author": random.choice(["Local Reporter", "NDRF Official", "District Admin", "Eyewitness"]),
                "timestamp": _ts(random.randint(30, 300)),
            })

        created = _ts(random.randint(60, 720))
        report = {
            "id": f"rpt_{uuid.uuid4().hex[:12]}",
            "title": inc["title"],
            "description": inc["description"],
            "disaster_type": inc["disaster_type"],
            "severity": inc["severity"],
            "location": {
                "name": loc_name,
                "state": loc_info["state"],
                "lat": loc_info["lat"],
                "lng": loc_info["lng"],
            },
            "verification": {
                "status": "verified",
                "confidence": round(random.uniform(0.82, 0.98), 2),
                "source_count": len(sources),
                "platforms": list({s["platform"] for s in sources}),
                "verified_by": random.choice(["system", "cross_reference"]),
            },
            "impact": inc["impact"],
            "sources": sources,
            "timeline": [
                {"timestamp": _ts(random.randint(400, 700)), "event": "First social media report detected"},
                {"timestamp": _ts(random.randint(300, 399)), "event": "Cross-platform correlation confirmed"},
                {"timestamp": _ts(random.randint(200, 299)), "event": f"Official {inc['disaster_type']} alert issued"},
                {"timestamp": _ts(random.randint(100, 199)), "event": "Rescue / response operations initiated"},
                {"timestamp": _ts(random.randint(10, 99)), "event": "Report verified and published"},
            ],
            "response_actions": inc.get("response_actions", []),
            "created_at": created,
            "updated_at": _ts(random.randint(5, 55)),
        }
        reports.append(report)

    return reports


# ═══════════════════════════════════════════════════════════════════════════
# APP FACTORY
# ═══════════════════════════════════════════════════════════════════════════

def create_app() -> FastAPI:
    app = FastAPI(
        title="Sentinel — Social Feed & Disaster Map API",
        version="2.0.0",
        description=(
            "REST + WebSocket API powering two frontend pages:\n"
            "1. **Social Media Feed** — live-streamed disaster-related content\n"
            "2. **Disaster Map** — verified reports pinned on a map of India"
        ),
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Initialise stores ──
    feed_store = FeedStore()
    report_store = ReportStore()
    ws_mgr = WSManager()

    # ── Seed demo data ──
    seed_posts = _seed_feed_posts()
    feed_store.add_many(seed_posts)
    seed_reports = _seed_map_reports(seed_posts)
    for r in seed_reports:
        report_store.add(r)

    # Expose for external pipeline integration
    app.state.feed_store = feed_store
    app.state.report_store = report_store
    app.state.ws_mgr = ws_mgr

    # ── Helper ──

    def _ok(data: Any, **meta) -> Dict:
        return {"ok": True, "data": data, "meta": {"timestamp": datetime.now(timezone.utc).isoformat(), **meta}}

    # ─────────────────────────────────────────────────────────────────────
    #  SHARED ENDPOINTS
    # ─────────────────────────────────────────────────────────────────────

    @app.get("/", tags=["Health"])
    def root():
        return {"status": "ok", "api": "Sentinel Feed & Map API", "version": "2.0.0"}

    @app.get("/api/health", tags=["Health"])
    def health():
        return _ok({
            "status": "healthy",
            "feed_posts": feed_store.count,
            "verified_reports": report_store.count,
            "ws_connections": len(ws_mgr.active),
        })

    @app.get("/api/stats", tags=["Health"])
    def global_stats():
        disaster_posts = len([p for p in feed_store._posts if p.get("analysis", {}).get("is_disaster")])
        critical_reports = len([r for r in report_store.all() if r.get("severity") == "critical"])
        return _ok({
            "total_posts": feed_store.count,
            "disaster_posts": disaster_posts,
            "verified_reports": report_store.count,
            "critical_reports": critical_reports,
            "platforms": feed_store.platform_counts(),
        })

    # ─────────────────────────────────────────────────────────────────────
    #  PAGE 1 — SOCIAL MEDIA FEED
    # ─────────────────────────────────────────────────────────────────────

    @app.get("/api/feed", tags=["Feed"], summary="Paginated social media posts")
    def list_feed(
        platform: Optional[str] = Query(None, description="Filter by platform: twitter, facebook, youtube, news, instagram"),
        disaster_type: Optional[str] = Query(None, description="Filter by disaster type: flood, cyclone, earthquake, ..."),
        urgency: Optional[str] = Query(None, description="Filter by urgency: critical, high, medium, low"),
        disasters_only: bool = Query(False, description="Show only disaster-classified posts"),
        language: Optional[str] = Query(None, description="Filter by language: english, hindi, ..."),
        search: Optional[str] = Query(None, description="Free-text search in post content"),
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ):
        """
        Returns a paginated list of social media posts, newest first.
        Supports filtering by platform, disaster type, urgency, and free-text search.
        """
        results = feed_store.query(
            platform=platform,
            disaster_type=disaster_type,
            urgency=urgency,
            is_disaster=True if disasters_only else None,
            language=language,
            search=search,
            limit=limit,
            offset=offset,
        )
        return _ok(results, count=len(results), total=feed_store.count)

    @app.get("/api/feed/platforms", tags=["Feed"], summary="Platform breakdown")
    def feed_platforms():
        """Returns available platforms and the count of posts for each."""
        return _ok(feed_store.platform_counts())

    @app.get("/api/feed/trending", tags=["Feed"], summary="Trending hashtags")
    def feed_trending(top_n: int = Query(15, ge=1, le=50)):
        """Returns the most-used hashtags across all feed posts."""
        return _ok(feed_store.trending_hashtags(top_n))

    @app.get("/api/feed/{post_id}", tags=["Feed"], summary="Single post detail")
    def get_feed_post(post_id: str):
        """Returns full detail for a single social media post."""
        post = feed_store.get(post_id)
        if not post:
            raise HTTPException(404, detail="Post not found")
        return _ok(post)

    # ─────────────────────────────────────────────────────────────────────
    #  PAGE 2 — DISASTER MAP (pins on India)
    # ─────────────────────────────────────────────────────────────────────

    @app.get("/api/map/reports", tags=["Map"], summary="All map pins (lightweight)")
    def map_pins(
        disaster_type: Optional[str] = Query(None, description="Filter by disaster type"),
        severity: Optional[str] = Query(None, description="Filter by severity"),
        state: Optional[str] = Query(None, description="Filter by Indian state"),
    ):
        """
        Returns lightweight pin data for every verified report — enough to
        render markers on the map without loading full detail.

        Each pin includes: id, lat, lng, disaster_type, severity, title,
        location_name, source_count, created_at.
        """
        reports = report_store.query(
            disaster_type=disaster_type,
            severity=severity,
            state=state,
        )
        pins = []
        for r in reports:
            loc = r.get("location", {})
            pins.append({
                "id": r["id"],
                "lat": loc.get("lat"),
                "lng": loc.get("lng"),
                "disaster_type": r["disaster_type"],
                "severity": r["severity"],
                "title": r["title"],
                "location_name": loc.get("name", ""),
                "source_count": r.get("verification", {}).get("source_count", 0),
                "created_at": r["created_at"],
            })
        return _ok(pins, count=len(pins))

    @app.get("/api/map/reports/{report_id}", tags=["Map"], summary="Full report card")
    def map_report_detail(report_id: str):
        """
        Returns the full report card for a single disaster report.
        Contains description, impact assessment, verification details,
        source posts, timeline, and response actions.
        Triggered when user clicks a map pin.
        """
        report = report_store.get(report_id)
        if not report:
            raise HTTPException(404, detail="Report not found")
        return _ok(report)

    @app.get("/api/map/clusters", tags=["Map"], summary="Clustered pins for zoom")
    def map_clusters(
        zoom: int = Query(5, ge=1, le=18, description="Map zoom level (higher = more granular)"),
    ):
        """
        Returns clustered report pins for a given zoom level.
        At low zoom (zoomed-out), nearby reports are merged into clusters
        with a count and dominant disaster type.
        """
        reports = report_store.all()
        if not reports:
            return _ok([], count=0)

        # Grid-based clustering: cell size decreases as zoom increases
        cell_size = 180.0 / (2 ** zoom)  # degrees
        buckets: Dict[str, List[Dict]] = defaultdict(list)
        for r in reports:
            loc = r.get("location", {})
            lat, lng = loc.get("lat", 0), loc.get("lng", 0)
            key = f"{int(lat / cell_size)}_{int(lng / cell_size)}"
            buckets[key].append(r)

        clusters = []
        for _, group in buckets.items():
            lats = [r["location"]["lat"] for r in group]
            lngs = [r["location"]["lng"] for r in group]
            types = [r["disaster_type"] for r in group]
            severities = [r["severity"] for r in group]
            # Dominant type = most frequent
            type_counts = defaultdict(int)
            for t in types:
                type_counts[t] += 1
            dominant = max(type_counts, key=type_counts.get)
            # Worst severity in cluster
            sev_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
            worst_sev = min(severities, key=lambda s: sev_rank.get(s, 4))
            clusters.append({
                "lat": sum(lats) / len(lats),
                "lng": sum(lngs) / len(lngs),
                "count": len(group),
                "dominant_type": dominant,
                "severity": worst_sev,
                "report_ids": [r["id"] for r in group],
            })
        return _ok(clusters, count=len(clusters))

    @app.get("/api/map/heatmap", tags=["Map"], summary="Heatmap density data")
    def map_heatmap():
        """
        Returns an array of [lat, lng, intensity] points for rendering
        a heat-map layer on the frontend map.
        Intensity is based on severity: critical=1.0, high=0.7, medium=0.4, low=0.2.
        """
        intensity_map = {"critical": 1.0, "high": 0.7, "medium": 0.4, "low": 0.2}
        points = []
        for r in report_store.all():
            loc = r.get("location", {})
            sev = r.get("severity", "low")
            points.append([loc.get("lat", 0), loc.get("lng", 0), intensity_map.get(sev, 0.2)])
        return _ok(points, count=len(points))

    @app.get("/api/map/states", tags=["Map"], summary="Per-state disaster summary")
    def map_states():
        """
        Aggregated disaster statistics per Indian state.
        Useful for a choropleth layer or state-level summary sidebar.
        """
        state_data: Dict[str, Dict] = defaultdict(lambda: {
            "total_reports": 0,
            "active_reports": 0,
            "disaster_types": defaultdict(int),
            "severity_breakdown": defaultdict(int),
            "latest_report_at": None,
        })
        for r in report_store.all():
            st = r.get("location", {}).get("state", "Unknown")
            d = state_data[st]
            d["total_reports"] += 1
            d["active_reports"] += 1   # Treat all seeded reports as active
            d["disaster_types"][r["disaster_type"]] += 1
            d["severity_breakdown"][r["severity"]] += 1
            ts = r.get("created_at", "")
            if not d["latest_report_at"] or ts > d["latest_report_at"]:
                d["latest_report_at"] = ts

        result = []
        for state, d in sorted(state_data.items()):
            result.append({
                "state": state,
                "total_reports": d["total_reports"],
                "active_reports": d["active_reports"],
                "disaster_types": dict(d["disaster_types"]),
                "severity_breakdown": dict(d["severity_breakdown"]),
                "latest_report_at": d["latest_report_at"],
            })
        return _ok(result, count=len(result))

    @app.post("/api/map/reports", tags=["Map"], summary="Create / update a verified report")
    def create_report(req: CreateReportRequest):
        """
        Admin endpoint to upsert a verified disaster report.
        If `location_name` matches a known Indian city, lat/lng/state auto-fill.
        """
        loc = INDIA_LOCATIONS.get(req.location_name)
        lat = req.lat or (loc["lat"] if loc else 20.5)
        lng = req.lng or (loc["lng"] if loc else 78.9)
        state = req.state or (loc["state"] if loc else "Unknown")

        report = {
            "id": f"rpt_{uuid.uuid4().hex[:12]}",
            "title": req.title,
            "description": req.description,
            "disaster_type": req.disaster_type,
            "severity": req.severity,
            "location": {"name": req.location_name, "state": state, "lat": lat, "lng": lng},
            "verification": {
                "status": "verified",
                "confidence": 0.90,
                "source_count": len(req.source_post_ids),
                "platforms": [],
                "verified_by": "manual",
            },
            "impact": {
                "affected_area_km2": req.affected_area_km2,
                "estimated_affected_people": req.estimated_affected_people,
                "casualties": req.casualties.dict() if req.casualties else {"deaths": 0, "injured": 0, "missing": 0, "rescued": 0, "displaced": 0},
                "infrastructure_damage": req.infrastructure_damage,
                "economic_loss_crore": 0.0,
            },
            "sources": [],
            "timeline": [
                {"timestamp": datetime.now(timezone.utc).isoformat(), "event": "Report created manually"},
            ],
            "response_actions": req.response_actions,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }

        # Link source posts
        for pid in req.source_post_ids:
            fp = feed_store.get(pid)
            if fp:
                report["sources"].append({
                    "platform": fp["platform"],
                    "post_id": fp["id"],
                    "snippet": (fp.get("content", {}).get("text") or fp.get("content", {}).get("headline") or "")[:120],
                    "author": fp.get("author", {}).get("name", ""),
                    "timestamp": fp["timestamp"],
                })
                if fp["platform"] not in report["verification"]["platforms"]:
                    report["verification"]["platforms"].append(fp["platform"])

        report_store.add(report)
        return _ok(report, message="Report created")

    # ─────────────────────────────────────────────────────────────────────
    #  PUSH ENDPOINT — for pipeline to push new processed posts
    # ─────────────────────────────────────────────────────────────────────

    @app.post("/api/feed/push", tags=["Feed"], summary="Push a new post (internal)")
    async def push_feed_post(post: Dict = Body(...)):
        """
        Internal endpoint for the processing pipeline to push new posts
        into the feed store and broadcast to WebSocket subscribers.
        """
        if "id" not in post:
            post["id"] = f"push_{uuid.uuid4().hex[:12]}"
        feed_store.add(post)
        await ws_mgr.broadcast({"type": "new_post", "post": post})
        return _ok({"id": post["id"]}, message="Post added and broadcast")

    # ─────────────────────────────────────────────────────────────────────
    #  WEBSOCKET — real-time feed stream
    # ─────────────────────────────────────────────────────────────────────

    @app.websocket("/ws/feed", name="ws_feed")
    async def ws_feed(ws: WebSocket):
        """
        WebSocket endpoint for real-time social media feed.

        After connection, the server pushes:
          { "type": "new_post", "post": { ... } }

        Client can send:
          "ping"          → receives { "type": "pong" }
          JSON filters    → receives only matching posts (TODO)
        """
        await ws_mgr.connect(ws)
        try:
            while True:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_json({"type": "pong"})
        except WebSocketDisconnect:
            ws_mgr.disconnect(ws)

    return app


# ═══════════════════════════════════════════════════════════════════════════
# MODULE-LEVEL APP INSTANCE (for uvicorn)
# ═══════════════════════════════════════════════════════════════════════════

app = create_app()


# ═══════════════════════════════════════════════════════════════════════════
# STANDALONE RUNNER
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 64)
    print("  SENTINEL — Social Feed & Disaster Map API")
    print("  " + "─" * 60)
    print("  Swagger UI     : http://localhost:8000/docs")
    print("  Health         : http://localhost:8000/api/health")
    print("  Feed           : http://localhost:8000/api/feed")
    print("  Map pins       : http://localhost:8000/api/map/reports")
    print("  Report detail  : http://localhost:8000/api/map/reports/{id}")
    print("  WebSocket      : ws://localhost:8000/ws/feed")
    print("=" * 64 + "\n")

    uvicorn.run("backend.feed_and_map_api:app", host="0.0.0.0", port=8000, reload=True)
