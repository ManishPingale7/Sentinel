# Sentinel — Complete Frontend Developer Guide

> **Give this entire document to your AI agent.** It contains every API endpoint, data shape, TypeScript interface, setup instruction, and design reference needed to build the Next.js frontend.

---

## Table of Contents

1. [What Is Sentinel](#1-what-is-sentinel)
2. [Pages to Build](#2-pages-to-build)
3. [Backend Setup (Run Before Frontend)](#3-backend-setup)
4. [API 1 — Social Media Feed & Disaster Map](#4-api-1--social-media-feed--disaster-map)
5. [API 2 — Satellite Flood Detection & Maps](#5-api-2--satellite-flood-detection--maps)
6. [TypeScript Interfaces](#6-typescript-interfaces)
7. [Complete API Reference Table](#7-complete-api-reference-table)
8. [Detailed Response Examples](#8-detailed-response-examples)
9. [WebSocket Integration](#9-websocket-integration)
10. [Recommended Libraries & Design](#10-recommended-libraries--design)
11. [Page-by-Page Build Guide](#11-page-by-page-build-guide)
12. [Color Schemes & Icons](#12-color-schemes--icons)
13. [End-to-End Workflow](#13-end-to-end-workflow)

---

## 1. What Is Sentinel

Sentinel is a **disaster monitoring system** with two independent capabilities:

**A) Social Media Intelligence** — Ingests posts from Twitter, Facebook, YouTube, News outlets, and Instagram. Each post is analyzed by an NLP pipeline that classifies disaster type, urgency, credibility, and sentiment. Verified disaster reports are aggregated and pinned on a map of India.

**B) Satellite Flood Detection** — Uses AI models (NASA Prithvi + Microsoft AI4Good) to analyze Sentinel satellite imagery and detect flood extent. Compares "before" and "during" images to map new flooding. Serves GeoJSON polygons and satellite imagery via API.

The backend is **100% built and running**. Your job is the **Next.js frontend only**.

---

## 2. Pages to Build

### Page 1: Social Media Feed
A scrollable, filterable feed of social media posts from multiple platforms. Each card shows the post content, author info, engagement metrics, and an AI analysis badge (disaster type, urgency, confidence). Supports real-time updates via WebSocket.

### Page 2: Disaster Map of India
A full-screen interactive map of India with pins/markers for verified disaster reports. Clicking a pin opens a detailed report card showing impact assessment, casualties, timeline, source posts, and response actions. Supports heatmap layer, clustering, and state-level choropleth.

### Page 3: Satellite Flood Map (Optional but impressive)
A map showing real satellite flood detection results from 3 CEMS emergency events (Croatia, Italy, Spain). Renders GeoJSON flood polygons (blue = existing water, red = new flooding) overlaid on satellite imagery. Shows before/after image comparison and area statistics.

### Page 4: Dashboard / Landing (Optional)
A landing page showing global stats: total posts monitored, disaster posts detected, verified reports, critical alerts, platform breakdown, trending hashtags.

---

## 3. Backend Setup

The frontend developer needs to start the backend APIs before the frontend can fetch data.

### Prerequisites
- **Python 3.10+** installed
- **Git clone** the project to local machine
- The project uses a Python virtual environment at `myenv/`

### Start API Server 1 — Social Media Feed + Disaster Map (Port 8000)

```bash
cd G:\Sentinel
myenv\Scripts\activate          # Windows
# source myenv/bin/activate     # Mac/Linux

uvicorn backend.feed_and_map_api:app --reload --port 8000
```

Verify: Open http://localhost:8000/docs — you should see Swagger UI.
Quick test: http://localhost:8000/api/health should return:
```json
{ "ok": true, "data": { "status": "healthy", "feed_posts": 120, "verified_reports": 10 } }
```

### Start API Server 2 — Satellite Flood Detection (Port 8001)

```bash
# In a SECOND terminal:
cd G:\Sentinel
myenv\Scripts\activate

uvicorn backend.api:app --reload --port 8001
```

Verify: http://localhost:8001/api/summary should return:
```json
{ "total_events": 3, "total_tiles": 44, "total_new_flood_km2": 537.57 }
```

### CORS
Both APIs allow `*` origins. No proxy config needed in Next.js during development.

### Environment Variables (suggested for Next.js `.env.local`)
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_FLOOD_API_URL=http://localhost:8001
NEXT_PUBLIC_MAPBOX_TOKEN=pk.your_mapbox_token_here
```

---

## 4. API 1 — Social Media Feed & Disaster Map

**Base URL:** `http://localhost:8000`
**Swagger:** http://localhost:8000/docs

All responses use this envelope:
```json
{
  "ok": true,
  "data": <payload>,
  "meta": {
    "timestamp": "2026-02-17T10:30:00+00:00",
    "count": 50,
    "total": 120
  }
}
```

### 4.1 Feed Endpoints (Page 1)

#### `GET /api/feed` — Paginated Social Media Posts

The primary endpoint for the feed page.

| Query Param | Type | Default | Options |
|-------------|------|---------|---------|
| `platform` | string | — | `twitter`, `facebook`, `youtube`, `news`, `instagram` |
| `disaster_type` | string | — | `flood`, `cyclone`, `earthquake`, `landslide`, `tsunami`, `drought`, `fire`, `heavy_rain` |
| `urgency` | string | — | `critical`, `high`, `medium`, `low` |
| `disasters_only` | bool | `false` | `true` to hide non-disaster posts |
| `language` | string | — | `english`, `hindi`, `marathi`, `tamil`, `telugu` |
| `search` | string | — | Free-text search in post content |
| `limit` | int | `50` | 1–200 |
| `offset` | int | `0` | Pagination offset |

**Response `data`:** Array of post objects (see TypeScript interfaces below).

#### `GET /api/feed/platforms` — Platform Breakdown
```json
{ "ok": true, "data": { "twitter": 35, "facebook": 20, "youtube": 15, "news": 25, "instagram": 25 } }
```

#### `GET /api/feed/trending` — Trending Hashtags
| Param | Type | Default |
|-------|------|---------|
| `top_n` | int | `15` |

```json
{ "ok": true, "data": [{ "hashtag": "#floodalert", "count": 18 }, { "hashtag": "#cyclonealert", "count": 12 }] }
```

#### `GET /api/feed/{post_id}` — Single Post Detail
Returns full post object. 404 if not found.

### 4.2 Map Endpoints (Page 2)

#### `GET /api/map/reports` — All Map Pins (Lightweight)

Returns lightweight pin data for rendering markers. Does NOT include full report detail.

| Query Param | Type | Description |
|-------------|------|-------------|
| `disaster_type` | string | Filter by type |
| `severity` | string | Filter by severity |
| `state` | string | Filter by Indian state name |

**Response `data`:**
```json
[
  {
    "id": "rpt_a1b2c3d4e5f6",
    "lat": 19.076,
    "lng": 72.877,
    "disaster_type": "flood",
    "severity": "critical",
    "title": "Severe Flooding in Mumbai - Red Alert",
    "location_name": "Mumbai",
    "source_count": 4,
    "created_at": "2026-02-17T05:30:00+00:00"
  }
]
```

#### `GET /api/map/reports/{report_id}` — Full Report Card

Called when user clicks a map pin. Returns detailed report.

**Response `data`:**
```json
{
  "id": "rpt_a1b2c3d4e5f6",
  "title": "Severe Flooding in Mumbai - Red Alert",
  "description": "Multiple areas of Mumbai experiencing severe waterlogging and flooding following 72 hours of continuous heavy rainfall. Mithi River has breached its banks. Over 50,000 people affected across Dharavi, Sion, Kurla and Andheri areas. NDRF has deployed 12 teams. BMC has opened 87 relief camps.",
  "disaster_type": "flood",
  "severity": "critical",
  "location": {
    "name": "Mumbai",
    "state": "Maharashtra",
    "lat": 19.076,
    "lng": 72.877
  },
  "verification": {
    "status": "verified",
    "confidence": 0.94,
    "source_count": 4,
    "platforms": ["twitter", "news", "facebook"],
    "verified_by": "cross_reference"
  },
  "impact": {
    "affected_area_km2": 45.5,
    "estimated_affected_people": 52000,
    "casualties": {
      "deaths": 8,
      "injured": 43,
      "missing": 5,
      "rescued": 1200,
      "displaced": 15000
    },
    "infrastructure_damage": ["roads", "railway_tracks", "bridges", "power_lines", "water_supply"],
    "economic_loss_crore": 850.0
  },
  "sources": [
    {
      "platform": "twitter",
      "post_id": "tw_abc123",
      "snippet": "🌊 Severe coastal flooding in Mumbai! Water levels rising fast...",
      "author": "NDRF India",
      "timestamp": "2026-02-17T06:00:00+00:00"
    }
  ],
  "timeline": [
    { "timestamp": "2026-02-17T04:00:00+00:00", "event": "First social media report detected" },
    { "timestamp": "2026-02-17T04:30:00+00:00", "event": "Cross-platform correlation confirmed" },
    { "timestamp": "2026-02-17T05:00:00+00:00", "event": "Official flood alert issued" },
    { "timestamp": "2026-02-17T05:30:00+00:00", "event": "Rescue operations initiated" },
    { "timestamp": "2026-02-17T08:00:00+00:00", "event": "Report verified and published" }
  ],
  "response_actions": [
    "NDRF teams deployed",
    "Relief camps opened",
    "Army on standby",
    "Emergency helpline activated: 1916",
    "Schools closed for 3 days"
  ],
  "created_at": "2026-02-17T05:30:00+00:00",
  "updated_at": "2026-02-17T09:00:00+00:00"
}
```

#### `GET /api/map/clusters` — Clustered Pins by Zoom

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `zoom` | int | `5` | Map zoom (1=world, 18=street). Higher = more granularity |

```json
{
  "ok": true,
  "data": [
    { "lat": 19.5, "lng": 73.1, "count": 3, "dominant_type": "flood", "severity": "critical", "report_ids": ["rpt_1", "rpt_2", "rpt_3"] }
  ]
}
```
Use at low zoom to show cluster bubbles. At high zoom (>10), switch to individual pins.

#### `GET /api/map/heatmap` — Heatmap Layer Data
```json
{ "ok": true, "data": [[19.076, 72.877, 1.0], [13.082, 80.270, 0.7], [22.572, 88.363, 0.4]] }
```
Each item: `[lat, lng, intensity]`. Intensity: critical=1.0, high=0.7, medium=0.4, low=0.2.

#### `GET /api/map/states` — State-Level Summary
```json
{
  "ok": true,
  "data": [
    {
      "state": "Maharashtra",
      "total_reports": 3,
      "active_reports": 3,
      "disaster_types": { "flood": 2, "drought": 1 },
      "severity_breakdown": { "critical": 1, "high": 1, "medium": 1 },
      "latest_report_at": "2026-02-17T09:00:00+00:00"
    }
  ]
}
```

#### `POST /api/map/reports` — Create a New Report

```json
{
  "title": "Flash Floods in Patna",
  "description": "Heavy monsoon rain causes flash flooding...",
  "disaster_type": "flood",
  "severity": "high",
  "location_name": "Patna",
  "affected_area_km2": 15.0,
  "estimated_affected_people": 20000,
  "casualties": { "deaths": 2, "injured": 15, "missing": 0, "rescued": 200, "displaced": 5000 },
  "infrastructure_damage": ["roads", "power_lines"],
  "source_post_ids": ["tw_abc123", "nw_def456"],
  "response_actions": ["NDRF deployed", "Relief camps opened"]
}
```
If `location_name` matches a known Indian city (35 cities built in), lat/lng/state auto-fill.

### 4.3 Shared Endpoints

#### `GET /api/health`
```json
{ "ok": true, "data": { "status": "healthy", "feed_posts": 120, "verified_reports": 10, "ws_connections": 2 } }
```

#### `GET /api/stats`
```json
{
  "ok": true,
  "data": {
    "total_posts": 120,
    "disaster_posts": 85,
    "verified_reports": 10,
    "critical_reports": 3,
    "platforms": { "twitter": 35, "facebook": 20, "youtube": 15, "news": 25, "instagram": 25 }
  }
}
```

---

## 5. API 2 — Satellite Flood Detection & Maps

**Base URL:** `http://localhost:8001`
**Swagger:** http://localhost:8001/docs

This API serves **real satellite imagery and flood prediction results** from 3 CEMS emergency events. Responses are raw JSON (no envelope wrapper).

### Available Events

| CEMS ID | Location | Country | Tiles | New Flood Detected |
|---------|----------|---------|-------|--------------------|
| EMSR766 | Danube River | Croatia/Serbia | 14 | 344.59 km² |
| EMSR771 | Po Valley | Italy | 15 | 51.07 km² |
| EMSR773 | Valencia | Spain | 15 | 141.91 km² |

### 5.1 Endpoints

#### `GET /api/summary` — Global Dashboard Stats
```json
{
  "total_events": 3,
  "total_tiles": 44,
  "total_new_flood_km2": 537.57,
  "total_flooded_km2": 648.27,
  "events": ["EMSR766", "EMSR771", "EMSR773"]
}
```

#### `GET /api/events` — List All Events with Map Coordinates
```json
[
  {
    "cems_id": "EMSR773",
    "tile_count": 15,
    "center": [39.40, -0.33],
    "bbox": [-0.59, 39.21, -0.27, 39.42],
    "total_new_flood_km2": 141.91,
    "total_flooded_km2": 224.84
  }
]
```
- `center` is `[lat, lng]` — use for placing markers
- `bbox` is `[west, south, east, north]` — use for `map.fitBounds()`

#### `GET /api/events/{cems_id}` — Event Detail with All Tiles
```json
{
  "cems_id": "EMSR773",
  "tile_count": 15,
  "bbox": [-0.59, 39.21, -0.27, 39.42],
  "tiles": [
    {
      "tile": "000006",
      "bounds": [-0.31, 39.25, -0.27, 39.28],
      "center": [39.26, -0.29],
      "stats": {
        "before_km2": 1.6168,
        "during_km2": 23.1053,
        "new_flood_km2": 21.5048,
        "pct_increase": 1330.08
      }
    }
  ]
}
```

#### `GET /api/events/{cems_id}/tiles` — List Tiles
```json
[
  { "tile": "000006", "bounds": [-0.31, 39.25, -0.27, 39.28], "new_flood_km2": 21.50, "pct_increase": 1330.08 }
]
```

#### `GET /api/events/{cems_id}/tiles/{tile}/geojson` — GeoJSON Flood Polygons ⭐

**This is the most important endpoint for map rendering.**

| Query Param | Values | Default | Description |
|-------------|--------|---------|-------------|
| `phase` | `before`, `during` | `during` | Temporal phase |
| `layer` | `flood`, `new_flood`, `change` | `flood` | What to render |

**Use `layer=change`** for the best visual. Returns two feature types:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "MultiPolygon", "coordinates": [...] },
      "properties": { "layer": "existing_water", "color": "#0066ff" }
    },
    {
      "type": "Feature",
      "geometry": { "type": "MultiPolygon", "coordinates": [...] },
      "properties": { "layer": "new_flood", "color": "#ff0000" }
    }
  ]
}
```

#### `GET /api/events/{cems_id}/geojson?layer=change` — Full Event GeoJSON (All Tiles Merged)

Same as above but merges ALL tiles in one response. Each feature also gets `tile`, `new_flood_km2`, `pct_increase` in properties.

**Best for:** Rendering the full flood extent of an event on one map layer.

#### `GET /api/events/{cems_id}/tiles/{tile}/stats` — Tile Change Metrics
```json
{
  "before_km2": 1.6168,
  "during_km2": 23.1053,
  "new_flood_km2": 21.5048,
  "receded_km2": 0.0163,
  "pct_increase": 1330.08,
  "bounds": [-0.31, 39.25, -0.27, 39.28]
}
```

#### `GET /api/events/{cems_id}/tiles/{tile}/rgb/{phase}` — Satellite Image PNG
- `phase` = `before` or `during`
- Returns **512×512 PNG image** (Sentinel-2 true-color)
- Use as image overlay or in before/after slider

#### `GET /api/events/{cems_id}/tiles/{tile}/mask/{phase}` — Flood Mask PNG
| Param | Values | Default | Description |
|-------|--------|---------|-------------|
| `overlay` | `true`, `false` | `false` | Composite mask on satellite image |

- `overlay=false` → transparent RGBA PNG (blue flood pixels, rest transparent) — layer on top of satellite
- `overlay=true` → RGB with blue flood overlay composited

#### `GET /api/events/{cems_id}/tiles/{tile}/change_overlay` — Composite Change PNG
Returns **512×512 PNG**: satellite RGB + blue (existing water) + red (new flood) composited.
Great for popups, panels, or before/after sliders.

---

## 6. TypeScript Interfaces

Copy these directly into your Next.js project (e.g., `types/sentinel.ts`):

```typescript
// ═══════════════════════════════════════════════════════
//  API Response Envelope (API 1 only)
// ═══════════════════════════════════════════════════════

interface ApiResponse<T> {
  ok: boolean;
  data: T;
  meta: {
    timestamp: string;
    count?: number;
    total?: number;
    message?: string;
  };
}

// ═══════════════════════════════════════════════════════
//  Social Media Feed (Page 1)
// ═══════════════════════════════════════════════════════

interface Author {
  name: string;
  handle: string;
  avatar_url: string | null;
  verified: boolean;
  followers: number;
  account_type: 'personal' | 'journalist' | 'government' | 'ngo' | 'news_outlet';
}

interface Engagement {
  likes: number;
  shares: number;
  comments: number;
  views: number;
}

interface PostLocation {
  name: string;
  lat: number;
  lng: number;
  state: string | null;
}

interface Analysis {
  is_disaster: boolean;
  disaster_type: DisasterType | null;
  confidence: number;        // 0.0 – 1.0
  urgency: Urgency;
  sentiment: 'negative' | 'neutral' | 'positive';
  credibility_score: number; // 0.0 – 1.0
}

interface FeedPostContent {
  text?: string;             // Twitter, Facebook, Instagram, YouTube description
  headline?: string;         // News articles
  title?: string;            // YouTube video title
  hashtags: string[];
  media_urls: string[];
}

interface FeedPost {
  id: string;                // e.g. "tw_a1b2c3", "fb_d4e5f6", "yt_...", "nw_...", "ig_..."
  platform: Platform;
  content: FeedPostContent;
  author: Author;
  engagement: Engagement;
  analysis: Analysis;
  location: PostLocation | null;
  timestamp: string;         // ISO 8601
  language: string;
}

// ═══════════════════════════════════════════════════════
//  Disaster Map (Page 2)
// ═══════════════════════════════════════════════════════

interface MapPin {
  id: string;
  lat: number;
  lng: number;
  disaster_type: DisasterType;
  severity: Severity;
  title: string;
  location_name: string;
  source_count: number;
  created_at: string;
}

interface ReportLocation {
  name: string;
  state: string;
  lat: number;
  lng: number;
}

interface Verification {
  status: 'verified' | 'unverified' | 'disputed';
  confidence: number;
  source_count: number;
  platforms: Platform[];
  verified_by: 'system' | 'manual' | 'cross_reference';
}

interface Casualties {
  deaths: number;
  injured: number;
  missing: number;
  rescued: number;
  displaced: number;
}

interface Impact {
  affected_area_km2: number;
  estimated_affected_people: number;
  casualties: Casualties;
  infrastructure_damage: string[];
  economic_loss_crore: number;
}

interface SourceSnippet {
  platform: Platform;
  post_id: string;
  snippet: string;
  author: string;
  timestamp: string;
}

interface TimelineEvent {
  timestamp: string;
  event: string;
}

interface MapReport {
  id: string;
  title: string;
  description: string;
  disaster_type: DisasterType;
  severity: Severity;
  location: ReportLocation;
  verification: Verification;
  impact: Impact;
  sources: SourceSnippet[];
  timeline: TimelineEvent[];
  response_actions: string[];
  created_at: string;
  updated_at: string;
}

interface ClusterPoint {
  lat: number;
  lng: number;
  count: number;
  dominant_type: DisasterType;
  severity: Severity;
  report_ids: string[];
}

interface StateStats {
  state: string;
  total_reports: number;
  active_reports: number;
  disaster_types: Record<string, number>;
  severity_breakdown: Record<string, number>;
  latest_report_at: string | null;
}

interface TrendingHashtag {
  hashtag: string;
  count: number;
}

// ═══════════════════════════════════════════════════════
//  Satellite Flood Detection (Page 3)
// ═══════════════════════════════════════════════════════

interface FloodSummary {
  total_events: number;
  total_tiles: number;
  total_new_flood_km2: number;
  total_flooded_km2: number;
  events: string[];
}

interface FloodEvent {
  cems_id: string;
  tile_count: number;
  center: [number, number];   // [lat, lng]
  bbox: [number, number, number, number]; // [west, south, east, north]
  total_new_flood_km2: number;
  total_flooded_km2: number;
}

interface TileStats {
  before_km2: number;
  during_km2: number;
  new_flood_km2: number;
  receded_km2?: number;
  pct_increase: number;
  bounds?: [number, number, number, number];
}

interface FloodEventDetail {
  cems_id: string;
  tile_count: number;
  bbox: [number, number, number, number];
  tiles: {
    tile: string;
    bounds: [number, number, number, number];
    center: [number, number];
    stats: TileStats;
  }[];
}

// ═══════════════════════════════════════════════════════
//  Enums / Union Types
// ═══════════════════════════════════════════════════════

type Platform = 'twitter' | 'facebook' | 'youtube' | 'news' | 'instagram';

type DisasterType = 'flood' | 'cyclone' | 'earthquake' | 'landslide' | 'tsunami' | 'drought' | 'fire' | 'heavy_rain';

type Severity = 'critical' | 'high' | 'medium' | 'low';

type Urgency = 'critical' | 'high' | 'medium' | 'low';

// ═══════════════════════════════════════════════════════
//  WebSocket Messages
// ═══════════════════════════════════════════════════════

interface WSNewPost {
  type: 'new_post';
  post: FeedPost;
}

interface WSPong {
  type: 'pong';
}

type WSMessage = WSNewPost | WSPong;
```

---

## 7. Complete API Reference Table

### API 1 — Social Media + Map (Port 8000)

| Method | Endpoint | Page | Description |
|--------|----------|------|-------------|
| GET | `/api/health` | Shared | Health check + counts |
| GET | `/api/stats` | Dashboard | Global statistics |
| GET | `/api/feed` | Feed | Paginated posts (filters: platform, disaster_type, urgency, language, search, disasters_only) |
| GET | `/api/feed/platforms` | Feed | Platform name → post count |
| GET | `/api/feed/trending` | Feed | Top hashtags |
| GET | `/api/feed/{post_id}` | Feed | Single post detail |
| POST | `/api/feed/push` | Internal | Pipeline pushes new post |
| WS | `/ws/feed` | Feed | Real-time post stream |
| GET | `/api/map/reports` | Map | Lightweight pins (filters: disaster_type, severity, state) |
| GET | `/api/map/reports/{id}` | Map | Full report card |
| GET | `/api/map/clusters?zoom=N` | Map | Clustered pins for zoom level |
| GET | `/api/map/heatmap` | Map | `[lat, lng, intensity]` array |
| GET | `/api/map/states` | Map | Per-state summary |
| POST | `/api/map/reports` | Map | Create new report |

### API 2 — Satellite Flood Detection (Port 8001)

| Method | Endpoint | Returns | Description |
|--------|----------|---------|-------------|
| GET | `/api/summary` | JSON | Global flood stats |
| GET | `/api/events` | JSON | All events with coordinates |
| GET | `/api/events/{cems_id}` | JSON | Event detail + all tiles |
| GET | `/api/events/{cems_id}/tiles` | JSON | Tile list with bounds |
| GET | `/api/events/{cems_id}/tiles/{tile}/geojson?phase=during&layer=change` | GeoJSON | **Flood polygons for map** |
| GET | `/api/events/{cems_id}/geojson?layer=change` | GeoJSON | **All tiles merged** |
| GET | `/api/events/{cems_id}/tiles/{tile}/stats` | JSON | Area metrics |
| GET | `/api/events/{cems_id}/tiles/{tile}/rgb/{phase}` | PNG | Satellite image |
| GET | `/api/events/{cems_id}/tiles/{tile}/mask/{phase}?overlay=false` | PNG | Flood mask |
| GET | `/api/events/{cems_id}/tiles/{tile}/change_overlay` | PNG | RGB + flood composite |

---

## 8. Detailed Response Examples

### Feed Post — Twitter (disaster)
```json
{
  "id": "tw_617be9e7e69d",
  "platform": "twitter",
  "content": {
    "text": "🌊 Severe coastal flooding in Mumbai! Water levels rising fast. Stay safe everyone. #FloodAlert #DisasterResponse",
    "hashtags": ["#FloodAlert", "#DisasterResponse"],
    "media_urls": []
  },
  "author": {
    "name": "NDRF India",
    "handle": "@ndrfindia",
    "avatar_url": "https://ui-avatars.com/api/?name=NDRF+India&background=random",
    "verified": true,
    "followers": 450000,
    "account_type": "government"
  },
  "engagement": { "likes": 4520, "shares": 1800, "comments": 320, "views": 125000 },
  "analysis": {
    "is_disaster": true,
    "disaster_type": "flood",
    "confidence": 0.94,
    "urgency": "critical",
    "sentiment": "negative",
    "credibility_score": 0.92
  },
  "location": { "name": "Mumbai", "lat": 19.076, "lng": 72.877, "state": "Maharashtra" },
  "timestamp": "2026-02-17T08:15:00+00:00",
  "language": "english"
}
```

### Feed Post — YouTube (non-disaster)
```json
{
  "id": "yt_abc123def456",
  "platform": "youtube",
  "content": {
    "title": "Chennai Street Food Tour | Best Places to Eat",
    "text": "Exploring the amazing street food scene in Chennai!",
    "hashtags": ["#Chennai", "#StreetFood"],
    "media_urls": ["https://img.youtube.com/vi/placeholder/hqdefault.jpg"]
  },
  "author": {
    "name": "Foodie Traveler",
    "handle": "@foodietraveler",
    "avatar_url": "https://ui-avatars.com/api/?name=Foodie+Traveler&background=random",
    "verified": false,
    "followers": 12500,
    "account_type": "personal"
  },
  "engagement": { "likes": 850, "shares": 120, "comments": 95, "views": 45000 },
  "analysis": {
    "is_disaster": false,
    "disaster_type": null,
    "confidence": 0.08,
    "urgency": "low",
    "sentiment": "positive",
    "credibility_score": 0.65
  },
  "location": { "name": "Chennai", "lat": 13.082, "lng": 80.270, "state": "Tamil Nadu" },
  "timestamp": "2026-02-17T06:30:00+00:00",
  "language": "english"
}
```

### Feed Post — News (disaster)
```json
{
  "id": "nw_d38334d7d3d3",
  "platform": "news",
  "content": {
    "headline": "Mumbai floods: Death toll rises to 26, thousands displaced",
    "text": "The situation in Mumbai remains grim as rescue operations continue for the fourth day. NDRF boats are being used to evacuate stranded families.",
    "hashtags": ["#Mumbai"],
    "media_urls": []
  },
  "author": {
    "name": "NDTV",
    "handle": "@ndtv",
    "avatar_url": null,
    "verified": true,
    "followers": 3500000,
    "account_type": "news_outlet"
  },
  "engagement": { "likes": 8500, "shares": 4200, "comments": 1100, "views": 520000 },
  "analysis": {
    "is_disaster": true,
    "disaster_type": "flood",
    "confidence": 0.97,
    "urgency": "critical",
    "sentiment": "negative",
    "credibility_score": 0.95
  },
  "location": { "name": "Mumbai", "lat": 19.076, "lng": 72.877, "state": "Maharashtra" },
  "timestamp": "2026-02-17T09:00:00+00:00",
  "language": "english"
}
```

---

## 9. WebSocket Integration

### Real-Time Feed (API 1)

```typescript
// hooks/useRealtimeFeed.ts
import { useEffect, useRef, useCallback, useState } from 'react';

export function useRealtimeFeed(onNewPost: (post: FeedPost) => void) {
  const wsRef = useRef<WebSocket | null>(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws/feed');
    wsRef.current = ws;

    ws.onopen = () => setConnected(true);
    ws.onclose = () => {
      setConnected(false);
      // Reconnect after 3 seconds
      setTimeout(() => wsRef.current = new WebSocket('ws://localhost:8000/ws/feed'), 3000);
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (data.type === 'new_post') {
        onNewPost(data.post);
      }
    };

    // Keep alive
    const interval = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) ws.send('ping');
    }, 30000);

    return () => {
      clearInterval(interval);
      ws.close();
    };
  }, [onNewPost]);

  return { connected };
}
```

### Usage in a Feed Page
```tsx
const [posts, setPosts] = useState<FeedPost[]>([]);

const handleNewPost = useCallback((post: FeedPost) => {
  setPosts(prev => [post, ...prev].slice(0, 200)); // Keep latest 200
}, []);

const { connected } = useRealtimeFeed(handleNewPost);
```

---

## 10. Recommended Libraries & Design

### NPM Packages

```bash
# Map (choose one)
npm install react-leaflet leaflet                   # Lightweight, free
npm install react-map-gl mapbox-gl                  # Polished, needs token

# Charts
npm install recharts                                # Simple bar/pie/line charts
# OR
npm install chart.js react-chartjs-2                # More chart types

# UI
npm install @radix-ui/react-dialog                  # Modals/cards for report detail
npm install lucide-react                            # Icons
npm install framer-motion                           # Animations
npm install date-fns                                # Date formatting
npm install clsx tailwind-merge                     # Tailwind utilities

# Data fetching
npm install @tanstack/react-query                   # Server state management
# OR use Next.js built-in fetch with SWR
npm install swr
```

### Map of India — Center & Bounds

```typescript
// Map default view — centered on India
const INDIA_CENTER = { lat: 22.5, lng: 82.0 };
const INDIA_ZOOM = 5;
const INDIA_BOUNDS = {
  south: 6.0,
  west: 68.0,
  north: 37.0,
  east: 98.0,
};
```

### Suggested Next.js Project Structure

```
app/
├── layout.tsx                  # Root layout with nav
├── page.tsx                    # Dashboard / landing
├── feed/
│   └── page.tsx                # Social media feed (Page 1)
├── map/
│   └── page.tsx                # Disaster map of India (Page 2)
├── satellite/
│   └── page.tsx                # Satellite flood map (Page 3, optional)
├── api/                        # (empty if using external backend)
components/
├── feed/
│   ├── FeedCard.tsx            # Single post card
│   ├── FeedFilters.tsx         # Platform/type/urgency filters
│   ├── TrendingHashtags.tsx    # Sidebar trending
│   └── PlatformIcon.tsx        # Twitter/FB/YT/News/IG icons
├── map/
│   ├── IndiaMap.tsx            # Map component
│   ├── MapPin.tsx              # Custom marker
│   ├── ReportCard.tsx          # Report detail modal/drawer
│   ├── StatePanel.tsx          # State-level sidebar
│   └── HeatmapLayer.tsx       # Heatmap overlay
├── satellite/
│   ├── FloodMap.tsx            # Satellite flood map
│   ├── EventSelector.tsx       # EMSR event picker
│   ├── TileStats.tsx           # Stats panel
│   └── BeforeAfter.tsx         # Image comparison slider
├── shared/
│   ├── Navbar.tsx              # Navigation
│   ├── StatsCard.tsx           # Metric card
│   └── SeverityBadge.tsx       # Color-coded badge
lib/
├── api.ts                      # API fetch functions
├── types.ts                    # TypeScript interfaces (from section 6)
└── constants.ts                # Colors, map config
```

---

## 11. Page-by-Page Build Guide

### Page 1: Social Media Feed

**Layout:**
```
┌──────────────────────────────────────────────────┐
│  Navbar: [Dashboard] [Feed] [Map] [Satellite]    │
├──────────┬───────────────────────────────────────┤
│ Filters  │  Feed Cards (scrollable)              │
│          │  ┌─────────────────────────────────┐  │
│ Platform │  │ 🐦 @ndrfindia  ✓  •  2h ago    │  │
│ ☐ Twitter│  │ 🌊 Severe coastal flooding in   │  │
│ ☐ FB     │  │ Mumbai! Water levels rising...   │  │
│ ☐ YT     │  │ ❤ 4520  🔁 1800  💬 320        │  │
│ ☐ News   │  │ 🔴 CRITICAL  Flood  94% conf   │  │
│ ☐ Insta  │  └─────────────────────────────────┘  │
│          │  ┌─────────────────────────────────┐  │
│ Urgency  │  │ 📰 NDTV  ✓  •  3h ago          │  │
│ ☐ Crit   │  │ Mumbai floods: Death toll rises │  │
│ ☐ High   │  │ to 26, thousands displaced...   │  │
│ ☐ Medium │  │ ❤ 8500  🔁 4200  💬 1100       │  │
│ ☐ Low    │  │ 🔴 CRITICAL  Flood  97% conf   │  │
│          │  └─────────────────────────────────┘  │
│ Trending │  ┌─────────────────────────────────┐  │
│ #flood.. │  │ 📷 @priya_patel  •  5h ago      │  │
│ #cyclone │  │ Beautiful sunrise today 🌅       │  │
│ #mumbai  │  │ ❤ 230  🔁 12  💬 8             │  │
│          │  │ 🟢 LOW  Non-disaster  8% conf   │  │
│ 🟢 Live  │  └─────────────────────────────────┘  │
├──────────┴───────────────────────────────────────┤
│  Showing 50 of 120 posts  •  [Load More]         │
└──────────────────────────────────────────────────┘
```

**Data flow:**
1. On page load: `GET /api/feed?limit=50` + `GET /api/feed/platforms` + `GET /api/feed/trending`
2. When filter changes: `GET /api/feed?platform=twitter&urgency=critical&limit=50`
3. Real-time: Connect WebSocket `ws://localhost:8000/ws/feed`, prepend new posts
4. On "Load More": `GET /api/feed?offset=50&limit=50`

**Card rendering logic:**
- `post.platform` → show platform icon (Twitter bird, FB logo, etc.)
- `post.author.verified` → show checkmark badge
- `post.analysis.is_disaster` → add red/orange border
- `post.analysis.urgency` → severity badge color
- `post.content.text || post.content.headline` → main text
- `post.engagement` → like/share/comment counts
- `post.location` → small location tag if available
- News posts: use `content.headline` as bold title, `content.text` as body

### Page 2: Disaster Map of India

**Layout:**
```
┌────────────────────────────────────────────────────────┐
│  Navbar   │  Filters: [Type ▾] [Severity ▾] [State ▾] │
├───────────┴────────────────────────┬───────────────────┤
│                                    │                   │
│         INDIA MAP                  │  Report Card      │
│     (full height)                  │  (when pin clicked)│
│                                    │                   │
│     📍 Mumbai (🔴 critical)       │  ┌──────────────┐ │
│     📍 Paradip (🔴 critical)      │  │ 🔴 CRITICAL  │ │
│     📍 Delhi (🟠 high)            │  │ Flood        │ │
│     📍 Wayanad (🔴 critical)      │  │ Mumbai, MH   │ │
│     📍 Chennai (🟠 high)          │  │              │ │
│     📍 Uttarkashi (🟠 high)       │  │ 52,000 ppl   │ │
│     📍 Nagpur (🟡 medium)         │  │ 45.5 km²     │ │
│     📍 Kolkata (🟡 medium)        │  │ 8 deaths     │ │
│     📍 Surat (🟠 high)            │  │              │ │
│     📍 Guwahati (🟠 high)         │  │ ✅ Verified  │ │
│                                    │  │ 4 sources    │ │
│  [Toggle: Pins | Heatmap | Both]   │  │              │ │
│                                    │  │ Timeline...  │ │
│                                    │  └──────────────┘ │
├────────────────────────────────────┴───────────────────┤
│  10 verified reports  •  3 critical  •  4 high         │
└────────────────────────────────────────────────────────┘
```

**Data flow:**
1. On load: `GET /api/map/reports` → place all pins on map
2. On pin click: `GET /api/map/reports/{id}` → show report card in sidebar/modal
3. On zoom change: `GET /api/map/clusters?zoom={currentZoom}` → cluster nearby pins
4. For heatmap toggle: `GET /api/map/heatmap` → render heat-map layer
5. For state panel: `GET /api/map/states` → show in sidebar

**Pin icons by disaster type:**
- 🌊 Flood → Blue pin
- 🌀 Cyclone → Purple pin
- 🏔️ Earthquake → Brown pin
- ⛰️ Landslide → Orange pin
- 🌊 Tsunami → Cyan pin
- ☀️ Drought → Amber pin
- 🔥 Fire → Red pin
- 🌧️ Heavy Rain → Grey pin

### Page 3: Satellite Flood Map (Optional)

**Layout:**
```
┌──────────────────────────────────────────────────────┐
│  Navbar   │  Event: [EMSR766 ▾]  │  537 km² flooded  │
├───────────┴────────────────────────┬──────────────────┤
│                                    │ Stats Panel      │
│     WORLD MAP                      │ ┌──────────────┐│
│   (Mapbox/Leaflet)                 │ │ EMSR773      ││
│                                    │ │ Valencia, ES  ││
│   🔵 Blue = existing water        │ │ Tiles: 15    ││
│   🔴 Red = NEW flooding           │ │ New: 141 km² ││
│                                    │ │ +1330% ⬆     ││
│   [GeoJSON polygons overlaid]      │ └──────────────┘│
│                                    │                  │
│                                    │ Tile Detail      │
│                                    │ ┌──────────────┐│
│                                    │ │ Before|After  ││
│                                    │ │ [img slider]  ││
│                                    │ │              ││
│                                    │ │ Before: 1.6km²│
│                                    │ │ During: 23 km²│
│                                    │ │ New: 21.5 km² │
│                                    │ └──────────────┘│
├────────────────────────────────────┴──────────────────┤
│  3 events • 44 tiles • 537.57 km² new flooding        │
└──────────────────────────────────────────────────────┘
```

**Data flow:**
1. On load: `GET /api/events` (port 8001) → place event markers on world map
2. On event click: `GET /api/events/{cems_id}` → zoom to bbox, list tiles
3. `GET /api/events/{cems_id}/geojson?layer=change` → render flood polygons (blue + red)
4. On tile click: `GET /api/events/{cems_id}/tiles/{tile}/stats` → show stats panel
5. For before/after slider:
   - Before: `GET /api/events/{cems_id}/tiles/{tile}/rgb/before` (PNG)
   - After: `GET /api/events/{cems_id}/tiles/{tile}/change_overlay` (PNG)

**Map rendering with react-map-gl:**
```tsx
import Map, { Source, Layer } from 'react-map-gl';

// Fetch GeoJSON
const geojson = await fetch(`${FLOOD_API}/api/events/${cemsId}/geojson?layer=change`).then(r => r.json());

// Render
<Map initialViewState={{ longitude: -0.33, latitude: 39.40, zoom: 10 }}>
  <Source id="flood" type="geojson" data={geojson}>
    <Layer
      id="existing-water"
      type="fill"
      filter={['==', ['get', 'layer'], 'existing_water']}
      paint={{ 'fill-color': '#0066ff', 'fill-opacity': 0.5 }}
    />
    <Layer
      id="new-flood"
      type="fill"
      filter={['==', ['get', 'layer'], 'new_flood']}
      paint={{ 'fill-color': '#ff0000', 'fill-opacity': 0.6 }}
    />
  </Source>
</Map>
```

**Image overlays:**
```tsx
// Satellite images as <img> in a panel or popup
<img src={`${FLOOD_API}/api/events/${cemsId}/tiles/${tile}/rgb/before`} alt="Before" />
<img src={`${FLOOD_API}/api/events/${cemsId}/tiles/${tile}/change_overlay`} alt="After with flood overlay" />
```

---

## 12. Color Schemes & Icons

### Severity Colors
| Severity | Hex | Tailwind |
|----------|-----|----------|
| Critical | `#D32F2F` | `bg-red-700 text-white` |
| High | `#F57C00` | `bg-orange-600 text-white` |
| Medium | `#FBC02D` | `bg-yellow-500 text-black` |
| Low | `#388E3C` | `bg-green-700 text-white` |

### Disaster Type Colors
| Type | Hex | Emoji |
|------|-----|-------|
| Flood | `#2196F3` | 🌊 |
| Cyclone | `#9C27B0` | 🌀 |
| Earthquake | `#795548` | 🏔️ |
| Landslide | `#FF9800` | ⛰️ |
| Tsunami | `#00BCD4` | 🌊 |
| Drought | `#FFC107` | ☀️ |
| Fire | `#F44336` | 🔥 |
| Heavy Rain | `#607D8B` | 🌧️ |

### Platform Colors & Icons
| Platform | Color | Icon suggestion |
|----------|-------|-----------------|
| Twitter | `#1DA1F2` | Bird / X logo |
| Facebook | `#4267B2` | F logo |
| YouTube | `#FF0000` | Play button |
| News | `#333333` | Newspaper |
| Instagram | `#E4405F` | Camera |

### Sentiment Colors
| Sentiment | Color |
|-----------|-------|
| Negative | `text-red-500` |
| Neutral | `text-gray-500` |
| Positive | `text-green-500` |

---

## 13. End-to-End Workflow

### How the system works (for context):

```
┌─────────────────────────────────────────────────────────────────┐
│                        SENTINEL SYSTEM                         │
│                                                                 │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────┐    │
│  │ Simulator │───▶│ Redis Stream │───▶│ Pipeline Engine    │    │
│  │ (5 social │    │              │    │ (NLP + ML + LLM)   │    │
│  │ platforms)│    └──────────────┘    │                    │    │
│  └──────────┘                        │ 1. Disaster detect │    │
│                                      │ 2. Urgency score   │    │
│                                      │ 3. Credibility     │    │
│                                      │ 4. Location extract│    │
│                                      │ 5. Cross-reference │    │
│                                      └────────┬───────────┘    │
│                                               │                 │
│                    ┌──────────────────────────┼──────────┐      │
│                    ▼                          ▼          ▼      │
│           ┌──────────────┐          ┌──────────┐  ┌─────────┐  │
│           │ Feed API     │          │ Firebase │  │WebSocket│  │
│           │ (port 8000)  │          │ Storage  │  │broadcast│  │
│           │              │          └──────────┘  └────┬────┘  │
│           │ /api/feed    │                             │       │
│           │ /api/map/*   │                             │       │
│           └──────┬───────┘                             │       │
│                  │                                     │       │
│                  ▼                                     ▼       │
│           ┌─────────────────────────────────────────────┐      │
│           │           NEXT.JS FRONTEND (YOU)            │      │
│           │  Page 1: Social Media Feed                  │      │
│           │  Page 2: Disaster Map (India pins)          │      │
│           └─────────────────────────────────────────────┘      │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐      │
│  │ SATELLITE PIPELINE (separate)                        │      │
│  │                                                      │      │
│  │  Sentinel-2 imagery → Prithvi AI model → flood mask  │      │
│  │  Sentinel-1 SAR     → AI4Good model   → flood mask   │      │
│  │                                                      │      │
│  │  ┌─────────────┐                                    │      │
│  │  │ Flood API   │──▶ NEXT.JS Page 3: Satellite Map   │      │
│  │  │ (port 8001) │                                    │      │
│  │  └─────────────┘                                    │      │
│  └──────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

### What data is pre-loaded (no setup needed):
- **120 social media posts** across all 5 platforms with realistic Indian disaster content
- **10 verified disaster reports** pinned across India (Mumbai floods, Odisha cyclone, Delhi earthquake, Wayanad landslide, Chennai floods, Marathwada drought, Kolkata waterlogging, Surat fire, Guwahati floods, Uttarkashi landslide)
- **3 satellite flood events** with real GeoJSON polygons and satellite imagery (44 tiles, 537 km² flood area)
- **All data refreshes on server restart** (in-memory stores with seed data)

### Quick Verification Checklist

After starting both servers, verify these URLs in your browser:

| URL | Expected |
|-----|----------|
| http://localhost:8000/docs | Swagger UI for Feed+Map API |
| http://localhost:8000/api/health | `{ ok: true, data: { feed_posts: 120, verified_reports: 10 } }` |
| http://localhost:8000/api/feed?limit=3 | 3 social media posts |
| http://localhost:8000/api/map/reports | 10 map pins with lat/lng |
| http://localhost:8001/docs | Swagger UI for Flood API |
| http://localhost:8001/api/summary | `{ total_events: 3, total_tiles: 44 }` |
| http://localhost:8001/api/events | 3 events with coordinates |

### API Fetch Utility (copy to `lib/api.ts`)

```typescript
const API = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
const FLOOD_API = process.env.NEXT_PUBLIC_FLOOD_API_URL || 'http://localhost:8001';

// ── Feed (Page 1) ──

export async function getFeed(params?: Record<string, string | number | boolean>) {
  const query = params ? '?' + new URLSearchParams(params as any).toString() : '';
  const res = await fetch(`${API}/api/feed${query}`);
  return res.json() as Promise<ApiResponse<FeedPost[]>>;
}

export async function getFeedPost(id: string) {
  const res = await fetch(`${API}/api/feed/${id}`);
  return res.json() as Promise<ApiResponse<FeedPost>>;
}

export async function getPlatforms() {
  const res = await fetch(`${API}/api/feed/platforms`);
  return res.json() as Promise<ApiResponse<Record<string, number>>>;
}

export async function getTrending(topN = 15) {
  const res = await fetch(`${API}/api/feed/trending?top_n=${topN}`);
  return res.json() as Promise<ApiResponse<TrendingHashtag[]>>;
}

// ── Map (Page 2) ──

export async function getMapPins(params?: Record<string, string>) {
  const query = params ? '?' + new URLSearchParams(params).toString() : '';
  const res = await fetch(`${API}/api/map/reports${query}`);
  return res.json() as Promise<ApiResponse<MapPin[]>>;
}

export async function getReportDetail(id: string) {
  const res = await fetch(`${API}/api/map/reports/${id}`);
  return res.json() as Promise<ApiResponse<MapReport>>;
}

export async function getMapClusters(zoom: number) {
  const res = await fetch(`${API}/api/map/clusters?zoom=${zoom}`);
  return res.json() as Promise<ApiResponse<ClusterPoint[]>>;
}

export async function getHeatmapData() {
  const res = await fetch(`${API}/api/map/heatmap`);
  return res.json() as Promise<ApiResponse<[number, number, number][]>>;
}

export async function getStateStats() {
  const res = await fetch(`${API}/api/map/states`);
  return res.json() as Promise<ApiResponse<StateStats[]>>;
}

// ── Dashboard ──

export async function getStats() {
  const res = await fetch(`${API}/api/stats`);
  return res.json();
}

export async function getHealth() {
  const res = await fetch(`${API}/api/health`);
  return res.json();
}

// ── Satellite Flood (Page 3) ──

export async function getFloodSummary() {
  const res = await fetch(`${FLOOD_API}/api/summary`);
  return res.json() as Promise<FloodSummary>;
}

export async function getFloodEvents() {
  const res = await fetch(`${FLOOD_API}/api/events`);
  return res.json() as Promise<FloodEvent[]>;
}

export async function getFloodEventDetail(cemsId: string) {
  const res = await fetch(`${FLOOD_API}/api/events/${cemsId}`);
  return res.json() as Promise<FloodEventDetail>;
}

export async function getFloodGeoJSON(cemsId: string, layer = 'change') {
  const res = await fetch(`${FLOOD_API}/api/events/${cemsId}/geojson?layer=${layer}`);
  return res.json() as Promise<GeoJSON.FeatureCollection>;
}

export async function getTileStats(cemsId: string, tile: string) {
  const res = await fetch(`${FLOOD_API}/api/events/${cemsId}/tiles/${tile}/stats`);
  return res.json() as Promise<TileStats>;
}

export function getSatelliteImageUrl(cemsId: string, tile: string, phase: 'before' | 'during') {
  return `${FLOOD_API}/api/events/${cemsId}/tiles/${tile}/rgb/${phase}`;
}

export function getFloodOverlayUrl(cemsId: string, tile: string) {
  return `${FLOOD_API}/api/events/${cemsId}/tiles/${tile}/change_overlay`;
}

export function getFloodMaskUrl(cemsId: string, tile: string, phase: 'before' | 'during', overlay = false) {
  return `${FLOOD_API}/api/events/${cemsId}/tiles/${tile}/mask/${phase}?overlay=${overlay}`;
}
```

---

**That's everything.** Both APIs are fully functional with pre-loaded demo data. Start the two servers and begin building the frontend. Use http://localhost:8000/docs and http://localhost:8001/docs to explore the APIs interactively.
