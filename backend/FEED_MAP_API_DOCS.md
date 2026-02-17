# Sentinel — Social Feed & Disaster Map API Documentation

**Base URL:** `http://localhost:8000`  
**Swagger UI:** `http://localhost:8000/docs`  
**Version:** 2.0.0

## Quick Start

```bash
cd G:\Sentinel
api_env\Scripts\activate
uvicorn backend.feed_and_map_api:app --reload --port 8000
```

---

## Response Format

All endpoints return a consistent JSON envelope:

```json
{
  "ok": true,
  "data": ...,
  "meta": {
    "timestamp": "2026-02-17T10:30:00+00:00",
    "count": 50,
    "total": 120
  }
}
```

---

## Page 1 — Social Media Feed

### `GET /api/feed` — Paginated Social Media Posts

The main endpoint for the feed page. Returns social media content from all platforms.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `platform` | string | — | Filter: `twitter`, `facebook`, `youtube`, `news`, `instagram` |
| `disaster_type` | string | — | Filter: `flood`, `cyclone`, `earthquake`, `landslide`, `tsunami`, `drought`, `fire`, `heavy_rain` |
| `urgency` | string | — | Filter: `critical`, `high`, `medium`, `low` |
| `disasters_only` | bool | `false` | Show only disaster-classified posts |
| `language` | string | — | Filter: `english`, `hindi`, `marathi`, `tamil`, `telugu` |
| `search` | string | — | Free-text search within post content |
| `limit` | int | `50` | Page size (1–200) |
| `offset` | int | `0` | Pagination offset |

**Response example:**

```json
{
  "ok": true,
  "data": [
    {
      "id": "tw_a1b2c3d4e5f6",
      "platform": "twitter",
      "content": {
        "text": "🌊 Severe coastal flooding in Mumbai! Water levels rising fast. #FloodAlert",
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
      "engagement": {
        "likes": 4520,
        "shares": 1800,
        "comments": 320,
        "views": 125000
      },
      "analysis": {
        "is_disaster": true,
        "disaster_type": "flood",
        "confidence": 0.94,
        "urgency": "critical",
        "sentiment": "negative",
        "credibility_score": 0.92
      },
      "location": {
        "name": "Mumbai",
        "lat": 19.0760,
        "lng": 72.8777,
        "state": "Maharashtra"
      },
      "timestamp": "2026-02-17T08:15:00+00:00",
      "language": "english"
    }
  ],
  "meta": { "count": 50, "total": 120, "timestamp": "..." }
}
```

**Frontend usage:**
- Render each post as a card with platform icon, author info, content, engagement stats
- Use `analysis.is_disaster` to apply disaster badge/highlight
- Use `analysis.urgency` for color coding (critical=red, high=orange, medium=yellow, low=green)
- Use filter dropdowns to set query params

---

### `GET /api/feed/platforms` — Platform Breakdown

```json
{
  "ok": true,
  "data": {
    "twitter": 35,
    "facebook": 20,
    "youtube": 15,
    "news": 25,
    "instagram": 25
  }
}
```

### `GET /api/feed/trending` — Trending Hashtags

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `top_n` | int | `15` | Number of top hashtags to return |

```json
{
  "ok": true,
  "data": [
    { "hashtag": "#floodalert", "count": 18 },
    { "hashtag": "#cyclonealert", "count": 12 },
    { "hashtag": "#mumbai", "count": 9 }
  ]
}
```

### `GET /api/feed/{post_id}` — Single Post Detail

Returns the full post object (same shape as items in `/api/feed`).

**404** if post not found.

---

### `WebSocket /ws/feed` — Real-time Feed Stream

Connect to receive new posts in real-time as they're processed.

**Messages from server:**
```json
{ "type": "new_post", "post": { /* full post object */ } }
```

**Messages from client:**
```
"ping"  →  server replies { "type": "pong" }
```

**Next.js usage example:**
```typescript
const ws = new WebSocket('ws://localhost:8000/ws/feed');

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'new_post') {
    // Prepend to feed state
    setPosts(prev => [data.post, ...prev]);
  }
};

// Keep alive
setInterval(() => ws.send('ping'), 30000);
```

---

## Page 2 — Disaster Map (India)

### `GET /api/map/reports` — All Map Pins (Lightweight)

Returns lightweight pin data for rendering markers on the map. No full report detail.

**Query Parameters:**

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `disaster_type` | string | — | Filter by disaster type |
| `severity` | string | — | Filter by severity |
| `state` | string | — | Filter by Indian state name |

**Response:**

```json
{
  "ok": true,
  "data": [
    {
      "id": "rpt_a1b2c3d4e5f6",
      "lat": 19.0760,
      "lng": 72.8777,
      "disaster_type": "flood",
      "severity": "critical",
      "title": "Severe Flooding in Mumbai - Red Alert",
      "location_name": "Mumbai",
      "source_count": 4,
      "created_at": "2026-02-17T05:30:00+00:00"
    }
  ],
  "meta": { "count": 10 }
}
```

**Frontend usage:**
- Place a marker for each pin at `[lat, lng]`
- Color markers by severity or disaster_type
- Show `title` as tooltip on hover
- On click → fetch full detail from `/api/map/reports/{id}`

---

### `GET /api/map/reports/{report_id}` — Full Report Card

Returns complete detail for a verified disaster report. This powers the card/modal that opens when a user clicks a pin.

**Response:**

```json
{
  "ok": true,
  "data": {
    "id": "rpt_a1b2c3d4e5f6",
    "title": "Severe Flooding in Mumbai - Red Alert",
    "description": "Multiple areas of Mumbai experiencing severe waterlogging...",
    "disaster_type": "flood",
    "severity": "critical",
    "location": {
      "name": "Mumbai",
      "state": "Maharashtra",
      "lat": 19.0760,
      "lng": 72.8777
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
      "infrastructure_damage": ["roads", "railway_tracks", "bridges", "power_lines"],
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
}
```

**Frontend card layout suggestion:**
```
┌─────────────────────────────────────────────┐
│ 🔴 CRITICAL        Flood         Mumbai, MH │
│                                             │
│ Severe Flooding in Mumbai - Red Alert       │
│ ─────────────────────────────────────────── │
│ Multiple areas experiencing severe water... │
│                                             │
│ ┌─ Impact ────────────────────────────────┐ │
│ │ 👥 52,000 affected  📐 45.5 km²        │ │
│ │ 💀 8 deaths  🤕 43 injured  🔍 5 missing│ │
│ │ 🏗️ roads, railways, bridges, power      │ │
│ │ 💰 ₹850 Cr estimated loss              │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ ✅ Verified (94% confidence)                │
│    4 sources across twitter, news, facebook │
│                                             │
│ ┌─ Timeline ──────────────────────────────┐ │
│ │ 04:00  First report detected            │ │
│ │ 04:30  Cross-platform confirmed         │ │
│ │ 05:00  Official alert issued            │ │
│ │ 05:30  Rescue operations started        │ │
│ │ 08:00  Report verified                  │ │
│ └─────────────────────────────────────────┘ │
│                                             │
│ Response: NDRF deployed, Relief camps...    │
└─────────────────────────────────────────────┘
```

---

### `GET /api/map/clusters` — Clustered Pins

Returns clustered pins for map zoom levels. At low zoom, nearby reports merge.

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `zoom` | int | `5` | Map zoom level (1=world, 18=street) |

```json
{
  "ok": true,
  "data": [
    {
      "lat": 19.5,
      "lng": 73.1,
      "count": 3,
      "dominant_type": "flood",
      "severity": "critical",
      "report_ids": ["rpt_abc1", "rpt_abc2", "rpt_abc3"]
    }
  ]
}
```

**Frontend usage:**
- At low zoom: show cluster circles with `count` inside
- At high zoom (>10): switch to individual pins
- On cluster click: zoom in or show list of reports

---

### `GET /api/map/heatmap` — Heatmap Data

Returns `[lat, lng, intensity]` triples for a heat-map layer.

```json
{
  "ok": true,
  "data": [
    [19.076, 72.877, 1.0],
    [13.082, 80.270, 0.7],
    [22.572, 88.363, 0.4]
  ]
}
```

Intensity: critical=1.0, high=0.7, medium=0.4, low=0.2

---

### `GET /api/map/states` — State-Level Summary

Aggregated stats per Indian state. Useful for choropleth / sidebar.

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

---

### `POST /api/map/reports` — Create a Verified Report

**Request body:**

```json
{
  "title": "Flash Floods in Patna",
  "description": "Heavy monsoon rain causes flash flooding across Patna...",
  "disaster_type": "flood",
  "severity": "high",
  "location_name": "Patna",
  "affected_area_km2": 15.0,
  "estimated_affected_people": 20000,
  "casualties": {
    "deaths": 2,
    "injured": 15,
    "missing": 0,
    "rescued": 200,
    "displaced": 5000
  },
  "infrastructure_damage": ["roads", "power_lines"],
  "source_post_ids": ["tw_abc123", "nw_def456"],
  "response_actions": ["NDRF deployed", "Relief camps opened"]
}
```

If `location_name` matches a known Indian city, `lat`/`lng`/`state` auto-fill.

---

## Shared Endpoints

### `GET /api/health`

```json
{
  "ok": true,
  "data": {
    "status": "healthy",
    "feed_posts": 120,
    "verified_reports": 10,
    "ws_connections": 2
  }
}
```

### `GET /api/stats`

```json
{
  "ok": true,
  "data": {
    "total_posts": 120,
    "disaster_posts": 78,
    "verified_reports": 10,
    "critical_reports": 3,
    "platforms": {
      "twitter": 35,
      "facebook": 20,
      "youtube": 15,
      "news": 25,
      "instagram": 25
    }
  }
}
```

---

## Internal Pipeline Endpoint

### `POST /api/feed/push` — Push New Post

Used by the processing pipeline to push newly processed posts into the feed + broadcast via WebSocket.

```json
// Body: same shape as a feed post object
{
  "id": "tw_newpost123",
  "platform": "twitter",
  "content": { "text": "...", "hashtags": [...] },
  "author": { ... },
  "engagement": { ... },
  "analysis": { ... },
  "location": { ... },
  "timestamp": "...",
  "language": "english"
}
```

---

## Data Types Reference

### Platforms
`twitter` | `facebook` | `youtube` | `news` | `instagram`

### Disaster Types
`flood` | `cyclone` | `earthquake` | `landslide` | `tsunami` | `drought` | `fire` | `heavy_rain`

### Severity / Urgency
`critical` | `high` | `medium` | `low`

### Sentiments
`negative` | `neutral` | `positive`

### Account Types
`personal` | `journalist` | `government` | `ngo` | `news_outlet`

### Verification Status
`verified` | `unverified` | `disputed`

---

## Next.js Integration Tips

### Fetching Feed (React Query / SWR)
```typescript
// lib/api.ts
const API = 'http://localhost:8000';

export async function getFeed(params?: {
  platform?: string;
  disasters_only?: boolean;
  limit?: number;
  offset?: number;
}) {
  const query = new URLSearchParams(params as any).toString();
  const res = await fetch(`${API}/api/feed?${query}`);
  return res.json();
}

export async function getMapPins(params?: {
  disaster_type?: string;
  severity?: string;
  state?: string;
}) {
  const query = new URLSearchParams(params as any).toString();
  const res = await fetch(`${API}/api/map/reports?${query}`);
  return res.json();
}

export async function getReportDetail(id: string) {
  const res = await fetch(`${API}/api/map/reports/${id}`);
  return res.json();
}
```

### Recommended Map Libraries
- **react-leaflet** — lightweight, good for markers + heatmap
- **@react-google-maps/api** — Google Maps with Indian map data  
- **mapbox-gl** + **react-map-gl** — best for clusters + heatmap layers

### Color Scheme Suggestion
| Disaster Type | Color |
|---------------|-------|
| Flood | `#2196F3` (blue) |
| Cyclone | `#9C27B0` (purple) |
| Earthquake | `#795548` (brown) |
| Landslide | `#FF9800` (orange) |
| Tsunami | `#00BCD4` (cyan) |
| Drought | `#FFC107` (amber) |
| Fire | `#F44336` (red) |
| Heavy Rain | `#607D8B` (blue-grey) |

| Severity | Color |
|----------|-------|
| Critical | `#D32F2F` |
| High | `#F57C00` |
| Medium | `#FBC02D` |
| Low | `#388E3C` |
