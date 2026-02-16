# Sentinel Flood Detection API

**Base URL:** `http://localhost:8000`  
**Docs:** `http://localhost:8000/docs` (auto-generated Swagger UI)

## Start the server
```bash
cd G:\Sentinel
api_env\Scripts\activate
python -m uvicorn backend.api:app --host 0.0.0.0 --port 8000
```

---

## Endpoints

### 1. Global Summary
```
GET /api/summary
```
**Response:**
```json
{
  "total_events": 3,
  "total_tiles": 44,
  "total_new_flood_km2": 537.57,
  "total_flooded_km2": 648.27,
  "events": ["EMSR766", "EMSR771", "EMSR773"]
}
```

---

### 2. List All Events (with map coordinates)
```
GET /api/events
```
**Response:** Array of events with center lat/lng and bounding boxes.
```json
[
  {
    "cems_id": "EMSR773",
    "tile_count": 15,
    "center": [39.40, -0.33],       // [lat, lng]
    "bbox": [-0.59, 39.21, -0.27, 39.42],  // [west, south, east, north]
    "total_new_flood_km2": 141.91,
    "total_flooded_km2": 224.84
  }
]
```
**Frontend use:** Place markers on the globe/map for each event. Use `bbox` to `fitBounds()`.

---

### 3. Event Detail
```
GET /api/events/{cems_id}
```
**Response:** Full event info with per-tile bounds and stats.
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

---

### 4. List Tiles for an Event
```
GET /api/events/{cems_id}/tiles
```
**Response:** Array of tiles with bounds and key metrics.

---

### 5. GeoJSON Flood Polygons (for map rendering)
```
GET /api/events/{cems_id}/tiles/{tile}/geojson?phase=during&layer=change
```
**Query params:**
| Param | Values | Default | Description |
|-------|--------|---------|-------------|
| `phase` | `before`, `during` | `during` | Which temporal phase |
| `layer` | `flood`, `new_flood`, `change` | `flood` | What to vectorize |

**`layer=change`** is what you want for the map — returns 2 feature types:
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": { "type": "MultiPolygon", "coordinates": [...] },
      "properties": {
        "layer": "existing_water",
        "color": "#0066ff"
      }
    },
    {
      "type": "Feature",
      "geometry": { "type": "MultiPolygon", "coordinates": [...] },
      "properties": {
        "layer": "new_flood",
        "color": "#ff0000"
      }
    }
  ]
}
```
**Frontend use:** Add as a GeoJSON source in Mapbox/Leaflet. Style by `properties.color`.

---

### 6. Full Event GeoJSON (all tiles merged)
```
GET /api/events/{cems_id}/geojson?layer=change
```
Returns GeoJSON for ALL tiles in one call. Each feature has `tile`, `new_flood_km2`, `pct_increase` in properties.

**Frontend use:** Best for rendering the full flood extent of an event on the map in one layer.

---

### 7. Tile Statistics
```
GET /api/events/{cems_id}/tiles/{tile}/stats
```
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

---

### 8. RGB Satellite Image (PNG)
```
GET /api/events/{cems_id}/tiles/{tile}/rgb/{phase}
```
`phase` = `before` or `during`  
Returns: **512x512 PNG image**

**Frontend use:** Use as an image overlay on the map positioned at the tile's bounds.

---

### 9. Flood Mask (transparent PNG)
```
GET /api/events/{cems_id}/tiles/{tile}/mask/{phase}?overlay=false
```
| Param | Values | Default | Description |
|-------|--------|---------|-------------|
| `overlay` | `true`, `false` | `false` | Composite mask on RGB |

- `overlay=false` → transparent RGBA PNG (blue flood pixels, rest transparent)
- `overlay=true` → RGB with blue flood overlay composited

**Frontend use:** Layer the transparent PNG on top of satellite imagery.

---

### 10. Change Detection Overlay (PNG)
```
GET /api/events/{cems_id}/tiles/{tile}/change_overlay
```
Returns: **512x512 PNG** with RGB satellite + blue (existing water) + red (new flood) composited.

**Frontend use:** Great for popups, side panels, or "before/after" sliders.

---

## Suggested Frontend Architecture (Next.js + Mapbox GL JS)

### Map Library
Use **Mapbox GL JS** (`react-map-gl`) or **Leaflet** (`react-leaflet`). Mapbox looks more polished for hackathons.

### Suggested Page Layout
```
┌─────────────────────────────────────────────────────┐
│  HEADER: "Sentinel Flood Monitor"    [Event Picker] │
├────────────────────────────────┬────────────────────┤
│                                │  Stats Panel       │
│                                │  ┌──────────────┐  │
│         MAP                    │  │ Event: EMSR773│  │
│   (full height, interactive)   │  │ New flood:    │  │
│                                │  │  141.91 km²   │  │
│   Event markers (red dots)     │  │ Tiles: 15     │  │
│   Click → zoom + load GeoJSON  │  │               │  │
│   Blue poly = existing water   │  │ [Bar chart]   │  │
│   Red poly = new flood         │  │               │  │
│                                │  └──────────────┘  │
│                                │                    │
│                                │  Tile Popup        │
│                                │  ┌──────────────┐  │
│                                │  │ Before / After│  │
│                                │  │ image slider  │  │
│                                │  └──────────────┘  │
├────────────────────────────────┴────────────────────┤
│  FOOTER: Total monitored: 3 events, 537 km² flooded │
└─────────────────────────────────────────────────────┘
```

### Key NPM Packages
```bash
npm install react-map-gl mapbox-gl @turf/turf recharts
# or for Leaflet:
npm install react-leaflet leaflet
```

### Quick Integration Example (react-map-gl)
```tsx
// 1. Fetch events on load
const events = await fetch('http://localhost:8000/api/events').then(r => r.json())

// 2. On event click, fetch GeoJSON
const geojson = await fetch(
  `http://localhost:8000/api/events/${cems_id}/geojson?layer=change`
).then(r => r.json())

// 3. Add to map as a source + layers
<Source id="flood" type="geojson" data={geojson}>
  <Layer id="existing-water" type="fill" 
    filter={['==', ['get', 'layer'], 'existing_water']}
    paint={{ 'fill-color': '#0066ff', 'fill-opacity': 0.5 }} />
  <Layer id="new-flood" type="fill"
    filter={['==', ['get', 'layer'], 'new_flood']}
    paint={{ 'fill-color': '#ff0000', 'fill-opacity': 0.6 }} />
</Source>

// 4. Image overlays for satellite tiles (optional)
// Use the /rgb/ and /change_overlay endpoints as image sources
```

### Available CEMS Events & Locations
| Event | Location | Tiles | New Flood |
|-------|----------|-------|-----------|
| EMSR766 | Croatia/Serbia (Danube) | 14 | 344.59 km² |
| EMSR771 | Northern Italy (Po Valley) | 15 | 51.07 km² |
| EMSR773 | Eastern Spain (Valencia) | 15 | 141.91 km² |
