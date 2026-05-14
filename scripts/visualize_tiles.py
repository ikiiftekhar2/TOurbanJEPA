#!/usr/bin/env python3
"""
Generate an interactive Leaflet map showing ALL downloaded ortho tiles
plotted on a map by their real geographic positions, color-coded by area.
"""

import json
import math
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ORTHO_DIR = PROJECT_ROOT / "data" / "ortho"
OUTPUT_HTML = PROJECT_ROOT / "notebooks" / "tile_map_full.html"

# Web Mercator constants
ORIGIN_X = -20037508.342787
ORIGIN_Y = 20037508.342787
TILE_SIZE = 256

# Area display names
AREA_DISPLAY = {
    "downtown":        "Downtown Core",
    "downtown_core":   "Downtown Core",
    "midtown":         "Midtown",
    "scarborough":     "Scarborough",
    "north_york":      "North York",
    "etobicoke":       "Etobicoke",
    "airport":         "Pearson Airport",
    "industrial":      "Port Lands",
    "parkland":        "High Park",
    "waterfront":      "Waterfront",
    "ravine":          "Don Valley",
    "residential_mid": "The Annex",
    "suburban_low":    "Suburban Low-rise",
    "east_york":       "East York",
    "york":            "York",
    "don_valley":      "Don Valley",
    "high_park":       "High Park",
    "port_lands":      "Port Lands",
    "test_tiles":      "Test Tiles",
}

AREA_COLORS = {
    "downtown":        "#e6194b",
    "downtown_core":   "#e6194b",
    "midtown":         "#f58231",
    "scarborough":     "#ffe119",
    "north_york":      "#3cb44b",
    "etobicoke":       "#4363d8",
    "airport":         "#911eb4",
    "industrial":      "#f032e6",
    "parkland":        "#42d4f4",
    "waterfront":      "#bfef45",
    "ravine":          "#fabed4",
    "residential_mid": "#dcbeff",
    "suburban_low":    "#800000",
    "east_york":       "#469990",
    "york":            "#9A6324",
    "don_valley":      "#aaffc3",
    "high_park":       "#808000",
    "port_lands":      "#ffd8b1",
    "test_tiles":      "#aaaaaa",
}


def tile_to_bbox(zoom, tx, ty):
    """Convert tile coordinates to Web Mercator bbox [xmin, ymin, xmax, ymax]."""
    res = 156543.03392800014 / (2 ** zoom)
    xmin = ORIGIN_X + tx * TILE_SIZE * res
    ymin = ORIGIN_Y - (ty + 1) * TILE_SIZE * res
    xmax = ORIGIN_X + (tx + 1) * TILE_SIZE * res
    ymax = ORIGIN_Y - ty * TILE_SIZE * res
    return xmin, ymin, xmax, ymax


def mercator_to_latlon(x, y):
    """Web Mercator (EPSG:3857) to WGS84 lat/lon."""
    lon = x / 20037508.34 * 180.0
    lat = 180.0 / math.pi * (2 * math.atan(math.exp(y / 20037508.34 * math.pi)) - math.pi / 2)
    return lat, lon


def gather_all_tiles():
    """Gather ALL tiles with their geographic bounds. Returns per-area tile data."""
    area_tiles = {}
    total = 0

    for area_dir in sorted(ORTHO_DIR.iterdir()):
        if not area_dir.is_dir():
            continue
        area = area_dir.name
        tiles = sorted(area_dir.glob("*.jpg"))
        if not tiles:
            continue

        tile_data = []
        for t in tiles:
            try:
                parts = t.stem.split("_")
                zoom, tx, ty = int(parts[0]), int(parts[1]), int(parts[2])
            except (ValueError, IndexError):
                continue
            xmin, ymin, xmax, ymax = tile_to_bbox(zoom, tx, ty)
            sw_lat, sw_lon = mercator_to_latlon(xmin, ymin)
            ne_lat, ne_lon = mercator_to_latlon(xmax, ymax)
            # Leaflet wants [lat, lng] pairs
            tile_data.append({
                "tx": tx, "ty": ty, "zoom": zoom,
                "bounds": [[sw_lat, sw_lon], [ne_lat, ne_lon]],
                "center": [(sw_lat + ne_lat) / 2, (sw_lon + ne_lon) / 2],
            })
        area_tiles[area] = tile_data
        total += len(tile_data)

    return area_tiles, total


def build_geojson(area_tiles):
    """Build GeoJSON FeatureCollection with one polygon per tile bounding box."""
    features = []
    for area, tiles in area_tiles.items():
        for t in tiles:
            s, n = t["bounds"][0], t["bounds"][1]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [s[1], s[0]], [n[1], s[0]], [n[1], n[0]], [s[1], n[0]], [s[1], s[0]]
                    ]]
                },
                "properties": {
                    "area": area,
                    "display": AREA_DISPLAY.get(area, area),
                    "color": AREA_COLORS.get(area, "#aaaaaa"),
                    "zoom": t["zoom"],
                    "tx": t["tx"],
                    "ty": t["ty"],
                }
            })
    return {"type": "FeatureCollection", "features": features}


def generate_html(area_tiles, total_tiles, output_path):
    """Generate interactive map HTML."""
    geojson = build_geojson(area_tiles)
    geojson_str = json.dumps(geojson)

    # Area stats for sidebar
    area_items = []
    area_order = sorted(area_tiles.keys(), key=lambda a: -len(area_tiles[a]))
    for area in area_order:
        count = len(area_tiles[area])
        color = AREA_COLORS.get(area, "#aaa")
        display = AREA_DISPLAY.get(area, area)
        area_items.append(f'''
            <div class="area-row" data-area="{area}" style="border-left: 4px solid {color};">
                <span class="area-dot" style="background:{color}"></span>
                <span class="area-label">{display}</span>
                <span class="area-n">{count:,}</span>
            </div>''')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Toronto Ortho Tile Coverage — {total_tiles:,} tiles</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
  #map {{ position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 1; }}
  #sidebar {{
    position: absolute; top: 10px; right: 10px; z-index: 1000;
    background: rgba(22, 33, 62, 0.95); color: #e0e0e0;
    border-radius: 8px; padding: 15px; max-height: calc(100vh - 20px);
    overflow-y: auto; width: 260px; font-size: 0.82rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
  }}
  #sidebar h2 {{
    color: #fff; font-size: 1rem; margin-bottom: 4px;
  }}
  #sidebar .subt {{ color: #888; font-size: 0.75rem; margin-bottom: 12px; }}
  #sidebar h3 {{ color: #7ec8e3; font-size: 0.85rem; margin: 10px 0 6px; }}
  .area-row {{
    display: flex; align-items: center; gap: 6px; padding: 4px 8px;
    margin: 2px 0; border-radius: 3px; cursor: pointer;
    background: rgba(255,255,255,0.03);
  }}
  .area-row:hover {{ background: rgba(255,255,255,0.08); }}
  .area-row.active {{ background: rgba(255,255,255,0.12); }}
  .area-dot {{ width: 10px; height: 10px; border-radius: 2px; flex-shrink: 0; }}
  .area-label {{ flex: 1; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
  .area-n {{ color: #888; font-size: 0.75rem; font-variant-numeric: tabular-nums; }}
  .stats {{ margin-bottom: 10px; }}
  .stat {{ display: flex; justify-content: space-between; padding: 2px 0; }}
  .stat-val {{ color: #7ec8e3; font-weight: 600; }}
  .btn {{
    display: block; width: 100%; margin-top: 8px; padding: 6px;
    background: #0f3460; color: #7ec8e3; border: 1px solid #2a2a4a;
    border-radius: 4px; cursor: pointer; font-size: 0.78rem; text-align: center;
  }}
  .btn:hover {{ background: #1a4a7a; }}
  .info-hover {{
    position: absolute; bottom: 20px; left: 50%; transform: translateX(-50%);
    z-index: 1000; background: rgba(0,0,0,0.8); color: #fff;
    padding: 8px 16px; border-radius: 4px; font-size: 0.8rem;
    pointer-events: none; display: none;
  }}
  @media (max-width: 600px) {{
    #sidebar {{ width: 200px; font-size: 0.7rem; padding: 10px; }}
  }}
</style>
</head>
<body>
<div id="map"></div>
<div id="sidebar">
  <h2>Toronto Ortho Coverage</h2>
  <div class="subt">{total_tiles:,} tiles · Zoom 19 · ~30cm/px</div>
  <div class="stats">
    <div class="stat"><span>Total Tiles</span><span class="stat-val">{total_tiles:,}</span></div>
    <div class="stat"><span>Areas</span><span class="stat-val">{len(area_tiles)}</span></div>
  </div>
  <h3>Areas</h3>
  {''.join(area_items)}
  <button class="btn" onclick="resetView()">Reset View (Toronto)</button>
  <button class="btn" onclick="toggleAll(true)">Show All</button>
  <button class="btn" onclick="toggleAll(false)">Hide All</button>
</div>
<div class="info-hover" id="info-hover"></div>
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<script>
// Tile GeoJSON data
var tileData = {geojson_str};

// Color-coded area layers
var areaLayers = {{}};
var allBounds = L.latLngBounds([]);

// Create layer for each area
tileData.features.forEach(function(f) {{
  var a = f.properties.area;
  if (!areaLayers[a]) {{
    areaLayers[a] = {{
      group: L.layerGroup(),
      color: f.properties.color,
      display: f.properties.display,
      features: []
    }};
  }}
  areaLayers[a].features.push(f);

  // Extend bounds
  var c = f.geometry.coordinates[0];
  allBounds.extend([c[0][1], c[0][0]]);
  allBounds.extend([c[2][1], c[2][0]]);
}});

// Render tiles as canvas rectangles (fast for 40K+ tiles)
var map = L.map('map', {{
  preferCanvas: true,
  center: [43.70, -79.38],
  zoom: 12,
  minZoom: 10,
  maxZoom: 18
}});

// Add tile layer for basemap context
L.tileLayer('https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
  attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a>, &copy; <a href="https://carto.com/">CARTO</a>',
  subdomains: 'abcd',
  maxZoom: 20
}}).addTo(map);

// Render all tile polygons using canvas
Object.keys(areaLayers).forEach(function(area) {{
  var al = areaLayers[area];
  var geojson = {{type: "FeatureCollection", features: al.features}};

  var layer = L.geoJSON(geojson, {{
    style: function(f) {{
      return {{
        color: f.properties.color,
        weight: 0.5,
        opacity: 0.4,
        fillColor: f.properties.color,
        fillOpacity: 0.15,
      }};
    }},
    onEachFeature: function(f, l) {{
      l.on('mouseover', function(e) {{
        var p = f.properties;
        document.getElementById('info-hover').style.display = 'block';
        document.getElementById('info-hover').innerHTML =
          '<strong>' + p.display + '</strong> · z' + p.zoom +
          ' · tile (' + p.tx + ', ' + p.ty + ')';
      }});
      l.on('mouseout', function() {{
        document.getElementById('info-hover').style.display = 'none';
      }});
    }}
  }}).addTo(map);

  al.layer = layer;
}});

// Click sidebar to toggle area
document.querySelectorAll('.area-row').forEach(function(row) {{
  row.addEventListener('click', function() {{
    var area = this.dataset.area;
    var al = areaLayers[area];
    if (map.hasLayer(al.layer)) {{
      map.removeLayer(al.layer);
      this.classList.remove('active');
    }} else {{
      map.addLayer(al.layer);
      this.classList.add('active');
    }}
  }});
  // All visible by default
  row.classList.add('active');
}});

function resetView() {{
  map.fitBounds(allBounds, {{padding: [30, 30]}});
}}

function toggleAll(show) {{
  document.querySelectorAll('.area-row').forEach(function(row) {{
    var al = areaLayers[row.dataset.area];
    if (show) {{
      if (!map.hasLayer(al.layer)) map.addLayer(al.layer);
      row.classList.add('active');
    }} else {{
      if (map.hasLayer(al.layer)) map.removeLayer(al.layer);
      row.classList.remove('active');
    }}
  }});
}}

// Fit to data bounds on load
map.fitBounds(allBounds, {{padding: [30, 30]}});
</script>
</body>
</html>'''

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path


def main():
    print("Gathering ALL tiles with geographic bounds...")
    area_tiles, total = gather_all_tiles()

    for area in sorted(area_tiles.keys(), key=lambda a: -len(area_tiles[a])):
        print(f"  {area:20s}: {len(area_tiles[area]):,} tiles")

    print(f"\n  TOTAL: {total:,} tiles across {len(area_tiles)} areas")

    print(f"\nGenerating interactive map...")
    path = generate_html(area_tiles, total, OUTPUT_HTML)
    print(f"  Saved: {path}")
    print(f"  File size: {path.stat().st_size / 1024:.0f} KB")
    print(f"\n  Open in browser: file://{path}")


if __name__ == "__main__":
    main()
