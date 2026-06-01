"""
Cache the small offline map tile set used by Demo/demo.html.

Current area: Shanghai Lujiazui / Century Avenue demo area.
Output: Demo/map-tiles/voyager/{z}/{x}/{y}@2x.png
"""
import math
import os
import sys
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "Demo" / "map-tiles" / "voyager"
BBOX = (121.497, 31.219, 121.533, 31.242)  # west, south, east, north
ZOOMS = range(14, 18)
SUBDOMAINS = ("a", "b", "c", "d")


def lon_to_x(lon, z):
    return int((lon + 180.0) / 360.0 * (2**z))


def lat_to_y(lat, z):
    lat_rad = math.radians(lat)
    return int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * (2**z))


def iter_tiles():
    west, south, east, north = BBOX
    for z in ZOOMS:
        x_min = lon_to_x(west, z)
        x_max = lon_to_x(east, z)
        y_min = lat_to_y(north, z)
        y_max = lat_to_y(south, z)
        for x in range(x_min, x_max + 1):
            for y in range(y_min, y_max + 1):
                yield z, x, y


def main():
    tiles = list(iter_tiles())
    print(f"cache {len(tiles)} tiles into {OUT_ROOT}")
    for idx, (z, x, y) in enumerate(tiles, 1):
        dest_dir = OUT_ROOT / str(z) / str(x)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{y}@2x.png"
        if dest.exists() and dest.stat().st_size > 100:
            continue
        subdomain = SUBDOMAINS[(x + y) % len(SUBDOMAINS)]
        url = f"https://{subdomain}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}@2x.png"
        try:
            urllib.request.urlretrieve(url, dest)
        except Exception as exc:
            print(f"failed {url}: {exc}", file=sys.stderr)
        if idx % 40 == 0:
            print(f"{idx}/{len(tiles)}")
    print("done")


if __name__ == "__main__":
    main()
