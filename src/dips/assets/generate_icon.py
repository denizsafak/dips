"""Generate crisp vector and raster application icons for DIPS.

Design: Restraint & precision. Dark-slate rounded squircle with a subtle
1px hairline border, an architectural optical viewfinder frame, and a minimal
geometric aperture/pixel-prism emblem in silver-white (#E6EAF0) and slate-blue
(#4C8BF5).
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from PIL import Image

SVG_SOURCE = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <defs>
    <linearGradient id="squircleBg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#1E2533"/>
      <stop offset="55%" stop-color="#141923"/>
      <stop offset="100%" stop-color="#0D1017"/>
    </linearGradient>
    <radialGradient id="topGlow" cx="32%" cy="24%" r="65%">
      <stop offset="0%" stop-color="#4C8BF5" stop-opacity="0.22"/>
      <stop offset="100%" stop-color="#4C8BF5" stop-opacity="0"/>
    </radialGradient>
    <linearGradient id="rimGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#58657D" stop-opacity="0.9"/>
      <stop offset="50%" stop-color="#2B3242" stop-opacity="0.7"/>
      <stop offset="100%" stop-color="#1E232E" stop-opacity="0.9"/>
    </linearGradient>
    <linearGradient id="blueGrad" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#7CB4FF"/>
      <stop offset="50%" stop-color="#4C8BF5"/>
      <stop offset="100%" stop-color="#2255D6"/>
    </linearGradient>
    <linearGradient id="silverGrad" x1="0%" y1="0%" x2="0%" y2="100%">
      <stop offset="0%" stop-color="#F8FAFC"/>
      <stop offset="100%" stop-color="#CBD5E1"/>
    </linearGradient>
    <linearGradient id="emeraldGlow" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#60A5FA" stop-opacity="0.34"/>
      <stop offset="100%" stop-color="#2563EB" stop-opacity="0.06"/>
    </linearGradient>
  </defs>

  <!-- Outer Rounded Squircle -->
  <rect x="24" y="24" width="464" height="464" rx="108" ry="108" fill="url(#squircleBg)" stroke="url(#rimGrad)" stroke-width="5"/>
  <rect x="24" y="24" width="464" height="464" rx="108" ry="108" fill="url(#topGlow)"/>

  <!-- Subtle Pixel Matrix Grid -->
  <g stroke="#293245" stroke-width="2" opacity="0.55">
    <rect x="92" y="92" width="328" height="328" rx="44" fill="none"/>
    <line x1="200" y1="92" x2="200" y2="420"/>
    <line x1="312" y1="92" x2="312" y2="420"/>
    <line x1="92" y1="200" x2="420" y2="200"/>
    <line x1="92" y1="312" x2="420" y2="312"/>
  </g>

  <!-- Outer Optical Lens Ring -->
  <circle cx="256" cy="256" r="144" fill="#111620" stroke="url(#silverGrad)" stroke-width="11"/>
  <circle cx="256" cy="256" r="122" fill="none" stroke="#313C52" stroke-width="3"/>

  <!-- Split Processed Side Accent (Left Raw Dark vs Right Enhanced Blue Glow) -->
  <path d="M 256 138 A 118 118 0 0 1 256 374 Z" fill="url(#emeraldGlow)"/>
  <line x1="256" y1="126" x2="256" y2="386" stroke="#60A5FA" stroke-width="4" stroke-linecap="round" opacity="0.9"/>

  <!-- Crisp Inner Aperture Ring Arc -->
  <path d="M 164 218 A 98 98 0 0 1 298 166" fill="none" stroke="url(#blueGrad)" stroke-width="8" stroke-linecap="round"/>
  <path d="M 348 294 A 98 98 0 0 1 214 346" fill="none" stroke="url(#silverGrad)" stroke-width="6" stroke-linecap="round" opacity="0.75"/>

  <!-- Center Geometric Diamond Prism -->
  <polygon points="256,184 328,256 256,328 184,256" fill="#131924" stroke="url(#silverGrad)" stroke-width="9" stroke-linejoin="round"/>
  <polygon points="256,208 304,256 256,304 208,256" fill="url(#blueGrad)"/>
  <polygon points="256,228 284,256 256,284 228,256" fill="#EBF3FF" opacity="0.92"/>

  <!-- Optical Specular Spark -->
  <path d="M 358 136 L 364 152 L 380 158 L 364 164 L 358 180 L 352 164 L 336 158 L 352 152 Z" fill="#93C5FD"/>
</svg>
"""


def generate_icons(output_dir: Path | None = None) -> tuple[Path, Path, Path, Path]:
    """Write dips.svg, dips.png (512x512), dips.ico, and dips.icns."""
    from PyQt6.QtCore import QByteArray, Qt
    from PyQt6.QtGui import QImage, QPainter
    from PyQt6.QtSvg import QSvgRenderer

    out = output_dir or Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)

    svg_path = out / "dips.svg"
    png_path = out / "dips.png"
    ico_path = out / "dips.ico"
    icns_path = out / "dips.icns"

    svg_path.write_text(SVG_SOURCE, encoding="utf-8")

    renderer = QSvgRenderer(QByteArray(SVG_SOURCE.encode("utf-8")))
    img = QImage(512, 512, QImage.Format.Format_RGBA8888)
    img.fill(Qt.GlobalColor.transparent)
    painter = QPainter(img)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    img.save(str(png_path), "PNG")

    pil_img = Image.open(png_path).convert("RGBA")
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)]
    pil_img.save(str(ico_path), format="ICO", sizes=sizes)
    pil_img.save(str(icns_path), format="ICNS")

    for cname, stroke in (("chevron_dark.svg", "#B8C0CE"), ("chevron_light.svg", "#3E4654")):
        (out / cname).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" '
            f'fill="none" stroke="{stroke}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
            f'<polyline points="6 9 12 15 18 9"/></svg>\n',
            encoding="utf-8",
        )
    for cname, stroke in (("chevron_up_dark.svg", "#B8C0CE"), ("chevron_up_light.svg", "#3E4654")):
        (out / cname).write_text(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" '
            f'fill="none" stroke="{stroke}" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
            f'<polyline points="18 15 12 9 6 15"/></svg>\n',
            encoding="utf-8",
        )
    (out / "check_white.svg").write_text(
        '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10" viewBox="0 0 24 24" '
        'fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
        '<polyline points="6.5 12.5 10.2 16.2 17.5 8.2"/></svg>\n',
        encoding="utf-8",
    )
    return svg_path, png_path, ico_path, icns_path


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys

    app = QApplication(sys.argv)
    paths = generate_icons()
    print("Generated icons:", [str(p) for p in paths])
