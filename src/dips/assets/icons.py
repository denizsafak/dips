"""Crisp 16x16 Lucide-style stroke SVG UI icons tinted dynamically to monochrome theme colors."""

from __future__ import annotations

from functools import lru_cache
from PyQt6.QtCore import QByteArray, QSize, Qt
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer

_SVGS: dict[str, str] = {
    "open": '<path d="M3 7v10a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-7l-2-2H5a2 2 0 0 0-2 2Z"/>',
    "save": '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2Z"/><polyline points="17 21 17 13 7 13 7 21"/><polyline points="7 3 7 8 15 8"/>',
    "export": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "undo": '<path d="M3 7v6h6"/><path d="M21 17a9 9 0 0 0-9-9 9 9 0 0 0-6 2.3L3 13"/>',
    "redo": '<path d="M21 7v6h-6"/><path d="M3 17a9 9 0 0 1 9-9 9 9 0 0 1 6 2.3L21 13"/>',
    "reset": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    "fit": '<path d="M15 3h6v6"/><path d="M9 21H3v-6"/><path d="M21 3l-7 7"/><path d="M3 21l7-7"/>',
    "zoom_in": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="11" y1="8" x2="11" y2="14"/><line x1="8" y1="11" x2="14" y2="11"/>',
    "zoom_out": '<circle cx="11" cy="11" r="8"/><line x1="21" y1="21" x2="16.65" y2="16.65"/><line x1="8" y1="11" x2="14" y2="11"/>',
    "eye": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    "eye_off": '<path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"/><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"/><path d="M6.61 6.61A13.526 13.526 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"/><line x1="2" y1="2" x2="22" y2="22"/>',
    "up": '<polyline points="18 15 12 9 6 15"/>',
    "down": '<polyline points="6 9 12 15 18 9"/>',
    "trash": '<polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
    "check": '<polyline points="6.5 12.5 10.2 16.2 17.5 8.2"/>',
    "close": '<line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>',
    "copy": '<rect x="9" y="9" width="13" height="13" rx="2"/><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"/>',
    "theme": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2"/><path d="M12 20v2"/><path d="m4.93 4.93 1.41 1.41"/><path d="m17.66 17.66 1.41 1.41"/><path d="M2 12h2"/><path d="M20 12h2"/><path d="m6.34 17.66-1.41 1.41"/><path d="m19.07 4.93-1.41 1.41"/>',
    "batch": '<rect x="2" y="6" width="14" height="14" rx="2"/><path d="M6 2h12a2 2 0 0 1 2 2v12"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "preset_save": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><polyline points="14 2 14 8 20 8"/><line x1="12" y1="18" x2="12" y2="12"/><line x1="9" y1="15" x2="15" y2="15"/>',
    "preset_load": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><polyline points="14 2 14 8 20 8"/>',

    # Categories
    "cat_ai": '<path d="M12 3l1.8 4.2L18 9l-4.2 1.8L12 15l-1.8-4.2L6 9l4.2-1.8Z"/><path d="M19 15l.9 2.1L22 18l-2.1.9L19 21l-.9-2.1L16 18l2.1-.9Z"/>',
    "cat_basics": '<rect x="3" y="3" width="18" height="18" rx="2"/><path d="M12 3v18"/>',
    "cat_affine": '<polygon points="4 6 20 4 18 18 2 20"/>',
    "cat_geometry": '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
    "cat_intensity": '<path d="M3 19C8 19 10 5 21 5"/><line x1="3" y1="20" x2="21" y2="20"/>',
    "cat_spatial": '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/>',
    "cat_frequency": '<path d="M2 12c2-6 4-6 6 0s4 6 6 0 4-6 6 0"/>',
    "cat_histogram": '<path d="M3 20h18"/><rect x="5" y="12" width="3" height="8"/><rect x="10.5" y="6" width="3" height="14"/><rect x="16" y="10" width="3" height="10"/>',
    "cat_morphology": '<rect x="4" y="4" width="7" height="7" rx="1"/><rect x="13" y="13" width="7" height="7" rx="1"/><rect x="13" y="4" width="7" height="7" rx="1"/>',
    "cat_restoration": '<path d="m12 3 1.9 3.9L18 8.8l-3 2.9.7 4.1-3.7-2-3.7 2 .7-4.1-3-2.9 4.1-1.9Z"/>',
    "cat_color": '<circle cx="12" cy="9" r="5"/><circle cx="9" cy="14" r="5"/><circle cx="15" cy="14" r="5"/>',
    "cat_artistic": '<path d="M12 2a10 10 0 1 0 10 10 4 4 0 0 1-5-5 4 4 0 0 1-5-5Z"/><circle cx="8.5" cy="10.5" r="1"/><circle cx="13.5" cy="15.5" r="1"/>',
    "cat_features": '<path d="M4 8V4h4"/><path d="M20 8V4h-4"/><path d="M4 16v4h4"/><path d="M20 16v4h-4"/><circle cx="12" cy="12" r="3"/>',

    # Individual monochrome operation icons
    "op_upscale": '<polyline points="15 3 21 3 21 9"/><polyline points="9 21 3 21 3 15"/><line x1="21" y1="3" x2="14" y2="10"/><line x1="3" y1="21" x2="10" y2="14"/>',
    "op_cutout": '<circle cx="6" cy="6" r="3"/><circle cx="6" cy="18" r="3"/><line x1="20" y1="4" x2="8.12" y2="15.88"/><line x1="14.47" y1="14.48" x2="20" y2="20"/><line x1="8.12" y1="8.12" x2="12" y2="12"/>',
    "op_bokeh": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="4"/>',
    "op_face": '<circle cx="12" cy="12" r="9"/><circle cx="9" cy="10" r="1"/><circle cx="15" cy="10" r="1"/><path d="M9 15c1 1 2 1.5 3 1.5s2-.5 3-1.5"/>',
    "op_shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>',
    "op_wand": '<path d="m15 4-1 2 2 1-2 1 1 2-2-1-1 2-1-2-2 1 1-2-2-1 2-1-1-2 2 1 1-2 1 2Z"/><path d="m3 21 9-9"/>',
    "op_mono": '<circle cx="12" cy="12" r="9"/><path d="M12 3v18"/>',
    "op_flip": '<path d="M12 3v18"/><polyline points="8 7 4 12 8 17"/><polyline points="16 7 20 12 16 17"/>',
    "op_rotate": '<path d="M21 12a9 9 0 1 1-3-6.7"/><polyline points="21 3 21 8 16 8"/>',
    "op_layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "op_move": '<polyline points="5 9 2 12 5 15"/><polyline points="9 5 12 2 15 5"/><polyline points="15 19 12 22 9 19"/><polyline points="19 9 22 12 19 15"/><line x1="2" y1="12" x2="22" y2="12"/><line x1="12" y1="2" x2="12" y2="22"/>',
    "op_shear": '<polygon points="6 5 20 5 18 19 4 19"/>',
    "op_crop": '<path d="M6 2v14a2 2 0 0 0 2 2h14"/><path d="M18 22V8a2 2 0 0 0-2-2H2"/>',
    "op_lens": '<circle cx="12" cy="12" r="9"/><path d="M8 5c2.5 4 2.5 10 0 14"/><path d="M16 5c-2.5 4-2.5 10 0 14"/>',
    "op_frame": '<rect x="3" y="3" width="18" height="18" rx="1"/><rect x="7" y="7" width="10" height="10" rx="1"/>',
    "op_polar": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1"/>',
    "op_curve": '<path d="M3 20C9 20 11 4 21 4"/><line x1="3" y1="20" x2="21" y2="20"/>',
    "op_sun": '<circle cx="12" cy="12" r="4"/><line x1="12" y1="2" x2="12" y2="5"/><line x1="12" y1="19" x2="12" y2="22"/><line x1="2" y1="12" x2="5" y2="12"/><line x1="19" y1="12" x2="22" y2="12"/>',
    "op_invert": '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="3" y1="21" x2="21" y2="3"/>',
    "op_blur": '<circle cx="12" cy="12" r="3"/><circle cx="12" cy="12" r="7" stroke-dasharray="2 2"/><circle cx="12" cy="12" r="10" stroke-dasharray="1 3"/>',
    "op_edges": '<path d="M4 20 L10 6 L14 14 L20 4"/>',
    "op_grid": '<rect x="3" y="3" width="18" height="18" rx="2"/><line x1="9" y1="3" x2="9" y2="21"/><line x1="15" y1="3" x2="15" y2="21"/><line x1="3" y1="9" x2="21" y2="9"/><line x1="3" y1="15" x2="21" y2="15"/>',
    "op_wave": '<path d="M2 12c2-5 4-5 6 0s4 5 6 0 4-5 6 0"/>',
    "op_bars": '<line x1="4" y1="20" x2="20" y2="20"/><rect x="5" y="11" width="3" height="9"/><rect x="10.5" y="5" width="3" height="15"/><rect x="16" y="8" width="3" height="12"/>',
    "op_thresh": '<path d="M3 18h6V6h12"/>',
    "op_morph": '<rect x="4" y="4" width="7" height="7"/><rect x="13" y="13" width="7" height="7"/><path d="M11 7h6v6"/>',
    "op_noise": '<circle cx="6" cy="7" r="1"/><circle cx="15" cy="6" r="1"/><circle cx="10" cy="12" r="1"/><circle cx="18" cy="13" r="1"/><circle cx="7" cy="17" r="1"/><circle cx="14" cy="18" r="1"/>',
    "op_repair": '<path d="m14.7 6.3 3 3-9.4 9.4H5.3v-3Z"/><path d="m12.5 8.5 3 3"/>',
    "op_cloud": '<path d="M17.5 19H7a4 4 0 0 1-.5-7.96A5.5 5.5 0 0 1 17.4 9.5a3.5 3.5 0 0 1 .1 9.5Z"/>',
    "op_color": '<circle cx="12" cy="9" r="4"/><circle cx="9" cy="14" r="4"/><circle cx="15" cy="14" r="4"/>',
    "op_shapes": '<circle cx="8" cy="8" r="4"/><rect x="13" y="13" width="7" height="7" rx="1"/>',
    "op_lines": '<line x1="4" y1="18" x2="20" y2="6"/><circle cx="15" cy="15" r="4"/>',
    "op_cube": '<path d="M12 2 3 7v10l9 5 9-5V7Z"/><polyline points="3.3 7 12 12 20.7 7"/><line x1="12" y1="22" x2="12" y2="12"/>',
    "op_moon": '<path d="M12 3a6 6 0 0 0 9 9 9 9 0 1 1-9-9Z"/>',
}


@lru_cache(maxsize=512)
def get_icon(name: str, color: str = "#CBD5E1", size: int = 16, stroke_width: float = 1.8) -> QIcon:
    """Return a crisp monochrome vector QIcon tinted to the given hex color."""
    body = _SVGS.get(name, _SVGS["info"])
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-linecap="round" stroke-linejoin="round">{body}</svg>'
    )
    renderer = QSvgRenderer(QByteArray(svg.encode("utf-8")))
    pm = QPixmap(QSize(size * 2, size * 2))
    pm.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pm)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    pm.setDevicePixelRatio(2.0)
    return QIcon(pm)
