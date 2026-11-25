"""Professional Dark and Light theme definitions and QSS stylesheets for DIPS."""

from __future__ import annotations

from dataclasses import dataclass
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QGuiApplication, QPalette
from PyQt6.QtWidgets import QApplication, QComboBox, QListView

from ..assets import asset_path


@dataclass(frozen=True)
class ThemePalette:
    name: str
    bg_app: str
    bg_panel: str
    bg_elevated: str
    bg_canvas: str
    bg_hover: str
    bg_pressed: str
    border: str
    border_strong: str
    text_primary: str
    text_secondary: str
    text_muted: str
    accent: str
    accent_hover: str
    accent_subtle: str
    danger: str
    success: str


DARK_PALETTE = ThemePalette(
    name="dark",
    bg_app="#15171C",
    bg_panel="#1B1E25",
    bg_elevated="#232730",
    bg_canvas="#101216",
    bg_hover="#2A2F3A",
    bg_pressed="#323846",
    border="#2C313C",
    border_strong="#3C4352",
    text_primary="#E6EAF0",
    text_secondary="#B8C0CE",
    text_muted="#7D8696",
    accent="#4C8BF5",
    accent_hover="#6098F7",
    accent_subtle="#21314C",
    danger="#D9534F",
    success="#22C55E",
)

LIGHT_PALETTE = ThemePalette(
    name="light",
    bg_app="#ECEFF4",
    bg_panel="#F8F9FB",
    bg_elevated="#FFFFFF",
    bg_canvas="#E2E6ED",
    bg_hover="#E5E9F0",
    bg_pressed="#D8DEE9",
    border="#D5DBE4",
    border_strong="#B8C2D1",
    text_primary="#1E232B",
    text_secondary="#3E4654",
    text_muted="#6B7485",
    accent="#2F6FED",
    accent_hover="#255CD0",
    accent_subtle="#DCE7FC",
    danger="#C9413D",
    success="#16A34A",
)


def build_stylesheet(p: ThemePalette) -> str:
    """Return a clean, consistent QSS stylesheet for the given palette."""
    chev_name = "chevron_dark.svg" if p.name == "dark" else "chevron_light.svg"
    chev_url = asset_path(chev_name).resolve().as_posix()
    chev_up_name = "chevron_up_dark.svg" if p.name == "dark" else "chevron_up_light.svg"
    chev_up_url = asset_path(chev_up_name).resolve().as_posix()
    check_url = asset_path("check_white.svg").resolve().as_posix()
    return f"""
    QMainWindow, QDialog {{
        background-color: {p.bg_app};
        color: {p.text_primary};
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", "Inter", "Noto Sans", sans-serif;
        font-size: 12px;
    }}
    QWidget {{
        color: {p.text_primary};
    }}
    QMenuBar {{
        background-color: {p.bg_panel};
        border-bottom: 1px solid {p.border};
        padding: 2px 6px;
    }}
    QMenuBar::item {{
        padding: 4px 10px;
        border-radius: 4px;
        background: transparent;
    }}
    QMenuBar::item:selected {{
        background-color: {p.bg_hover};
    }}
    QMenu {{
        background-color: {p.bg_elevated};
        border: 1px solid {p.border_strong};
        border-radius: 6px;
        padding: 4px;
    }}
    QMenu::item {{
        padding: 5px 24px 5px 12px;
        border-radius: 4px;
    }}
    QMenu::item:selected {{
        background-color: {p.accent_subtle};
        color: {p.text_primary};
    }}
    QMenu::item:disabled {{
        color: {p.text_muted};
        background: transparent;
    }}
    QMenu::separator {{
        height: 1px;
        background: {p.border};
        margin: 4px 8px;
    }}
    QToolTip {{
        background-color: {p.bg_elevated};
        color: {p.text_primary};
        border: 1px solid {p.border_strong};
        border-radius: 4px;
        padding: 4px 8px;
    }}
    QMessageBox {{
        background-color: {p.bg_app};
        color: {p.text_primary};
    }}
    QProgressBar {{
        background-color: {p.bg_elevated};
        border: 1px solid {p.border};
        border-radius: 5px;
        text-align: center;
        color: {p.text_primary};
        min-height: 14px;
    }}
    QProgressBar::chunk {{
        background-color: {p.accent};
        border-radius: 4px;
    }}
    QToolBar {{
        background-color: {p.bg_panel};
        border-bottom: 1px solid {p.border};
        spacing: 4px;
        padding: 4px 8px;
    }}
    QToolBar::separator {{
        width: 1px;
        background-color: {p.border_strong};
        margin: 6px 6px;
    }}
    QToolButton {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 5px;
        padding: 4px 8px;
        color: {p.text_secondary};
    }}
    QToolButton:hover {{
        background-color: {p.bg_hover};
        border-color: {p.border};
        color: {p.text_primary};
    }}
    QToolButton:pressed, QToolButton:checked {{
        background-color: {p.accent_subtle};
        border-color: {p.accent};
        color: {p.text_primary};
    }}
    QToolButton:disabled {{
        color: {p.text_muted};
    }}
    QToolButton#RowBtn {{
        padding: 2px;
        border: 1px solid transparent;
        border-radius: 4px;
    }}
    QToolButton#RowBtn:hover {{
        background-color: {p.bg_hover};
        border-color: {p.border};
    }}
    QStatusBar {{
        background-color: {p.bg_panel};
        border-top: 1px solid {p.border};
        color: {p.text_secondary};
        font-size: 11px;
    }}
    QStatusBar::item {{
        border: none;
    }}
    QSplitter::handle {{
        background-color: {p.border};
    }}
    QSplitter::handle:horizontal {{
        width: 1px;
    }}
    QSplitter::handle:vertical {{
        height: 1px;
    }}
    QFrame#PanelCard, QWidget#PanelPane {{
        background-color: {p.bg_panel};
    }}
    QFrame#SurfaceCard {{
        background-color: {p.bg_elevated};
        border: 1px solid {p.border};
        border-radius: 6px;
    }}
    QFrame#HeaderBar {{
        background-color: {p.bg_panel};
        border-bottom: 1px solid {p.border};
    }}
    QLabel#SectionTitle {{
        font-size: 11px;
        font-weight: 600;
        color: {p.text_muted};
        letter-spacing: 0.6px;
        text-transform: uppercase;
    }}
    QLabel#TitleLabel {{
        font-size: 13px;
        font-weight: 600;
        color: {p.text_primary};
    }}
    QLabel#MutedLabel {{
        font-size: 11px;
        color: {p.text_muted};
    }}
    QLineEdit, QSpinBox, QDoubleSpinBox {{
        background-color: {p.bg_elevated};
        border: 1px solid {p.border};
        border-radius: 5px;
        padding: 3px 7px;
        color: {p.text_primary};
        selection-background-color: {p.accent};
        selection-color: #FFFFFF;
    }}
    QSpinBox, QDoubleSpinBox {{
        padding-right: 18px;
    }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border;
        subcontrol-position: top right;
        width: 16px;
        border-left: 1px solid {p.border};
        border-top-right-radius: 4px;
        background: transparent;
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border;
        subcontrol-position: bottom right;
        width: 16px;
        border-left: 1px solid {p.border};
        border-bottom-right-radius: 4px;
        background: transparent;
    }}
    QSpinBox::up-button:hover, QSpinBox::down-button:hover,
    QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
        background-color: {p.bg_hover};
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url("{chev_up_url}");
        width: 8px;
        height: 8px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url("{chev_url}");
        width: 8px;
        height: 8px;
    }}
    QComboBox {{
        background-color: {p.bg_elevated};
        border: 1px solid {p.border_strong};
        border-radius: 5px;
        padding: 4px 28px 4px 9px;
        min-height: 20px;
        color: {p.text_primary};
    }}
    QComboBox:hover {{
        background-color: {p.bg_hover};
        border-color: {p.accent};
    }}
    QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
        border-color: {p.accent};
    }}
    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border-left: 1px solid {p.border};
        border-top-right-radius: 5px;
        border-bottom-right-radius: 5px;
        background: transparent;
    }}
    QComboBox::down-arrow {{
        image: url("{chev_url}");
        width: 12px;
        height: 12px;
    }}
    QComboBoxPrivateContainer {{
        background-color: {p.bg_elevated};
        border: 1px solid {p.border_strong};
        border-radius: 6px;
    }}
    QComboBox QAbstractItemView, QComboBox QAbstractItemView::viewport {{
        background-color: {p.bg_elevated};
        color: {p.text_primary};
    }}
    QComboBox QAbstractItemView {{
        border: 1px solid {p.border_strong};
        border-radius: 6px;
        padding: 4px;
        selection-background-color: {p.accent_subtle};
        selection-color: {p.text_primary};
        outline: none;
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 24px;
        padding: 4px 10px;
        border-radius: 4px;
        background-color: transparent;
        color: {p.text_primary};
    }}
    QComboBox QAbstractItemView::item:hover {{
        background-color: {p.accent_subtle};
        color: {p.text_primary};
    }}
    QComboBox QAbstractItemView::item:selected {{
        background-color: {p.accent_subtle};
        color: {p.text_primary};
    }}
    QPushButton {{
        background-color: {p.bg_elevated};
        border: 1px solid {p.border_strong};
        border-radius: 5px;
        padding: 5px 12px;
        color: {p.text_primary};
        font-weight: 500;
    }}
    QPushButton:hover {{
        background-color: {p.bg_hover};
        border-color: {p.border_strong};
    }}
    QPushButton:pressed {{
        background-color: {p.bg_pressed};
    }}
    QPushButton:disabled {{
        color: {p.text_muted};
        border-color: {p.border};
        background-color: {p.bg_panel};
    }}
    QPushButton#PrimaryButton {{
        background-color: {p.accent};
        border-color: {p.accent};
        color: #FFFFFF;
        font-weight: 600;
    }}
    QPushButton#PrimaryButton:hover {{
        background-color: {p.accent_hover};
        border-color: {p.accent_hover};
    }}
    QPushButton#PrimaryButton:disabled {{
        background-color: {p.bg_hover};
        border-color: {p.border};
        color: {p.text_muted};
    }}
    QFrame#SegmentedTrack {{
        background-color: {p.bg_canvas};
        border: 1px solid {p.border};
        border-radius: 6px;
    }}
    QPushButton#SegmentedBtn {{
        background-color: transparent;
        border: 1px solid transparent;
        border-radius: 4px;
        padding: 3px 10px;
        font-size: 11px;
        color: {p.text_muted};
    }}
    QPushButton#SegmentedBtn:hover {{
        color: {p.text_primary};
    }}
    QPushButton#SegmentedBtn:checked {{
        background-color: {p.bg_elevated};
        border: 1px solid {p.border_strong};
        color: {p.text_primary};
        font-weight: 600;
    }}
    QPushButton#CompactBtn {{
        background-color: {p.bg_elevated};
        border: 1px solid {p.border_strong};
        border-radius: 5px;
        padding: 0px;
        font-weight: 600;
        color: {p.text_primary};
    }}
    QPushButton#CompactBtn:hover {{
        background-color: {p.bg_hover};
    }}
    QPushButton#LinkButton {{
        background: transparent;
        border: none;
        padding: 2px 4px;
        color: {p.accent};
        font-weight: 600;
        font-size: 11px;
    }}
    QPushButton#LinkButton:hover {{
        background: transparent;
        color: {p.accent_hover};
        text-decoration: underline;
    }}
    QPushButton#LinkButton:pressed {{
        background: transparent;
    }}
    QPushButton#ChipButton {{
        background-color: {p.bg_elevated};
        border: 1px solid {p.border};
        border-radius: 13px;
        padding: 4px 11px;
        font-size: 11px;
        color: {p.text_secondary};
    }}
    QPushButton#ChipButton:hover {{
        border-color: {p.accent};
        color: {p.text_primary};
        background-color: {p.accent_subtle};
    }}
    QSlider {{
        min-height: 22px;
        background: transparent;
    }}
    QSlider:focus {{
        outline: none;
    }}
    QSlider::groove:horizontal {{
        height: 4px;
        background: {p.border_strong};
        border-radius: 2px;
        margin: 0 7px;
    }}
    QSlider::sub-page:horizontal {{
        height: 4px;
        background: {p.accent};
        border-radius: 2px;
        margin: 0 7px;
    }}
    QSlider::add-page:horizontal {{
        height: 4px;
        background: {p.border_strong};
        border-radius: 2px;
        margin: 0 7px;
    }}
    QSlider::handle:horizontal {{
        background: {p.bg_elevated};
        border: 2px solid {p.accent};
        width: 10px;
        height: 10px;
        margin: -5px -7px;
        border-radius: 7px;
    }}
    QSlider::handle:horizontal:hover,
    QSlider::handle:horizontal:focus {{
        background: {p.accent_subtle};
        border: 2px solid {p.accent_hover};
    }}
    QSlider::handle:horizontal:pressed {{
        background: {p.accent};
        border: 2px solid {p.accent_hover};
    }}
    QCheckBox {{
        spacing: 7px;
        color: {p.text_primary};
    }}
    QCheckBox::indicator {{
        width: 14px;
        height: 14px;
        border-radius: 3px;
        border: 1px solid {p.border_strong};
        background-color: {p.bg_elevated};
    }}
    QCheckBox::indicator:hover {{
        border-color: {p.accent};
        background-color: {p.bg_hover};
    }}
    QCheckBox::indicator:checked {{
        background-color: {p.accent};
        border-color: {p.accent};
        image: url("{check_url}");
    }}
    QCheckBox::indicator:checked:hover {{
        background-color: {p.accent_hover};
        border-color: {p.accent_hover};
        image: url("{check_url}");
    }}
    QCheckBox::indicator:disabled {{
        background-color: {p.bg_panel};
        border-color: {p.border};
    }}
    QTreeWidget, QListWidget {{
        background-color: {p.bg_panel};
        border: none;
        outline: none;
        show-decoration-selected: 0;
    }}
    QTreeWidget::branch,
    QTreeWidget::branch:selected,
    QTreeWidget::branch:hover,
    QTreeWidget::branch:!selected {{
        background: transparent;
        border: none;
        outline: none;
    }}
    QTreeWidget::item, QListWidget::item {{
        padding: 4px 6px;
        border-radius: 4px;
        border: none;
        outline: none;
    }}
    QTreeWidget::item:focus, QListWidget::item:focus {{
        outline: none;
        border: none;
    }}
    QTreeWidget::item:hover, QListWidget::item:hover {{
        background-color: {p.bg_hover};
    }}
    QTreeWidget::item:selected, QListWidget::item:selected {{
        background-color: {p.accent_subtle};
        color: {p.text_primary};
        border: none;
        outline: none;
    }}
    QScrollArea, QScrollArea > QWidget, QScrollArea > QWidget > QWidget {{
        border: none;
        background: transparent;
    }}
    QScrollBar:vertical {{
        background: transparent;
        width: 10px;
        margin: 2px;
    }}
    QScrollBar::handle:vertical {{
        background: {p.border_strong};
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{
        background: {p.text_muted};
    }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
    QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
        background: none;
        height: 0px;
    }}
    QScrollBar:horizontal {{
        background: transparent;
        height: 8px;
        margin: 2px;
    }}
    QScrollBar::handle:horizontal {{
        background: {p.border_strong};
        border-radius: 4px;
        min-width: 24px;
    }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
    QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {{
        background: none;
        width: 0px;
    }}
    QTabWidget::pane {{
        border: 1px solid {p.border};
        border-radius: 5px;
        background-color: {p.bg_panel};
    }}
    QTabBar::tab {{
        background: transparent;
        padding: 6px 12px;
        color: {p.text_muted};
        border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:selected {{
        color: {p.text_primary};
        border-bottom: 2px solid {p.accent};
        font-weight: 600;
    }}
    """


def configure_combobox_popup(combo: QComboBox) -> QComboBox:
    """Attach a mouse-tracking QListView so QComboBox popup items highlight cleanly on hover."""
    view = QListView(combo)
    view.setMouseTracking(True)
    view.viewport().setMouseTracking(True)
    view.setAutoFillBackground(True)
    view.viewport().setAutoFillBackground(True)
    combo.setView(view)
    return combo


def _build_qpalette(p: ThemePalette) -> QPalette:
    """Build a full QPalette matching ThemePalette so native popups/dialogs follow Light/Dark."""
    pal = QPalette()
    bg_app = QColor(p.bg_app)
    bg_panel = QColor(p.bg_panel)
    bg_elev = QColor(p.bg_elevated)
    text_pri = QColor(p.text_primary)
    text_sec = QColor(p.text_secondary)
    text_mut = QColor(p.text_muted)
    accent = QColor(p.accent)
    accent_sub = QColor(p.accent_subtle)

    for grp in (QPalette.ColorGroup.Active, QPalette.ColorGroup.Inactive):
        pal.setColor(grp, QPalette.ColorRole.Window, bg_app)
        pal.setColor(grp, QPalette.ColorRole.WindowText, text_pri)
        pal.setColor(grp, QPalette.ColorRole.Base, bg_elev)
        pal.setColor(grp, QPalette.ColorRole.AlternateBase, bg_panel)
        pal.setColor(grp, QPalette.ColorRole.ToolTipBase, bg_elev)
        pal.setColor(grp, QPalette.ColorRole.ToolTipText, text_pri)
        pal.setColor(grp, QPalette.ColorRole.PlaceholderText, text_mut)
        pal.setColor(grp, QPalette.ColorRole.Text, text_pri)
        pal.setColor(grp, QPalette.ColorRole.Button, bg_elev)
        pal.setColor(grp, QPalette.ColorRole.ButtonText, text_pri)
        pal.setColor(grp, QPalette.ColorRole.BrightText, QColor("#FFFFFF"))
        pal.setColor(grp, QPalette.ColorRole.Highlight, accent_sub)
        pal.setColor(grp, QPalette.ColorRole.HighlightedText, text_pri)
        pal.setColor(grp, QPalette.ColorRole.Link, accent)
        pal.setColor(grp, QPalette.ColorRole.Dark, QColor(p.border_strong))
        pal.setColor(grp, QPalette.ColorRole.Mid, QColor(p.border))
        pal.setColor(grp, QPalette.ColorRole.Light, bg_elev)

    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.WindowText, text_mut)
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, text_mut)
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, text_mut)
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Button, bg_panel)
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, bg_panel)
    return pal


def detect_system_theme() -> str:
    """Detect OS light/dark preference via Qt styleHints with palette fallback."""
    app = QGuiApplication.instance()
    if app is not None:
        try:
            scheme = app.styleHints().colorScheme()
            if scheme == Qt.ColorScheme.Light:
                return "light"
            if scheme == Qt.ColorScheme.Dark:
                return "dark"
        except Exception:
            pass
        try:
            if app.palette().window().color().lightness() >= 128:
                return "light"
        except Exception:
            pass
    return "dark"


_SYSTEM_PALETTE = None


def apply_theme(app: QApplication, mode: str = "dark", target: QWidget | None = None) -> ThemePalette:
    """Apply Dark or Light palette and QSS stylesheet to the app/target window."""
    global _SYSTEM_PALETTE
    if _SYSTEM_PALETTE is None:
        _SYSTEM_PALETTE = app.palette()
    palette = DARK_PALETTE if mode == "dark" else LIGHT_PALETTE
    app.setStyle("Fusion")
    app.setPalette(_build_qpalette(palette))
    qss = build_stylesheet(palette)
    if target is not None:
        target.setStyleSheet(qss)
    else:
        app.setStyleSheet(qss)
    return palette


def _make_unthemed_file_dialog(caption: str, directory: str, name_filter: str = "") -> QFileDialog:
    from PyQt6.QtWidgets import QFileDialog

    dlg = QFileDialog(None, caption, directory, name_filter)
    app = QApplication.instance()
    if _SYSTEM_PALETTE is not None:
        dlg.setPalette(_SYSTEM_PALETTE)
    elif isinstance(app, QApplication):
        dlg.setPalette(app.style().standardPalette())
    return dlg


def open_file_dialog(caption: str, directory: str, name_filter: str = "") -> str:
    from PyQt6.QtWidgets import QFileDialog

    dlg = _make_unthemed_file_dialog(caption, directory, name_filter)
    dlg.setFileMode(QFileDialog.FileMode.ExistingFile)
    dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptOpen)
    if dlg.exec():
        files = dlg.selectedFiles()
        return files[0] if files else ""
    return ""


def save_file_dialog(caption: str, directory: str, name_filter: str = "") -> str:
    from PyQt6.QtWidgets import QFileDialog

    dlg = _make_unthemed_file_dialog(caption, directory, name_filter)
    dlg.setFileMode(QFileDialog.FileMode.AnyFile)
    dlg.setAcceptMode(QFileDialog.AcceptMode.AcceptSave)
    if dlg.exec():
        files = dlg.selectedFiles()
        return files[0] if files else ""
    return ""


def select_directory_dialog(caption: str, directory: str) -> str:
    from PyQt6.QtWidgets import QFileDialog

    dlg = _make_unthemed_file_dialog(caption, directory)
    dlg.setFileMode(QFileDialog.FileMode.Directory)
    dlg.setOption(QFileDialog.Option.ShowDirsOnly, True)
    if dlg.exec():
        files = dlg.selectedFiles()
        return files[0] if files else ""
    return ""
