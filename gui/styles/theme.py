"""
DeStripe GUI — Dark Teal Medical Imaging Theme (QSS)
Design system: OLED navy + electric teal accent, glassmorphism panels.
"""


def get_qss() -> str:
    return """
/* ───────────────────────────── GLOBAL ───────────────────────────── */
* {
    font-family: "Segoe UI", "Inter", "SF Pro Text", sans-serif;
    font-size: 13px;
    color: #E2EAF4;
    selection-background-color: #00D4D8;
    selection-color: #0A0F1E;
    outline: none;
}

QMainWindow, QDialog {
    background-color: #0A0F1E;
}

QWidget {
    background-color: transparent;
}

/* ───────────────────────────── TAB BAR ──────────────────────────── */
QTabWidget::pane {
    background-color: #0D1526;
    border: 1px solid rgba(0, 212, 216, 0.12);
    border-radius: 12px;
    margin-top: -1px;
}

QTabBar {
    background: transparent;
}

QTabBar::tab {
    background: rgba(15, 24, 41, 0.6);
    color: #64748B;
    border: 1px solid rgba(0, 212, 216, 0.08);
    border-bottom: none;
    padding: 10px 22px;
    min-width: 110px;
    margin-right: 4px;
    border-radius: 8px 8px 0 0;
    font-weight: 500;
    font-size: 13px;
    letter-spacing: 0.3px;
}

QTabBar::tab:selected {
    background: rgba(0, 212, 216, 0.10);
    color: #00D4D8;
    border-color: rgba(0, 212, 216, 0.35);
    border-bottom: 2px solid #00D4D8;
    font-weight: 600;
}

QTabBar::tab:hover:!selected {
    background: rgba(0, 212, 216, 0.05);
    color: #A8D4D4;
    border-color: rgba(0, 212, 216, 0.15);
}

/* ──────────────────────────── BUTTONS ───────────────────────────── */
QPushButton {
    background: rgba(0, 212, 216, 0.08);
    color: #00D4D8;
    border: 1px solid rgba(0, 212, 216, 0.28);
    border-radius: 6px;
    padding: 5px 14px;
    font-weight: 500;
    font-size: 12px;
    min-height: 26px;
}

QPushButton:hover {
    background: rgba(0, 212, 216, 0.16);
    border-color: rgba(0, 212, 216, 0.60);
    color: #FFFFFF;
}

QPushButton:pressed {
    background: rgba(0, 212, 216, 0.25);
    border-color: #00D4D8;
}

QPushButton:disabled {
    background: rgba(30, 40, 60, 0.4);
    color: #2D3D55;
    border-color: rgba(45, 61, 85, 0.4);
}

QPushButton#btn_primary {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #00B8BC, stop:1 #0EA5E9);
    color: #0A0F1E;
    border: none;
    font-weight: 600;
}

QPushButton#btn_primary:hover {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #00D4D8, stop:1 #38BDF8);
    color: #020810;
}

QPushButton#btn_primary:pressed {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #009EA2, stop:1 #0284C7);
}

QPushButton#btn_danger {
    background: rgba(239, 68, 68, 0.10);
    color: #F87171;
    border-color: rgba(239, 68, 68, 0.30);
}

QPushButton#btn_danger:hover {
    background: rgba(239, 68, 68, 0.20);
    border-color: #EF4444;
}

/* ─────────────────────────── LINE EDIT ──────────────────────────── */
QLineEdit {
    background: rgba(15, 24, 41, 0.8);
    border: 1px solid rgba(0, 212, 216, 0.18);
    border-radius: 7px;
    padding: 8px 12px;
    color: #E2EAF4;
    selection-background-color: #00D4D8;
    font-size: 13px;
}

QLineEdit:focus {
    border-color: rgba(0, 212, 216, 0.65);
    background: rgba(0, 212, 216, 0.04);
}

QLineEdit:disabled {
    color: #2D3D55;
    border-color: rgba(45, 61, 85, 0.4);
}

/* ─────────────────────────── PROGRESS BAR ───────────────────────── */
QProgressBar {
    background: rgba(15, 24, 41, 0.9);
    border: 1px solid rgba(0, 212, 216, 0.15);
    border-radius: 5px;
    height: 7px;
    text-align: center;
    color: transparent;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #00B8BC, stop:1 #0EA5E9);
    border-radius: 5px;
}

/* ──────────────────────────── LABELS ────────────────────────────── */
QLabel#lbl_title {
    font-size: 22px;
    font-weight: 700;
    color: #00D4D8;
    letter-spacing: -0.5px;
}

QLabel#lbl_section {
    font-size: 11px;
    font-weight: 600;
    color: #4A6080;
    letter-spacing: 1.2px;
    text-transform: uppercase;
}

QLabel#lbl_metric {
    font-size: 12px;
    font-weight: 600;
    color: #00D4D8;
    background: rgba(0, 212, 216, 0.08);
    border: 1px solid rgba(0, 212, 216, 0.20);
    border-radius: 5px;
    padding: 4px 10px;
}

QLabel#lbl_badge_gpu {
    font-size: 11px;
    font-weight: 700;
    color: #0A0F1E;
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
        stop:0 #00B8BC, stop:1 #0EA5E9);
    border-radius: 4px;
    padding: 3px 9px;
}

QLabel#lbl_badge_cpu {
    font-size: 11px;
    font-weight: 700;
    color: #0A0F1E;
    background: #64748B;
    border-radius: 4px;
    padding: 3px 9px;
}

QLabel#lbl_status_ok {
    color: #10B981;
    font-weight: 600;
}

QLabel#lbl_status_err {
    color: #F87171;
    font-weight: 600;
}

/* ─────────────────────────── SCROLL BARS ────────────────────────── */
QScrollBar:vertical {
    background: rgba(10, 15, 30, 0.6);
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: rgba(0, 212, 216, 0.25);
    border-radius: 4px;
    min-height: 24px;
}

QScrollBar::handle:vertical:hover {
    background: rgba(0, 212, 216, 0.50);
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0;
}

QScrollBar:horizontal {
    background: rgba(10, 15, 30, 0.6);
    height: 8px;
    border-radius: 4px;
}

QScrollBar::handle:horizontal {
    background: rgba(0, 212, 216, 0.25);
    border-radius: 4px;
    min-width: 24px;
}

QScrollBar::handle:horizontal:hover {
    background: rgba(0, 212, 216, 0.50);
}

QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0;
}

/* ──────────────────────────── TABLES ────────────────────────────── */
QTableWidget {
    background: rgba(13, 21, 38, 0.95);
    border: 1px solid rgba(0, 212, 216, 0.12);
    border-radius: 10px;
    gridline-color: rgba(0, 212, 216, 0.08);
    alternate-background-color: rgba(0, 212, 216, 0.03);
}

QTableWidget::item {
    padding: 8px 12px;
    border: none;
    color: #C8D8EC;
}

QTableWidget::item:selected {
    background: rgba(0, 212, 216, 0.12);
    color: #E2EAF4;
}

QHeaderView::section {
    background: rgba(10, 15, 30, 0.95);
    color: #4A6080;
    border: none;
    border-bottom: 1px solid rgba(0, 212, 216, 0.15);
    padding: 8px 12px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

/* ─────────────────────────── LIST WIDGET ────────────────────────── */
QListWidget {
    background: rgba(13, 21, 38, 0.95);
    border: 1px solid rgba(0, 212, 216, 0.12);
    border-radius: 10px;
    padding: 4px;
}

QListWidget::item {
    padding: 9px 12px;
    border-radius: 6px;
    color: #C8D8EC;
    margin: 2px 0;
}

QListWidget::item:selected {
    background: rgba(0, 212, 216, 0.12);
    color: #00D4D8;
}

QListWidget::item:hover:!selected {
    background: rgba(0, 212, 216, 0.05);
}

/* ─────────────────────────── FRAME / CARD ───────────────────────── */
QFrame#card {
    background: rgba(15, 24, 41, 0.7);
    border: 1px solid rgba(0, 212, 216, 0.12);
    border-radius: 12px;
}

QFrame#card_elevated {
    background: rgba(22, 32, 53, 0.85);
    border: 1px solid rgba(0, 212, 216, 0.20);
    border-radius: 12px;
}

/* ──────────────────────────── SPLITTER ──────────────────────────── */
QSplitter::handle {
    background: rgba(0, 212, 216, 0.10);
    width: 4px;
    border-radius: 2px;
}

QSplitter::handle:hover {
    background: rgba(0, 212, 216, 0.35);
}

/* ───────────────────────── STATUS BAR ───────────────────────────── */
QStatusBar {
    background: rgba(6, 10, 20, 0.95);
    border-top: 1px solid rgba(0, 212, 216, 0.10);
    color: #4A6080;
    font-size: 12px;
}

QStatusBar::item {
    border: none;
}

/* ─────────────────────────── TOOLTIPS ───────────────────────────── */
QToolTip {
    background: #0D1526;
    color: #E2EAF4;
    border: 1px solid rgba(0, 212, 216, 0.35);
    border-radius: 6px;
    padding: 6px 10px;
    font-size: 12px;
}

/* ──────────────────────────── COMBO BOX ─────────────────────────── */
QComboBox {
    background: rgba(15, 24, 41, 0.8);
    border: 1px solid rgba(0, 212, 216, 0.18);
    border-radius: 7px;
    padding: 7px 12px;
    color: #E2EAF4;
    min-height: 32px;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background: #0D1526;
    border: 1px solid rgba(0, 212, 216, 0.25);
    border-radius: 8px;
    selection-background-color: rgba(0, 212, 216, 0.15);
    selection-color: #00D4D8;
    padding: 4px;
}

/* ─────────────────────────── SPIN BOX ───────────────────────────── */
QSpinBox, QDoubleSpinBox {
    background: rgba(15, 24, 41, 0.8);
    border: 1px solid rgba(0, 212, 216, 0.18);
    border-radius: 7px;
    padding: 7px 10px;
    color: #E2EAF4;
}

QSpinBox:focus, QDoubleSpinBox:focus {
    border-color: rgba(0, 212, 216, 0.65);
}

/* ─────────────────────────── CHECK BOX ──────────────────────────── */
QCheckBox {
    spacing: 8px;
    color: #C8D8EC;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 1.5px solid rgba(0, 212, 216, 0.35);
    background: rgba(15, 24, 41, 0.8);
}

QCheckBox::indicator:checked {
    background: rgba(0, 212, 216, 0.20);
    border-color: #00D4D8;
    image: none;
}

QCheckBox::indicator:hover {
    border-color: rgba(0, 212, 216, 0.65);
}

/* ─────────────────────────── GROUP BOX ──────────────────────────── */
QGroupBox {
    background: rgba(13, 21, 38, 0.6);
    border: 1px solid rgba(0, 212, 216, 0.12);
    border-radius: 10px;
    margin-top: 18px;
    padding-top: 10px;
    font-weight: 600;
    color: #4A6080;
    font-size: 11px;
    letter-spacing: 0.8px;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: -8px;
    background: #0A0F1E;
    padding: 2px 8px;
    border-radius: 4px;
    color: rgba(0, 212, 216, 0.70);
    font-size: 10px;
    letter-spacing: 1.0px;
    text-transform: uppercase;
}
"""
