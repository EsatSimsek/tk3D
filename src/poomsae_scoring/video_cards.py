"""Typography and layout for evidence videos; no measurement or scoring logic."""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
from PIL import Image, ImageDraw, ImageFont

INK = "#21373D"
MUTED = "#63757A"
PAPER = "#EEF1EF"
LINE = "#D7DFDC"
ACCENTS = {"red": "#B5483F", "amber": "#986417", "green": "#287158", "blue": "#346E91", "gray": "#64747A"}


@lru_cache(maxsize=24)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    filename = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / filename), size)


def _text(draw: ImageDraw.ImageDraw, text: Any, x: int, y: int, width: int,
          *, size: int = 18, color: str = INK, bold: bool = False, lines: int = 1) -> None:
    """Wrap within a fixed region, including unusually long metric identifiers."""
    font = _font(size, bold)
    pending = str(text or "—").strip()
    for row in range(lines):
        if not pending:
            break
        stop = len(pending)
        while stop > 0 and draw.textlength(pending[:stop], font=font) > width:
            stop -= 1
        if stop < len(pending) and " " in pending[:stop]:
            stop = pending.rfind(" ", 0, stop) or stop
        stop = max(stop, 1)
        part, pending = pending[:stop], pending[stop:].lstrip()
        if row == lines - 1 and pending:
            while part and draw.textlength(part + "…", font=font) > width:
                part = part[:-1]
            part += "…"
        draw.text((x, y + row * (size + 5)), part, font=font, fill=color)


def _foot_schematic(draw: ImageDraw.ImageDraw, event: dict, x: int, y: int, width: int) -> None:
    draw.rounded_rectangle((x, y, x + width, y + 130), radius=7, fill=PAPER)
    _text(draw, "Üstten 3B şema", x + 10, y + 8, width - 20, size=13, bold=True)
    _text(draw, "Kamera görüntüsü değil", x + 10, y + 26, width - 20, size=11, color=MUTED)
    measurement = event.get("measurement") or {}
    limits = measurement.get("rule_limits") or []
    if not limits:
        return
    angle = float(limits[0])
    ox, oy, radius = x + width // 2, y + 116, 63

    def endpoint(degrees: float) -> tuple[float, float]:
        rad = math.radians(degrees)
        return ox + radius * math.sin(rad), oy - radius * math.cos(rad)

    points = [(ox, oy)] + [endpoint(a) for a in np.linspace(-angle, angle, 25)]
    draw.polygon(points, fill="#D3E5DC")
    for a in (-angle, angle):
        draw.line([(ox, oy), endpoint(a)], fill=ACCENTS["green"], width=2)
    for offset in range(0, 63, 9):
        draw.line((ox, oy - offset, ox, oy - offset - 4), fill=MUTED, width=2)
    value = measurement.get("value")
    if value is not None:
        draw.line([(ox, oy), endpoint(float(np.clip(value, -85, 85)))],
                  fill=ACCENTS.get(event.get("display_color"), MUTED), width=4)
    draw.ellipse((ox - 3, oy - 3, ox + 3, oy + 3), fill=INK)


@lru_cache(maxsize=48)
def _card(serialized: str, number: int, width: int, height: int) -> Image.Image:
    event = json.loads(serialized)
    explanation = event.get("user_explanation") or {}
    measurement = event.get("measurement") or {}
    color = ACCENTS.get(event.get("display_color"), MUTED)
    card = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(card)
    draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=10, outline=LINE, width=1)
    draw.rounded_rectangle((18, 17, 47, 46), radius=7, fill=color)
    _text(draw, number, 26, 20, 23, size=17, color="white", bold=True)
    title = explanation.get("title") or event.get("description") or event.get("metric_id")
    title = {
        "ARKA AYAK ACISI": "Arka ayak açısı",
        "ARAE-MAKKI YUMRUK-UYLUK MESAFESI": "Arae-makki · yumruk–uyluk mesafesi",
    }.get(title, title)
    _text(draw, title, 58, 18, width - 76, size=20, bold=True)
    points = event.get("deduction_points")
    status = event.get("display_label") or "İnceleme"
    if points is not None:
        status += f" · −{float(points):g}"
    elif "puan yok" not in status.lower():
        status += " · puan kesintisi yok"
    _text(draw, status, 58, 46, width - 76, size=15, color=color)
    measured = explanation.get("measured") or str(measurement.get("value", "Ölçüm yok"))
    if measurement.get("value") is None:
        measured = "Ölçülemedi"
    _text(draw, measured, 20, 77, width - 40, size=29, bold=True)
    _text(draw, explanation.get("expected") or event.get("description"),
          20, 119, width - 40, size=17, lines=2)
    _text(draw, explanation.get("interval") or "Belirsizlik bilgisi yok.",
          20, 164, width - 40, size=15, lines=2, color=MUTED)
    _text(draw, explanation.get("comparison") or "Fark bilgisi yok.",
          20, 205, width - 40, size=16, lines=2, color=color)
    _text(draw, explanation.get("result") or status,
          20, 248, width - 40, size=16, lines=2)
    draw.line((20, 295, width - 20, 295), fill=LINE)
    has_schematic = (event.get("visual_geometry") or {}).get("kind") == "foot_direction_angle"
    text_width = width - (226 if has_schematic else 40)
    _text(draw, "İNCELEME NOTU", 20, 307, text_width, size=11, bold=True, color=MUTED)
    _text(draw, explanation.get("correction") or "Hareketi kaynak tanımıyla karşılaştırın.",
          20, 326, text_width, size=15, lines=2)
    _text(draw, explanation.get("source_note") or "Kaynak ayrıntıları karar raporundadır.",
          20, 374, text_width, size=12, lines=3, color=MUTED)
    if has_schematic:
        _foot_schematic(draw, event, width - 192, 302, 174)
    return card


def draw_evidence_panel(canvas: np.ndarray, events: list[dict], frame: int, fps: float, top: int) -> None:
    width, height = canvas.shape[1], canvas.shape[0] - top
    panel = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(panel)
    _text(draw, "TK3D  /  HAREKET İNCELEMESİ", 24, 16, 480, size=15, bold=True, color=MUTED)
    _text(draw, f"{frame / fps:06.3f} s   ·   Kare {frame}", width - 310, 16, 286, size=16, color=MUTED)
    if not events:
        _text(draw, "Bu kare için aktif inceleme bulgusu yok.", 24, 103, width - 48, size=28, bold=True)
        _text(draw, "Bu ifade hareketin doğru olduğu anlamına gelmez. Kanıt pencereleri hareket boyunca değişir.",
              24, 155, width - 48, size=19, color=MUTED)
    else:
        movement = events[0]
        _text(draw, f"{movement.get('movement_id') or 'Performans'}  ·  {movement.get('movement_name') or ''}",
              24, 43, width - 420, size=20, bold=True)
        if len(events) > 3:
            _text(draw, f"+{len(events) - 3} bulgu HTML raporunda", width - 350, 46, 326, size=15, color=MUTED)
        visible = events[:3]
        gap, margin = 16, 24
        card_width = (width - 2 * margin - gap * (len(visible) - 1)) // len(visible)
        for index, event in enumerate(visible):
            card = _card(json.dumps(event, sort_keys=True, ensure_ascii=False), index + 1, card_width, height - 104)
            panel.paste(card, (margin + index * (card_width + gap), 78))
    _text(draw, "Kamera çizgileri: gözlenen 2B iz  ·  Ölçümler: kalibre 3B  ·  Şemada yeşil: kural aralığı  ·  Aday kararlar resmî puan değildir",
          24, height - 23, width - 48, size=13, color=MUTED)
    canvas[top:] = np.asarray(panel)[:, :, ::-1]


def draw_pause_banner(canvas: np.ndarray, seconds_left: float) -> None:
    width = canvas.shape[1]
    banner = Image.new("RGB", (width, 48), "#21373D")
    draw = ImageDraw.Draw(banner)
    _text(draw, "KESİNTİ ADAYI  ·  Resmî puan değil", 24, 11, width - 390, size=20, color="white", bold=True)
    _text(draw, f"Okuma arası · {seconds_left:.1f} s", width - 330, 12, 310, size=18, color="#D9E4E1")
    canvas[42:90] = np.asarray(banner)[:, :, ::-1]
