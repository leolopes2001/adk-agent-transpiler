"""A synthetic order printed on ruled notebook paper, for the line-removal tests."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = Path(__file__).resolve().parent / "fixtures" / "fonts" / "DejaVuSans.ttf"
TEXT_ROWS = [
    "Paciente em jejum de 8 horas",
    "1. Hemograma completo",
    "2. Glicemia de jejum",
    "3. TSH",
    "4. Ferritina",
    "Solicito os seguintes exames",
]
WIDTH, HEIGHT = 1500, 800
ROW_PITCH = 110
FIRST_ROW = 60
FONT_SIZE = 40
LINE_THICKNESS = 3
LINE_GRAY = 90
MARGIN = 20
# The ruled line crosses the letters halfway up.
LINE_OFFSET = 22


def line_rows() -> list[range]:
    starts = [FIRST_ROW + row * ROW_PITCH + LINE_OFFSET for row in range(len(TEXT_ROWS))]
    return [range(start, start + LINE_THICKNESS) for start in starts]


def text_page() -> np.ndarray:
    page = Image.new("L", (WIDTH, HEIGHT), 255)
    draw = ImageDraw.Draw(page)
    font = ImageFont.truetype(str(FONT), FONT_SIZE)
    for row, text in enumerate(TEXT_ROWS):
        draw.text((80, FIRST_ROW + row * ROW_PITCH), text, font=font, fill=0)

    return np.asarray(page)


def ruled_page() -> np.ndarray:
    page = text_page().copy()
    for rows in line_rows():
        band = page[rows.start : rows.stop, MARGIN : WIDTH - MARGIN]
        np.minimum(band, LINE_GRAY, out=band)

    return page
