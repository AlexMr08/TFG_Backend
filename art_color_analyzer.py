"""
art_color_analyzer.py
---------------------
Recibe los 5 colores dominantes (RGB + porcentaje) y construye
el payload listo para enviar a un LLM.

Dependencias: ninguna (solo stdlib)
"""

from __future__ import annotations
import colorsys
from dataclasses import dataclass


@dataclass
class ColorInfo:
    hex: str
    rgb: tuple[int, int, int]
    hue: float          # 0–360°
    saturation: float   # 0–100%
    brightness: float   # 0–100%
    percentage: float   # % del lienzo
    temperature: str    # "warm" | "cool" | "neutral"
    harmony_role: str   # "dominant" | "secondary" | "accent"


def _classify_temperature(hue: float) -> str:
    if hue <= 60 or hue >= 300:
        return "warm"
    elif 160 <= hue < 300:
        return "cool"
    else:
        return "neutral"


def _classify_harmony_role(percentage: float, rank: int) -> str:
    if rank == 0:
        return "dominant"
    elif percentage >= 15:
        return "secondary"
    else:
        return "accent"


def _classify_harmony_type(colors: list[ColorInfo]) -> str:
    hues = [c.hue for c in colors if c.saturation > 15]
    if len(hues) < 2:
        return "monochromatic"
    hue_range = max(hues) - min(hues)
    if hue_range < 30:
        return "monochromatic"
    elif hue_range < 60:
        return "analogous"
    for i, h1 in enumerate(hues):
        for h2 in hues[i + 1:]:
            if 150 <= abs(h1 - h2) <= 210:
                return "complementary"
    return "triadic"


def enrich_colors(
    raw_colors: list[tuple[tuple[int, int, int], float]]
) -> list[ColorInfo]:
    """
    Enriquece los colores con metadata semántica.

    Args:
        raw_colors: lista de ((R, G, B), percentage) ordenada por porcentaje desc.

    Returns:
        Lista de ColorInfo lista para construir el prompt.
    """
    result = []
    for rank, ((r, g, b), pct) in enumerate(raw_colors):
        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        result.append(ColorInfo(
            hex=f"#{r:02x}{g:02x}{b:02x}",
            rgb=(r, g, b),
            hue=round(h * 360, 1),
            saturation=round(s * 100, 1),
            brightness=round(v * 100, 1),
            percentage=round(pct, 1),
            temperature=_classify_temperature(round(h * 360, 1)),
            harmony_role=_classify_harmony_role(pct, rank),
        ))
    return result


def build_prompt(  # noqa: keep for future LLM integration
    raw_colors: list[tuple[tuple[int, int, int], float]],
    artwork_context: str = "",
) -> str:
    """
    Construye el prompt completo listo para enviar al LLM.

    Args:
        raw_colors:      Lista de ((R, G, B), percentage) desc. por porcentaje.
        artwork_context: Info adicional (título, artista, año...).

    Returns:
        String con el prompt listo para usar.
    """
    colors = enrich_colors(raw_colors)
    harmony = _classify_harmony_type(colors)
    temps = [c.temperature for c in colors]
    dominant_temp = max(set(temps), key=temps.count)
    avg_sat = round(sum(c.saturation * c.percentage for c in colors) / 100, 1)
    avg_bright = round(sum(c.brightness * c.percentage for c in colors) / 100, 1)

    color_lines = "\n".join([
        f"  {i+1}. {c.hex}  |  {c.harmony_role}  |  {c.temperature}"
        f"  |  sat {c.saturation}%  brightness {c.brightness}%  →  {c.percentage}% of canvas"
        for i, c in enumerate(colors)
    ])

    context_block = f"\nAdditional context: {artwork_context}" if artwork_context else ""

    return f"""You are an art critic specialized in chromatic analysis.
Analyze the following dominant color palette extracted from an artwork.{context_block}

Palette (ordered by presence, most to least):
{color_lines}

Global palette stats:
  - Dominant temperature: {dominant_temp}
  - Harmony type: {harmony}
  - Average saturation: {avg_sat}%
  - Average brightness: {avg_bright}%

Provide a brief analysis (3–4 sentences) covering:
- The emotional and psychological impact of this palette
- The artistic movement or style it may suggest
- The relationship between the colors (harmony or tension)
- The possible mood or narrative the artist is communicating

Reply with the analysis only, no headers or bullet points."""
