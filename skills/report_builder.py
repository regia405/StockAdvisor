"""
report_builder.py — Generates radar/spider charts and PDF reports for stock analysis.

Exports:
    generate_radar_chart(ticker, scores, output_dir="data") -> str
    export_pdf(ticker, report_sections, chart_paths, output_dir="data") -> str
"""

import subprocess
import sys

try:
    from fpdf import FPDF
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fpdf2", "--quiet"])
    from fpdf import FPDF

import os
import math
from datetime import date


def _ascii_safe(text: str) -> str:
    """Replace common Unicode characters with ASCII equivalents for fpdf Helvetica font."""
    if not text:
        return ""
    replacements = {
        "\u2014": "--", "\u2013": "-", "\u2012": "-",   # em/en dashes
        "\u201c": '"',  "\u201d": '"',                   # curly double quotes
        "\u2018": "'",  "\u2019": "'",                   # curly single quotes
        "\u2022": "-",  "\u2023": "-", "\u25cf": "-",    # bullets
        "\u00a0": " ",                                   # non-breaking space
        "\u2026": "...",                                 # ellipsis
        "\u00b7": ".",                                   # middle dot
        "\u2192": "->", "\u2190": "<-",                 # arrows
        "\u00d7": "x",  "\u00f7": "/",                  # multiply/divide
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    # Drop anything still outside latin-1
    return text.encode("latin-1", errors="replace").decode("latin-1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def generate_radar_chart(ticker, scores, output_dir="data"):
    """
    Generate a radar/spider chart for the given scores dict and save as PNG.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (used in title and filename).
    scores : dict
        Mapping of category label -> numeric score (0-10).
        Example: {"Fundamental": 6, "Technical": 2, ...}
    output_dir : str
        Directory to save the PNG file. Defaults to "data".

    Returns
    -------
    str
        Absolute path to the saved PNG file.
    """
    os.makedirs(output_dir, exist_ok=True)

    labels = list(scores.keys())
    values = [float(scores[k]) for k in labels]
    n = len(labels)

    if n < 3:
        raise ValueError("At least 3 score categories are required for a radar chart.")

    # Compute evenly-spaced angles; close the polygon by repeating the first point
    angles = [2 * math.pi * i / n for i in range(n)]
    angles_closed = angles + [angles[0]]
    values_closed = values + [values[0]]

    fig = plt.figure(figsize=(7, 7), facecolor="#1a1a2e")
    ax = fig.add_subplot(111, polar=True, facecolor="#16213e")

    # --- Grid rings at 2, 4, 6, 8, 10 ---
    ring_levels = [2, 4, 6, 8, 10]
    for level in ring_levels:
        ring_angles = np.linspace(0, 2 * math.pi, 360)
        ring_r = [level] * 360
        ax.plot(ring_angles, ring_r, color="#444466", linewidth=0.6, linestyle="--", zorder=1)
        # Label the ring value at 90 degrees (top)
        ax.text(
            math.pi / 2, level + 0.3,
            str(level),
            color="#888899",
            fontsize=7,
            ha="center",
            va="bottom",
        )

    # --- Spoke lines ---
    for angle in angles:
        ax.plot([angle, angle], [0, 10], color="#444466", linewidth=0.8, zorder=1)

    # --- Filled polygon ---
    ax.fill(
        angles_closed,
        values_closed,
        color="#4fc3f7",
        alpha=0.30,
        zorder=2,
    )
    ax.plot(
        angles_closed,
        values_closed,
        color="#4fc3f7",
        linewidth=2,
        zorder=3,
    )
    ax.scatter(
        angles,
        values,
        color="#4fc3f7",
        s=40,
        zorder=4,
    )

    # --- Axis labels with score ---
    ax.set_xticks(angles)
    ax.set_xticklabels([])  # We will draw custom labels below

    for i, (angle, label, value) in enumerate(zip(angles, labels, values)):
        # Determine horizontal alignment based on position
        x_deg = math.degrees(angle) % 360
        if x_deg < 10 or x_deg > 350:
            ha = "center"
        elif x_deg <= 180:
            ha = "left"
        else:
            ha = "right"

        # Radial offset beyond the max ring
        label_radius = 11.5
        ax.text(
            angle,
            label_radius,
            f"{label}\n{int(value)}/10",
            color="#e0e0e0",
            fontsize=9,
            fontweight="bold",
            ha=ha,
            va="center",
        )

    # Hide default polar ticks/labels
    ax.set_yticklabels([])
    ax.set_ylim(0, 13)
    ax.set_yticks([])
    ax.spines["polar"].set_visible(False)

    # --- Title ---
    ax.set_title(
        f"{ticker} -- Score Radar",
        color="#ffffff",
        fontsize=14,
        fontweight="bold",
        pad=30,
    )

    plt.tight_layout()

    output_path = os.path.join(output_dir, f"{ticker}_radar.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    return os.path.abspath(output_path)


def export_pdf(ticker, report_sections, chart_paths, output_dir="data"):
    """
    Export a full PDF report containing text sections and embedded chart images.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol.
    report_sections : dict
        Ordered mapping of section heading -> body text.
    chart_paths : list of str
        File paths to PNG images to embed (one per page, after text sections).
    output_dir : str
        Directory to save the PDF file. Defaults to "data".

    Returns
    -------
    str
        Absolute path to the saved PDF file.
    """
    os.makedirs(output_dir, exist_ok=True)

    report_date = date.today().strftime("%Y-%m-%d")

    class StockReport(FPDF):
        def header(self):
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(30, 30, 30)
            self.cell(0, 8, f"{ticker}  |  Stock Advisor Report", align="L", new_x="LMARGIN", new_y="NEXT")
            self.set_font("Helvetica", "", 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, f"Date: {report_date}", align="L", new_x="LMARGIN", new_y="NEXT")
            self.ln(2)
            # Horizontal rule
            self.set_draw_color(180, 180, 180)
            self.set_line_width(0.4)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(4)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            disclaimer = "This is not financial advice. Always do your own research."
            self.cell(0, 5, disclaimer, align="L")
            self.set_y(-15)
            self.cell(0, 5, f"Page {self.page_no()}", align="R")

    pdf = StockReport(orientation="P", unit="mm", format="A4")
    pdf.set_margins(left=15, top=20, right=15)
    pdf.set_auto_page_break(auto=True, margin=18)

    # --- Text sections ---
    pdf.add_page()

    for heading, body in report_sections.items():
        # Section heading
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(20, 60, 120)
        pdf.cell(0, 8, _ascii_safe(heading), new_x="LMARGIN", new_y="NEXT")

        # Thin underline beneath heading
        pdf.set_draw_color(20, 60, 120)
        pdf.set_line_width(0.3)
        pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
        pdf.ln(2)

        # Body text
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)
        if body:
            pdf.multi_cell(0, 5.5, _ascii_safe(body))
        else:
            pdf.multi_cell(0, 5.5, "No data available.")
        pdf.ln(5)

    # --- Chart pages ---
    for chart_path in chart_paths:
        chart_path = os.path.abspath(chart_path)
        if not os.path.isfile(chart_path):
            continue

        pdf.add_page()

        # Available content area
        usable_w = pdf.w - pdf.l_margin - pdf.r_margin
        usable_h = pdf.h - pdf.t_margin - 18  # leave room for footer

        # Place image centred, scaled to fit
        pdf.image(
            chart_path,
            x=pdf.l_margin,
            y=pdf.get_y(),
            w=usable_w,
            h=usable_h,
            keep_aspect_ratio=True,
        )

    output_path = os.path.join(output_dir, f"{ticker}_report.pdf")
    pdf.output(output_path)

    return os.path.abspath(output_path)


# ---------------------------------------------------------------------------
# Demo / self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    TICKER = "AAPL"

    demo_scores = {
        "Fundamental": 6,
        "Technical":   2,
        "News":        3,
        "Patterns":    5,
        "Analyst":     7,
        "Peer":        4,
    }

    demo_sections = {
        "Fundamental Analysis": (
            "Revenue grew 8% YoY. EPS of $6.43 beat estimates by $0.12. "
            "P/E ratio of 28.4 is slightly above the sector median of 24.1. "
            "Free cash flow remains strong at $92B TTM. DCF fair value estimated at $178."
        ),
        "Technical Analysis": (
            "Price is trading below the 50-day moving average ($189.20) and the 200-day MA ($194.05), "
            "indicating a bearish trend. RSI at 38 suggests the stock is approaching oversold territory. "
            "MACD is negative with a bearish crossover 5 sessions ago."
        ),
        "News & Sentiment": (
            "Recent headlines carry a mildly negative tone. Concerns around iPhone demand in China "
            "weighed on sentiment this week. Services revenue guidance was reaffirmed, providing "
            "some offset. Aggregate sentiment score: 3/10."
        ),
        "Historical Patterns": (
            "A descending triangle pattern has formed over the past 30 trading days. "
            "Historically, breakdowns from this pattern have led to a 5-10% decline. "
            "No strong bullish reversal signals detected at current levels."
        ),
        "Analyst Data": (
            "Consensus rating: Overweight (14 Buy, 6 Hold, 2 Sell). "
            "Average 12-month price target: $210.00 (+11% upside from current price). "
            "Most recent upgrade: Morgan Stanley raised to Overweight, target $220."
        ),
        "Peer Comparison": (
            "AAPL trades at a premium to peers MSFT (P/E 32) and GOOGL (P/E 22) on a blended basis. "
            "Revenue growth lags MSFT but margins remain superior. "
            "Relative strength vs. XLK sector index is -4.2% over 30 days."
        ),
        "Claude Recommendation": (
            "Overall signal score: 4.5/10 — Cautious / Hold.\n"
            "The stock faces near-term technical headwinds and soft sentiment, "
            "though strong fundamentals and analyst support provide a floor. "
            "A position re-evaluation is warranted if price breaks below the $182 support level.\n\n"
            "This is not financial advice. Always do your own research."
        ),
    }

    print("Generating radar chart...")
    radar_path = generate_radar_chart(TICKER, demo_scores, output_dir="data")
    print(f"  Saved: {radar_path}")

    print("Exporting PDF report...")
    pdf_path = export_pdf(
        TICKER,
        demo_sections,
        chart_paths=[radar_path],
        output_dir="data",
    )
    print(f"  Saved: {pdf_path}")

    print("Done.")
