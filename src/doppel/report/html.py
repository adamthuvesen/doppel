"""HTML rendering of a QualityReport — self-contained single file with inline CSS.

Visual language: the "doppelgaenger readout". A light, editorial precision theme built
around the real-vs-synth twin motif — real data reads warm (amber), synthetic reads cool
(teal), mirrored throughout. The centrepiece is the side-by-side correlation heatmaps
(real / synth / delta) so a viewer literally sees the synthetic structure shadow the
original. The file is fully self-contained — no external fonts, scripts, or stylesheets —
so it renders identically offline and survives archival. Typography leans on a
high-contrast system serif for display and a monospace for every figure.
"""

from __future__ import annotations

import math
from html import escape

from doppel.quality.aggregate import QualityReport

# perceptual tints used by the heatmaps and quality ramp, as raw RGB triples
_REAL_RGB = (194, 116, 26)
_SYNTH_RGB = (14, 155, 115)
_DELTA_RGB = (214, 73, 63)

_CSS = """
:root {
  --bg: #f3f0e9;
  --panel: rgba(255,255,255,0.62);
  --panel-2: rgba(255,255,255,0.9);
  --ink: #1b1f26;
  --muted: #5d636e;
  --faint-ink: #9a9aa0;
  --line: rgba(20,24,30,0.12);
  --track: rgba(20,24,30,0.08);
  --real: #c2741a;
  --synth: #0e9b73;
  --bad: #d6493f;
  --warn: #c8861a;
  --good: #1aa06e;
  --display: 'Iowan Old Style', 'Palatino Linotype', Palatino, 'Hoefler Text', Georgia, serif;
  --body: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
  --mono: ui-monospace, 'SF Mono', SFMono-Regular, Menlo, 'Cascadia Code', monospace;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  font-family: var(--body);
  color: var(--ink);
  background: var(--bg);
  margin: 0;
  padding: 0 1.5rem 6rem;
  line-height: 1.55;
  -webkit-font-smoothing: antialiased;
  background-image:
    radial-gradient(60rem 40rem at 12% -8%, rgba(194,116,26,0.10), transparent 60%),
    radial-gradient(60rem 40rem at 92% 8%, rgba(14,155,115,0.10), transparent 60%),
    radial-gradient(80rem 60rem at 50% 120%, rgba(14,155,115,0.05), transparent 60%);
  background-attachment: fixed;
}
body::before {
  content: "";
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  opacity: 0.45; mix-blend-mode: multiply;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='160' height='160'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.04'/%3E%3C/svg%3E");
}
main { position: relative; z-index: 1; max-width: 1040px; margin: 0 auto; }

/* ---- masthead ---- */
header { padding: 4.5rem 0 2.5rem; }
.kicker {
  font-family: var(--mono);
  font-size: 0.72rem; letter-spacing: 0.42em; text-transform: uppercase;
  color: var(--synth); margin: 0 0 1.1rem;
}
.wordmark { position: relative; width: fit-content; }
.wordmark h1 {
  font-family: var(--display);
  font-weight: 900; font-size: clamp(3.4rem, 11vw, 7rem);
  letter-spacing: -0.03em; line-height: 0.86; margin: 0;
  background: linear-gradient(92deg, var(--real), #1b1f26 48%, var(--synth));
  -webkit-background-clip: text; background-clip: text; color: transparent;
}
.wordmark .ghost {
  position: absolute; left: 0; top: 100%;
  font-family: var(--display); font-weight: 900;
  font-size: clamp(3.4rem, 11vw, 7rem); letter-spacing: -0.03em; line-height: 0.86;
  margin: 0; color: rgba(27,31,38,0.07);
  transform: scaleY(-1); transform-origin: top;
  -webkit-mask-image: linear-gradient(to bottom, rgba(0,0,0,0.55), transparent 60%);
  mask-image: linear-gradient(to bottom, rgba(0,0,0,0.55), transparent 60%);
  pointer-events: none; user-select: none;
}
.lede {
  margin: 3.2rem 0 0; max-width: 46ch;
  font-size: 1.05rem; color: var(--muted);
}
.pair { display: flex; flex-wrap: wrap; align-items: baseline; gap: 0.6rem 1rem; margin-top: 1.6rem; }
.tag {
  font-family: var(--mono); font-size: 0.82rem;
  padding: 0.32rem 0.7rem; border-radius: 999px;
  border: 1px solid var(--line); background: var(--panel-2);
}
.tag .rows { color: var(--faint-ink); }
.tag.real { border-color: rgba(194,116,26,0.45); color: var(--real); }
.tag.synth { border-color: rgba(14,155,115,0.45); color: var(--synth); }
.mirror { color: var(--faint-ink); font-size: 1.2rem; }

/* ---- section scaffolding ---- */
section { margin-top: 4.5rem; }
.eyebrow {
  font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.28em;
  text-transform: uppercase; color: var(--faint-ink); margin: 0 0 0.5rem;
}
h2 {
  font-family: var(--display); font-weight: 600;
  font-size: 1.9rem; letter-spacing: -0.01em; margin: 0 0 0.4rem;
}
.note { color: var(--muted); margin: 0 0 1.4rem; max-width: 62ch; font-size: 0.95rem; }

/* ---- verdict cards ---- */
.verdict { display: grid; grid-template-columns: repeat(3, 1fr); gap: 1rem; }
.card {
  border: 1px solid var(--line); border-radius: 16px;
  background: linear-gradient(180deg, var(--panel-2), var(--panel));
  box-shadow: 0 1px 2px rgba(20,24,30,0.04), 0 12px 30px -22px rgba(20,24,30,0.35);
  padding: 1.4rem 1.4rem 1.5rem; position: relative; overflow: hidden;
}
.card .label {
  font-family: var(--mono); font-size: 0.72rem; letter-spacing: 0.16em;
  text-transform: uppercase; color: var(--muted);
}
.card .value {
  font-family: var(--mono); font-weight: 700;
  font-size: 2.6rem; letter-spacing: -0.02em; margin: 0.5rem 0 0.1rem;
  font-variant-numeric: tabular-nums;
}
.card .unit { color: var(--faint-ink); font-size: 0.85rem; }
.chip {
  display: inline-block; font-family: var(--mono); font-size: 0.68rem;
  letter-spacing: 0.08em; text-transform: uppercase; font-weight: 700;
  padding: 0.24rem 0.6rem; border-radius: 6px; margin-top: 0.9rem;
}
.meter { height: 4px; border-radius: 99px; background: var(--track); margin-top: 1rem; overflow: hidden; }
.meter > i { display: block; height: 100%; border-radius: 99px; }

/* ---- marginals bars ---- */
.bars { display: flex; flex-direction: column; gap: 0; border-top: 1px solid var(--line); }
.bar-row {
  display: grid;
  grid-template-columns: minmax(9rem, 1.4fr) 5.2rem minmax(8rem, 2fr) 4.5rem 6rem;
  align-items: center; gap: 0.9rem;
  padding: 0.62rem 0; border-bottom: 1px solid var(--line);
}
.bar-row .col { font-weight: 600; font-size: 0.95rem; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.bar-row .meta { font-family: var(--mono); font-size: 0.68rem; color: var(--faint-ink); }
.bar-row .meta .mtype { text-transform: uppercase; letter-spacing: 0.06em; }
.track { height: 8px; border-radius: 99px; background: var(--track); position: relative; overflow: hidden; }
.track > i { position: absolute; left: 0; top: 0; bottom: 0; border-radius: 99px; }
.bar-row .score { font-family: var(--mono); font-size: 0.92rem; text-align: right; font-variant-numeric: tabular-nums; }
.bar-row .nulls { font-family: var(--mono); font-size: 0.72rem; color: var(--muted); text-align: right; line-height: 1.35; }
.bar-row .nulls .leak { color: var(--warn); font-weight: 700; }
.legend-min { display: flex; justify-content: space-between; font-family: var(--mono); font-size: 0.68rem; color: var(--faint-ink); margin-top: 0.7rem; }

/* ---- heatmaps ---- */
.maps { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 1.6rem; margin-top: 0.5rem; }
@media (max-width: 820px) { .maps { grid-template-columns: 1fr; } .verdict { grid-template-columns: 1fr; } }
.map h3 {
  font-family: var(--mono); font-size: 0.74rem; letter-spacing: 0.18em;
  text-transform: uppercase; margin: 0 0 0.8rem; display: flex; align-items: center; gap: 0.5rem; color: var(--muted);
}
.map h3 .dot { width: 9px; height: 9px; border-radius: 99px; display: inline-block; }
.grid { display: grid; gap: 2px; padding: 4px; border-radius: 8px; background: var(--panel); border: 1px solid var(--line); }
.cell { aspect-ratio: 1 / 1; border-radius: 2px; box-shadow: inset 0 0 0 1px rgba(20,24,30,0.04); }
.frob {
  display: flex; align-items: baseline; gap: 0.8rem; margin-top: 2rem;
  border: 1px solid var(--line); border-radius: 14px; padding: 1.1rem 1.4rem;
  background: var(--panel-2); box-shadow: 0 1px 2px rgba(20,24,30,0.04);
}
.frob .big { font-family: var(--mono); font-weight: 700; font-size: 2rem; font-variant-numeric: tabular-nums; }
.frob .cap { color: var(--muted); font-size: 0.9rem; }

/* ---- dcr strip ---- */
.strip { margin-top: 1rem; padding: 2.4rem 0.5rem 0.5rem; }
.rail { position: relative; height: 3px; background: linear-gradient(90deg, var(--bad), var(--warn) 35%, var(--good)); border-radius: 99px; }
.pin { position: absolute; top: 50%; transform: translate(-50%, -50%); text-align: center; }
.pin > b { display: block; width: 11px; height: 11px; border-radius: 99px; background: var(--ink); border: 2px solid var(--bg); margin: 0 auto; }
.pin > .lab { font-family: var(--mono); font-size: 0.62rem; letter-spacing: 0.08em; color: var(--faint-ink); text-transform: uppercase; margin-top: 0.55rem; }
.pin > .num { font-family: var(--mono); font-size: 0.82rem; font-variant-numeric: tabular-nums; display: block; }
.pin.hi { top: -1.7rem; }

/* ---- tables (calendar / issues) ---- */
table { border-collapse: collapse; width: 100%; font-size: 0.92rem; margin: 0.6rem 0 0; }
th, td { padding: 0.5rem 0.7rem; border-bottom: 1px solid var(--line); text-align: left; }
th { font-family: var(--mono); font-size: 0.68rem; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 500; }
td.num, th.num { text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; }
h3.sub { font-family: var(--mono); font-size: 0.78rem; letter-spacing: 0.12em; text-transform: uppercase; color: var(--muted); margin: 1.8rem 0 0; }
code { font-family: var(--mono); font-size: 0.88em; color: var(--synth); }

/* ---- callout ---- */
.warn {
  border: 1px solid rgba(200,134,26,0.35); border-left: 3px solid var(--warn);
  background: rgba(200,134,26,0.08); border-radius: 10px;
  padding: 1rem 1.2rem; margin: 1.2rem 0; color: #6b4e16; font-size: 0.92rem;
}
.warn strong { color: #8a6411; }
.warn ul { margin: 0.6rem 0 0; padding-left: 1.2rem; }
.warn code { color: #8a6411; }

footer {
  margin-top: 6rem; padding-top: 1.6rem; border-top: 1px solid var(--line);
  display: flex; justify-content: space-between; align-items: center;
  font-family: var(--mono); font-size: 0.72rem; color: var(--faint-ink); letter-spacing: 0.06em;
}
footer .brand { color: var(--muted); }
"""


def to_html(report: QualityReport) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>doppel quality report</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<header>
  <p class="kicker">doppel · fidelity report</p>
  <div class="wordmark">
    <h1>doppel&shy;ganger</h1>
    <div class="ghost" aria-hidden="true">doppel&shy;ganger</div>
  </div>
  <p class="lede">How faithfully the synthetic twin reproduces the statistical fingerprint
  of the source — marginals, correlation structure, and the privacy margin between them.</p>
  <div class="pair">
    <span class="tag real">{escape(report.real_label)} <span class="rows">· {report.real_rows:,} rows</span></span>
    <span class="mirror">⟷</span>
    <span class="tag synth">{escape(report.synth_label)} <span class="rows">· {report.synth_rows:,} rows</span></span>
  </div>
</header>

{_verdict(report)}

<section>
  <p class="eyebrow">per-column fidelity</p>
  <h2>Marginals</h2>
  <p class="note">Distance between each column's real and synthetic distribution
  (KS for numeric/temporal, total-variation for categorical). Shorter, greener bars are
  closer to the source; 0 is identical.</p>
  {_marginals(report)}
  {_verbatim_warning(report)}
</section>

{_correlation_section(report)}

{_calendar_section(report)}

{_privacy_section(report)}

{_issues(report)}

<footer>
  <span class="brand">generated by doppel</span>
  <span>statistical twin · not real records</span>
</footer>
</main>
</body>
</html>
"""


# --------------------------------------------------------------------------- #
# formatting helpers
# --------------------------------------------------------------------------- #
def _fmt(v: float) -> str:
    """Render a float as 4-decimal text, or an em-dash when it isn't finite."""
    return f"{v:.4f}" if math.isfinite(v) else "&mdash;"


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def _quality_rgb(fraction: float) -> tuple[int, int, int]:
    """Green (1.0) → amber (0.5) → red (0.0) across a quality fraction, tuned for a
    light background so every step stays legible."""
    f = _clamp(fraction)
    good, mid, bad = (26, 160, 110), (200, 134, 26), (214, 73, 63)
    if f >= 0.5:
        t = (f - 0.5) / 0.5
        a, b = mid, good
    else:
        t = f / 0.5
        a, b = bad, mid
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))  # type: ignore[return-value]


def _rgb(c: tuple[int, int, int]) -> str:
    return f"rgb({c[0]},{c[1]},{c[2]})"


def _rgba(c: tuple[int, int, int], alpha: float) -> str:
    return f"rgba({c[0]},{c[1]},{c[2]},{alpha:.3f})"


# --------------------------------------------------------------------------- #
# verdict cards
# --------------------------------------------------------------------------- #
def _verdict(report: QualityReport) -> str:
    marg = report.avg_marginal
    corr = report.correlations.frobenius_distance
    dcr = report.privacy.percentile_5

    cards = [
        _card(
            "marginal distance",
            marg,
            _clamp(1 - marg / 0.2),
            _band(marg, (0.05, 0.10, 0.20), ("Excellent", "Strong", "Fair", "Investigate")),
            "avg across columns",
        ),
        _card(
            "correlation Δ",
            corr,
            _clamp(1 - corr / 0.35),
            _band(corr, (0.10, 0.20, 0.35), ("Excellent", "Strong", "Fair", "Investigate")),
            "normalised Frobenius",
        ),
        _card(
            "privacy margin",
            dcr,
            _clamp(dcr / 0.15),
            _band(-dcr, (-0.10, -0.05, -0.02), ("Strong", "Adequate", "Thin", "Close")),
            "5th-pctile DCR",
        ),
    ]
    return f'<section class="verdict">{"".join(cards)}</section>'


def _band(v: float, cuts: tuple[float, float, float], labels: tuple[str, str, str, str]) -> str:
    if not math.isfinite(v):
        return labels[3]
    if v <= cuts[0]:
        return labels[0]
    if v <= cuts[1]:
        return labels[1]
    if v <= cuts[2]:
        return labels[2]
    return labels[3]


def _card(label: str, value: float, fraction: float, verdict: str, unit: str) -> str:
    c = _quality_rgb(fraction)
    color = _rgb(c)
    return (
        '<div class="card">'
        f'<div class="label">{escape(label)}</div>'
        f'<div class="value" style="color:{color}">{_fmt(value)}</div>'
        f'<div class="unit">{escape(unit)}</div>'
        f'<span class="chip" style="background:{_rgba(c, 0.14)};color:{color}">{escape(verdict)}</span>'
        f'<div class="meter"><i style="width:{fraction * 100:.0f}%;background:{color}"></i></div>'
        "</div>"
    )


# --------------------------------------------------------------------------- #
# marginals
# --------------------------------------------------------------------------- #
def _marginals(report: QualityReport) -> str:
    if not report.marginals:
        return "<p class='note'><em>no comparable columns</em></p>"
    ordered = sorted(report.marginals, key=lambda m: m.value, reverse=True)
    finite = [m.value for m in ordered if math.isfinite(m.value)]
    scale = max(0.1, max(finite) if finite else 0.1)

    rows: list[str] = []
    for m in ordered:
        v = m.value
        frac = _clamp(1 - v / 0.2) if math.isfinite(v) else 0.0
        color = _rgb(_quality_rgb(frac))
        width = (_clamp(v / scale) * 100) if math.isfinite(v) else 0.0
        rows.append(
            '<div class="bar-row">'
            f'<div class="col" title="{escape(m.column)}">{escape(m.column)}</div>'
            f'<div class="meta"><span class="mtype">{escape(m.type.value)}</span><br>{escape(m.metric)}</div>'
            f'<div class="track"><i style="width:{width:.1f}%;background:{color}"></i></div>'
            f'<div class="score" style="color:{color}">{_fmt(v)}</div>'
            f'<div class="nulls">{_nulls_cell(m.null_rate_real, m.null_rate_synth, m.verbatim_rate)}</div>'
            "</div>"
        )
    legend = (
        '<div class="legend-min"><span>0 · identical</span>'
        f"<span>bar scale → {scale:.3f}</span></div>"
    )
    return f'<div class="bars">{"".join(rows)}</div>{legend}'


def _nulls_cell(real: float, synth: float, verbatim: float | None) -> str:
    base = f"null {real * 100:.0f}/{synth * 100:.0f}%"
    if verbatim is not None and verbatim > 0:
        return f'{base}<br><span class="leak">{verbatim * 100:.0f}% verbatim</span>'
    return base


def _verbatim_warning(report: QualityReport) -> str:
    leaks = [
        (m.column, m.verbatim_rate)
        for m in report.marginals
        if m.verbatim_rate is not None and m.verbatim_rate > 0
    ]
    if not leaks:
        return ""
    items = "\n".join(
        f"<li><code>{escape(column)}</code> — {rate * 100:.1f}% of synthetic values are verbatim copies</li>"
        for column, rate in leaks
    )
    return (
        '<div class="warn"><strong>Free-text resampled with replacement.</strong> '
        "Synthetic values may be verbatim copies of the source — rerun with "
        "<code>--text-policy hash</code> to mitigate."
        f"<ul>{items}</ul></div>"
    )


# --------------------------------------------------------------------------- #
# correlation heatmaps — the centrepiece
# --------------------------------------------------------------------------- #
def _correlation_section(report: QualityReport) -> str:
    c = report.correlations
    n = len(c.columns)
    if n < 2:
        return (
            "<section><p class='eyebrow'>joint structure</p><h2>Correlation structure</h2>"
            "<p class='note'>Not enough modeled columns to compare pairwise associations.</p></section>"
        )
    real = _heatmap(c.real_matrix, _REAL_RGB)
    synth = _heatmap(c.synth_matrix, _SYNTH_RGB)
    delta = _delta_map(c.real_matrix, c.synth_matrix)
    frob_color = _rgb(_quality_rgb(_clamp(1 - c.frobenius_distance / 0.35)))
    return f"""<section>
  <p class="eyebrow">joint structure</p>
  <h2>Correlation structure</h2>
  <p class="note">Pairwise association matrices over the {n} modeled columns. The synthetic
  grid should shadow the real one; the Δ grid lights up exactly where the twin's joint
  structure drifts from the source.</p>
  <div class="maps">
    <div class="map"><h3><span class="dot" style="background:var(--real)"></span>real</h3>{real}</div>
    <div class="map"><h3><span class="dot" style="background:var(--synth)"></span>synthetic</h3>{synth}</div>
    <div class="map"><h3><span class="dot" style="background:var(--bad)"></span>Δ divergence</h3>{delta}</div>
  </div>
  <div class="frob">
    <span class="big" style="color:{frob_color}">{_fmt(c.frobenius_distance)}</span>
    <span class="cap">normalised Frobenius distance between the two matrices — 0 means the
    synthetic data preserves every pairwise association exactly.</span>
  </div>
</section>"""


def _heatmap(matrix: list[list[float]], tint: tuple[int, int, int]) -> str:
    n = len(matrix)
    cells: list[str] = []
    for row in matrix:
        for v in row:
            a = 0.06 + _clamp(abs(v)) * 0.92 if math.isfinite(v) else 0.0
            cells.append(f'<div class="cell" style="background:{_rgba(tint, a)}"></div>')
    return f'<div class="grid" style="grid-template-columns:repeat({n},1fr)">{"".join(cells)}</div>'


def _delta_map(real: list[list[float]], synth: list[list[float]]) -> str:
    n = len(real)
    cells: list[str] = []
    for i in range(n):
        for j in range(n):
            rv, sv = real[i][j], synth[i][j]
            d = _clamp(abs(rv - sv) * 2.5) if math.isfinite(rv) and math.isfinite(sv) else 0.0
            a = 0.05 + d * 0.93
            cells.append(f'<div class="cell" style="background:{_rgba(_DELTA_RGB, a)}"></div>')
    return f'<div class="grid" style="grid-template-columns:repeat({n},1fr)">{"".join(cells)}</div>'


# --------------------------------------------------------------------------- #
# calendar fidelity
# --------------------------------------------------------------------------- #
def _calendar_section(report: QualityReport) -> str:
    if not report.calendar_fidelity:
        return ""
    blocks: list[str] = [
        "<section><p class='eyebrow'>temporal patterns</p><h2>Calendar fidelity</h2>",
        "<p class='note'>Per-feature KS distance between real and synthetic temporal patterns "
        "(hour-of-day, day-of-week, month-of-year). Lower is better.</p>",
    ]
    for column, scores in report.calendar_fidelity.items():
        rows = "\n".join(
            f"<tr><td>{escape(s.feature)}</td>"
            f"<td class='num'>{_fmt(s.value)}</td>"
            f"<td class='num'>{s.n_real:,}</td>"
            f"<td class='num'>{s.n_synth:,}</td></tr>"
            for s in scores
        )
        blocks.append(
            f"<h3 class='sub'>{escape(column)}</h3>"
            "<table><tr><th>feature</th><th class='num'>KS</th>"
            "<th class='num'>n real</th><th class='num'>n synth</th></tr>"
            f"{rows}</table>"
        )
    blocks.append("</section>")
    return "\n".join(blocks)


# --------------------------------------------------------------------------- #
# privacy — DCR distribution strip
# --------------------------------------------------------------------------- #
def _privacy_section(report: QualityReport) -> str:
    p = report.privacy
    points = [
        ("min", p.min_distance),
        ("p5", p.percentile_5),
        ("p25", p.percentile_25),
        ("p50", p.percentile_50),
        ("mean", p.mean_distance),
    ]
    finite = [v for _, v in points if math.isfinite(v)]
    strip = ""
    if finite:
        hi = max(finite) or 1.0
        # alternate label height to avoid collisions on a crowded rail
        pins = "".join(
            f'<div class="pin {"hi" if k % 2 else ""}" style="left:{_clamp(v / hi) * 100:.1f}%">'
            f'<span class="num">{_fmt(v)}</span><b></b><span class="lab">{name}</span></div>'
            if math.isfinite(v)
            else ""
            for k, (name, v) in enumerate(points)
        )
        strip = f'<div class="strip"><div class="rail">{pins}</div></div>'

    return f"""<section>
  <p class="eyebrow">memorisation risk</p>
  <h2>Distance to closest record</h2>
  <p class="note">For every synthetic row, the L2 distance to its nearest real neighbour in
  encoded space. Values bunched near zero mean the twin is echoing real rows.</p>
  <div class="warn">Heuristic, not a formal privacy guarantee — differential privacy lands
  post-v1. Compared {p.n_synth:,} synthetic vs {p.n_real:,} real rows over {p.n_features}
  encoded features.</div>
  {strip}
</section>"""


# --------------------------------------------------------------------------- #
# issues
# --------------------------------------------------------------------------- #
def _issues(report: QualityReport) -> str:
    if not report.dtype_mismatches and not report.invariant_issues:
        return ""
    blocks: list[str] = ["<section><p class='eyebrow'>diagnostics</p><h2>Likely issues</h2>"]
    if report.dtype_mismatches:
        rows = "\n".join(
            f"<tr><td><code>{escape(i.column)}</code></td>"
            f"<td>{escape(i.real_dtype)}</td><td>{escape(i.synth_dtype)}</td></tr>"
            for i in report.dtype_mismatches
        )
        blocks.append(
            "<h3 class='sub'>dtype mismatches</h3>"
            "<table><tr><th>column</th><th>real dtype</th><th>synth dtype</th></tr>"
            f"{rows}</table>"
        )
    if report.invariant_issues:
        rows = "\n".join(
            f"<tr><td>{escape(i.label)}</td>"
            f"<td class='num'>{i.real_violations:,}</td>"
            f"<td class='num'>{i.synth_violations:,}</td></tr>"
            for i in report.invariant_issues
        )
        blocks.append(
            "<h3 class='sub'>count invariant drift</h3>"
            "<table><tr><th>relationship</th><th class='num'>real violations</th>"
            "<th class='num'>synth violations</th></tr>"
            f"{rows}</table>"
        )
    blocks.append("</section>")
    return "\n".join(blocks)
