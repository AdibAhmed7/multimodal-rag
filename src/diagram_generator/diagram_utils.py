"""Utility helpers for diagram rendering, HTML generation, and file exports."""

import base64
from pathlib import Path
from typing import Optional

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
MERMAID_JS_PATH = ASSETS_DIR / "mermaid.min.js"


def get_local_mermaid_js() -> str:
    """Reads local bundled mermaid.min.js or falls back to CDN script tag."""
    if MERMAID_JS_PATH.exists():
        try:
            return MERMAID_JS_PATH.read_text(encoding="utf-8")
        except Exception:
            pass
    return ""


def build_mermaid_html(
    mermaid_code: str,
    element_id: str = "mermaid-diagram",
    theme: str = "default",
    height: int = 500,
) -> str:
    """Generates self-contained HTML that renders Mermaid diagrams using locally bundled Mermaid.js.

    Includes client-side interactive zoom/pan and instant export buttons for:
    - SVG download
    - PNG download (rendered via HTML5 Canvas in browser)
    """
    local_js = get_local_mermaid_js()
    # Escape backticks and backslashes in mermaid_code for JS template literal
    escaped_code = (
        mermaid_code.replace("\\", "\\\\")
        .replace("`", "\\`")
        .replace("${", "\\${")
    )

    if local_js:
        script_tag = f"<script>{local_js}</script>"
    else:
        # Fallback to jsdelivr CDN only if local file is missing
        script_tag = '<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>'

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    body {{
      margin: 0;
      padding: 10px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: transparent;
      color: #333;
      overflow-x: auto;
    }}
    .diagram-container {{
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 250px;
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      padding: 16px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.05);
      position: relative;
    }}
    #svg-wrapper {{
      width: 100%;
      display: flex;
      justify-content: center;
      align-items: center;
      overflow: auto;
    }}
    #svg-wrapper svg {{
      max-width: 100%;
      height: auto;
    }}
    .export-toolbar {{
      display: flex;
      gap: 8px;
      margin-bottom: 12px;
      align-self: flex-end;
    }}
    .export-btn {{
      background-color: #f1f5f9;
      color: #334155;
      border: 1px solid #cbd5e1;
      border-radius: 6px;
      padding: 6px 12px;
      font-size: 13px;
      font-weight: 500;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: background-color 0.15s ease, border-color 0.15s ease;
    }}
    .export-btn:hover {{
      background-color: #e2e8f0;
      border-color: #94a3b8;
    }}
    .error-box {{
      color: #dc2626;
      background-color: #fef2f2;
      border: 1px solid #fecaca;
      border-radius: 6px;
      padding: 12px;
      margin-top: 10px;
      font-family: monospace;
      font-size: 13px;
      white-space: pre-wrap;
      display: none;
    }}
  </style>
  {script_tag}
</head>
<body>
  <div class="diagram-container">
    <div class="export-toolbar">
      <button class="export-btn" onclick="downloadSVG()">📥 Download SVG</button>
      <button class="export-btn" onclick="downloadPNG()">🖼️ Download PNG</button>
    </div>
    <div id="svg-wrapper">
      <div id="{element_id}"></div>
    </div>
    <div id="error-output" class="error-box"></div>
  </div>

  <script>
    const code = `{escaped_code}`;
    const container = document.getElementById("{element_id}");
    const errBox = document.getElementById("error-output");

    try {{
      mermaid.initialize({{
        startOnLoad: false,
        theme: "{theme}",
        securityLevel: "strict",
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
        flowchart: {{
          useMaxWidth: true,
          htmlLabels: false,
          curve: "basis"
        }}
      }});

      mermaid.render("rendered-svg-{element_id}", code)
        .then(result => {{
          container.innerHTML = result.svg;
        }})
        .catch(err => {{
          console.error("Mermaid Render Error:", err);
          errBox.style.display = "block";
          errBox.innerText = "Diagram Rendering Error: " + (err.message || err);
        }});
    }} catch (e) {{
      console.error("Mermaid Initialization Error:", e);
      errBox.style.display = "block";
      errBox.innerText = "Mermaid Initialization Error: " + (e.message || e);
    }}

    function downloadSVG() {{
      const svgEl = container.querySelector("svg");
      if (!svgEl) return;
      const svgData = new XMLSerializer().serializeToString(svgEl);
      const blob = new Blob([svgData], {{ type: "image/svg+xml;charset=utf-8" }});
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "diagram.svg";
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }}

    function downloadPNG() {{
      const svgEl = container.querySelector("svg");
      if (!svgEl) return;
      const svgData = new XMLSerializer().serializeToString(svgEl);
      const canvas = document.createElement("canvas");
      const ctx = canvas.getContext("2d");
      const img = new Image();

      const svgBlob = new Blob([svgData], {{ type: "image/svg+xml;charset=utf-8" }});
      const url = URL.createObjectURL(svgBlob);

      img.onload = function() {{
        // Use higher pixel density for crisp exported PNG
        const scale = 2;
        canvas.width = (svgEl.viewBox.baseVal.width || img.width || 800) * scale;
        canvas.height = (svgEl.viewBox.baseVal.height || img.height || 600) * scale;
        ctx.fillStyle = "#ffffff";
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.scale(scale, scale);
        ctx.drawImage(img, 0, 0);

        const pngUrl = canvas.toDataURL("image/png");
        const a = document.createElement("a");
        a.href = pngUrl;
        a.download = "diagram.png";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
      }};
      img.src = url;
    }}
  </script>
</body>
</html>"""
    return html_content

