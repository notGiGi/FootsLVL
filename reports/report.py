import os, time
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

def export_pdf(out_dir: str, session_path: str, metrics_summary: dict):
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(out_dir, f"report_{ts}.pdf")

    c = canvas.Canvas(path, pagesize=A4)
    w, h = A4

    c.setFont("Helvetica-Bold", 16)
    c.drawString(40, h-40, "FootLab — Session Report")

    c.setFont("Helvetica", 11)
    c.drawString(40, h-70, f"Session file: {os.path.basename(session_path)}")
    c.drawString(40, h-90, f"Duration (s): {metrics_summary.get('duration_s',0):.2f}")
    c.drawString(40, h-110, f"Cadence L (spm): {metrics_summary.get('cadence_L',0):.1f}")
    c.drawString(40, h-130, f"Cadence R (spm): {metrics_summary.get('cadence_R',0):.1f}")
    c.drawString(40, h-150, f"PTI L (heel/mid/fore): {metrics_summary.get('pti_L','-')}")
    c.drawString(40, h-170, f"PTI R (heel/mid/fore): {metrics_summary.get('pti_R','-')}")

    c.setFont("Helvetica-Oblique", 10)
    c.drawString(40, 40, time.strftime("Generated: %Y-%m-%d %H:%M:%S"))

    c.showPage()
    c.save()
    return path
