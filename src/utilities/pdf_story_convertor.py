import os
import asyncio
import unicodedata
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER


class PDFStoryBuilder:
    # ---------------------------------------------------------
    # Character cleaning map for characters ReportLab can't render
    # ---------------------------------------------------------
    REPLACEMENTS = {
        "\u2011": "-",   # non-breaking hyphen
        "\u2013": "-",   # en dash
        "\u2014": "--",  # em dash
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2026": "...",
    }

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize and replace characters that break ReportLab."""
        if not text:
            return ""
        for bad, good in PDFStoryBuilder.REPLACEMENTS.items():
            text = text.replace(bad, good)
        return unicodedata.normalize("NFKC", text)

    # ---------------------------------------------------------
    # Main PDF build function
    # ---------------------------------------------------------
    @staticmethod
    async def build_story_pdf(memory_system, total_chapters: int):
        """
        Asynchronously fetches chapter clusters and builds a formatted PDF.
        """
        # Register Unicode-capable font
        pdfmetrics.registerFont(TTFont('DejaVu', 'fonts/DejaVuSerif.ttf'))

        chapters = []
        acts = {}

        # ---------------------------------------------------------
        # Fetch Story Data (All Chapters)
        # ---------------------------------------------------------
        for chapter_id in range(1, total_chapters + 1):
            cluster = await memory_system.get_story_cluster(chapter_id)
            text_list = cluster.get("text", [])
            metadata = cluster.get("metadata", {})

            act_id = metadata.get("act_id")

            entry = {
                "chapter_id": chapter_id,
                "act_id": act_id,
                "act_title": metadata.get("act_title"),
                "story_title": metadata.get("story_title"),
                "text_list": text_list,
            }
            chapters.append(entry)

            if act_id not in acts:
                acts[act_id] = {
                    "title": entry["act_title"],
                    "chapters": []
                }

            acts[act_id]["chapters"].append(entry)

        story_title = chapters[0]["story_title"].strip()
        pdf_filename = f"{story_title}.pdf"

        PDFStoryBuilder._create_pdf(pdf_filename, story_title, acts)
        return pdf_filename

    # ---------------------------------------------------------
    # PDF Body Builder
    # ---------------------------------------------------------
    @staticmethod
    def _create_pdf(filename: str, story_title: str, acts: dict):
        doc = SimpleDocTemplate(
            filename,
            pagesize=LETTER,
            title=story_title
        )

        # ---------------------------------------------------------
        # Styles
        # ---------------------------------------------------------
        styles = getSampleStyleSheet()

        styles.add(ParagraphStyle(
            name="UnicodeTitle",
            parent=styles["Title"],
            fontName="DejaVu",
            fontSize=28,
            leading=32,
            alignment=TA_CENTER
        ))

        styles.add(ParagraphStyle(
            name="UnicodeHeading1",
            parent=styles["Heading1"],
            fontName="DejaVu",
            fontSize=20
        ))

        styles.add(ParagraphStyle(
            name="UnicodeHeading2",
            parent=styles["Heading2"],
            fontName="DejaVu",
            fontSize=16
        ))

        styles.add(ParagraphStyle(
            name="UnicodeBody",
            parent=styles["BodyText"],
            fontName="DejaVu",
            fontSize=11,
            leading=14
        ))

        title_style = styles["UnicodeTitle"]
        heading_style = styles["UnicodeHeading1"]
        sub_heading_style = styles["UnicodeHeading2"]
        body_style = styles["UnicodeBody"]

        story_flow = []

        # ---------------------------------------------------------
        # Title Page
        # ---------------------------------------------------------
        story_flow.append(Spacer(1, 100))
        story_flow.append(Paragraph(story_title, title_style))
        story_flow.append(PageBreak())

        # ---------------------------------------------------------
        # Chapter Index Page
        # ---------------------------------------------------------
        story_flow.append(Paragraph("Chapter Index", heading_style))
        story_flow.append(Spacer(1, 20))

        for act_id in sorted(acts.keys()):
            act = acts[act_id]
            act_title = act["title"] or f"Act {act_id}"

            story_flow.append(Paragraph(f"Act {act_id}: {act_title}", sub_heading_style))

            for ch in act["chapters"]:
                story_flow.append(Paragraph(f"Chapter {ch['chapter_id']}", body_style))

            story_flow.append(Spacer(1, 12))

        story_flow.append(PageBreak())

        # ---------------------------------------------------------
        # Story Content
        # ---------------------------------------------------------
        for act_id in sorted(acts.keys()):
            act = acts[act_id]
            act_title = act["title"] or f"Act {act_id}"

            # Act header
            story_flow.append(Paragraph(f"Act {act_id}: {act_title}", heading_style))
            story_flow.append(Spacer(1, 20))

            # Chapters
            for ch in act["chapters"]:
                story_flow.append(Paragraph(f"Chapter {ch['chapter_id']}", sub_heading_style))
                story_flow.append(Spacer(1, 12))

                # Scenes
                for scene in ch["text_list"]:
                    if scene.get("type") == "text":
                        cleaned = PDFStoryBuilder.clean_text(scene.get("scene_text", ""))
                        cleaned = cleaned.replace("\n", "<br/>")

                        story_flow.append(Paragraph(cleaned, body_style))
                        story_flow.append(Spacer(1, 12))

                story_flow.append(PageBreak())

        # ---------------------------------------------------------
        # Final Build
        # ---------------------------------------------------------
        doc.build(story_flow)
