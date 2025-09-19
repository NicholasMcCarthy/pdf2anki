#!/usr/bin/env python3
from pathlib import Path

def main():
    try:
        from reportlab.lib.pagesizes import LETTER
        from reportlab.lib.styles import getSampleStyleSheet
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    except ImportError:
        print("Please `pip install reportlab` to generate sample PDFs.")
        return

    samples_dir = Path("samples")
    samples_dir.mkdir(parents=True, exist_ok=True)

    # 1) research_paper.pdf — one-page abstract on Ancient Roman philosophy
    doc1 = SimpleDocTemplate(str(samples_dir / "research_paper.pdf"), pagesize=LETTER)
    styles = getSampleStyleSheet()
    story = []
    story.append(Paragraph("Ancient Roman Philosophy — Abstract", styles["Title"]))
    story.append(Spacer(1, 12))
    abstract = (
        "This abstract surveys the reception and transformation of Greek philosophical "
        "traditions in Rome, focusing on Stoicism, Epicureanism, and Academic Skepticism. "
        "Roman thinkers like Cicero, Seneca, and Marcus Aurelius adapted ethical and "
        "political doctrines to civic life, emphasizing virtue, duty, and tranquility "
        "amid public responsibilities. The Roman synthesis prioritized practical ethics "
        "and rhetoric, framing philosophy as a guide for governance and personal conduct."
    )
    story.append(Paragraph(abstract, styles["BodyText"]))
    doc1.build(story)

    # 2) textbook.pdf — 3 pages, each page a chapter
    def chapter(title, text):
        story = []
        story.append(Paragraph(title, styles["Title"]))
        story.append(Spacer(1, 12))
        story.append(Paragraph(text, styles["BodyText"]))
        return story

    ch1 = (
        "Chapter 1: Monarchy to Republic — From legendary kings to the establishment "
        "of the Republic (509 BCE), Rome forged its identity through civic institutions, "
        "legal customs, and an expanding citizen body."
    )
    ch2 = (
        "Chapter 2: The Republic Expands — Military innovation and alliances extended "
        "Roman influence across the Mediterranean, but social tensions and reform "
        "movements exposed deep fractures."
    )
    ch3 = (
        "Chapter 3: The Empire — Augustus consolidated power, birthing an imperial "
        "system that balanced tradition with centralized authority; subsequent emperors "
        "oversaw periods of prosperity and crisis, leaving a vast cultural legacy."
    )

    doc2 = SimpleDocTemplate(str(samples_dir / "textbook.pdf"), pagesize=LETTER)
    output = []
    for title, text in [
        ("A Brief History of Ancient Rome — I", ch1),
        ("A Brief History of Ancient Rome — II", ch2),
        ("A Brief History of Ancient Rome — III", ch3),
    ]:
        output += chapter(title, text)
        # Force a new page by adding enough space
        output.append(Spacer(1, 800))
    doc2.build(output)

    print("Generated samples/research_paper.pdf and samples/textbook.pdf")

if __name__ == "__main__":
    main()
