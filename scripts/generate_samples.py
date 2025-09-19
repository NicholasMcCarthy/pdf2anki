#!/usr/bin/env python3
"""
Generate sample PDFs for testing pdf2anki functionality.

This script creates sample PDF documents with various layouts and content types
to test different aspects of the pdf2anki pipeline. It's a soft dependency on
reportlab for PDF generation.

Usage:
    python scripts/generate_samples.py [--output-dir samples/]
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.platypus import Table, TableStyle
    from reportlab.lib import colors
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def create_basic_academic_paper(output_path: Path) -> None:
    """Create a basic academic paper with sections and references."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required to generate sample PDFs")
    
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=16,
        alignment=1,  # Center
        spaceAfter=30
    )
    story.append(Paragraph("Introduction to Machine Learning Algorithms", title_style))
    story.append(Spacer(1, 12))
    
    # Abstract
    story.append(Paragraph("Abstract", styles['Heading2']))
    abstract_text = """
    This paper provides an introduction to machine learning algorithms, focusing on 
    supervised learning techniques. We discuss linear regression, decision trees, 
    and neural networks, providing mathematical foundations and practical applications.
    Key findings suggest that ensemble methods often outperform individual algorithms
    in complex datasets.
    """
    story.append(Paragraph(abstract_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Introduction section
    story.append(Paragraph("1. Introduction", styles['Heading2']))
    intro_text = """
    Machine learning is a subset of artificial intelligence that focuses on algorithms
    that can learn and make predictions from data. The field has grown exponentially
    in recent years, with applications ranging from image recognition to natural language
    processing. This paper examines three fundamental approaches: supervised learning,
    unsupervised learning, and reinforcement learning.
    """
    story.append(Paragraph(intro_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Methodology section  
    story.append(Paragraph("2. Methodology", styles['Heading2']))
    method_text = """
    We evaluated each algorithm using standard benchmark datasets including the
    Iris dataset, Boston Housing prices, and MNIST handwritten digits. Performance
    metrics included accuracy, precision, recall, and F1-score. Cross-validation
    was used to ensure robust evaluation.
    """
    story.append(Paragraph(method_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Results with a simple table
    story.append(Paragraph("3. Results", styles['Heading2']))
    results_text = """
    The following table summarizes the performance of different algorithms
    across our test datasets:
    """
    story.append(Paragraph(results_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Create a results table
    data = [
        ['Algorithm', 'Accuracy', 'Precision', 'Recall'],
        ['Linear Regression', '0.87', '0.85', '0.89'],
        ['Decision Tree', '0.92', '0.91', '0.93'],
        ['Neural Network', '0.95', '0.94', '0.96'],
        ['Random Forest', '0.97', '0.96', '0.98']
    ]
    
    table = Table(data)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))
    
    # Conclusion
    story.append(Paragraph("4. Conclusion", styles['Heading2']))
    conclusion_text = """
    Our analysis demonstrates that ensemble methods like Random Forest consistently
    outperform individual algorithms. Neural networks show promise but require
    careful tuning of hyperparameters. Future work should explore deep learning
    architectures and their applicability to domain-specific problems.
    """
    story.append(Paragraph(conclusion_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # References
    story.append(Paragraph("References", styles['Heading2']))
    refs = [
        "Smith, J. (2023). Machine Learning Fundamentals. Academic Press.",  
        "Johnson, A. et al. (2022). Ensemble Methods in Practice. ML Journal, 15(3), 45-67.",
        "Brown, M. (2021). Neural Networks: Theory and Applications. Tech Publishers."
    ]
    
    for ref in refs:
        story.append(Paragraph(f"• {ref}", styles['Normal']))
        story.append(Spacer(1, 6))
    
    doc.build(story)


def create_textbook_chapter(output_path: Path) -> None:
    """Create a textbook-style chapter with exercises."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required to generate sample PDFs")
    
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Chapter title
    title_style = ParagraphStyle(
        'ChapterTitle',
        parent=styles['Heading1'],
        fontSize=18,
        alignment=0,  # Left
        spaceAfter=20
    )
    story.append(Paragraph("Chapter 5: Data Structures and Algorithms", title_style))
    story.append(Spacer(1, 20))
    
    # Learning objectives
    story.append(Paragraph("Learning Objectives", styles['Heading3']))
    objectives = [
        "Understand fundamental data structures (arrays, lists, trees)",
        "Analyze time and space complexity of algorithms", 
        "Implement sorting and searching algorithms",
        "Apply data structures to solve computational problems"
    ]
    
    for obj in objectives:
        story.append(Paragraph(f"• {obj}", styles['Normal']))
        story.append(Spacer(1, 6))
    
    story.append(Spacer(1, 20))
    
    # Section 5.1
    story.append(Paragraph("5.1 Introduction to Data Structures", styles['Heading2']))
    ds_text = """
    A data structure is a way of organizing and storing data in a computer so that
    it can be accessed and modified efficiently. Different data structures are suited
    for different kinds of applications, and some are highly specialized for specific tasks.
    
    The choice of data structure affects the efficiency of algorithms that manipulate
    the data. Understanding the trade-offs between different data structures is crucial
    for writing efficient programs.
    """
    story.append(Paragraph(ds_text, styles['Normal']))
    story.append(Spacer(1, 12))
    
    # Definition box
    definition_style = ParagraphStyle(
        'Definition',
        parent=styles['Normal'],
        leftIndent=20,
        rightIndent=20,
        borderWidth=1,
        borderColor=colors.black,
        backColor=colors.lightgrey,
        borderPadding=10
    )
    
    story.append(Paragraph(
        "<b>Definition:</b> An <i>algorithm</i> is a finite sequence of well-defined "
        "instructions for solving a computational problem.",
        definition_style
    ))
    story.append(Spacer(1, 20))
    
    # Section 5.2
    story.append(Paragraph("5.2 Array Operations", styles['Heading2']))
    array_text = """
    Arrays are the most fundamental data structure, consisting of elements stored
    in contiguous memory locations. Key operations include:
    
    1. <b>Access:</b> O(1) - Direct access using index
    2. <b>Search:</b> O(n) - Linear search through elements  
    3. <b>Insertion:</b> O(n) - May require shifting elements
    4. <b>Deletion:</b> O(n) - May require shifting elements
    
    Arrays provide excellent cache locality but have fixed size in many languages.
    """
    story.append(Paragraph(array_text, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Exercise section
    story.append(Paragraph("Exercises", styles['Heading3']))
    
    exercises = [
        "Implement a function to find the maximum element in an array.",
        "Write an algorithm to reverse an array in-place.",
        "Design a method to remove duplicates from a sorted array.",
        "Calculate the time complexity of bubble sort algorithm."
    ]
    
    for i, exercise in enumerate(exercises, 1):
        story.append(Paragraph(f"{i}. {exercise}", styles['Normal']))
        story.append(Spacer(1, 8))
    
    story.append(PageBreak())
    
    # Solutions section
    story.append(Paragraph("Solutions", styles['Heading3']))
    story.append(Paragraph(
        "1. Maximum element can be found by iterating through the array once, "
        "keeping track of the largest value seen so far. Time complexity: O(n).",
        styles['Normal']
    ))
    
    doc.build(story)


def create_research_paper(output_path: Path) -> None:
    """Create a research paper with mathematical content."""
    if not REPORTLAB_AVAILABLE:
        raise ImportError("reportlab is required to generate sample PDFs")
    
    doc = SimpleDocTemplate(str(output_path), pagesize=A4)
    story = []
    styles = getSampleStyleSheet()
    
    # Title and authors
    title_style = ParagraphStyle(
        'ResearchTitle',
        parent=styles['Heading1'],
        fontSize=14,
        alignment=1,
        spaceAfter=20
    )
    
    story.append(Paragraph(
        "Optimization Techniques in Deep Neural Networks: A Comparative Study",
        title_style
    ))
    
    author_style = ParagraphStyle(
        'Authors',
        parent=styles['Normal'],
        alignment=1,
        fontSize=10
    )
    
    story.append(Paragraph(
        "Jane Smith¹, Robert Johnson², Maria Garcia¹<br/>"
        "¹University of Technology, ²Research Institute",
        author_style
    ))
    story.append(Spacer(1, 30))
    
    # Abstract
    story.append(Paragraph("ABSTRACT", styles['Heading3']))
    abstract = """
    We present a comprehensive comparison of optimization algorithms for training
    deep neural networks. Our study evaluates Stochastic Gradient Descent (SGD),
    Adam, RMSprop, and AdaGrad on various architectures including CNNs and LSTMs.
    Results indicate that Adam consistently achieves faster convergence while
    maintaining competitive final accuracy across different domains.
    """
    story.append(Paragraph(abstract, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Keywords
    story.append(Paragraph(
        "<b>Keywords:</b> deep learning, optimization, neural networks, machine learning",
        styles['Normal']
    ))
    story.append(Spacer(1, 30))
    
    # Introduction
    story.append(Paragraph("1. INTRODUCTION", styles['Heading2']))
    intro = """
    The success of deep neural networks depends heavily on the optimization algorithm
    used during training. While traditional gradient descent provides theoretical
    guarantees, practical considerations such as convergence speed and computational
    efficiency have led to the development of adaptive optimization methods.
    
    This paper investigates four prominent optimization algorithms:
    • Stochastic Gradient Descent with momentum
    • Adam (Adaptive Moment Estimation)  
    • RMSprop (Root Mean Square Propagation)
    • AdaGrad (Adaptive Gradient)
    """
    story.append(Paragraph(intro, styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Mathematical notation (simplified since we can't easily render LaTeX)
    story.append(Paragraph("2. MATHEMATICAL BACKGROUND", styles['Heading2']))
    math_text = """
    The general form of gradient-based optimization can be expressed as:
    
    θ(t+1) = θ(t) - α * g(t)
    
    where θ represents the parameters, α is the learning rate, and g(t) is the
    gradient or a modified version thereof. Each algorithm differs in how it
    computes or modifies g(t).
    
    Adam combines momentum and adaptive learning rates:
    m(t) = β1 * m(t-1) + (1 - β1) * ∇f(θ(t))
    v(t) = β2 * v(t-1) + (1 - β2) * (∇f(θ(t)))²
    """
    story.append(Paragraph(math_text, styles['Normal']))
    
    doc.build(story)


def main():
    """Main function to generate sample PDFs."""
    parser = argparse.ArgumentParser(description="Generate sample PDFs for pdf2anki testing")
    parser.add_argument(
        "--output-dir", 
        type=Path, 
        default=Path("samples"),
        help="Output directory for generated PDFs (default: samples/)"
    )
    
    args = parser.parse_args()
    
    if not REPORTLAB_AVAILABLE:
        print("Error: reportlab is not installed. Please install it with:")
        print("  pip install reportlab")
        sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate sample PDFs
    samples = [
        ("academic_paper.pdf", create_basic_academic_paper),
        ("textbook_chapter.pdf", create_textbook_chapter), 
        ("research_paper.pdf", create_research_paper)
    ]
    
    print(f"Generating sample PDFs in {args.output_dir}/")
    
    for filename, generator_func in samples:
        output_path = args.output_dir / filename
        try:
            generator_func(output_path)
            print(f"  ✓ Created {filename}")
        except Exception as e:
            print(f"  ✗ Failed to create {filename}: {e}")
    
    print(f"\nGenerated {len(samples)} sample PDFs in {args.output_dir}/")
    print("You can now test pdf2anki with these files!")


if __name__ == "__main__":
    main()