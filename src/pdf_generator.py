import io
from datetime import datetime
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(df: pd.DataFrame, owner: str, repo_name: str, health_score: int,
                        overall_risk: str, avg_confidence: float, sim_df: pd.DataFrame) -> bytes:
    """
    Generates a professional PDF risk assessment report and returns it as bytes.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )

    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=colors.HexColor('#1e293b'),
        spaceAfter=12
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        leading=14,
        textColor=colors.HexColor('#64748b'),
        spaceAfter=25
    )

    h1_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        leading=18,
        textColor=colors.HexColor('#1e293b'),
        spaceBefore=15,
        spaceAfter=10,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        spaceAfter=8
    )

    code_style = ParagraphStyle(
        'CodeStyleCustom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#0f172a'),
    )

    bullet_style = ParagraphStyle(
        'BulletCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        firstLineIndent=-10,
        spaceAfter=4
    )

    story = []

    # Title & Subtitle
    story.append(Paragraph(f"Repository Risk Assessment Report", title_style))
    story.append(Paragraph(f"Repository: {owner}/{repo_name} | Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} UTC", subtitle_style))
    
    story.append(Spacer(1, 10))

    # Executive Summary Card
    story.append(Paragraph("1. Executive Summary", h1_style))
    
    # Calculate counts
    n_total = len(df)
    n_high = int((df["predicted_risk"] == "HIGH").sum())
    n_med = int((df["predicted_risk"] == "MEDIUM").sum())
    n_low = int((df["predicted_risk"] == "LOW").sum())
    
    # Map trust level
    conf_pct = avg_confidence * 100
    if conf_pct >= 90:
        trust_level = "HIGH TRUST"
    elif conf_pct >= 80:
        trust_level = "GOOD TRUST"
    elif conf_pct >= 70:
        trust_level = "MODERATE TRUST"
    else:
        trust_level = "MANUAL REVIEW RECOMMENDED"

    summary_text = (
        f"A repository-level risk assessment was performed on <b>{owner}/{repo_name}</b> analyzing "
        f"<b>{n_total:,}</b> source code files. The repository exhibits an overall <b>{overall_risk}</b> "
        f"risk profile, with a <b>Repository Health Score of {health_score}/100</b> (where 100 represents zero risk). "
        f"The prediction models returned an average confidence score of <b>{conf_pct:.1f}%</b>, "
        f"classifying the reliability of this analysis as <b>{trust_level}</b>."
    )
    story.append(Paragraph(summary_text, body_style))
    story.append(Spacer(1, 10))

    # Executive Info Table
    exec_data = [
        [Paragraph("<b>Metric</b>", body_style), Paragraph("<b>Value</b>", body_style), Paragraph("<b>Assessment</b>", body_style)],
        [Paragraph("Repository Health Score", body_style), Paragraph(f"{health_score}/100", body_style), Paragraph("Excellent" if health_score >= 85 else ("Good" if health_score >= 70 else "At Risk"), body_style)],
        [Paragraph("Risk Classification", body_style), Paragraph(overall_risk, body_style), Paragraph("Requires attention" if overall_risk in ["HIGH", "CRITICAL"] else "Stable", body_style)],
        [Paragraph("Total Files Analyzed", body_style), Paragraph(f"{n_total:,}", body_style), Paragraph("Source files (Py/JS/TS)", body_style)],
        [Paragraph("Average Prediction Confidence", body_style), Paragraph(f"{conf_pct:.1f}%", body_style), Paragraph(trust_level, body_style)],
    ]
    t_exec = Table(exec_data, colWidths=[2.2*inch, 1.5*inch, 3.2*inch])
    t_exec.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(t_exec)
    
    story.append(Spacer(1, 15))

    # 2. Risk Distribution
    story.append(Paragraph("2. Risk Distribution", h1_style))
    pct_high = n_high / n_total * 100 if n_total > 0 else 0
    pct_med  = n_med / n_total * 100 if n_total > 0 else 0
    pct_low  = n_low / n_total * 100 if n_total > 0 else 0

    dist_data = [
        [Paragraph("<b>Risk Class</b>", body_style), Paragraph("<b>File Count</b>", body_style), Paragraph("<b>Percentage</b>", body_style)],
        [Paragraph("<font color='red'><b>HIGH</b></font>", body_style), Paragraph(f"{n_high:,}", body_style), Paragraph(f"{pct_high:.1f}%", body_style)],
        [Paragraph("<font color='orange'><b>MEDIUM</b></font>", body_style), Paragraph(f"{n_med:,}", body_style), Paragraph(f"{pct_med:.1f}%", body_style)],
        [Paragraph("<font color='green'><b>LOW</b></font>", body_style), Paragraph(f"{n_low:,}", body_style), Paragraph(f"{pct_low:.1f}%", body_style)],
    ]
    t_dist = Table(dist_data, colWidths=[2.2*inch, 2.2*inch, 2.5*inch])
    t_dist.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('TOPPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(t_dist)

    story.append(Spacer(1, 15))

    # 3. Repository Similarity (estimation of reliability)
    story.append(Paragraph("3. Repository Similarity & Reliability Estimation", h1_style))
    if sim_df is not None and not sim_df.empty:
        sim_rows = [
            [Paragraph("<b>Rank</b>", body_style), Paragraph("<b>Similar Training Repository</b>", body_style), Paragraph("<b>Cosine Similarity</b>", body_style)]
        ]
        # Filter similarity for this specific repository if mixed
        repo_sim = sim_df[sim_df["external_repo"] == repo_name].head(3)
        if repo_sim.empty:
            repo_sim = sim_df.head(3)
            
        for _, r in repo_sim.iterrows():
            sim_rows.append([
                Paragraph(str(r.get("rank", 1)), body_style),
                Paragraph(str(r.get("similar_training_repo", "N/A")), body_style),
                Paragraph(f"{r.get('cosine_similarity', 0.0) * 100:.1f}%", body_style)
            ])
        t_sim = Table(sim_rows, colWidths=[1.0*inch, 3.5*inch, 2.4*inch])
        t_sim.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('BOTTOMPADDING', (0,0), (-1,0), 6),
            ('TOPPADDING', (0,0), (-1,0), 6),
            ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ]))
        story.append(t_sim)
        
        # Explain similarity
        top_sim_val = repo_sim.iloc[0].get('cosine_similarity', 0.0) * 100 if not repo_sim.empty else 0
        top_sim_repo = repo_sim.iloc[0].get('similar_training_repo', 'N/A') if not repo_sim.empty else 'N/A'
        
        sim_explanation = (
            f"The repository matches closest to the training repository <b>{top_sim_repo}</b> with a cosine similarity "
            f"of <b>{top_sim_val:.1f}%</b>. High similarity (>85%) indicates that risk predictions are highly reliable "
            f"since the target repository's code structures match profiles successfully verified during cross-validation."
        )
        story.append(Spacer(1, 5))
        story.append(Paragraph(sim_explanation, body_style))
    else:
        story.append(Paragraph("Similarity statistics unavailable.", body_style))

    # Page Break for details
    story.append(PageBreak())

    # 4. Top 10 High-Risk Files
    story.append(Paragraph("4. Top 10 High-Risk Files", h1_style))
    story.append(Paragraph("The files below represent the highest-risk modules based on code complexity, change intensity, and contributor metrics.", body_style))
    story.append(Spacer(1, 5))

    # Sort high-risk files
    df_sorted = df.sort_values("risk_score_val", ascending=False).head(10)
    
    hr_headers = [
        Paragraph("<b>Rank</b>", body_style),
        Paragraph("<b>File Path</b>", body_style),
        Paragraph("<b>Score</b>", body_style),
        Paragraph("<b>Conf</b>", body_style),
        Paragraph("<b>Complexity</b>", body_style),
        Paragraph("<b>LOC</b>", body_style),
        Paragraph("<b>Mods</b>", body_style)
    ]
    hr_rows = [hr_headers]
    for idx, (_, r) in enumerate(df_sorted.iterrows(), 1):
        fp_p = Paragraph(r.get("file_path", "N/A"), code_style)
        hr_rows.append([
            Paragraph(str(idx), body_style),
            fp_p,
            Paragraph(f"{r.get('risk_score_val', 0.0):.0f}", body_style),
            Paragraph(f"{r.get('confidence', 0.0)*100:.0f}%", body_style),
            Paragraph(f"{r.get('complexity', 0):.0f}", body_style),
            Paragraph(f"{r.get('loc', 0):.0f}", body_style),
            Paragraph(f"{r.get('modification_count', 0):.0f}", body_style),
        ])
    t_hr = Table(hr_rows, colWidths=[0.5*inch, 3.2*inch, 0.6*inch, 0.6*inch, 0.8*inch, 0.6*inch, 0.6*inch])
    t_hr.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(t_hr)

    story.append(Spacer(1, 15))

    # 5. Technical Debt Hotspots (Top 15 Candidate Files)
    story.append(Paragraph("5. Technical Debt Hotspots & Refactoring Candidates", h1_style))
    story.append(Paragraph("The following table lists the top candidate files for refactoring, ordered by total risk score and prioritized by maintainability and churn indicators.", body_style))
    story.append(Spacer(1, 5))

    # Sort hotspots
    df_hot = df.sort_values("risk_score_val", ascending=False).head(15)
    
    hot_headers = [
        Paragraph("<b>Rank</b>", body_style),
        Paragraph("<b>File Path</b>", body_style),
        Paragraph("<b>Risk Score</b>", body_style),
        Paragraph("<b>Confidence</b>", body_style),
        Paragraph("<b>Priority</b>", body_style),
    ]
    hot_rows = [hot_headers]
    for idx, (_, r) in enumerate(df_hot.iterrows(), 1):
        fp_p = Paragraph(r.get("file_path", "N/A"), code_style)
        
        # Calculate tech debt priority
        score = r.get("risk_score_val", 0)
        if score >= 80:
            priority = "<font color='red'><b>CRITICAL</b></font>"
        elif score >= 60:
            priority = "<font color='orange'><b>HIGH</b></font>"
        elif score >= 40:
            priority = "<font color='yellow'><b>MEDIUM</b></font>"
        else:
            priority = "<font color='green'><b>LOW</b></font>"
            
        hot_rows.append([
            Paragraph(str(idx), body_style),
            fp_p,
            Paragraph(f"{score:.1f}", body_style),
            Paragraph(f"{r.get('confidence', 0.0)*100:.1f}%", body_style),
            Paragraph(priority, body_style)
        ])
    t_hot = Table(hot_rows, colWidths=[0.6*inch, 4.0*inch, 0.9*inch, 0.9*inch, 1.1*inch])
    t_hot.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#f1f5f9')),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#cbd5e1')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, colors.HexColor('#f8fafc')])
    ]))
    story.append(t_hot)

    story.append(Spacer(1, 15))

    # 6. Strategic Recommendations
    story.append(Paragraph("6. Deployment Recommendations", h1_style))
    rec1 = (
        "<b>Focus Refactoring on Hotspots:</b> Refactoring should prioritize files flagged with "
        "<i>Critical</i> and <i>High</i> Technical Debt Priority. These files represent disproportionate "
        "risk because they are modified frequently, have low maintainability indices, and high complexity."
    )
    rec2 = (
        "<b>Mitigate Knowledge Concentration (Bus Factor):</b> Files showing high ownership concentration "
        "and low bus factor should be reviewed to distribute knowledge. Consider peer programming or cross-team "
        "reviews for the top high-risk files identified in Section 4."
    )
    rec3 = (
        "<b>Continuous Risk Monitoring:</b> Integrating risk assessment directly into the CI/CD pipeline "
        "can block merges that introduce excessive file risk (e.g. cyclomatic complexity spikes or severe maintainability index drops)."
    )
    story.append(Paragraph(f"• {rec1}", bullet_style))
    story.append(Paragraph(f"• {rec2}", bullet_style))
    story.append(Paragraph(f"• {rec3}", bullet_style))

    # Build the document
    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
