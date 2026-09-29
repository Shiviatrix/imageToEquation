#!/usr/bin/env python3
"""
update_comparison_readmes.py
Formats LaTeX equations with proper MathJax typography and line-breaks
under each comparison image in markdown files.
"""

import re
from pathlib import Path

def split_top_level_terms(expr: str):
    terms = []
    current = []
    depth = 0
    i = 0
    while i < len(expr):
        c = expr[i]
        if c in "({[":
            depth += 1
        elif c in ")}]":
            depth = max(0, depth - 1)
        elif depth == 0 and c in "+-" and i > 0 and (expr[i-1].isspace() or expr[i-1] in ") }"):
            if current:
                terms.append("".join(current).strip())
            current = [c]
            i += 1
            continue
        current.append(c)
        i += 1
    if current:
        terms.append("".join(current).strip())
    return [t for t in terms if t]

def clean_signs(s: str) -> str:
    s = re.sub(r"(\))\s*(\\frac|\\cos)", r"\1 + \2", s)
    s = re.sub(r"-\s*-\s*", "+ ", s)
    s = re.sub(r"\+\s*-\s*", "- ", s)
    s = re.sub(r"-\s*\+", "- ", s)
    s = re.sub(r"\+\s*\+", "+ ", s)
    s = re.sub(r"\\frac\{([^}]+)\}\{1\}", r"\1", s)
    s = re.sub(r"\(\s*\((X[^)]+)\)\s*\)", r"(\1)", s)
    s = re.sub(r"\(\s*\((Y[^)]+)\)\s*\)", r"(\1)", s)
    return s

def chunk_terms(terms, max_len=95):
    lines = []
    curr = []
    curr_len = 0
    for t in terms:
        t_len = len(t)
        if curr and curr_len + t_len > max_len:
            lines.append(" ".join(curr))
            curr = [t]
            curr_len = t_len
        else:
            curr.append(t)
            curr_len += t_len
    if curr:
        lines.append(" ".join(curr))
    return lines

def format_equation_properly(tex_path: Path):
    if not tex_path.exists():
        return ""
    content = tex_path.read_text()
    
    # coordinate transform
    coord_match = re.search(r"\\subsection\*\{(?:Scale-Invariant )?Coordinate (?:System|Transformation)\}\s*\\\[\s*(.*?)\s*\\\]", content, re.DOTALL)
    coord_raw = coord_match.group(1).strip() if coord_match else ""
    coord_raw = clean_signs(coord_raw)
    
    coord_parts = [p.strip().rstrip("\\").strip() for p in re.split(r"\\\\(?:\[\d+pt\])?|\n+", coord_raw) if p.strip()]
    
    # layer 1 formulation
    l1_match = re.search(r"% Layer 1: Primitive_1\s*\\begin\{align\*\}\s*(.*?)\s*\\end\{align\*\}", content, re.DOTALL)
    l1_raw = l1_match.group(1).strip() if l1_match else ""
    l1_raw = clean_signs(l1_raw)
    
    rows = []
    # coordinates
    for cp in coord_parts:
        if "=" in cp:
            p = cp.replace("=", "&=", 1)
            rows.append(p)
        else:
            rows.append(f"&\\quad {cp}")
            
    # layer 1 formulation lines
    for raw_line in l1_raw.splitlines():
        line = raw_line.strip().rstrip("\\").strip()
        line = re.sub(r"^\s*&\s*", "", line)
        line = re.sub(r"\s*&\s*", " ", line)
        if not line:
            continue
        line = clean_signs(line)
        if "=" in line:
            var_part, expr_part = line.split("=", 1)
            var_part = var_part.strip()
            expr_part = expr_part.strip()
            
            if len(expr_part) <= 90:
                rows.append(f"{var_part} &= {expr_part}")
                continue
                
            terms = split_top_level_terms(expr_part)
            
            if var_part.startswith("\\Phi"):
                base_terms = []
                cos_terms = []
                for t in terms:
                    if "\\cos" in t:
                        cos_terms.append(t)
                    else:
                        base_terms.append(t)
                base_str = " ".join(base_terms)
                rows.append(f"{var_part} &= {base_str}")
                for ch in chunk_terms(cos_terms, 85):
                    rows.append(f"&\\quad {ch}")
            elif var_part.startswith("\\tau"):
                rows.append(f"{var_part} &= {terms[0]}")
                for t in terms[1:]:
                    rows.append(f"&\\quad {t}")
            else:
                chunked = chunk_terms(terms[1:], 85)
                rows.append(f"{var_part} &= {terms[0]}")
                for ch in chunked:
                    rows.append(f"&\\quad {ch}")
        else:
            rows.append(f"&\\quad {line}")
            
    res = ["$$", "\\begin{aligned}"]
    for idx, r in enumerate(rows):
        term_suffix = " \\\\[3pt]" if idx < len(rows) - 1 else ""
        res.append(f"  {r}{term_suffix}")
    res.append("\\end{aligned}")
    res.append("$$")
    return "\n".join(res)

def update_comparisons_readme():
    targets = [
        ("testImage1029", Path("hpc_transcriptions/test_image1029_equations.tex"), "512x384", "0.9104", "22.16 dB", "0.0961", "64"),
        ("trainImage1008", Path("hpc_transcriptions/train_image1008_equations.tex"), "512x512", "0.8780", "22.97 dB", "0.0990", "64"),
        ("testImage1013", Path("test_comparisons/test_image1013_equations.tex"), "362x512", "0.8437", "22.09 dB", "0.1356", "16"),
        ("testImage1023", Path("hpc_transcriptions/test_image1023_equations.tex"), "512x384", "0.8431", "24.29 dB", "0.1325", "64"),
        ("testImage10", Path("test_comparisons/test_image10_equations.tex"), "512x339", "0.8125", "19.15 dB", "0.1522", "16"),
        ("testImage1040", Path("hpc_transcriptions/test_image1040_equations.tex"), "313x512", "0.7991", "21.45 dB", "0.1622", "64"),
        ("testImage1014", Path("test_comparisons/test_image1014_equations.tex"), "341x512", "0.7487", "20.14 dB", "0.2055", "16"),
        ("testImage1043", Path("hpc_transcriptions/test_image1043_equations.tex"), "512x384", "0.7181", "20.59 dB", "0.2263", "64"),
    ]
    
    lines = [
        "# comparisons",
        "",
        "reconstructions matching yeganeh equations with full canvas normalized coords.",
        "",
        "## metrics",
        "",
        "| image | res | ssim | psnr | loss | layers |",
        "|:---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for name, tex, res, ssim, psnr, loss, layers in targets:
        lines.append(f"| {name} | {res} | {ssim} | {psnr} | {loss} | {layers} |")
    
    lines.append("")
    lines.append("## samples")
    lines.append("")
    
    for name, tex, _, _, _, _, _ in targets:
        lines.append(f"### {name}")
        lines.append(f"![{name}]({name}_comparison.png)")
        lines.append("")
        snippet = format_equation_properly(tex)
        if snippet:
            lines.append(snippet)
            lines.append("")
            
    Path("comparisons/README.md").write_text("\n".join(lines).strip() + "\n")
    print("Updated comparisons/README.md")

def update_root_readme():
    targets = [
        ("testImage1029", Path("hpc_transcriptions/test_image1029_equations.tex"), "0.9104", "22.16 dB"),
        ("trainImage1008", Path("hpc_transcriptions/train_image1008_equations.tex"), "0.8780", "22.97 dB"),
        ("testImage1013", Path("test_comparisons/test_image1013_equations.tex"), "0.8437", "22.09 dB"),
    ]
    
    comp_lines = [
        "## comparisons",
        "",
        "reconstructions matching yeganeh equations with full canvas normalized coords. see [comparisons/](comparisons/) for all samples.",
        "",
    ]
    for name, tex, ssim, psnr in targets:
        comp_lines.append(f"### {name} (ssim: {ssim}, psnr: {psnr})")
        comp_lines.append(f"![{name}](comparisons/{name}_comparison.png)")
        comp_lines.append("")
        snippet = format_equation_properly(tex)
        if snippet:
            comp_lines.append(snippet)
            comp_lines.append("")
            
    comp_block = "\n".join(comp_lines).strip()
    
    root_text = Path("README.md").read_text()
    pattern = r"## comparisons\s+reconstructions matching yeganeh equations.*?(?=## CLI Options)"
    new_root = re.sub(pattern, lambda m: comp_block + "\n\n", root_text, flags=re.DOTALL)
    Path("README.md").write_text(new_root)
    print("Updated root README.md")

def update_test_comparisons_readme():
    targets = [
        ("testImage1013", Path("test_comparisons/test_image1013_equations.tex"), "362x512", "0.8437", "22.09 dB", "0.13566", "16", "150", "32.5s"),
        ("testImage10", Path("test_comparisons/test_image10_equations.tex"), "512x339", "0.8125", "19.15 dB", "0.15227", "16", "150", "37.3s"),
        ("testImage1014", Path("test_comparisons/test_image1014_equations.tex"), "341x512", "0.7487", "20.14 dB", "0.20557", "16", "150", "33.4s"),
        ("testImage100", Path("test_comparisons/test_image100_equations.tex"), "512x384", "0.7002", "18.07 dB", "0.24753", "17", "150", "35.7s"),
        ("testImage1019", Path("test_comparisons/test_image1019_equations.tex"), "418x512", "0.6590", "15.53 dB", "0.32018", "16", "150", "40.4s"),
    ]
    
    lines = [
        "# testComparisons",
        "",
        "test split runs from ezzzio/random-images with normalized coords.",
        "",
        "## metrics",
        "",
        "| image | res | ssim | psnr | loss | layers | steps | time | tex | pdf |",
        "|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    for name, tex, res, ssim, psnr, loss, layers, steps, t in targets:
        tex_file = tex.name
        pdf_file = tex.stem + ".pdf"
        lines.append(f"| {name} | {res} | {ssim} | {psnr} | {loss} | {layers} | {steps} | {t} | [tex]({tex_file}) | [pdf]({pdf_file}) |")
        
    lines.append("")
    lines.append("## samples")
    lines.append("")
    
    for name, tex, _, _, _, _, _, _, _ in targets:
        lines.append(f"### {name}")
        lines.append(f"![{name}]({name}_comparison.png)")
        lines.append("")
        snippet = format_equation_properly(tex)
        if snippet:
            lines.append(snippet)
            lines.append("")
            
    Path("test_comparisons/README.md").write_text("\n".join(lines).strip() + "\n")
    print("Updated test_comparisons/README.md")

if __name__ == "__main__":
    update_comparisons_readme()
    update_root_readme()
    update_test_comparisons_readme()
