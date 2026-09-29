#!/usr/bin/env python3
"""
update_comparison_readmes.py
Injects concise LaTeX equation snippets under each comparison image in markdown files.
"""

import re
from pathlib import Path

def extract_equation_snippet(tex_path: Path):
    if not tex_path.exists():
        return ""
    content = tex_path.read_text()
    
    # coordinate transform
    coord_match = re.search(r"\\subsection\*\{(?:Scale-Invariant )?Coordinate (?:System|Transformation)\}\s*\\\[\s*(.*?)\s*\\\]", content, re.DOTALL)
    coord_eq = ""
    if coord_match:
        lines = [l.strip() for l in coord_match.group(1).splitlines() if l.strip()]
        coord_eq = " ".join(lines)
    
    # layer 1 formulation
    l1_match = re.search(r"% Layer 1: Primitive_1\s*\\begin\{align\*\}\s*(.*?)\s*\\end\{align\*\}", content, re.DOTALL)
    l1_clean = ""
    if l1_match:
        lines = []
        for l in l1_match.group(1).splitlines():
            l_clean = l.strip().rstrip("\\").strip()
            l_clean = re.sub(r"^\s*&\s*", "", l_clean)
            l_clean = re.sub(r"\s*&\s*", " ", l_clean)
            if l_clean:
                lines.append(l_clean)
        l1_clean = "\n".join(lines)
        
    out = "```latex\n"
    if coord_eq:
        out += f"% coordinate system\n{coord_eq}\n\n"
    if l1_clean:
        out += f"% layer formulation\n{l1_clean}\n"
    out += "```"
    return out

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
        snippet = extract_equation_snippet(tex)
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
        snippet = extract_equation_snippet(tex)
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
        snippet = extract_equation_snippet(tex)
        if snippet:
            lines.append(snippet)
            lines.append("")
            
    Path("test_comparisons/README.md").write_text("\n".join(lines).strip() + "\n")
    print("Updated test_comparisons/README.md")

if __name__ == "__main__":
    update_comparisons_readme()
    update_root_readme()
    update_test_comparisons_readme()
