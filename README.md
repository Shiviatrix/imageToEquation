# imageToEquation

Converts raster images into closed-form algebraic and trigonometric equations matching Hamid Naderi Yeganeh's mathematical artwork formulation. Outputs standalone LaTeX documents and rendered images.

## Requirements

- Python 3.10+
- PyTorch 2.0+
- `pdflatex` (TeX Live or MacTeX, optional for PDF compilation)

Install Python dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Single Image

Transcribe an image into LaTeX equations and render the output:

```bash
python transcribeImage.py --input path/to/image.png --outputDir ./output --compilePdf
```

### Batch Processing

Transcribe all images in a directory:

```bash
python batchTranscribe.py --inputDir ./test_images --outputDir ./transcriptions_batch --compilePdf
```

### Continuous Training / HPC Worker

Run the continuous optimization worker on local GPU or cluster:

```bash
# Continuous streaming from curated online artworks
python runHpcTrainer.py --mode curated --targetPsnr 26.0

# Transcribe a specific online image URL
python runHpcTrainer.py --mode url --url "https://example.com/artwork.png" --targetPsnr 28.0

# Run in background via shell manager
./submitHpc.sh start
./submitHpc.sh status
./submitHpc.sh stop
```

To submit to a Slurm cluster:

```bash
sbatch hpcJob.slurm
```

## comparisons

reconstructions matching yeganeh equations with full canvas normalized coords. see [comparisons/](comparisons/) for all samples.

### testImage1029 (ssim: 0.9104, psnr: 22.16 dB)
![testImage1029](comparisons/testImage1029_comparison.png)

### trainImage1008 (ssim: 0.8780, psnr: 22.97 dB)
![trainImage1008](comparisons/trainImage1008_comparison.png)

### testImage1013 (ssim: 0.8437, psnr: 22.09 dB)
![testImage1013](comparisons/testImage1013_comparison.png)

## CLI Options

| Argument | Description | Default |
|---|---|---|
| `--input` | Path to target image | Required |
| `--outputDir` | Directory for exported `.tex`, `.pdf`, and `.png` files | `./output` |
| `--numLayers` | Initial number of mathematical layers | `16` |
| `--numIterations` | Number of optimization iterations | `250` |
| `--learningRate` | Base optimization learning rate | `0.035` |
| `--compilePdf` | Compile the exported `.tex` file to `.pdf` via `pdflatex` | `False` |
| `--device` | Compute device (`cuda`, `mps`, `cpu`) | Auto-detect |

## Project Structure

```
├── transcribeImage.py     # Single image transcription CLI
├── batchTranscribe.py     # Batch image transcription CLI
├── runHpcTrainer.py       # Continuous training worker
├── submitHpc.sh           # Daemon / Slurm runner script
├── hpcJob.slurm           # Slurm job submission script
├── comparisons/           # Reconstructions and error residual maps
└── src/
    └── imageToEquation/   # Core library
        ├── differentiableYeganeh.py
        ├── yeganehAst.py
        ├── yeganehTranscriber.py
        ├── perceptualLoss.py
        ├── onlineFetcher.py
        └── hpcEngine.py
```

## License

MIT
