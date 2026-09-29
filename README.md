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

$$
\begin{aligned}
  \tilde{X} &= 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[3pt]
  X &= \frac{\tilde{X} + \frac{95}{146}}{\frac{7}{20}}, \quad Y = \frac{\tilde{Y} - \frac{27}{164}}{\frac{91}{200}}, \quad \theta = \arctan\left(\frac{Y}{X}\right) \\[3pt]
  \tau_{1}(X,Y) &= \frac{13}{51}\cos^{2}\left(\frac{2819}{178}(X - \frac{95}{174}) + \frac{629}{113}(Y + \frac{2}{171}) - \frac{144}{199}\cos\left(\frac{171}{70}(X - \frac{95}{174}) + \frac{534}{191}(Y + \frac{2}{171})\right)\right) \\[3pt]
  &\quad +\frac{19}{56}\cos^{4}\left(\frac{512}{103}(X - \frac{95}{174}) + \frac{1801}{88}(Y + \frac{2}{171}) + \frac{177}{163}\cos\left(\frac{230}{197}(X - \frac{95}{174}) + \frac{687}{148}(Y + \frac{2}{171})\right)\right) \\[3pt]
  &\quad +\frac{31}{103}\cos^{6}\left(\frac{4018}{141}(X - \frac{95}{174}) + \frac{1707}{155}(Y + \frac{2}{171}) + \frac{103}{68}\cos\left(\frac{1154}{173}(X - \frac{95}{174}) + \frac{348}{101}(Y + \frac{2}{171})\right)\right) \\[3pt]
  &\quad +\frac{10}{103}\cos^{8}\left(\frac{2059}{166}(X - \frac{95}{174}) + \frac{3780}{107}(Y + \frac{2}{171}) - \frac{39}{32}\cos\left(\frac{310}{109}(X - \frac{95}{174}) + \frac{581}{94}(Y + \frac{2}{171})\right)\right) \\[3pt]
  &\quad + \frac{5}{91}\cos^{12}\left(\frac{6443}{135}(X - \frac{95}{174}) + \frac{3226}{167}(Y + \frac{2}{171}) + \frac{116}{173}\cos\left(\frac{579}{71}(X - \frac{95}{174}) + \frac{17}{4}(Y + \frac{2}{171})\right)\right) \\[3pt]
  &\quad +\frac{27}{191}\cos^{16}\left(\frac{3933}{191}(X - \frac{95}{174}) + \frac{10983}{197}(Y + \frac{2}{171}) + \frac{11}{9}\cos\left(\frac{316}{99}(X - \frac{95}{174}) + \frac{728}{85}(Y + \frac{2}{171})\right)\right) \\[3pt]
  C_{1}(X,Y,v) &= \frac{39-3v^2}{40} \cdot \left(1 + \frac{35}{53}(X - \frac{95}{174}) + \frac{88}{183}(Y + \frac{2}{171})\right) \\[3pt]
  &\quad + \tau_{1}(X,Y) \\[3pt]
  \Phi_{1}(X,Y) &= 1 - \left(\frac{(X - \frac{95}{174})}{0.002}\right)^{2} - \left(\frac{(Y + \frac{2}{171})}{\frac{5}{119}}\right)^{2} \\[3pt]
  &\quad +\frac{52}{87}\cos(1\theta + \frac{16}{17}) \\[3pt]
  &\quad +\frac{68}{107}\cos(2\theta + \frac{11}{148}) \\[3pt]
  &\quad +\frac{74}{177}\cos(3\theta + \frac{133}{181}) \\[3pt]
  &\quad +\frac{40}{199}\cos(4\theta + \frac{53}{54}) \\[3pt]
  &\quad +\frac{30}{61}\cos(5\theta + \frac{451}{172}) \\[3pt]
  &\quad +\frac{10}{147}\cos(6\theta + \frac{241}{132}) \\[3pt]
  &\quad -\frac{9}{190}\cos(7\theta + \frac{447}{158}) \\[3pt]
  &\quad +\frac{37}{104}\cos(8\theta + \frac{634}{191}) \\[3pt]
  W_{1}(X,Y) &= \exp\left(-\exp\left(-\frac{7229}{167}\cdot\Phi_{1}(X,Y)\right)\right)
\end{aligned}
$$

### trainImage1008 (ssim: 0.8780, psnr: 22.97 dB)
![trainImage1008](comparisons/trainImage1008_comparison.png)

$$
\begin{aligned}
  \tilde{X} &= 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[3pt]
  X &= \frac{\tilde{X} - \frac{94}{95}}{\frac{227}{147}}, \quad Y = \frac{\tilde{Y} - \frac{25}{199}}{\frac{60}{77}}, \quad \theta = \arctan\left(\frac{Y}{X}\right) \\[3pt]
  \tau_{1}(X,Y) &= -\frac{12}{55}\cos^{2}\left(\frac{2675}{177}(X + \frac{83}{182}) + \frac{269}{45}(Y + \frac{37}{36}) + \frac{83}{141}\cos\left(\frac{929}{198}(X + \frac{83}{182}) + \frac{575}{186}(Y + \frac{37}{36})\right)\right) \\[3pt]
  &\quad +\frac{1}{14}\cos^{4}\left(\frac{929}{155}(X + \frac{83}{182}) + \frac{947}{47}(Y + \frac{37}{36}) + \frac{25}{26}\cos\left(\frac{334}{157}(X + \frac{83}{182}) + \frac{925}{192}(Y + \frac{37}{36})\right)\right) \\[3pt]
  &\quad +\frac{27}{161}\cos^{6}\left(\frac{4441}{163}(X + \frac{83}{182}) + \frac{688}{53}(Y + \frac{37}{36}) + \frac{182}{193}\cos\left(\frac{941}{152}(X + \frac{83}{182}) + \frac{358}{175}(Y + \frac{37}{36})\right)\right) \\[3pt]
  &\quad +\frac{23}{194}\cos^{8}\left(\frac{2101}{174}(X + \frac{83}{182}) + \frac{5505}{154}(Y + \frac{37}{36}) + \frac{115}{146}\cos\left(\frac{543}{199}(X + \frac{83}{182}) + \frac{1393}{180}(Y + \frac{37}{36})\right)\right) \\[3pt]
  &\quad + \frac{5}{174}\cos^{12}\left(\frac{9320}{189}(X + \frac{83}{182}) + \frac{1279}{66}(Y + \frac{37}{36}) + \frac{69}{106}\cos\left(\frac{1469}{189}(X + \frac{83}{182}) + \frac{237}{59}(Y + \frac{37}{36})\right)\right) \\[3pt]
  &\quad +\frac{7}{36}\cos^{16}\left(\frac{2939}{150}(X + \frac{83}{182}) + \frac{2121}{38}(Y + \frac{37}{36}) - \frac{127}{193}\cos\left(\frac{594}{155}(X + \frac{83}{182}) + \frac{1604}{197}(Y + \frac{37}{36})\right)\right) \\[3pt]
  C_{1}(X,Y,v) &= \frac{52+3v-5v^2}{40} \cdot \left(1 - \frac{1}{3}(X + \frac{83}{182}) - \frac{49}{178}(Y + \frac{37}{36})\right) \\[3pt]
  &\quad + \tau_{1}(X,Y) \\[3pt]
  \Phi_{1}(X,Y) &= 1 - \left(\frac{(X + \frac{83}{182})}{\frac{37}{136}}\right)^{2} - \left(\frac{(Y + \frac{37}{36})}{\frac{2}{167}}\right)^{2} \\[3pt]
  &\quad -\frac{11}{129}\cos(1\theta - \frac{75}{131}) \\[3pt]
  &\quad -\frac{72}{73}\cos(2\theta - \frac{109}{189}) \\[3pt]
  &\quad -\frac{5}{18}\cos(3\theta + \frac{22}{17}) \\[3pt]
  &\quad -\frac{146}{199}\cos(4\theta + \frac{94}{63}) \\[3pt]
  &\quad -\frac{72}{131}\cos(5\theta + \frac{343}{106}) \\[3pt]
  &\quad -\frac{1}{113}\cos(6\theta + \frac{265}{117}) \\[3pt]
  &\quad +\frac{163}{171}\cos(7\theta + \frac{639}{184}) \\[3pt]
  &\quad +\frac{3}{22}\cos(8\theta + \frac{145}{47}) \\[3pt]
  W_{1}(X,Y) &= \exp\left(-\exp\left(-\frac{2887}{67}\cdot\Phi_{1}(X,Y)\right)\right)
\end{aligned}
$$

### testImage1013 (ssim: 0.8437, psnr: 22.09 dB)
![testImage1013](comparisons/testImage1013_comparison.png)

$$
\begin{aligned}
  \tilde{X} &= 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[3pt]
  X &= \frac{\left(\tilde{X} - \frac{5}{67}\right)}{\frac{83}{84}}, \quad Y = \frac{\left(\tilde{Y} + \frac{13}{176}\right)}{\frac{201}{182}}, \quad \theta = \arctan\left(\frac{Y}{X}\right) \\[3pt]
  X_{1} &= \left(X - \frac{35}{159}\right), \quad Y_{1} = \left(Y - \frac{99}{193}\right) \\[3pt]
  \tau_{1}(X,Y) &= -\frac{7}{187}\cos^{2}\left(\frac{351}{22}X_{1} + \frac{365}{69}Y_{1} - \frac{74}{141}\cos\left(\frac{581}{129}X_{1} + \frac{370}{191}Y_{1}\right)\right) \\[3pt]
  &\quad \quad + \frac{5}{76}\cos^{4}\left(\frac{1077}{200}X_{1} + \frac{2150}{107}Y_{1} + \frac{93}{158}\cos\left(\frac{55}{31}X_{1} + \frac{916}{169}Y_{1}\right)\right) + \frac{1}{28}\cos^{6}\left(\frac{1087}{39}X_{1} + \frac{1891}{150}Y_{1} + \frac{97}{151}\cos\left(\frac{1031}{179}X_{1} + \frac{467}{151}Y_{1}\right)\right) \\[3pt]
  &\quad \quad + \frac{1}{120}\cos^{8}\left(\frac{2138}{179}X_{1} + \frac{291}{8}Y_{1}\right) + \frac{5}{137}\cos^{12}\left(\frac{5153}{107}X_{1} + \frac{3566}{179}Y_{1} + \frac{15}{176}\cos\left(\frac{89}{11}X_{1} + \frac{354}{85}Y_{1}\right)\right) \\[3pt]
  &\quad \quad + \frac{2}{159}\cos^{16}\left(\frac{2089}{103}X_{1} + \frac{7957}{143}Y_{1} + \frac{53}{190}\cos\left(\frac{104}{23}X_{1} + \frac{1147}{150}Y_{1}\right)\right) \\[3pt]
  C_{1}(X,Y,v) &= \frac{9-2v}{40} \cdot \left(1 - \frac{13}{199}X_{1} + \frac{16}{149}Y_{1}\right) \\[3pt]
  &\quad + \tau_{1}(X,Y) \\[3pt]
  \Phi_{1}(X,Y) &= 1 - \left(\frac{X_{1}}{\frac{33}{38}}\right)^{2} - \left(\frac{Y_{1}}{\frac{36}{89}}\right)^{2} \\[3pt]
  &\quad \quad - \frac{1}{47}\cos\left(1\theta - \frac{1}{9}\right) + \frac{4}{29}\cos\left(2\theta + \frac{73}{85}\right) + \frac{13}{192}\cos\left(3\theta + \frac{138}{119}\right) + \frac{9}{182}\cos\left(4\theta + \frac{285}{182}\right) \\[3pt]
  &\quad \quad + \frac{20}{157}\cos\left(5\theta + 1.996\right) - \frac{30}{167}\cos\left(6\theta + \frac{535}{187}\right) + \frac{1}{44}\cos\left(7\theta + \frac{567}{199}\right) - \frac{3}{110}\cos\left(8\theta + \frac{513}{160}\right) \\[3pt]
  W_{1}(X,Y) &= \exp\left(-\exp\left(-\frac{6554}{151}\cdot\Phi_{1}(X,Y)\right)\right)
\end{aligned}
$$

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
