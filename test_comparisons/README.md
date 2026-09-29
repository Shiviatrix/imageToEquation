# testComparisons

test split runs from ezzzio/random-images with normalized coords.

## metrics

| image | res | ssim | psnr | loss | layers | steps | time | tex | pdf |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| testImage1013 | 362x512 | 0.8437 | 22.09 dB | 0.13566 | 16 | 150 | 32.5s | [tex](test_image1013_equations.tex) | [pdf](test_image1013_equations.pdf) |
| testImage10 | 512x339 | 0.8125 | 19.15 dB | 0.15227 | 16 | 150 | 37.3s | [tex](test_image10_equations.tex) | [pdf](test_image10_equations.pdf) |
| testImage1014 | 341x512 | 0.7487 | 20.14 dB | 0.20557 | 16 | 150 | 33.4s | [tex](test_image1014_equations.tex) | [pdf](test_image1014_equations.pdf) |
| testImage100 | 512x384 | 0.7002 | 18.07 dB | 0.24753 | 17 | 150 | 35.7s | [tex](test_image100_equations.tex) | [pdf](test_image100_equations.pdf) |
| testImage1019 | 418x512 | 0.6590 | 15.53 dB | 0.32018 | 16 | 150 | 40.4s | [tex](test_image1019_equations.tex) | [pdf](test_image1019_equations.pdf) |

## samples

### testImage1013
![testImage1013](testImage1013_comparison.png)

```latex
% coordinate system
\tilde{X} = 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[6pt] X = \frac{\left(\tilde{X} - \frac{5}{67}\right)}{\frac{83}{84}}, \quad Y = \frac{\left(\tilde{Y} + \frac{13}{176}\right)}{\frac{201}{182}}, \quad \theta = \arctan\left(\frac{Y}{X}\right)

% layer formulation
X_{1} = \left(X - \frac{35}{159}\right), \quad Y_{1} = \left(Y - \frac{99}{193}\right)
\tau_{1}(X,Y) = -\frac{7}{187}\cos^{2}\left(\frac{351}{22}X_{1} + \frac{365}{69}Y_{1} - \frac{74}{141}\cos\left(\frac{581}{129}X_{1} + \frac{370}{191}Y_{1}\right)\right)
\quad + \frac{5}{76}\cos^{4}\left(\frac{1077}{200}X_{1} + \frac{2150}{107}Y_{1} + \frac{93}{158}\cos\left(\frac{55}{31}X_{1} + \frac{916}{169}Y_{1}\right)\right) + \frac{1}{28}\cos^{6}\left(\frac{1087}{39}X_{1} + \frac{1891}{150}Y_{1} + \frac{97}{151}\cos\left(\frac{1031}{179}X_{1} + \frac{467}{151}Y_{1}\right)\right)
\quad + \frac{1}{120}\cos^{8}\left(\frac{2138}{179}X_{1} + \frac{291}{8}Y_{1}\right) + \frac{5}{137}\cos^{12}\left(\frac{5153}{107}X_{1} + \frac{3566}{179}Y_{1} + \frac{15}{176}\cos\left(\frac{89}{11}X_{1} + \frac{354}{85}Y_{1}\right)\right)
\quad + \frac{2}{159}\cos^{16}\left(\frac{2089}{103}X_{1} + \frac{7957}{143}Y_{1} + \frac{53}{190}\cos\left(\frac{104}{23}X_{1} + \frac{1147}{150}Y_{1}\right)\right)
C_{1}(X,Y,v) = \frac{9-2v}{40} \cdot \left(1 - \frac{13}{199}X_{1} + \frac{16}{149}Y_{1}\right) + \tau_{1}(X,Y)
\Phi_{1}(X,Y) = 1 - \left(\frac{X_{1}}{\frac{33}{38}}\right)^{2} - \left(\frac{Y_{1}}{\frac{36}{89}}\right)^{2}
\quad - \frac{1}{47}\cos\left(1\theta - \frac{1}{9}\right) + \frac{4}{29}\cos\left(2\theta + \frac{73}{85}\right) + \frac{13}{192}\cos\left(3\theta + \frac{138}{119}\right) + \frac{9}{182}\cos\left(4\theta + \frac{285}{182}\right)
\quad + \frac{20}{157}\cos\left(5\theta + 1.996\right) - \frac{30}{167}\cos\left(6\theta + \frac{535}{187}\right) + \frac{1}{44}\cos\left(7\theta + \frac{567}{199}\right) - \frac{3}{110}\cos\left(8\theta + \frac{513}{160}\right)
W_{1}(X,Y) = \exp\left(-\exp\left(-\frac{6554}{151}\cdot\Phi_{1}(X,Y)\right)\right)
```

### testImage10
![testImage10](testImage10_comparison.png)

```latex
% coordinate system
\tilde{X} = 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[6pt] X = \frac{\left(\tilde{X} + \frac{2}{139}\right)}{\frac{162}{167}}, \quad Y = \frac{\left(\tilde{Y} - \frac{32}{179}\right)}{1}, \quad \theta = \arctan\left(\frac{Y}{X}\right)

% layer formulation
X_{1} = \left(X - \frac{4}{177}\right), \quad Y_{1} = \left(Y - \frac{20}{41}\right)
\tau_{1}(X,Y) = \frac{69}{79}\cos^{2}\left(\frac{2713}{193}X_{1} + \frac{749}{118}Y_{1} - \frac{259}{151}\cos\left(\frac{489}{155}X_{1} + \frac{86}{29}Y_{1}\right)\right)
\quad + \frac{103}{129}\cos^{4}\left(\frac{1031}{179}X_{1} + \frac{2468}{121}Y_{1} + \frac{14}{17}\cos\left(\frac{606}{175}X_{1} + \frac{109}{26}Y_{1}\right)\right) + \frac{95}{139}\cos^{6}\left(\frac{4554}{155}X_{1} + \frac{352}{31}Y_{1} - \frac{21}{19}\cos\left(\frac{1067}{147}X_{1} + \frac{359}{159}Y_{1}\right)\right)
\quad + \frac{37}{52}\cos^{8}\left(\frac{880}{73}X_{1} + \frac{2257}{63}Y_{1} - \frac{47}{71}\cos\left(\frac{404}{115}X_{1} + \frac{923}{155}Y_{1}\right)\right) + \frac{164}{181}\cos^{12}\left(\frac{4747}{98}X_{1} + \frac{3769}{195}Y_{1} + \frac{79}{180}\cos\left(\frac{695}{87}X_{1} + \frac{397}{105}Y_{1}\right)\right)
\quad + \frac{131}{145}\cos^{16}\left(\frac{3505}{172}X_{1} + \frac{1121}{20}Y_{1} + \frac{45}{197}\cos\left(\frac{626}{171}X_{1} + \frac{772}{101}Y_{1}\right)\right)
C_{1}(X,Y,v) = \frac{66+23v+22v^2}{40} \cdot \left(1 + \frac{27}{46}X_{1} + \frac{70}{149}Y_{1}\right) + \tau_{1}(X,Y)
\Phi_{1}(X,Y) = 1 - \left(\frac{X_{1}}{\frac{159}{128}}\right)^{2} - \left(\frac{Y_{1}}{\frac{171}{158}}\right)^{2}
\quad + \frac{20}{77}\cos\left(1\theta - \frac{113}{183}\right) + \frac{13}{88}\cos\left(2\theta + \frac{11}{35}\right) - \frac{68}{173}\cos\left(3\theta + \frac{172}{103}\right) + \frac{23}{165}\cos\left(4\theta + \frac{170}{151}\right)
\quad + \frac{8}{125}\cos\left(5\theta + \frac{296}{173}\right) + \frac{37}{155}\cos\left(6\theta + \frac{299}{110}\right) - \frac{23}{90}\cos\left(7\theta + \frac{571}{178}\right) + \frac{17}{148}\cos\left(8\theta + \frac{613}{190}\right)
W_{1}(X,Y) = \exp\left(-\exp\left(-\frac{3643}{82}\cdot\Phi_{1}(X,Y)\right)\right)
```

### testImage1014
![testImage1014](testImage1014_comparison.png)

```latex
% coordinate system
\tilde{X} = 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[6pt] X = \frac{\left(\tilde{X} + \frac{8}{159}\right)}{\frac{78}{103}}, \quad Y = \frac{\left(\tilde{Y} - \frac{26}{197}\right)}{\frac{136}{149}}, \quad \theta = \arctan\left(\frac{Y}{X}\right)

% layer formulation
X_{1} = \left(X + \frac{15}{106}\right), \quad Y_{1} = \left(Y - \frac{22}{41}\right)
\tau_{1}(X,Y) = -\frac{5}{148}\cos^{2}\left(\frac{2736}{175}X_{1} + \frac{1234}{195}Y_{1} + \frac{10}{189}\cos\left(\frac{479}{128}X_{1} + \frac{208}{95}Y_{1}\right)\right)
\quad - \frac{11}{76}\cos^{4}\left(\frac{655}{112}X_{1} + \frac{2743}{133}Y_{1} + \frac{5}{6}\cos\left(\frac{176}{65}X_{1} + \frac{584}{133}Y_{1}\right)\right) - \frac{3}{121}\cos^{6}\left(\frac{4505}{162}X_{1} + \frac{1174}{97}Y_{1} - \frac{7}{29}\cos\left(\frac{713}{119}X_{1} + \frac{293}{106}Y_{1}\right)\right)
\quad + \frac{12}{107}\cos^{8}\left(\frac{2197}{183}X_{1} + \frac{3097}{86}Y_{1} - \frac{19}{87}\cos\left(\frac{331}{135}X_{1} + \frac{1493}{199}Y_{1}\right)\right) - \frac{10}{69}\cos^{12}\left(\frac{4101}{86}X_{1} + \frac{293}{15}Y_{1} - \frac{13}{79}\cos\left(\frac{122}{17}X_{1} + \frac{769}{196}Y_{1}\right)\right)
\quad - \frac{7}{121}\cos^{16}\left(\frac{779}{39}X_{1} + \frac{3518}{63}Y_{1} + \frac{57}{194}\cos\left(\frac{519}{124}X_{1} + \frac{1269}{158}Y_{1}\right)\right)
C_{1}(X,Y,v) = \frac{39+v-2v^2}{40} \cdot \left(1 + \frac{1}{42}Y_{1}\right) + \tau_{1}(X,Y)
\Phi_{1}(X,Y) = 1 - \left(\frac{X_{1}}{\frac{147}{187}}\right)^{2} - \left(\frac{Y_{1}}{\frac{9}{25}}\right)^{2}
\quad - \frac{29}{111}\cos\left(1\theta + \frac{63}{190}\right) + \frac{23}{117}\cos\left(2\theta + \frac{37}{146}\right) - \frac{80}{179}\cos\left(3\theta + \frac{98}{99}\right) + \frac{11}{113}\cos\left(4\theta + \frac{184}{131}\right)
\quad - \frac{73}{197}\cos\left(5\theta + \frac{223}{107}\right) + \frac{25}{199}\cos\left(6\theta + \frac{53}{24}\right) + \frac{9}{166}\cos\left(7\theta + \frac{501}{145}\right) - \frac{13}{192}\cos\left(8\theta + \frac{412}{129}\right)
W_{1}(X,Y) = \exp\left(-\exp\left(-\frac{2810}{63}\cdot\Phi_{1}(X,Y)\right)\right)
```

### testImage100
![testImage100](testImage100_comparison.png)

```latex
% coordinate system
\tilde{X} = 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[6pt] X = \frac{\left(\tilde{X} - \frac{8}{71}\right)}{\frac{146}{135}}, \quad Y = \frac{\left(\tilde{Y} - \frac{1}{86}\right)}{\frac{97}{115}}, \quad \theta = \arctan\left(\frac{Y}{X}\right)

% layer formulation
X_{1} = \left(X + \frac{58}{167}\right), \quad Y_{1} = \left(Y - \frac{33}{106}\right)
\tau_{1}(X,Y) = \frac{3}{157}\cos^{2}\left(\frac{639}{40}X_{1} + \frac{1153}{196}Y_{1} - \frac{34}{149}\cos\left(\frac{761}{183}X_{1} + \frac{217}{101}Y_{1}\right)\right)
\quad + \frac{14}{195}\cos^{4}\left(\frac{278}{45}X_{1} + \frac{3371}{167}Y_{1} + \frac{15}{88}\cos\left(\frac{372}{181}X_{1} + \frac{543}{109}Y_{1}\right)\right) + \frac{5}{121}\cos^{6}\left(\frac{5103}{181}X_{1} + \frac{1781}{146}Y_{1} - \frac{14}{71}\cos\left(\frac{1168}{195}X_{1} + \frac{217}{71}Y_{1}\right)\right)
\quad + \frac{8}{111}\cos^{8}\left(\frac{986}{81}X_{1} + \frac{217}{6}Y_{1} - \frac{25}{164}\cos\left(\frac{167}{55}X_{1} + \frac{1027}{148}Y_{1}\right)\right) + \frac{16}{191}\cos^{12}\left(\frac{8689}{182}X_{1} + \frac{3593}{182}Y_{1} + \frac{35}{141}\cos\left(\frac{1529}{195}X_{1} + \frac{593}{154}Y_{1}\right)\right)
\quad + \frac{1}{58}\cos^{16}\left(\frac{81}{4}X_{1} + \frac{1856}{33}Y_{1} - \frac{37}{145}\cos\left(\frac{410}{99}X_{1} + \frac{1128}{139}Y_{1}\right)\right)
C_{1}(X,Y,v) = \frac{17+3v+4v^2}{40} \cdot \left(1 + \frac{1}{12}X_{1} + \frac{7}{173}Y_{1}\right) + \tau_{1}(X,Y)
\Phi_{1}(X,Y) = 1 - \left(\frac{X_{1}}{\frac{28}{79}}\right)^{2} - \left(\frac{Y_{1}}{\frac{43}{195}}\right)^{2}
\quad - \frac{14}{57}\cos\left(1\theta + \frac{12}{115}\right) - \frac{13}{75}\cos\left(2\theta + \frac{58}{109}\right) + \frac{1}{17}\cos\left(3\theta + \frac{23}{21}\right) + \frac{11}{50}\cos\left(4\theta + \frac{130}{77}\right)
\quad + \frac{15}{56}\cos\left(5\theta + \frac{325}{173}\right) + \frac{17}{145}\cos\left(6\theta + \frac{377}{162}\right) + \frac{1}{9}\cos\left(7\theta + 3.002\right) + \frac{28}{195}\cos\left(8\theta + \frac{710}{199}\right)
W_{1}(X,Y) = \exp\left(-\exp\left(-\frac{3975}{89}\cdot\Phi_{1}(X,Y)\right)\right)
```

### testImage1019
![testImage1019](testImage1019_comparison.png)

```latex
% coordinate system
\tilde{X} = 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[6pt] X = \frac{\left(\tilde{X} + \frac{1}{8}\right)}{\frac{37}{45}}, \quad Y = \frac{\left(\tilde{Y} + \frac{6}{85}\right)}{\frac{151}{170}}, \quad \theta = \arctan\left(\frac{Y}{X}\right)

% layer formulation
X_{1} = \left(X + \frac{4}{61}\right), \quad Y_{1} = \left(Y + \frac{64}{77}\right)
\tau_{1}(X,Y) = -\frac{4}{123}\cos^{4}\left(\frac{531}{95}X_{1} + \frac{2698}{139}Y_{1} - \frac{53}{84}\cos\left(\frac{643}{195}X_{1} + \frac{747}{197}Y_{1}\right)\right)
\quad - \frac{8}{155}\cos^{6}\left(\frac{2089}{76}X_{1} + \frac{2229}{199}Y_{1} - \frac{18}{13}\cos\left(\frac{736}{101}X_{1} + \frac{632}{193}Y_{1}\right)\right) - \frac{1}{57}\cos^{8}\left(\frac{2271}{190}X_{1} + \frac{6459}{182}Y_{1} - \frac{50}{199}\cos\left(\frac{111}{38}X_{1} + \frac{497}{67}Y_{1}\right)\right)
\quad - \frac{2}{179}\cos^{12}\left(\frac{8777}{183}X_{1} + \frac{1574}{77}Y_{1}\right) - \frac{3}{151}\cos^{16}\left(\frac{2953}{150}X_{1} + \frac{7239}{130}Y_{1}\right)
C_{1}(X,Y,v) = \frac{3+2v^2}{40} \cdot \left(1 + \frac{19}{74}X_{1} - \frac{24}{83}Y_{1}\right) + \tau_{1}(X,Y)
\Phi_{1}(X,Y) = 1 - \left(\frac{X_{1}}{\frac{44}{73}}\right)^{2} - \left(\frac{Y_{1}}{\frac{59}{60}}\right)^{2}
\quad - \frac{18}{107}\cos\left(2\theta + \frac{83}{184}\right) + \frac{7}{60}\cos\left(3\theta + \frac{25}{27}\right) + \frac{16}{189}\cos\left(4\theta + \frac{37}{19}\right) - \frac{5}{64}\cos\left(5\theta + \frac{267}{113}\right)
\quad + \frac{7}{181}\cos\left(6\theta + \frac{221}{77}\right) + \frac{25}{191}\cos\left(7\theta + \frac{331}{114}\right) - \frac{3}{152}\cos\left(8\theta + \frac{642}{179}\right)
W_{1}(X,Y) = \exp\left(-\exp\left(-\frac{8225}{191}\cdot\Phi_{1}(X,Y)\right)\right)
```
