# comparisons

reconstructions matching yeganeh equations with full canvas normalized coords.

## metrics

| image | res | ssim | psnr | loss | layers |
|:---|:---:|:---:|:---:|:---:|:---:|
| testImage1029 | 512x384 | 0.9104 | 22.16 dB | 0.0961 | 64 |
| trainImage1008 | 512x512 | 0.8780 | 22.97 dB | 0.0990 | 64 |
| testImage1013 | 362x512 | 0.8437 | 22.09 dB | 0.1356 | 16 |
| testImage1023 | 512x384 | 0.8431 | 24.29 dB | 0.1325 | 64 |
| testImage10 | 512x339 | 0.8125 | 19.15 dB | 0.1522 | 16 |
| testImage1040 | 313x512 | 0.7991 | 21.45 dB | 0.1622 | 64 |
| testImage1014 | 341x512 | 0.7487 | 20.14 dB | 0.2055 | 16 |
| testImage1043 | 512x384 | 0.7181 | 20.59 dB | 0.2263 | 64 |

## samples

### testImage1029
![testImage1029](testImage1029_comparison.png)

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

### trainImage1008
![trainImage1008](trainImage1008_comparison.png)

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

### testImage1013
![testImage1013](testImage1013_comparison.png)

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

### testImage1023
![testImage1023](testImage1023_comparison.png)

$$
\begin{aligned}
  \tilde{X} &= 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[3pt]
  X &= \frac{\tilde{X} - \frac{119}{90}}{\frac{319}{190}}, \quad Y = \frac{\tilde{Y} - \frac{61}{130}}{\frac{8}{63}}, \quad \theta = \arctan\left(\frac{Y}{X}\right) \\[3pt]
  \tau_{1}(X,Y) &= -\frac{57}{197}\cos^{2}\left(\frac{3087}{197}(X - \frac{9}{155}) + \frac{1241}{197}(Y - \frac{91}{181}) + \frac{25}{56}\cos\left(\frac{896}{199}(X - \frac{9}{155}) + \frac{431}{182}(Y - \frac{91}{181})\right)\right) \\[3pt]
  &\quad + \frac{38}{161}\cos^{4}\left(\frac{232}{43}(X - \frac{9}{155}) + \frac{103}{5}(Y - \frac{91}{181}) + \frac{151}{169}\cos\left(\frac{25}{18}(X - \frac{9}{155}) + \frac{406}{79}(Y - \frac{91}{181})\right)\right) \\[3pt]
  &\quad +\frac{9}{193}\cos^{6}\left(\frac{5545}{189}(X - \frac{9}{155}) + \frac{617}{48}(Y - \frac{91}{181}) + \frac{7}{31}\cos\left(\frac{800}{147}(X - \frac{9}{155}) + \frac{376}{135}(Y - \frac{91}{181})\right)\right) \\[3pt]
  &\quad + \frac{17}{122}\cos^{8}\left(\frac{233}{20}(X - \frac{9}{155}) + \frac{2692}{75}(Y - \frac{91}{181}) - \frac{11}{64}\cos\left(\frac{9}{4}(X - \frac{9}{155}) + \frac{473}{68}(Y - \frac{91}{181})\right)\right) \\[3pt]
  &\quad + \frac{1}{104}\cos^{12}\left(\frac{8729}{181}(X - \frac{9}{155}) + \frac{3671}{200}(Y - \frac{91}{181}) - \frac{185}{198}\cos\left(7.997(X - \frac{9}{155}) + \frac{446}{101}(Y - \frac{91}{181})\right)\right) \\[3pt]
  &\quad +\frac{36}{181}\cos^{16}\left(\frac{1067}{53}(X - \frac{9}{155}) + \frac{2186}{39}(Y - \frac{91}{181}) - \frac{38}{179}\cos\left(\frac{229}{54}(X - \frac{9}{155}) + \frac{1072}{131}(Y - \frac{91}{181})\right)\right) \\[3pt]
  C_{1}(X,Y,v) &= \frac{26-4v+v^2}{40} \cdot \left(1 + \frac{46}{121}(X - \frac{9}{155}) + \frac{7}{16}(Y - \frac{91}{181})\right) \\[3pt]
  &\quad + \tau_{1}(X,Y) \\[3pt]
  \Phi_{1}(X,Y) &= 1 - \left(\frac{(X - \frac{9}{155})}{\frac{18}{199}}\right)^{2} - \left(\frac{(Y - \frac{91}{181})}{\frac{16}{191}}\right)^{2} \\[3pt]
  &\quad +\frac{47}{194}\cos(1\theta - \frac{85}{112}) \\[3pt]
  &\quad -\frac{1}{21}\cos(2\theta + \frac{217}{171}) \\[3pt]
  &\quad +\frac{25}{73}\cos(3\theta + \frac{98}{121}) \\[3pt]
  &\quad -\frac{41}{167}\cos(4\theta + \frac{71}{76}) \\[3pt]
  &\quad -\frac{9}{143}\cos(5\theta + \frac{59}{30}) \\[3pt]
  &\quad -\frac{10}{129}\cos(6\theta + \frac{160}{51}) \\[3pt]
  &\quad +\frac{2}{73}\cos(7\theta + \frac{564}{199}) \\[3pt]
  &\quad +\frac{8}{23}\cos(8\theta + \frac{695}{181}) \\[3pt]
  W_{1}(X,Y) &= \exp\left(-\exp\left(-\frac{1033}{24}\cdot\Phi_{1}(X,Y)\right)\right)
\end{aligned}
$$

### testImage10
![testImage10](testImage10_comparison.png)

$$
\begin{aligned}
  \tilde{X} &= 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[3pt]
  X &= \frac{\left(\tilde{X} + \frac{2}{139}\right)}{\frac{162}{167}}, \quad Y = \frac{\left(\tilde{Y} - \frac{32}{179}\right)}{1}, \quad \theta = \arctan\left(\frac{Y}{X}\right) \\[3pt]
  X_{1} &= \left(X - \frac{4}{177}\right), \quad Y_{1} = \left(Y - \frac{20}{41}\right) \\[3pt]
  \tau_{1}(X,Y) &= \frac{69}{79}\cos^{2}\left(\frac{2713}{193}X_{1} + \frac{749}{118}Y_{1} - \frac{259}{151}\cos\left(\frac{489}{155}X_{1} + \frac{86}{29}Y_{1}\right)\right) \\[3pt]
  &\quad \quad + \frac{103}{129}\cos^{4}\left(\frac{1031}{179}X_{1} + \frac{2468}{121}Y_{1} + \frac{14}{17}\cos\left(\frac{606}{175}X_{1} + \frac{109}{26}Y_{1}\right)\right) + \frac{95}{139}\cos^{6}\left(\frac{4554}{155}X_{1} + \frac{352}{31}Y_{1} - \frac{21}{19}\cos\left(\frac{1067}{147}X_{1} + \frac{359}{159}Y_{1}\right)\right) \\[3pt]
  &\quad \quad + \frac{37}{52}\cos^{8}\left(\frac{880}{73}X_{1} + \frac{2257}{63}Y_{1} - \frac{47}{71}\cos\left(\frac{404}{115}X_{1} + \frac{923}{155}Y_{1}\right)\right) + \frac{164}{181}\cos^{12}\left(\frac{4747}{98}X_{1} + \frac{3769}{195}Y_{1} + \frac{79}{180}\cos\left(\frac{695}{87}X_{1} + \frac{397}{105}Y_{1}\right)\right) \\[3pt]
  &\quad \quad + \frac{131}{145}\cos^{16}\left(\frac{3505}{172}X_{1} + \frac{1121}{20}Y_{1} + \frac{45}{197}\cos\left(\frac{626}{171}X_{1} + \frac{772}{101}Y_{1}\right)\right) \\[3pt]
  C_{1}(X,Y,v) &= \frac{66+23v+22v^2}{40} \cdot \left(1 + \frac{27}{46}X_{1} + \frac{70}{149}Y_{1}\right) \\[3pt]
  &\quad + \tau_{1}(X,Y) \\[3pt]
  \Phi_{1}(X,Y) &= 1 - \left(\frac{X_{1}}{\frac{159}{128}}\right)^{2} - \left(\frac{Y_{1}}{\frac{171}{158}}\right)^{2} \\[3pt]
  &\quad \quad + \frac{20}{77}\cos\left(1\theta - \frac{113}{183}\right) + \frac{13}{88}\cos\left(2\theta + \frac{11}{35}\right) - \frac{68}{173}\cos\left(3\theta + \frac{172}{103}\right) + \frac{23}{165}\cos\left(4\theta + \frac{170}{151}\right) \\[3pt]
  &\quad \quad + \frac{8}{125}\cos\left(5\theta + \frac{296}{173}\right) + \frac{37}{155}\cos\left(6\theta + \frac{299}{110}\right) - \frac{23}{90}\cos\left(7\theta + \frac{571}{178}\right) + \frac{17}{148}\cos\left(8\theta + \frac{613}{190}\right) \\[3pt]
  W_{1}(X,Y) &= \exp\left(-\exp\left(-\frac{3643}{82}\cdot\Phi_{1}(X,Y)\right)\right)
\end{aligned}
$$

### testImage1040
![testImage1040](testImage1040_comparison.png)

$$
\begin{aligned}
  \tilde{X} &= 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[3pt]
  X &= \frac{\tilde{X} - \frac{84}{181}}{\frac{9}{31}}, \quad Y = \frac{\tilde{Y} - \frac{71}{182}}{\frac{81}{152}}, \quad \theta = \arctan\left(\frac{Y}{X}\right) \\[3pt]
  \tau_{1}(X,Y) &= \frac{19}{177}\cos^{2}\left(\frac{1597}{103}(X - \frac{33}{193}) + \frac{284}{47}(Y - \frac{73}{174}) + \frac{105}{157}\cos\left(\frac{27}{7}(X - \frac{33}{193}) + \frac{25}{16}(Y - \frac{73}{174})\right)\right) \\[3pt]
  &\quad + \frac{59}{180}\cos^{4}\left(\frac{312}{55}(X - \frac{33}{193}) + \frac{142}{7}(Y - \frac{73}{174}) - \frac{17}{126}\cos\left(\frac{303}{193}(X - \frac{33}{193}) + \frac{580}{119}(Y - \frac{73}{174})\right)\right) \\[3pt]
  &\quad +\frac{2}{41}\cos^{6}\left(\frac{1111}{40}(X - \frac{33}{193}) + \frac{1247}{102}(Y - \frac{73}{174}) - \frac{11}{53}\cos\left(\frac{704}{117}(X - \frac{33}{193}) + \frac{494}{157}(Y - \frac{73}{174})\right)\right) \\[3pt]
  &\quad +\frac{9}{200}\cos^{8}\left(\frac{1868}{157}(X - \frac{33}{193}) + \frac{1524}{43}(Y - \frac{73}{174}) - \frac{1}{4}\cos\left(\frac{627}{196}(X - \frac{33}{193}) + \frac{1406}{199}(Y - \frac{73}{174})\right)\right) \\[3pt]
  &\quad + \frac{3}{103}\cos^{12}\left(\frac{383}{8}(X - \frac{33}{193}) + \frac{3768}{191}(Y - \frac{73}{174})\right) \\[3pt]
  &\quad +\frac{31}{138}\cos^{16}\left(\frac{1867}{94}(X - \frac{33}{193}) + \frac{6595}{118}(Y - \frac{73}{174}) + \frac{36}{97}\cos\left(\frac{391}{97}(X - \frac{33}{193}) + \frac{71}{9}(Y - \frac{73}{174})\right)\right) \\[3pt]
  C_{1}(X,Y,v) &= \frac{38+6v+8v^2}{40} \cdot \left(1 - \frac{46}{157}(X - \frac{33}{193}) + \frac{2}{83}(Y - \frac{73}{174})\right) \\[3pt]
  &\quad + \tau_{1}(X,Y) \\[3pt]
  \Phi_{1}(X,Y) &= 1 - \left(\frac{(X - \frac{33}{193})}{\frac{1}{105}}\right)^{2} - \left(\frac{(Y - \frac{73}{174})}{\frac{8}{141}}\right)^{2} \\[3pt]
  &\quad +\frac{18}{145}\cos(1\theta + \frac{6}{115}) \\[3pt]
  &\quad -\frac{2}{141}\cos(2\theta + \frac{195}{188}) \\[3pt]
  &\quad +\frac{6}{109}\cos(3\theta + \frac{129}{196}) \\[3pt]
  &\quad +\frac{1}{34}\cos(4\theta + \frac{222}{163}) \\[3pt]
  &\quad -\frac{8}{59}\cos(5\theta + \frac{157}{74}) \\[3pt]
  &\quad -\frac{63}{146}\cos(6\theta + \frac{359}{176}) \\[3pt]
  &\quad -\frac{4}{23}\cos(7\theta + \frac{57}{20}) \\[3pt]
  &\quad +\frac{55}{123}\cos(8\theta + \frac{223}{71}) \\[3pt]
  W_{1}(X,Y) &= \exp\left(-\exp\left(-\frac{4184}{95}\cdot\Phi_{1}(X,Y)\right)\right)
\end{aligned}
$$

### testImage1014
![testImage1014](testImage1014_comparison.png)

$$
\begin{aligned}
  \tilde{X} &= 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[3pt]
  X &= \frac{\left(\tilde{X} + \frac{8}{159}\right)}{\frac{78}{103}}, \quad Y = \frac{\left(\tilde{Y} - \frac{26}{197}\right)}{\frac{136}{149}}, \quad \theta = \arctan\left(\frac{Y}{X}\right) \\[3pt]
  X_{1} &= \left(X + \frac{15}{106}\right), \quad Y_{1} = \left(Y - \frac{22}{41}\right) \\[3pt]
  \tau_{1}(X,Y) &= -\frac{5}{148}\cos^{2}\left(\frac{2736}{175}X_{1} + \frac{1234}{195}Y_{1} + \frac{10}{189}\cos\left(\frac{479}{128}X_{1} + \frac{208}{95}Y_{1}\right)\right) \\[3pt]
  &\quad \quad - \frac{11}{76}\cos^{4}\left(\frac{655}{112}X_{1} + \frac{2743}{133}Y_{1} + \frac{5}{6}\cos\left(\frac{176}{65}X_{1} + \frac{584}{133}Y_{1}\right)\right) - \frac{3}{121}\cos^{6}\left(\frac{4505}{162}X_{1} + \frac{1174}{97}Y_{1} - \frac{7}{29}\cos\left(\frac{713}{119}X_{1} + \frac{293}{106}Y_{1}\right)\right) \\[3pt]
  &\quad \quad + \frac{12}{107}\cos^{8}\left(\frac{2197}{183}X_{1} + \frac{3097}{86}Y_{1} - \frac{19}{87}\cos\left(\frac{331}{135}X_{1} + \frac{1493}{199}Y_{1}\right)\right) - \frac{10}{69}\cos^{12}\left(\frac{4101}{86}X_{1} + \frac{293}{15}Y_{1} - \frac{13}{79}\cos\left(\frac{122}{17}X_{1} + \frac{769}{196}Y_{1}\right)\right) \\[3pt]
  &\quad \quad - \frac{7}{121}\cos^{16}\left(\frac{779}{39}X_{1} + \frac{3518}{63}Y_{1} + \frac{57}{194}\cos\left(\frac{519}{124}X_{1} + \frac{1269}{158}Y_{1}\right)\right) \\[3pt]
  C_{1}(X,Y,v) &= \frac{39+v-2v^2}{40} \cdot \left(1 + \frac{1}{42}Y_{1}\right) + \tau_{1}(X,Y) \\[3pt]
  \Phi_{1}(X,Y) &= 1 - \left(\frac{X_{1}}{\frac{147}{187}}\right)^{2} - \left(\frac{Y_{1}}{\frac{9}{25}}\right)^{2} \\[3pt]
  &\quad \quad - \frac{29}{111}\cos\left(1\theta + \frac{63}{190}\right) + \frac{23}{117}\cos\left(2\theta + \frac{37}{146}\right) - \frac{80}{179}\cos\left(3\theta + \frac{98}{99}\right) + \frac{11}{113}\cos\left(4\theta + \frac{184}{131}\right) \\[3pt]
  &\quad \quad - \frac{73}{197}\cos\left(5\theta + \frac{223}{107}\right) + \frac{25}{199}\cos\left(6\theta + \frac{53}{24}\right) + \frac{9}{166}\cos\left(7\theta + \frac{501}{145}\right) - \frac{13}{192}\cos\left(8\theta + \frac{412}{129}\right) \\[3pt]
  W_{1}(X,Y) &= \exp\left(-\exp\left(-\frac{2810}{63}\cdot\Phi_{1}(X,Y)\right)\right)
\end{aligned}
$$

### testImage1043
![testImage1043](testImage1043_comparison.png)

$$
\begin{aligned}
  \tilde{X} &= 2\left(\frac{x}{W}\right) - 1, \quad \tilde{Y} = 1 - 2\left(\frac{y}{H}\right) \\[3pt]
  X &= \frac{\tilde{X} + \frac{26}{73}}{\frac{137}{182}}, \quad Y = \frac{\tilde{Y} + \frac{3}{65}}{\frac{69}{106}}, \quad \theta = \arctan\left(\frac{Y}{X}\right) \\[3pt]
  \tau_{1}(X,Y) &= \frac{8}{171}\cos^{2}\left(\frac{2461}{192}(X - \frac{67}{94}) + \frac{185}{173}(Y - \frac{163}{79}) + \frac{63}{148}\cos\left(\frac{498}{71}(X - \frac{67}{94}) + \frac{58}{79}(Y - \frac{163}{79})\right)\right) \\[3pt]
  &\quad + \frac{1}{177}\cos^{4}\left(\frac{104}{41}(X - \frac{67}{94}) + \frac{1192}{73}(Y - \frac{163}{79}) + \frac{250}{81}\cos\left(\frac{45}{37}(X - \frac{67}{94}) + \frac{995}{188}(Y - \frac{163}{79})\right)\right) \\[3pt]
  &\quad + \frac{1}{120}\cos^{6}\left(\frac{4372}{185}(X - \frac{67}{94}) + \frac{1942}{197}(Y - \frac{163}{79}) + \frac{89}{33}\cos\left(\frac{814}{119}(X - \frac{67}{94}) + \frac{334}{81}(Y - \frac{163}{79})\right)\right) \\[3pt]
  C_{1}(X,Y,v) &= \frac{29}{40} \cdot \left(1 - \frac{5}{183}(X - \frac{67}{94}) - \frac{5}{134}(Y - \frac{163}{79})\right) \\[3pt]
  &\quad + \tau_{1}(X,Y) \\[3pt]
  \Phi_{1}(X,Y) &= 1 - \left(\frac{(X - \frac{67}{94})}{\frac{200}{147}}\right)^{2} - \left(\frac{(Y - \frac{163}{79})}{\frac{247}{167}}\right)^{2} \\[3pt]
  &\quad +\frac{214}{145}\cos(1\theta + \frac{49}{25}) \\[3pt]
  &\quad +\frac{36}{191}\cos(2\theta - \frac{119}{198}) \\[3pt]
  &\quad -\frac{47}{194}\cos(3\theta + \frac{697}{182}) \\[3pt]
  &\quad -\frac{29}{184}\cos(4\theta - \frac{16}{75}) \\[3pt]
  &\quad -\frac{3}{140}\cos(5\theta + \frac{293}{177}) \\[3pt]
  &\quad +\frac{17}{132}\cos(6\theta + \frac{221}{59}) \\[3pt]
  &\quad +\frac{57}{194}\cos(7\theta + \frac{204}{55}) \\[3pt]
  &\quad +\frac{1}{117}\cos(8\theta + \frac{97}{76}) \\[3pt]
  W_{1}(X,Y) &= \exp\left(-\exp\left(-\frac{5028}{181}\cdot\Phi_{1}(X,Y)\right)\right)
\end{aligned}
$$
