# AI Sourcing Strategies in Firms

Code for my Master's dissertation at Gisma University of Applied Sciences (MSc Data Science, AI and Digital Business).

The thesis looks at how firms get their AI, not just whether they use it. Some build it themselves (Make), some pay a provider for something tailored (Buy), and some use standard ready-made tools (Off-the-shelf, OTS). I wanted to know whether these choices depend on how AI-intensive a firm's sector is, and whether firms in different countries choose differently once sector and firm characteristics are taken into account. The countries are India, Kenya, Nigeria and the United States.

## Research questions

1. How is sectoral AI intensity associated with firms' AI sourcing modes?
2. To what extent do sourcing patterns differ across the four countries after accounting for sectoral AI intensity and firm characteristics?
3. How well can machine-learning models predict sourcing mode compared with a standard logistic regression?

## Data

The analysis uses the World Bank Enterprise Surveys. The outcome (`ai_b6c`) comes from the 2026 AI follow-up survey, and firm characteristics come from each country's baseline survey. The two are linked by the establishment ID `idstd`.

Sector AI intensity comes from the OECD sectoral taxonomy of AI intensity (Calvino, Dernis, Samek and Ughi, 2024). Each firm's ISIC Rev.4 sector is mapped to Low, Medium or High.

The WBES data is not included here, because it has to be downloaded from https://www.enterprisesurveys.org after a free registration. You need eight files:

| Country | Baseline | AI follow-up |
|---|---|---|
| India | 2025 | 2026 |
| Kenya | 2025 | 2026 |
| Nigeria | 2025 | 2026 |
| United States | 2024 | 2026 |

The final sample has 1,171 firms: India 517, Nigeria 315, United States 208 and Kenya 131.

## What the notebook does

The notebook starts by cleaning and merging the data. It checks that IDs are unique, merges each country's baseline and follow-up files, keeps only valid sourcing answers, and maps sectors to the OECD categories. Coverage is 1,186 of 1,193 firms. After that, every model uses the same complete-case sample, so differences between models are never caused by different firms dropping in or out.

For RQ1 and RQ2, I fit a survey-weighted multinomial logit with OTS as the reference outcome, Low intensity as the reference sector group, and India as the reference country. The controls are firm size, firm age, exporting, training, quality certification and having a website. Standard errors account for the survey strata. The main tests are joint Wald tests, and results are also shown as predicted probabilities. I checked robustness using equal-country weights, no weights, trimmed weights, sector-clustered standard errors and HC1 standard errors.

For RQ3, I compare a majority-class baseline, logistic regression, random forest and gradient boosting. I use repeated stratified cross-validation (5 folds, 3 repeats, 3 seeds), and the hyperparameters are tuned only on the training part of each split. The models are scored on balanced accuracy, log loss and calibration error. Differences from logistic regression are tested with the corrected resampled t-test (Nadeau and Bengio, 2003).

## Main findings

**Sector intensity matters.** AI intensity is jointly significant (Wald = 18.81, p < 0.001). Firms in high-intensity sectors are more likely to Make or Buy rather than use off-the-shelf tools. The predicted share of OTS drops from about 67% in low-intensity sectors to 37% in high-intensity ones.

**Country differences don't disappear.** They shrink once sector intensity is added, but they stay significant (Wald falls from 17.82 to 14.85, p = 0.021).

**Machine learning doesn't clearly beat logistic regression.** Gradient boosting has the highest balanced accuracy (0.423), but the gain is not significant, and its probabilities are overconfident. Random forest gives the most reliable probabilities.

One limitation should be kept in mind. The survey weights are quite uneven, so the effective sample size for the main model is around 137, even though there are 1,171 firms. All results are associations, not causal effects.

## Running it

The notebook runs in Google Colab. Open `WBES_AI_Sourcing_Pipeline.ipynb`, click Runtime → Run all, and upload the eight WBES files when prompted. Either the zip downloads or `.dta` files will work, and exact file names don't matter. The machine-learning part takes a few minutes.

Tables and figures are saved in `/content/wbes_ai_results/`. If everything ran correctly, you should see N = 1,171, an AI-intensity Wald statistic of 18.812 and a gradient boosting balanced accuracy of 0.4233.

The main libraries are pandas, NumPy, SciPy, patsy, scikit-learn and matplotlib.

## Author

Aditya Shinde, Gisma University of Applied Sciences
