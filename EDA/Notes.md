### Here are simpler versions of the 20 questions, plus what most experts usually recommend.

1. Outliers: detect vs remove
Simple question: If we find outliers, why don’t we remove or fix them?
Expert approach:

First, check if outliers are real (domain check, data quality).

For ML: try both with and without outliers, compare validation scores (e.g., tree models often robust, linear models more sensitive).

Use methods: winsorizing, robust scalers, log/Box‑Cox transforms, or modeling with robust loss (Huber, quantile, tree‑based).

2. Skewness not fixed
Simple question: If variables are skewed, why don’t we transform them?
Expert approach:

Transform when model assumes approximate normality or linearity (e.g., linear/logistic regression).

Use log, sqrt, Box‑Cox, Yeo‑Johnson.

For trees/boosting, skewness is often less critical; test with/without transformation.

3. Why detect outliers at all?
Simple question: What’s the use of finding outliers if we don’t do anything with them?
Expert approach:

EDA use: detect data quality issues, rare but important cases, and model risk.

Keep them for now but: mark them with flags, run sensitivity analyses, and document their effect on metrics.

4. Using df instead of dftemp in plots
Simple question: Why is the full data (df) used instead of the filtered data (dftemp) in plots that look the same?
Expert approach:

Always plot from the same filtered dataframe that analysis is based on, to avoid misleading graphs.

Standard: define a clear analysis dataset (df_eda / df_model) and use it consistently.

5. Wrong dataframe in fire vs no‑fire plot
Simple question: If the fire vs no‑fire plot should use dftemp, what happens if we use df by mistake?
Expert approach:

Mis-specified data can mix in rows that should be excluded, biasing class balance or relationships.

Best practice: write small tests (assert len(dftemp) < len(df), check unique labels) and encapsulate filtering steps in functions/pipelines.

6. Monthly analysis using df instead of dftemp
Simple question: Why is using df instead of dftemp in monthly plots a problem?
Expert approach:

It can show patterns that no longer apply after cleaning/filtering.

Experts maintain separate “raw”, “cleaned”, and “analysis” datasets and label plots accordingly.

7. Regional analysis with wrong data
Simple question: Why do both regional plots look the same when df is used instead of dftemp?
Expert approach:

Because the filter is ignored, you get no real comparison.

Use filtered subsets explicitly (df[df["Classes"]=="fire"], etc.) and verify counts before plotting.

8. When to use original vs filtered data
Simple question: When should we use the original dataframe, and when should we use a filtered/temporary one for EDA?
Expert approach:

Original: early sanity checks, missingness, basic distribution and data quality.

Filtered/cleaned: for any analysis that supports modeling decisions and reporting.

Keep a clear pipeline: raw → cleaned → filtered/model and never mix levels in one plot.

9. Why create a region column?
Simple question: What’s the point of adding a region column?
Expert approach:

Add grouping features (region) to: capture spatial/cluster effects, reveal group-wise differences, and improve interpretability.

Test value: compare models with/without the feature, and check feature importance/SHAP by region.

10. How to define the region column
Simple question: How should we decide how to define regions, and can different choices change conclusions?
Expert approach:

Use domain knowledge (admin regions, climate zones, business segments) or unsupervised clustering (k‑means on lat‑long, environment features).

Re‑run EDA and models under different region definitions to see how stable patterns are.

11. Why drop features with correlation > 0.85
Simple question: Why do we remove features that are highly correlated (above 0.85)?
Expert approach:

For linear models, high correlation causes multicollinearity, unstable coefficients, and wider confidence intervals.

Typical approaches:

Correlation‑based pruning (pick one of each highly correlated pair).

Regularization (ridge/lasso/elastic net) instead of manual dropping.

Tree‑based models where correlation is less of a problem but still monitored.

12. How to choose a correlation threshold
Simple question: How do we choose a threshold like 0.85 to drop features?
Expert approach:

Start with 0.8–0.9 as a heuristic, then:

Check VIF (Variance Inflation Factor) for linear models,

Compare model performance and stability with different thresholds,

Prefer regularization/feature selection algorithms over hard thresholds when possible.

13. Effect of dropping highly correlated features
Simple question: How does removing a correlated feature affect how well the model explains the target?
Expert approach:

Evaluate with and without the removed feature: check metrics and explanation tools (coefficients, SHAP, partial dependence).

If explanation is important, keep one variable per “concept” and choose the one that is most interpretable and stable.

14. EDA on wide (many‑column) datasets
Simple question: Can we see EDA and feature engineering on a dataset with many columns, like in real projects?
Expert approach:

Use automatic tools:

correlation and mutual information matrices,

embedded methods (L1 regularization, tree‑based feature importance, Boruta, permutation importance).

Combine with domain-driven grouping and dimensionality reduction (PCA, UMAP) for visualization.

15. Where to get the EDA code
Simple question: Where can students download the code/notebook used in the EDA?
Expert approach:

Host on a public repo, tag releases by video, and add clear links and version information.

Include environment files (requirements.txt, env.yml) and a short “run this first” section.

16. How to read regional/monthly countplots
Simple question: When we do regional or monthly fire analysis, what story are we trying to tell with the plots?
Expert approach:

Clarify: “We are comparing fire vs no‑fire counts across region/month to see where risk is higher.”

Always annotate: axis labels, baseline, relative vs absolute counts, and uncertainty.

For modeling, turn insights into features (season, region, interaction terms).

17. From plots to features
Simple question: How do we move from basic plots to concrete feature engineering steps?
Expert approach:

For each pattern in EDA, ask “Can this be turned into a feature?”

Non‑linear trend → transformations or binning.

Different behavior by group → interactions or group‑level features.

Validate every new feature with cross‑validation and feature importance/SHAP.

18. What to do with outliers in practice
Simple question: When we see outliers, how do we decide whether to drop, transform, or keep them?
Expert approach:

Check cause: data error → fix/drop; rare but real events → often keep but model carefully.

Try scenarios:

Model A: raw data,

Model B: transformed (log, robust scaling),

Model C: trimmed/winsorized.

Choose based on validation metrics and business impact.

19. Making EDA less boring
Simple question: How can beginners stop finding EDA boring and still do it properly?
Expert approach:

Tie every EDA step to a question (“Which months have more fires?”, “Which region is riskiest?”).

Automate repetitive parts using notebooks, EDA libraries (pandas-profiling/ydata-profiling, sweetviz), and focus human effort on questions and decisions, not just plots.

20. Simple checklist for good EDA
Simple question: Can we get a simple checklist to avoid confusion about outliers, df vs dftemp, and correlation thresholds?
Expert approach (sample checklist):

Step 1: Understand data (shape, types, missingness).

Step 2: Define and freeze an “analysis dataframe” (after basic cleaning).

Step 3: Explore distributions, outliers, skewness; decide on transformation strategy and document it.

Step 4: Check correlations, use regularization/feature selection instead of arbitrary drops where possible.

Step 5: Ensure all plots and stats use the same analysis dataframe or clearly labeled subsets.

Step 6: Convert EDA findings into concrete features and modeling hypotheses; validate with proper cross‑validation.