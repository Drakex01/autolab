# Comparative Analysis of Multi-Class Classifiers on Botanical Morphological Data

**Authors:** Dr. E. R. Anderson, Dr. K. M. Vance  
**Journal:** Journal of Computational Botany & Applied Machine Learning, 2024  
**Subject:** Empirical evaluation of linear vs non-linear discriminative models on the Fisher Iris taxonomy.

## Abstract
The Fisher Iris dataset represents a foundational benchmark in multivariate pattern recognition. In this study, we rigorously investigate the relative classification efficacy of linear models (multinomial Logistic Regression with L2 regularization) versus non-linear ensemble methods (Random Forest with varying tree estimators) across four canonical morphological dimensions: sepal length, sepal width, petal length, and petal width.

## 1. Introduction and Morphological Features
Botanical species discrimination within the genus *Iris* (specifically *Iris setosa*, *Iris versicolor*, and *Iris virginica*) relies on four continuous measurements:
1. Sepal Length (cm)
2. Sepal Width (cm)
3. Petal Length (cm)
4. Petal Width (cm)

Prior literature indicates that while *Iris setosa* is linearly separable from the other two species due to significantly smaller petal dimensions, the decision boundary separating *Iris versicolor* and *Iris virginica* exhibits non-linear overlap in feature space.

## 2. Experimental Methodology
We establish a standardized evaluation protocol:
- **Feature Preprocessing:** Features are mean-centered and unit-variance scaled using z-score normalization. Median imputation is utilized to ensure robustness against rare sensor dropouts.
- **Data Partitioning:** An 80/20 train-test partition is enforced with stratified sampling and a fixed seed (42) to guarantee reproducible test folds.
- **Baseline Model:** Multinomial Logistic Regression served as the primary linear baseline ($C=1.0$, L-BFGS solver, maximum iterations 1000).
- **Proposed Model:** Random Forest Classifier with bootstrap aggregation, Gini impurity splitting criterion, and $n=100$ estimators.

## 3. Findings & Evidence
Our experimental results demonstrate:
1. **Separability:** Petal length and petal width contribute over 85% of total feature importance in tree-based splits.
2. **Performance Comparison:** On standardized test partitions, Random Forest achieved an overall accuracy of 96.7% and a weighted F1-score of 0.967, correctly categorizing borderline *versicolor* and *virginica* specimens where linear hyperplanes incurred margin errors.
3. **Hyperparameter Sensitivity:** Parameter sweeps on ensemble size ($n \in [10, 50, 100, 200]$) revealed diminishing returns beyond 50 trees, with asymptotic performance plateauing around 100 estimators.
4. **Threats to Validity:** Due to the compact sample size of 150 specimens, minor sampling variance across different test folds can shift accuracy by 3–6 percentage points. Cross-validation and sample expansion are advised before general field deployment.

## 4. Conclusion
Non-linear tree ensembles provide robust margin separation for overlapping botanical morphology, demonstrating superior generalization over linear classifiers when distinguishing closely related floral taxa.
