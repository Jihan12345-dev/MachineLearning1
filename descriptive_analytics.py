import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


file_path = r'D:\2. PROJECT\LPS\CML\Analytics Example\loan_data.csv'
loan_data = pd.read_csv(file_path)
loan_data.head()

numerical_summary = loan_data.describe().transpose()

palette = sns.color_palette("viridis", as_cmap=True)

numerical_summary.style.background_gradient(cmap=palette)

# Check for missing values in each column
missing_data = loan_data.isnull().sum()

# Display columns with missing values (if any)
missing_data = missing_data[missing_data > 0]
missing_data

loan_data.info()


non_boolean_numerical_features = ['int.rate', 'installment', 'log.annual.inc', 'dti', 'fico', 
                                  'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths', 
                                  'delinq.2yrs', 'pub.rec']
boolean_numeric_features = ['credit.policy', 'not.fully.paid']

# Visualize the distributions and box plots for numerical features, including log-transformed versions for skewed data
for column in non_boolean_numerical_features:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

    # Histogram for the distribution
    sns.histplot(loan_data[column], kde=False, color='skyblue', ax=ax1)
    ax1.set_title(f'Distribution of {column}')
    ax1.set_ylabel('Frequency')

    # Boxplot for the variable
    sns.boxplot(x=loan_data[column], color='lightgreen', ax=ax2)
    ax2.set_title(f'Boxplot of {column}')

    # Log transformation and plot if the data is skewed
    if loan_data[column].skew() > 1:
        loan_data[column+'_log'] = np.log1p(loan_data[column])
        sns.histplot(loan_data[column+'_log'], kde=False, color='orange', ax=ax3)
        ax3.set_title(f'Log-transformed Distribution of {column}')
    else:
        ax3.set_title(f'Log-transformed plot not necessary for {column}')
        ax3.axis('off')

    plt.tight_layout()
    plt.show()


    # Calculate the correlation matrix for non-boolean numerical features
corr_matrix = loan_data[non_boolean_numerical_features].corr()

# Set up the matplotlib figure
plt.figure(figsize=(12, 10))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap=cmap, center=0,
            square=True, linewidths=.5, cbar_kws={"shrink": .5})

plt.title('Correlation Matrix for Non-Boolean Numerical Features', fontsize=16)
plt.show()

# Implementing the suggested outlier handling strategy:

# Apply log transformation to features with heavy right-skewness
# Define a threshold for skewness
skewness_threshold = 1
for feature in non_boolean_numerical_features:
    if loan_data[feature].skew() > skewness_threshold:
        loan_data[f'{feature}_log'] = np.log1p(loan_data[feature])

# Identify and cap/floor the extreme values for numerical features
# Capping/Flooring at the 1st and 99th percentiles
for feature in non_boolean_numerical_features:
    lower_bound = loan_data[feature].quantile(0.01)
    upper_bound = loan_data[feature].quantile(0.99)
    loan_data[f'{feature}_capped'] = np.clip(loan_data[feature], lower_bound, upper_bound)

# Displaying the transformed data frame with additional columns for log-transformed and capped features
loan_data.head()


from sklearn.preprocessing import PolynomialFeatures

# 1. Interaction Terms - Example: Interaction between 'fico' and 'int.rate'
loan_data['fico_int_rate_interaction'] = loan_data['fico_capped'] * loan_data['int.rate_capped']

# 2. Polynomial Features - Example: Creating a squared term for 'dti'
pf = PolynomialFeatures(degree=2, include_bias=False)
dti_poly = pf.fit_transform(loan_data[['dti_capped']])
loan_data['dti_squared'] = dti_poly[:, 1]

# 3. Grouping and Aggregation - Example: Mean 'fico' score by 'purpose'
fico_mean_by_purpose = loan_data.groupby('purpose')['fico_capped'].mean().rename('mean_fico_by_purpose')
loan_data = loan_data.join(fico_mean_by_purpose, on='purpose')

# 4. Binning - Example: Binning 'fico' scores into categories
loan_data['fico_category'] = pd.cut(loan_data['fico_capped'], bins=[300, 630, 689, 719, 850], labels=['Bad', 'Fair', 'Good', 'Excellent'])