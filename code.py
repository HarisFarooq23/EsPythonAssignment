import pandas as pd
import matplotlib.pyplot as plt
import numpy as np 

# Load the dataset (update the path if needed)
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\EsProject\cricket_bowlers_1000.csv")

column_name = "Wickets"

# Calculate mean and variance
mean_value = df[column_name].mean()
variance_value = df[column_name].var()

# Print results
print(f"Mean of '{column_name}': {mean_value:.2f}")
print(f"Variance of '{column_name}': {variance_value:.2f}")


# --- Histogram: Wickets Taken ---
plt.figure(figsize=(10, 6))
plt.hist(df["Wickets"], bins=20, color='skyblue', edgecolor='black')
plt.title("Distribution of Wickets Taken")
plt.xlabel("Wickets")
plt.ylabel("Number of Bowlers")
plt.grid(True)
plt.tight_layout()
plt.show()

# --- Pie Chart: Bowling Type Distribution ---
bowling_type_counts = df["Bowling Type"].value_counts()

plt.figure(figsize=(8, 8))
plt.pie(bowling_type_counts, labels=bowling_type_counts.index, autopct='%1.1f%%', startangle=140,    colors=["#1f77b4", "#ff7f0e", "#2ca02c"])
plt.title("Bowling Type Distribution")
plt.axis('equal')
plt.tight_layout()
plt.show()


num_bins = 10

# Create bins and get frequency distribution
counts, bin_edges = np.histogram(df[column_name], bins=num_bins)

# Display frequency distribution
print("Frequency Distribution ( wickets taken ):")
for i in range(len(counts)):
    print(f"{bin_edges[i]:.0f} to {bin_edges[i+1]:.0f} : {counts[i]}")




# Calculate bin midpoints
midpoints = (bin_edges[:-1] + bin_edges[1:]) / 2

# Calculate mean using frequency distribution
total_freq = np.sum(counts)
mean_fd = np.sum(midpoints * counts) / total_freq

# Calculate variance using frequency distribution
variance_fd = np.sum(((midpoints - mean_fd) ** 2) * counts) / total_freq



# Display results
print(f"Mean using frequency distribution: {mean_fd:.2f}")
print(f"Variance using frequency distribution: {variance_fd:.2f}")
print()



from scipy import stats


column = "Wickets"

# Split into 80% training and 20% testing
train = df.sample(frac=0.8, random_state=1)
test = df.drop(train.index)

# Sample statistics
x̄ = train[column].mean()
s = train[column].std()
n = len(train)

# 95% Confidence Interval for the Mean
z = 1.96
ci_lower = x̄ - z * s / np.sqrt(n)
ci_upper = x̄ + z * s / np.sqrt(n)

# 95% Confidence Interval for the Variance (Chi-Square method)
alpha = 0.05
dfree = n - 1
chi2_lower = stats.chi2.ppf(alpha / 2, dfree)
chi2_upper = stats.chi2.ppf(1 - alpha / 2, dfree)
var = train[column].var()
ci_var_lower = (dfree * var) / chi2_upper
ci_var_upper = (dfree * var) / chi2_lower

# Approximate 95% Tolerance Interval (for normal distribution)
# Using k ≈ 1.96 for simplicity, a more exact k can be calculated
k = 1.96
ti_lower = x̄ - k * s
ti_upper = x̄ + k * s

# Validation using 20% test set
within_ti = test[(test[column] >= ti_lower) & (test[column] <= ti_upper)]
tolerance_coverage = len(within_ti) / len(test) * 100

# Results
print(f"95% Confidence Interval for Mean: ({ci_lower:.2f}, {ci_upper:.2f})")
print(f"95% Confidence Interval for Variance: ({ci_var_lower:.2f}, {ci_var_upper:.2f})")
print(f"95% Tolerance Interval: ({ti_lower:.2f}, {ti_upper:.2f})")
print(f"Percentage of validation data within Tolerance Interval: {tolerance_coverage:.2f}%")



sample = df["Wickets"]
sample_mean = sample.mean()
hypothesized_mean = 100
sample_std = sample.std(ddof=1)
n = len(sample)

# t-statistic calculation
t_stat = (sample_mean - hypothesized_mean) / (sample_std / (n ** 0.5))

# One-tailed p-value (right tail)
p_value = 1 - stats.t.cdf(t_stat, df=n-1)

# Result
print(f"Sample Mean: {sample_mean:.2f}")
print(f"T-statistic: {t_stat:.4f}")
print(f"P-value: {p_value:.4f}")

# Decision
alpha = 0.05
if p_value < alpha:
    print("Reject the null hypothesis: The average number of wickets greater than 100")
else:
    print("Fail to reject the null hypothesis: Not enough evidence to say the average is greater than 100.")
