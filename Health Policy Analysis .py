#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:


## Health Policy Analysis: Leveraging Statistical Models and Programming to Support Healthcare Policy Implementation ##


# In[2]:


get_ipython().system('pip install linearmodels')
# Install the missing package 'linearmodels' using pip.


# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from linearmodels import PanelOLS
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# Correlation heatmap
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[6]:


# Load data
df = pd.read_csv('C:/Users/dgous/Downloads/HealthCare_Data_Final.csv')

# Convert 'Country' and 'Year' to categorical variables
df['Country'] = df['Country'].astype('category')
df['Year'] = df['Year'].astype('category')

# Print the columns of the dataframe
print(df.columns)


# In[9]:


# Drop unwanted variables
columns_to_drop = ['key', 'NationalIncome']
df_cleaned = df.drop(columns=columns_to_drop)

# Display the remaining columns to confirm the changes
print(df_cleaned.columns)


# In[10]:


# Filter data for the year 2015
df_2015 = df[df['Year'] == 2015]


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Assuming you have already loaded the data into a DataFrame 'df'
# and filtered it for the year 2015
df_2015 = df[df['Year'] == 2015]

# Create a scatter plot using seaborn
plt.figure(figsize=(10, 6))

# Scatter plot for Life Expectancy vs. Expenditure Per Capita
sns.scatterplot(
    x='ExpenditurePerCapita', 
    y='Life_Expectancy_at_Birth', 
    hue='Country',  # Use 'Country' as hue to color points by country
    size='ExpenditurePerCapita',  # Use 'ExpenditurePerCapita' to scale point sizes
    sizes=(50, 500),  # Define the minimum and maximum sizes for points
    data=df_2015,
    legend=False  # Remove the legend for better visual clarity
)

# Add title and labels
plt.title('Life Expectancy and Health Expenditure by Country (2015)', fontsize=16)
plt.xlabel('ExpenditurePerCapita (in USD)', fontsize=12)
plt.ylabel('Life Expectancy (in Years)', fontsize=12)

# Add text labels for each country
for line in range(0, df_2015.shape[0]):
    plt.text(
        df_2015.ExpenditurePerCapita.iloc[line], 
        df_2015.Life_Expectancy_at_Birth.iloc[line], 
        df_2015.Country.iloc[line], 
        horizontalalignment='left', 
        size='small', 
        color='black'
    )

# Display the plot
plt.tight_layout()
plt.show()


# In[11]:


# Plotting histograms
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df_2015['Life_Expectancy_at_Birth'], bins=30)
plt.title('Life Expectancy 2015')

plt.subplot(1, 2, 2)
plt.hist(np.log(df_2015['Life_Expectancy_at_Birth']), bins=30)
plt.title('Log of Life Expectancy 2015')
plt.show()

# Expenditure per capita
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.hist(df_2015['ExpenditurePerCapita'], bins=30)
plt.title('Expenditure Per Capita 2015')

plt.subplot(1, 2, 2)
plt.hist(np.log(df_2015['ExpenditurePerCapita']), bins=30)
plt.title('Log of Expenditure Per Capita 2015')
plt.show()


# In[12]:


# Subset data for correlation plot
correlation_cols_expense = ['ExpenditurePerCapita', 'Diagnostic_Exams', 'Hospitals', 'PercPopulationabove65', 'Private_Insurance', 'Public_Insurance', 'hospital_employment', 'tot_equipment']
cor_df_expense = df[correlation_cols_expense]

# Correlation heatmap for healthcare expenditure
plt.figure(figsize=(12, 8))
sns.heatmap(cor_df_expense.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation for Healthcare Expenditure')
plt.show()

# Subset data for life expectancy
correlation_cols_life = ['Life_Expectancy_at_Birth', 'Hospitals', 'ExpenditurePerCapita', 'Mean_Schooling_Years', 'hospital_employment', 'tot_equipment', 'medical_grads', 'nurse_grads', 'death_by_cancer', 'death_by_circular', 'death_by_accident', 'death_by_respirat']
cor_df_life = df[correlation_cols_life]

# Correlation heatmap for life expectancy
plt.figure(figsize=(12, 8))
sns.heatmap(cor_df_life.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation for Life Expectancy')
plt.show()


# In[14]:


import numpy as np
from linearmodels import PanelOLS

# Convert 'Year' to numeric (integer type) to avoid time index issues
df['Year'] = pd.to_numeric(df['Year'])

# Prepare panel data for healthcare expense model
df_expense = df[['Country', 'Year', 'Diagnostic_Exams', 'ExpenditurePerCapita', 'Hospitals', 'PercPopulationabove65', 'Private_Insurance', 'Public_Insurance', 'hospital_employment', 'tot_equipment']]

# Set the index for panel data
df_expense = df_expense.set_index(['Country', 'Year'])

# Define the panel OLS model (Pooling)
mod_pool_HE = PanelOLS.from_formula('np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment', data=df_expense)
pool_HE_result = mod_pool_HE.fit()
print(pool_HE_result)

# Fixed effects model
mod_fixed_HE = PanelOLS.from_formula('np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment + EntityEffects', data=df_expense)
fixed_HE_result = mod_fixed_HE.fit()
print(fixed_HE_result)

# Random effects model
mod_random_HE = PanelOLS.from_formula('np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment + RandomEffects', data=df_expense)
random_HE_result = mod_random_HE.fit()
print(random_HE_result)


# In[15]:


import numpy as np
from linearmodels import PanelOLS, RandomEffects

# Convert 'Year' to numeric (integer type) to avoid time index issues
df['Year'] = pd.to_numeric(df['Year'])

# Prepare panel data for healthcare expense model
df_expense = df[['Country', 'Year', 'Diagnostic_Exams', 'ExpenditurePerCapita', 'Hospitals', 'PercPopulationabove65', 'Private_Insurance', 'Public_Insurance', 'hospital_employment', 'tot_equipment']]

# Set the index for panel data
df_expense = df_expense.set_index(['Country', 'Year'])

# Define the panel OLS model (Pooling)
mod_pool_HE = PanelOLS.from_formula('np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment', data=df_expense)
pool_HE_result = mod_pool_HE.fit()
print(pool_HE_result)

# Fixed effects model
mod_fixed_HE = PanelOLS.from_formula('np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment + EntityEffects', data=df_expense)
fixed_HE_result = mod_fixed_HE.fit()
print(fixed_HE_result)

# Random effects model - use RandomEffects class directly
mod_random_HE = RandomEffects.from_formula('np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment', data=df_expense)
random_HE_result = mod_random_HE.fit()
print(random_HE_result)


# In[16]:


import pandas as pd
import numpy as np
from linearmodels import PanelOLS, RandomEffects
import statsmodels.api as sm

# Assuming df is already loaded and cleaned with all relevant variables.
# Ensure 'Year' is numeric
df['Year'] = pd.to_numeric(df['Year'])

# Prepare panel data for life expectancy model
df_life_expectancy = df[['Country', 'Year', 'Life_Expectancy_at_Birth', 'Hospitals', 
                         'ExpenditurePerCapita', 'Mean_Schooling_Years', 'hospital_employment', 
                         'tot_equipment', 'medical_grads', 'nurse_grads', 'death_by_cancer', 
                         'death_by_circular', 'death_by_accident', 'death_by_respirat']]

# Set the index for panel data
df_life_expectancy = df_life_expectancy.set_index(['Country', 'Year'])

# Pooling Model (Model 1)
mod_pool_LE = PanelOLS.from_formula(
    'Life_Expectancy_at_Birth ~ 1 + Hospitals + ExpenditurePerCapita + Mean_Schooling_Years + hospital_employment + tot_equipment + medical_grads + nurse_grads + death_by_cancer + death_by_circular + death_by_accident + death_by_respirat',
    data=df_life_expectancy
)
pool_LE_result = mod_pool_LE.fit()
print(pool_LE_result)

# Fixed Effects Model (Model 2)
mod_fixed_LE = PanelOLS.from_formula(
    'Life_Expectancy_at_Birth ~ 1 + Hospitals + ExpenditurePerCapita + Mean_Schooling_Years + hospital_employment + tot_equipment + medical_grads + nurse_grads + death_by_cancer + death_by_circular + death_by_accident + death_by_respirat + EntityEffects',
    data=df_life_expectancy
)
fixed_LE_result = mod_fixed_LE.fit()
print(fixed_LE_result)

# Random Effects Model (Model 3)
mod_random_LE = RandomEffects.from_formula(
    'Life_Expectancy_at_Birth ~ 1 + Hospitals + ExpenditurePerCapita + Mean_Schooling_Years + hospital_employment + tot_equipment + medical_grads + nurse_grads + death_by_cancer + death_by_circular + death_by_accident + death_by_respirat',
    data=df_life_expectancy
)
random_LE_result = mod_random_LE.fit()
print(random_LE_result)


# In[17]:


import pandas as pd
import numpy as np
from linearmodels import PanelOLS, RandomEffects

# Convert 'Year' to numeric (integer type) to avoid time index issues
df['Year'] = pd.to_numeric(df['Year'])

# Prepare panel data for healthcare expense model
df_expense = df[['Country', 'Year', 'ExpenditurePerCapita', 'Diagnostic_Exams', 'Hospitals', 
                 'PercPopulationabove65', 'Private_Insurance', 'Public_Insurance', 
                 'hospital_employment', 'tot_equipment']]

# Set the index for panel data
df_expense = df_expense.set_index(['Country', 'Year'])

# Pooling Model (Model 1)
mod_pool_HE = PanelOLS.from_formula(
    'np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment',
    data=df_expense
)
pool_HE_result = mod_pool_HE.fit()
print(pool_HE_result)

# Fixed Effects Model (Model 2)
mod_fixed_HE = PanelOLS.from_formula(
    'np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment + EntityEffects',
    data=df_expense
)
fixed_HE_result = mod_fixed_HE.fit()
print(fixed_HE_result)

# Random Effects Model (Model 3)
mod_random_HE = RandomEffects.from_formula(
    'np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment',
    data=df_expense
)
random_HE_result = mod_random_HE.fit()
print(random_HE_result)


# In[18]:


import pandas as pd
import numpy as np
from linearmodels.panel import PanelOLS, RandomEffects
from linearmodels.panel import compare
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Lagrange Multiplier Test for Panel Data (similar to LMTEST in R)
def breusch_pagan_test(residuals, exog):
    n = exog.shape[0]  # number of observations
    k = exog.shape[1]  # number of independent variables
    lm = n * (residuals.var() / residuals.mean() ** 2)
    p_value = 1 - stats.chi2.cdf(lm, k)
    return lm, p_value

# Convert 'Year' to numeric
df['Year'] = pd.to_numeric(df['Year'])

# Prepare panel data for healthcare expense model
df_expense = df[['Country', 'Year', 'ExpenditurePerCapita', 'Diagnostic_Exams', 'Hospitals',
                 'PercPopulationabove65', 'Private_Insurance', 'Public_Insurance',
                 'hospital_employment', 'tot_equipment']]

# Set the index for panel data
df_expense = df_expense.set_index(['Country', 'Year'])

# Model 1: Pooling Model
pool_HE = PanelOLS.from_formula('np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment', data=df_expense)
pool_HE_result = pool_HE.fit()
print(pool_HE_result)

# Model 2: Fixed Effects Model
fixed_HE = PanelOLS.from_formula('np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment + EntityEffects', data=df_expense)
fixed_HE_result = fixed_HE.fit()
print(fixed_HE_result)

# Model 3: Random Effects Model
random_HE = RandomEffects.from_formula('np.log(ExpenditurePerCapita) ~ 1 + Diagnostic_Exams + Hospitals + PercPopulationabove65 + Private_Insurance + Public_Insurance + hospital_employment + tot_equipment', data=df_expense)
random_HE_result = random_HE.fit()
print(random_HE_result)

# Breusch-Pagan Test (Lagrange Multiplier Test)
lm, p_value = breusch_pagan_test(pool_HE_result.resids, df_expense[['Diagnostic_Exams', 'Hospitals', 'PercPopulationabove65', 'Private_Insurance', 'Public_Insurance', 'hospital_employment', 'tot_equipment']])
print(f"Breusch-Pagan Test: LM = {lm}, p-value = {p_value}")

# Hausman Test
# The compare method in linearmodels can give us some insight, though the exact Hausman test isn't directly implemented in the library.
comparison = compare({'Fixed Effects': fixed_HE_result, 'Random Effects': random_HE_result})
print(comparison)

# Residual Plots
# Create residual plots for the Pooling, Fixed Effects, and Random Effects models

# Pooling Model Residuals
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df_expense.index.get_level_values(1), y=pool_HE_result.resids)
plt.title('Pooling Model Residuals by Year')
plt.subplot(1, 2, 2)
sns.boxplot(x=df_expense.index.get_level_values(0), y=pool_HE_result.resids)
plt.xticks(rotation=90)
plt.title('Pooling Model Residuals by Country')
plt.tight_layout()
plt.show()

# Fixed Effects Model Residuals
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df_expense.index.get_level_values(1), y=fixed_HE_result.resids)
plt.title('Fixed Effects Model Residuals by Year')
plt.subplot(1, 2, 2)
sns.boxplot(x=df_expense.index.get_level_values(0), y=fixed_HE_result.resids)
plt.xticks(rotation=90)
plt.title('Fixed Effects Model Residuals by Country')
plt.tight_layout()
plt.show()

# Random Effects Model Residuals
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.boxplot(x=df_expense.index.get_level_values(1), y=random_HE_result.resids)
plt.title('Random Effects Model Residuals by Year')
plt.subplot(1, 2, 2)
sns.boxplot(x=df_expense.index.get_level_values(0), y=random_HE_result.resids)
plt.xticks(rotation=90)
plt.title('Random Effects Model Residuals by Country')
plt.tight_layout()
plt.show()


# In[ ]:




