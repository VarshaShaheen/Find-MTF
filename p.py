import pandas as pd
from scipy.stats import pearsonr

csv_file_path = 'data.csv'


def find_p_value(csv_file):
    data = pd.read_csv(csv_file)
    mtf = data['MTF']
    perception = data['Perception']
    correlation, p_value = pearsonr(mtf, perception)

    print(f"Pearson's correlation coefficient: {correlation}, p-value: {p_value}")
    if p_value < 0.05:
        print("The relationship between MTF and perception is statistically significant.")
    else:
        print("There is no statistically significant relationship between MTF and perception.")


find_p_value(csv_file_path)
