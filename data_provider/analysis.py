import os.path
import pandas as pd
import pywt

class Analysis:
    def __init__(self, root_path, data_path):
        self.root_path = root_path
        self.data_path = data_path
        self.df = pd.read_csv(data_path)
        self.time_series = self.df['OT'].values
        self.level = 3

    def perform_WD(self):
        coeffs = pywt.wavedec(self.time_series, 'db4', level=self.level)
        num_coeffs = len(coeffs)
        WD_results = pd.DataFrame()

        for i in range(1, num_coeffs):
            WD_results[f'Detail_d{i}']  = pywt.upcoef('d', coeffs[num_coeffs - i], 'db4', level=i, take=len(self.time_series))
        WD_results[f'Approximation_a{self.level}'] = pywt.upcoef('a', coeffs[0], 'db4', level=self.level,
                                                                 take=len(self.time_series))

        WD_results['OT'] = self.df['OT']
        results = pd.concat([self.df['date'], WD_results], axis=1)
        return results

    def save_results_to_csv(self, results, output_path):
        self.data_path = os.path.basename(self.data_path)
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        results.to_csv(output_path + self.data_path, index=False)

