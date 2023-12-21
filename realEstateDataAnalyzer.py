import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats


class RealEstateDataAnalyzer:
    def __init__(self, file_path, sheet_name="Недвижимость СПБ"):
        self.df = pd.read_excel(file_path, sheet_name=sheet_name, header=None)
        self.years = self.df.iloc[0, 1:].tolist()
        self.months = self.df.iloc[1:, 0].tolist()
        self.data_by_year = self.populate_data_by_year()
        self.prices = self.calculate_prices()
        self.seasons = self.create_seasons()
        self.average_price_per_sqm_by_year = self.calculate_average_price_per_sqm_by_year()

    def populate_data_by_year(self):
        data_by_year = {year: [] for year in self.years}
        for i in range(1, len(self.months) + 1):
            for j in range(1, len(self.years) + 1):
                year = self.years[j - 1]
                price = self.df.iloc[i, j]
                data_by_year[year].append(price)
        return data_by_year

    def calculate_prices(self):
        prices = []
        for i in range(1, len(self.years) + 1):
            for j in range(1, len(self.months) + 1):
                price = self.df.iloc[j, i]
                prices.append(price)
        return prices

    def create_seasons(self):
        seasons = [f"{int(self.years[i])} {str(self.months[j])}" for i in range(len(self.years)) for j in range(len(self.months))]
        return seasons

    def plot_prices_over_time(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.seasons, self.prices, marker='o')
        plt.title('Изменение цен за 1 кв. м с января 2015 по декабрь 2022')
        plt.xlabel('Месяц')
        plt.ylabel('Цена за 1 кв. м')
        plt.xticks(range(0, len(self.seasons), 3), self.seasons[::3], rotation=45)
        plt.grid(True)
        plt.show()

    def calculate_average_price_per_sqm_by_year(self):
        start_col_index = 0
        average_price_per_sqm_by_year = {}
        for year, prices in self.data_by_year.items():
            total_price = sum(prices[start_col_index:])
            num_months = len(prices[start_col_index:])
            average_price_per_sqm = total_price / num_months
            average_price_per_sqm_by_year[year] = round(average_price_per_sqm, 3)
        return average_price_per_sqm_by_year

    def display_average_price_per_sqm(self):
        for year, avg_price_per_sqm in self.average_price_per_sqm_by_year.items():
            print(f"Год {year}: Средняя стоимость за 1 кв. м - {avg_price_per_sqm}")

    def calculate_absolute_growth(self, values):
        base_value = values[0]
        return [round(value - base_value, 2) for value in values[1:]]

    def calculate_growth_rate(self, values):
        base_value = values[0]
        return [round((value / base_value) * 100, 2) for value in values[1:]]

    def calculate_growth_rate_difference(self, values):
        base_value = values[0]
        return [round((value / base_value) * 100 - 100, 2) for value in values[1:]]

    def calculate_chain_absolute_growth(self, values):
        return [round(values[i] - values[i - 1], 2) for i in range(1, len(values))]

    def calculate_chain_growth_rate(self, values):
        return [round(values[i] / values[i - 1] * 100, 2) for i in range(1, len(values))]

    def calculate_chain_growth_rate_difference(self, values):
        return [round((values[i] / values[i - 1] * 100) - 100, 2) for i in range(1, len(values))]

    def calculate_comparison_indicator(self, values):
        standard_value = max(values)
        return [round((value / standard_value) * 100, 2) for value in values]

    def perform_statistical_analysis(self):
        values_avg = list(self.average_price_per_sqm_by_year.values())

        print("Базисный расчет")
        print("Абсолютный прирост:")
        absolute_growth = self.calculate_absolute_growth(values_avg)
        for growth in absolute_growth:
            print(growth)

        print("Темп роста:")
        growth_rate = self.calculate_growth_rate(values_avg)
        for rate in growth_rate:
            print(rate)

        print("Темп прироста:")
        growth_rate_diff = self.calculate_growth_rate_difference(values_avg)
        for diff in growth_rate_diff:
            print(diff)

        print("Цепной расчет")
        print("Абсолютный прирост:")
        chain_absolute_growth = self.calculate_chain_absolute_growth(values_avg)
        for chain_growth in chain_absolute_growth:
            print(chain_growth)

        print("Темп роста:")
        chain_growth_rate = self.calculate_chain_growth_rate(values_avg)
        for chain_rate in chain_growth_rate:
            print(chain_rate)

        print("Темп прироста:")
        chain_growth_rate_diff = self.calculate_chain_growth_rate_difference(values_avg)
        for chain_diff in chain_growth_rate_diff:
            print(chain_diff)

        print("Показатель сравнения:")
        comparison_indicator = self.calculate_comparison_indicator(values_avg)
        for indicator in comparison_indicator:
            print(indicator)

        data = {
            "Абсолютный прирост (Базисный расчет)": absolute_growth,
            "Темп роста (Базисный расчет)": growth_rate,
            "Темп прироста (Базисный расчет)": growth_rate_diff,
            "Абсолютный прирост (Цепной расчет)": chain_absolute_growth,
            "Темп роста (Цепной расчет)": chain_growth_rate,
            "Темп прироста (Цепной расчет)": chain_growth_rate_diff,
        }

        result_df = pd.DataFrame(data)
        result_df.to_excel("result_analysis.xlsx", index=False)

    def calculate_mean_and_std_deviation(self, prices):
        mean_price = sum(prices) / len(prices)
        std_deviation = (sum((x - mean_price) ** 2 for x in prices) / len(prices)) ** 0.5
        return mean_price, std_deviation

    def calculate_percentage_in_range(self, prices, mean_price, std_deviation, num_sigmas):
        lower_bound = mean_price - num_sigmas * std_deviation
        upper_bound = mean_price + num_sigmas * std_deviation
        prices_in_range = len([price for price in prices if lower_bound <= price <= upper_bound])
        percentage_in_range = (prices_in_range / len(prices)) * 100
        return percentage_in_range

    def perform_normal_distribution_check(self):
        prices = [value for values in self.data_by_year.values() for value in values]

        mean_price, std_deviation = self.calculate_mean_and_std_deviation(prices)
        print("Математическое ожидание =", mean_price)
        print("Стандартное отклонение =", std_deviation)

        for num_sigmas in [1, 2, 3]:
            percentage_in_range = self.calculate_percentage_in_range(prices, mean_price, std_deviation, num_sigmas)
            print(f"{num_sigmas}-сигма: {percentage_in_range:.2f}%")

        stat, p_value = stats.shapiro(prices)
        print(f"Тест Шапиро-Уилка: Statistic={stat}, p-value={p_value}")

    def perform_correlation_analysis(self, file_path):
        df_factors = pd.read_excel(file_path, sheet_name="Факторы")

        average_prices = list(self.average_price_per_sqm_by_year.values())
        df_factors["Средняя цена за кв. м., тыс. рубл."] = average_prices

        correlation_results = round(
            df_factors[["Ставка по ипотеке, %", "МРОТ, тыс. рубл.", "Курс доллара, рубл.",
                        "Объем выданных ипотечных кредитов, млрд руб", "Уровень занятости населения, %",
                        "Численность населения, тыс. чел.", "Средняя цена за кв. м., тыс. рубл."]].corr(), 3)
        print("Матрица парных корреляций:")
        print(correlation_results)
        correlation_results.to_excel("correlation_analysis_results.xlsx", sheet_name="Correlation Analysis")
        return correlation_results

    def prepare_regression_data(self, y, ipoteka_rate, minimum_wage, usd_exchange_rate, credit_volume, employment_rate,
                                population):
        ones_array = np.ones_like(y)
        X = np.array(
            [ones_array, ipoteka_rate, minimum_wage, usd_exchange_rate, credit_volume, employment_rate, population])
        return X

    def perform_regression_mnk(self, X, y):
        y = np.array(y)
        C_T = X
        C = C_T.T
        mul_C_T_and_C = np.dot(C_T, C)
        inverse_mul = np.linalg.inv(mul_C_T_and_C)
        mul_inv_and_C_T = np.dot(inverse_mul, C_T)
        result = np.dot(mul_inv_and_C_T, y.T)
        return result.tolist()

    def dis_analyze_and_sign_coeff(self, y, X, T_kr):
        y_mean = np.mean(y)
        S = np.sum((y - y_mean) ** 2)
        result = self.perform_regression_mnk(X, y)
        print("Результат МНК:", result)
        y_hat = result[0]
        for i in range(1, len(result)):
            y_hat += result[i] * X[i]
        S_fact = np.sum((y_hat - y_mean) ** 2)
        print("Sфакт:", S_fact)
        S_ost = S - S_fact
        print("Sост: ", S_ost)
        df_fact = len(result) - 1
        df_ost = len(y) - df_fact - 1
        s_square_fact = S_fact / df_fact
        print("s^2fact =", s_square_fact)
        s_square_ost = S_ost / df_ost
        print("s^2ost", s_square_ost)
        F = s_square_fact / s_square_ost
        print("F: ", F)
        p_value = 1 - stats.f.cdf(F, df_fact, df_ost)
        print("P-value: ", p_value)

        # Значимость коэффициентов
        s = math.sqrt(s_square_ost)
        y = np.array(y)
        C_T = X
        C = C_T.T
        C_T_C = np.dot(C_T, C)
        inv_C_T_C = np.linalg.inv(C_T_C)
        s_j = s * np.sqrt(np.diagonal(inv_C_T_C))
        np.set_printoptions(precision=4, suppress=True)
        print("Значение s_j:", s_j)
        Tj = result / s_j
        print("Tj:")
        for i, val in enumerate(Tj, start=1):
            print(f"Коэффициент {i}: {val:.4f}")

        interval_min = result - T_kr * s_j
        interval_max = result + T_kr * s_j
        print("Интервалы (θj − tγsj; θj + tγsj):")
        for i, (min_val, max_val) in enumerate(zip(interval_min, interval_max)):
            print(f"Коэффициент {i + 1}: [{min_val}, {max_val}]")

        p_values = 2 * (1 - stats.t.cdf(np.abs(Tj), df_ost))
        print("P-значения для Tj:")
        for i, p_value in enumerate(p_values):
            print(f"Коэффициент {i + 1}: {p_value}")

        # R^2 and R^2_adj
        s_squared = S / (len(y) - 1)
        r = 1 - ((df_ost * s_square_ost) / ((len(y) - 1) * s_squared))
        print("R^2 =", r)
        r_adj = 1 - s_square_ost / s_squared
        print("R^2adj =", r_adj)

