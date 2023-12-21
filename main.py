import numpy as np
import pandas as pd


from realEstateDataAnalyzer import RealEstateDataAnalyzer


file_path = r"C:\Users\Павел Фетисов\Downloads\СтатАнализ_Курсовая.xlsx"
real_estate_analyzer = RealEstateDataAnalyzer(file_path)
real_estate_analyzer.display_average_price_per_sqm()
real_estate_analyzer.plot_prices_over_time()
values_avg = list(real_estate_analyzer.average_price_per_sqm_by_year.values())
data_by_year = real_estate_analyzer.data_by_year

#Статистические показатели
real_estate_analyzer.perform_statistical_analysis()

#Проверка на нормальное распределение
real_estate_analyzer.perform_normal_distribution_check()

#Корреляционный анализ
correlation_results = real_estate_analyzer.perform_correlation_analysis(file_path)

#Параметрическая идентификация
df_factors = pd.read_excel(file_path, sheet_name="Факторы")
ipoteka_rate = df_factors["Ставка по ипотеке, %"]
minimum_wage = df_factors["МРОТ, тыс. рубл."]
usd_exchange_rate = df_factors["Курс доллара, рубл."]
credit_volume = df_factors["Объем выданных ипотечных кредитов, млрд руб"]
employment_rate = df_factors["Уровень занятости населения, %"]
population = df_factors["Численность населения, тыс. чел."]

average_prices = list(real_estate_analyzer.average_price_per_sqm_by_year.values())
X_regression = real_estate_analyzer.prepare_regression_data(average_prices, ipoteka_rate, minimum_wage,
                                                            usd_exchange_rate, credit_volume,
                                                            employment_rate, population)

result_regression = real_estate_analyzer.perform_regression_mnk(X_regression, average_prices)
print("Результат MNK:", result_regression)

#Дисперсионной анализ и исследование коэффициентов
y = values_avg
ones_array = np.ones_like(y)

print("Исследование модели с 6-ю факторами")
X = np.array([ones_array, ipoteka_rate, minimum_wage, usd_exchange_rate, credit_volume, employment_rate, population])
real_estate_analyzer.dis_analyze_and_sign_coeff(y, X, 2.776445)

print("Исследование модели с факторами x1, x4, x5, x6")
X = np.array([ones_array, ipoteka_rate, credit_volume, employment_rate, population])
real_estate_analyzer.dis_analyze_and_sign_coeff(y, X, 2.4469)

print("Исследование модели с факторами x4, x5, x6")
X = np.array([ones_array, credit_volume, employment_rate, population])
real_estate_analyzer.dis_analyze_and_sign_coeff(y, X, 2.3646)

print("Исследование модели с факторами x4, x5")
X = np.array([ones_array, credit_volume, employment_rate])
real_estate_analyzer.dis_analyze_and_sign_coeff(y, X, 2.306)

print("Исследование модели с фактором x4")
X = np.array([ones_array, credit_volume])
real_estate_analyzer.dis_analyze_and_sign_coeff(y, X, 2.2621)
