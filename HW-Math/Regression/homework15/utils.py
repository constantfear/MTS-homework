# Подключим нужные для базовых операций библиотеки
import seaborn as sb
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Подключим пакеты для использования OLS метода и тестов
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit, Logit
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy import stats
import scipy as sci
from statsmodels.stats.outliers_influence import variance_inflation_factor 

# Подгрузим библиотеки для репорт по бинарном таргету
from sklearn.metrics import roc_auc_score, classification_report, roc_curve, roc_auc_score, f1_score, precision_score, recall_score



def gen_uniform(mean, width, size, seed):

    # К сожалению, самый просто способ избавиться от случайности -
    # вызывать один и тот же random seed каждый раз перед запуском функции
    np.random.seed(seed)

    # Генерируем выборку
    x = np.random.uniform(low=mean - width/2, high=mean + width/2, size=size)

    return(x)


def gen_normal(mean, std, size, seed):

    # К сожалению, самый просто способ избавиться от случайности -
    # вызывать один и тот же random seed каждый раз перед запуском функции
    np.random.seed(seed)

    # Генерируем выборку
    x = np.random.normal(loc=mean, scale=std, size=size)

    return(x)


def gen_data(y_type, params, seed=None):

    # К сожалению, самый просто способ избавиться от случайности -
    # вызывать один и тот же random seed каждый раз перед запуском функции
    if seed:
        np.random.seed(seed)

    # Генерация датасета для задачи неверно определенной формы зависимости
    if y_type == 'linearity':
        
        # Сгенерируем значения факторов и форму зависимости y от регрессоров
        x1 = gen_uniform(params['x1_mean'], params['x1_width'], params['N'], seed)
        x2 = gen_normal(params['x2_mean'], params['x2_std'], params['N'], seed)
        x3 = np.exp(x1)

        # А также сгененируем случайную ошибку как величину из нормального распределения
        e = gen_normal(params['e_mean'], params['e_std'], params['N'], seed)

        # Здесь мы создаем таргет 'y' как некую функцию от x1, x2 и ошибки e
        y = params['beta0'] + params['beta1'] * np.exp(x1) + e

        # Для удобства сохраним вектора в pandas dataframe
        dataset = pd.DataFrame({'x1': x1, 'x2': x2, 'x3': x3, 'e': e, 'y': y})

    # Генерация датасета для задачи пропущенной переменной и мультиколлинеарности
    elif y_type == 'multivariate':

        # Сгенерируем значения факторов и форму зависимости y от регрессоров
        means = [params['x1_mean'], params['x2_mean'], params['x3_mean']]

        # Вычислим наполнение матрицы ковариаций 
        var_1 = params['x1_std']**2
        var_2 = params['x2_std']**2
        var_3 = params['x3_std']**2

        cov_12 = params['corr_12'] * params['x1_std'] * params['x2_std']
        cov_13 = params['corr_13'] * params['x1_std'] * params['x3_std']
        cov_23 = params['corr_23'] * params['x2_std'] * params['x3_std']
        
        covs = [[var_1, cov_12, cov_13],
                [cov_12, var_2, cov_23],
                [cov_13, cov_23, var_3]]

        # Сгененрируем требующуюся выборку
        X = np.random.multivariate_normal(
            mean=means,
            cov=covs,
            size=params['N']
        )

        # А также сгененируем случайную ошибку как величину из нормального распределения
        e = np.random.normal(loc=params['e_mean'], scale=params['e_std'], size=params['N'])

        # Здесь мы создаем таргет 'y' как некую функцию от x1, x2 и ошибки e
        y = params['beta0'] \
            + params['beta1'] * X[:, 0] \
            + params['beta2'] * X[:, 1] \
            + params['beta3'] * X[:, 2] \
            + e

        # Для удобства сохраним вектора в pandas dataframe
        dataset = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'x3': X[:, 2], 'e': e, 'y': y})

    # Генерация датасета для симуляции изменения какого-то параметра, например, размера выборки
    elif y_type == 'simul':

        # Сгенерируем значения факторов и форму зависимости y от регрессоров
        means = [params['x1_mean'], params['x2_mean']]

        # Вычислим наполнение матрицы ковариаций 
        var_1 = params['x1_std']**2
        var_2 = params['x2_std']**2

        cov_12 = params['corr_12'] * params['x1_std'] * params['x2_std']
        
        # Сгененрируем требующуюся выборку
        X = np.random.multivariate_normal(
            mean=means,
            cov=[[var_1, cov_12],
                 [cov_12, var_2]],
            size=params['N']
        )
        
        # А также сгененируем случайную ошибку как величину из нормального распределения
        e = np.random.normal(loc=params['e_mean'], scale=params['e_std'], size=params['N'])

        # Здесь мы создаем таргет 'y' как некую функцию от x1, x2 и ошибки e
        y = params['beta0'] \
            + params['beta1'] * X[:, 0] \
            + params['beta2'] * X[:, 1] \
            + e

        # Для удобства сохраним вектора в pandas dataframe
        dataset = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y, 'e': e})

    elif y_type == 'heteroskedasticity':
        
        # Сгенерируем значения факторов и форму зависимости y от регрессоров
        means = [params['x1_mean'], params['x2_mean']]

        # Вычислим наполнение матрицы ковариаций 
        var_1 = params['x1_std']**2
        var_2 = params['x2_std']**2
        cov_12 = params['corr_12'] * params['x1_std'] * params['x2_std']
        
        # Сгененрируем требующуюся выборку
        X = np.random.multivariate_normal(
            mean=means,
            cov=[[var_1, cov_12],
                 [cov_12, var_2]],
            size=params['N']
        )

        # А также сгененируем случайную ошибку как величину из нормального распределения
        # Но добавим корреляцию между значениями ошибки и регрессором х1
        e = np.random.normal(loc=params['e_mean'], scale=params['e_std'], size=params['N'])
        e = (X[:, 0] / np.mean(X[:, 0]))**2 * e 

        # Здесь мы создаем таргет 'y' как некую функцию от x1, x2 и ошибки e
        y = params['beta0'] \
            + params['beta1'] * X[:, 0] \
            + params['beta2'] * X[:, 1] \
            + e

        # Для удобства сохраним вектора в pandas dataframe
        dataset = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y, 'e': e})
    
    elif y_type == 'logit':
        
        # Сгенерируем значения факторов и форму зависимости y от регрессоров
        means = [params['x1_mean'], params['x2_mean']]

        # Вычислим наполнение матрицы ковариаций 
        var_1 = params['x1_std']**2
        var_2 = params['x2_std']**2
        cov_12 = params['corr_12'] * params['x1_std'] * params['x2_std']
        
        # Сгененрируем требующуюся выборку
        X = np.random.multivariate_normal(
            mean=means,
            cov=[[var_1, cov_12],
                 [cov_12, var_2]],
            size=params['N']
        )

        # А также сгененируем случайную ошибку как величину из нормального распределения
        e = np.random.normal(loc=params['e_mean'], scale=params['e_std'], size=params['N'])

        # Здесь мы создаем таргет 'y' как некую функцию от x1, x2 и ошибки e с бинарным исходом
        y_lin = params['beta0'] \
                + params['beta1'] * X[:, 0] \
                + params['beta2'] * X[:, 1] \
                + e
        
        y = (y_lin > np.mean(y_lin)).astype('int')
        #y = (np.exp(y_lin)/(1 + np.exp(y_lin)) > 0.5).astype('int')

        # Для удобства сохраним вектора в pandas dataframe
        dataset = pd.DataFrame({'x1': X[:, 0], 'x2': X[:, 1], 'y': y, 'e': e})
        
        
    return(dataset)


def plot_data(dataset, target, feature_names, plot_type = None, pairwise=False, model=None):

    dataset = dataset.copy()
    # Графики с результатами моделирования
    if plot_type == 'model':

        if pairwise == True:
            
            feat = feature_names[0]
            # Давайте посмотрим, как соотносится реальная выборка с предсказанными значениями
            sb.scatterplot(data=dataset, x=feat, y=target);
            sb.lineplot(x=dataset[feat], y=dataset[f'{target}_hat'], c = 'r');
            plt.xlabel(f'{feat} values')
            plt.ylabel(f'{target} values')
            plt.title(f'{target} vs {feat} distribution and model fit')
            plt.show()

        else:

            # Чтобы сделать функцию гибкой, пройдемся по фичам из списка
            for feat in feature_names:
                
                
                '''
                Предскажем значение таргета при фиксированных средних
                значениях всех перменных, кроме рассматриваемой. И отрисуем
                зависимость между этим фактором и предсказанным таргетом
                '''
                dataset_paired = pd.DataFrame(
                    data = [dataset[feature_names].mean()],
                    index = dataset.index
                    )
                dataset_paired[feat] = dataset[feat]
                dataset_paired['const'] = 1.0
                dataset_paired = dataset_paired[['const'] + feature_names]
                target_paired = model.predict(dataset_paired)

                # Давайте посмотрим, как соотносится реальная выборка с предсказанными значениями
                sb.scatterplot(data=dataset, x=feat, y=target);
                sb.lineplot(x=dataset[feat], y=target_paired, c = 'r');
                plt.xlabel(f'{feat} values')
                plt.ylabel(f'{target} values')
                plt.title(f'{target} vs {feat} distribution and model fit')
                plt.show()

                # Также посмотрим распределение остатков модели относительно факторов
                sb.scatterplot(data=dataset, x=feat, y='residuals');
                sb.lineplot(x=dataset[feat], y=dataset['residuals'].mean(), c='black', linestyle='--');
                plt.xlabel(f'{feat} values')
                plt.ylabel(f'Residuals values')
                plt.title(f'Residuals vs {feat} distribution and model fit')
                plt.show()

        # Отрисует предсказанные и реальные значения таргета
        sb.scatterplot(data=dataset, x=f'{target}_hat', y=target);
        sb.lineplot(data=dataset, x=target, y=target, c='black', linestyle='--');
        plt.xlabel('Predicted value')
        plt.ylabel('Actual value')
        plt.title('Model fit vs actual data')
        plt.show()

        # А также посмотрим на распределение остатков модели
        sb.displot(dataset['residuals'], bins=20, kde=True);
        plt.xlabel('Model residuals')
        plt.ylabel('Residuals frequency')
        plt.title('Model residuals histogram')
        plt.show()

    # Графики с базовыми распределениями факторов
    elif plot_type == 'EDA':
        
        # Чтобы сделать функцию гибкой, пройдемся по фичам из списка
        for feat in feature_names:

            # Отрисуем распределения нужных нам факторов
            sb.displot(dataset[feat], bins=20, kde=True);
            plt.xlabel(f'{feat} values')
            plt.ylabel(f'{feat} freq')
            plt.title(f'{feat} histogram and density plot')
            plt.show()

            # Также отрисуем зависимость между таргетом и интересующими факторами
            sb.scatterplot(data=dataset, x=feat, y='y');
            plt.xlabel(f'{feat} values')
            plt.ylabel('y values')
            plt.title(f'y vs {feat} distribution')
            plt.show()


def train_model(dataset, target, feature_names, show_results=False, pairwise=False,
                return_norm_tests=False, robust=False):

    dataset = dataset.copy()

    # Создадим матрицу фичей как фактор x1 и единичный вектор, на который будет фиттиться константа
    X = sm.add_constant(dataset[feature_names])
    
    # Зафитим модель на данные. y - наша целевая переменная, X - матрица факторов
    if robust == True:
        model = sm.OLS(dataset[target], X).fit(cov_type='HC0')
    else:
        model = sm.OLS(dataset[target], X).fit()
    
    dataset[f'{target}_hat'] = model.fittedvalues
    dataset['residuals'] = model.resid
    
    if show_results:
        # Выведем результат построения регрессии
        print(model.summary())
        
        # Выведем графики с результатами модели
        plot_data(dataset, target, feature_names, plot_type='model', pairwise=pairwise, model=model)
        
    # Выведем саммари по тестам
    if return_norm_tests:
        display(norm_distr_check(
            (dataset['residuals'] - np.mean(dataset['residuals']))\
                /np.std(dataset['residuals'])
            ))
    
    return(dataset, model)


def train_binary(dataset, target, feature_names, plot_feature, model_type='logit', pkg='sklearn', class_weight=None):

    dataset = dataset.copy()

    if pkg == 'statsmodels':
        # Зафитим модель на данные. y - наша целевая переменная, dataset - матрица факторов
        if model_type == 'logit':

            # Реализация через statsmodels
            X = sm.add_constant(dataset[feature_names])
            model = Logit(dataset[target], X).fit()
            print(model.summary())
            
        elif model_type == 'probit':
            X = sm.add_constant(dataset[feature_names])
            model = Probit(dataset[target], X).fit()
            print(model.summary())
        else:
            print('Please, choose between Logit and Probit models!')
    elif pkg == 'sklearn':

        # Реализация через sklearn
        X = dataset[feature_names]
        if class_weight is None:
            model = LogisticRegression().fit(X, dataset[target])
        else:
            model = LogisticRegression(class_weight=class_weight).fit(X, dataset[target])
        
    
    # Подготовим репорт с результами модели
    show_binary_res(model, dataset[target], X, plot_feature, pkg)

    return(dataset, model)


def show_binary_res(model, y, X, plot_feature, pkg='statsmodels'):

    # Получаем предсказанные скоры модели
    if pkg == 'statsmodels':
        pred_proba = model.predict(X)
        pred_val = (pred_proba > 0.5).astype('int')
    else:
        pred_proba = model.predict_proba(X)[:, 1]
        pred_val = model.predict(X)

    print('\n ================================================== TRAIN RESULTS ================================================== \n')
    print(f'ROC AUC score: {roc_auc_score(y.values, pred_proba).round(3)}\n')
    print(classification_report(y_pred=pred_val, y_true=y.values, zero_division=np.nan))
    
    match_df = pd.DataFrame({
        plot_feature: X[plot_feature],
        'Predicted': pred_val,
        'Predicted score': pred_proba,
        'Actual': y})

    # Давайте посмотрим, как соотносится реальная выборка с предсказанными значениями
    sb.scatterplot(data=match_df, x=plot_feature, y='Actual', label='Actual');
    sb.scatterplot(data=match_df, x=plot_feature, y='Predicted score', c = 'r', label='Predicted');
    plt.xlabel(f'{plot_feature} values')
    plt.ylabel('Actual & Predicted scores')
    plt.title('Actual target values and its predicted scores')
    plt.legend(loc='best')
    plt.show()

    # Отрисуем распределение скоров модели по классам
    sb.displot(data=match_df, x='Predicted score', hue='Actual')
    plt.show()

    # Покажем таблицу с Confusion Matrix    
    match_df = match_df.groupby(['Predicted', 'Actual'])[['Actual']].count()\
        .rename(columns={'Actual':'Value'}).reset_index()\
        .pivot(index='Actual', columns='Predicted', values='Value')
    display(match_df.style.background_gradient(cmap='coolwarm'))


# Посчитаем VIF для регрессоров нашей модели
def calc_VIF(dataset, feature_names):
    
    # Считаем VIF 
    X = sm.add_constant(dataset[feature_names])
    VIFs = pd.Series(
        [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
        index=X.columns)
    
    # Создадим и покажем датафрейм с VIF значениями
    dt = pd.DataFrame(VIFs).reset_index()
    dt.columns = ['Feature', 'VIF']
    display(dt)


# Функция анализа распределения остатков.
# Источник: https://github.com/AANazarov/MyModulePython/blob/master/my_module__stat.py
def norm_distr_check (
        data,
        p_level:    float = 0.95):

    """
    Проверка нормальности распределения: возвращает DataFrame, содержащий 
    результаты проверки нормальности распределения с использованием различных
    тестов.
    

    Args:
        data:                                     
            исходный массив данных.
            
        p_level (float, optional):           
            доверительная вероятность. 
            Defaults to 0.95.

    Returns:
        result (pd.core.frame.DataFrame):
            результат 
    
    Notes:
        1. Функция реализует следующие тесты:
            - тест Шапиро-Уилка (Shapiro-Wilk test) (при 8 <= N <= 1000)
            - тест Эппса-Палли (Epps_Pulley_test) (при N >= 8)
            - тест Д'Агостино (K2-test)
            - тест Андерсона-Дарлинга (Anderson-Darling test)
            - тест Колмогорова-Смирнова (Kolmogorov-Smirnov test) (при N >= 50)
            - тест Лиллиефорса (Lilliefors’ test)
            - тест Крамера-Мизеса-Смирнова (Cramér-von Mises test) (при N >= 40)
            - тест Пирсона (хи-квадрат) (chi-square test) (при N >= 100)
            - тест Харке-Бера (Jarque-Bera tes) (при N >= 2000)
            - тест асимметрии (при N >= 8)
            - тест эксцесса (при N >= 20)            
            
        2. Функция требует наличия файла table\Epps_Pulley_test_table.csv, 
            который содержит табличные значения критерия Эппса-Палли.
            
    """    
    
    a_level = 1 - p_level
    X = np.array(data)
    N = len(X)
       
    # тест Шапиро-Уилка (Shapiro-Wilk test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.shapiro.html
    if N >= 8:
        result_ShW = sci.stats.shapiro(X)
        s_calc_ShW = result_ShW.statistic
        a_calc_ShW = result_ShW.pvalue
        conclusion_ShW = 'gaussian distribution' if a_calc_ShW >= a_level \
            else 'not gaussian distribution'
    else:
        result_ShW = '-'
        s_calc_ShW = '-'
        a_calc_ShW = '-'
        conclusion_ShW = 'count less than 8'

    # тест Эппса-Палли (Epps_Pulley_test)
    сdf_beta_I = lambda x, a, b: sci.stats.beta.cdf(x, a, b, loc=0, scale=1)
    g_beta_III = lambda z, δ: δ*z / (1+(δ-1)*z)
    cdf_beta_III = \
        lambda x, θ0, θ1, θ2, θ3, θ4: \
            сdf_beta_I(g_beta_III((x - θ4)/θ3, θ2), θ0, θ1)
    
    θ_1 = (1.8645, 2.5155, 5.8256, 0.9216, 0.0008)    # для 15 < n < 50
    θ_2 = (1.7669, 2.1668, 6.7594, 0.91, 0.0016)    # для n >= 50
    
    if N >= 8 and N <= 1000:
        X_mean = X.mean()
        m2 = np.var(X, ddof = 0)
        # расчетное значение статистики критерия
        A = np.sqrt(2) * np.sum([np.exp(-(X[i] - X_mean)**2 / (4*m2)) 
                              for i in range(N)])
        B = 2/N * np.sum(
            [np.sum([np.exp(-(X[j] - X[k])**2 / (2*m2)) for j in range(0, k)]) 
             for k in range(1, N)])
        s_calc_EP = 1 + N / np.sqrt(3) + B - A
        # табличное значение статистики критерия
        Tep_table_df = pd.read_csv(
            filepath_or_buffer='table/Epps_Pulley_test_table.csv',
            sep=';',
            index_col='n')
        p_level_dict = {
            0.9:   Tep_table_df.columns[0],
            0.95:  Tep_table_df.columns[1],
            0.975: Tep_table_df.columns[2],
            0.99:  Tep_table_df.columns[3]}
        f_lin = sci.interpolate.interp1d(Tep_table_df.index, \
                                         Tep_table_df[p_level_dict[p_level]])
        critical_value_EP = float(f_lin(N))
        # проверка гипотезы
        if 15 < N < 50:
            a_calc_EP = 1 - cdf_beta_III(s_calc_EP, θ_1[0], θ_1[1], \
                                         θ_1[2], θ_1[3], θ_1[4])
            conclusion_EP = 'gaussian distribution' if a_calc_EP > a_level \
                else 'not gaussian distribution'            
        elif N >= 50:
            a_calc_EP = 1 - cdf_beta_III(s_calc_EP, θ_2[0], θ_2[1], \
                                         θ_2[2], θ_2[3], θ_2[4])
            conclusion_EP = 'gaussian distribution' if a_calc_EP > a_level \
                else 'not gaussian distribution'            
        else:
            a_calc_EP = ''              
            conclusion_EP = 'gaussian distribution' \
                if s_calc_EP <= critical_value_EP \
                    else 'not gaussian distribution'            
                
    elif N > 1000:
        s_calc_EP = '-'
        critical_value_EP = '-'
        a_calc_EP = '-'
        conclusion_EP = 'count more than 1000'
    else:
        s_calc_EP = '-'
        critical_value_EP = '-'
        a_calc_EP = '-'
        conclusion_EP = 'count less than 8'
    
    
    # тест Д'Агостино (K2-test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.normaltest.html
    if N >= 8:
        result_K2 = sci.stats.normaltest(X)
        s_calc_K2 = result_K2.statistic
        a_calc_K2 = result_K2.pvalue
        conclusion_K2 = 'gaussian distribution' if a_calc_K2 >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_K2 = '-'
        a_calc_K2 = '-'
        conclusion_K2 = 'count less than 8'
    
    # тест Андерсона-Дарлинга (Anderson-Darling test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.anderson.html
    result_AD = sci.stats.anderson(X)
    df_AD = pd.DataFrame({
        'a_level (%)': result_AD.significance_level,
        'statistic': [result_AD.statistic 
                      for i in range(len(result_AD.critical_values))],
        'critical_value': result_AD.critical_values
        })
    statistic_AD = float(df_AD[df_AD['a_level (%)'] == round((1 - p_level)*100, 1)]['statistic'].iloc[0])
    critical_value_AD = float(df_AD[df_AD['a_level (%)'] == round((1 - p_level)*100, 1)]['critical_value'].iloc[0])
    conclusion_AD = 'gaussian distribution' \
        if statistic_AD < critical_value_AD else 'not gaussian distribution'
    
    # тест Колмогорова-Смирнова (Kolmogorov-Smirnov test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kstest.html#scipy.stats.kstest
    if N >= 50:
        result_KS = sci.stats.kstest(X, 'norm')
        s_calc_KS = result_KS.statistic
        a_calc_KS = result_KS.pvalue
        conclusion_KS = 'gaussian distribution' if a_calc_KS >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_KS = '-'
        a_calc_KS = '-'
        conclusion_KS = 'count less than 50'
        
    # тест Лиллиефорса (Lilliefors’ test)
    # https://www.statsmodels.org/dev/generated/statsmodels.stats.diagnostic.lilliefors.html
    from statsmodels.stats.diagnostic import lilliefors
    s_calc_L, a_calc_L = sm.stats.diagnostic.lilliefors(X, 'norm')
    conclusion_L = 'gaussian distribution' if a_calc_L >= a_level \
        else 'not gaussian distribution'  
    
    # тест Крамера-Мизеса-Смирнова (Cramér-von Mises test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.cramervonmises.html#scipy-stats-cramervonmises
    if N >= 40:
        result_CvM = sci.stats.cramervonmises(X, 'norm')
        s_calc_CvM = result_CvM.statistic
        a_calc_CvM = result_CvM.pvalue
        conclusion_CvM = 'gaussian distribution' if a_calc_CvM >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_CvM = '-'
        a_calc_CvM = '-'
        conclusion_CvM = 'count less than 40'
    
    # тест Пирсона (хи-квадрат) (chi-square test)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html#scipy-stats-chisquare
    if N >= 100:
        ddof = 2    # поправка к числу степеней свободы 
                    # (число параметров распределения, оцениваемое по выборке)
        result_Chi2 = sci.stats.chisquare(X, ddof=ddof)
        s_calc_Chi2 = result_Chi2.statistic
        a_calc_Chi2 = result_Chi2.pvalue
        conclusion_Chi2 = 'gaussian distribution' if a_calc_Chi2 >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_Chi2 = '-'
        a_calc_Chi2 = '-'
        conclusion_Chi2 = 'count less than 100'
        
    # тест Харке-Бера (асимметрии и эксцесса) (Jarque-Bera tes)
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.jarque_bera.html#scipy-stats-jarque-bera
    if N >= 2000:
        result_JB = sci.stats.jarque_bera(X)
        s_calc_JB = result_JB.statistic
        a_calc_JB = result_JB.pvalue
        conclusion_JB = 'gaussian distribution' if a_calc_JB >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_JB = '-'
        a_calc_JB = '-'
        conclusion_JB = 'count less than 2000'
    
    # тест асимметрии
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skewtest.html#scipy-stats-skewtest
    if N >= 8:
        result_As = sci.stats.skewtest(X)
        s_calc_As = result_As.statistic
        a_calc_As = result_As.pvalue
        conclusion_As = 'gaussian distribution' if a_calc_As >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_As = '-'
        a_calc_As = '-'
        conclusion_As = 'count less than 8'
     
    # тест эксцесса
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kurtosistest.html#scipy-stats-kurtosistest
    if N > 20:
        result_Es = sci.stats.kurtosistest(X)
        s_calc_Es = result_Es.statistic
        a_calc_Es = result_Es.pvalue
        conclusion_Es = 'gaussian distribution' if a_calc_Es >= a_level \
            else 'not gaussian distribution'
    else:
        s_calc_Es = '-'
        a_calc_Es = '-'
        conclusion_Es = 'count less than 20'
    
    # Создадим DataFrame для сводки результатов    
    result = pd.DataFrame({
    'test': (
        'Shapiro-Wilk test',
        'Epps-Pulley test',
        "D'Agostino's K-squared test",
        'Anderson-Darling test',
        'Kolmogorov–Smirnov test',
        'Lilliefors test',
        'Cramér–von Mises test',
        'Chi-squared test',
        'Jarque–Bera test',
        'skewtest',
        'kurtosistest'),
    'p_level': (p_level),
    'a_level': (a_level),
    'a_calc': (
        a_calc_ShW,
        a_calc_EP,
        a_calc_K2,
        '',
        a_calc_KS,
        a_calc_L,
        a_calc_CvM,
        a_calc_Chi2,
        a_calc_JB,
        a_calc_As,
        a_calc_Es),
    'a_calc >= a_level': (
        a_calc_ShW >= a_level if N >= 8 else '-',
        a_calc_EP >= a_level if N > 15 and N <= 1000 else '-',
        a_calc_K2 >= a_level if N >= 8 else '-',
        '',
        a_calc_KS >= a_level if N >= 50 else '-',
        a_calc_L >= a_level,
        a_calc_CvM >= a_level if N >= 40 else '-',
        a_calc_Chi2 >= a_level if N >= 100 else '-',
        a_calc_JB >= a_level if N >= 2000 else '-',
        a_calc_As >= a_level if N >= 8 else '-',
        a_calc_Es >= a_level if N > 20 else '-'),
    'statistic': (
        s_calc_ShW,
        s_calc_EP,
        s_calc_K2,
        statistic_AD,
        s_calc_KS,
        s_calc_L,
        s_calc_CvM,
        s_calc_Chi2,
        s_calc_JB,
        s_calc_As,
        s_calc_Es),
    'critical_value': (
        '',
        critical_value_EP,
        '',
        critical_value_AD,
        '', '', '', '', '', '', ''),
    'statistic < critical_value': (
        '',
        s_calc_EP < critical_value_EP  if N >= 8 else '-',
        '',
        statistic_AD < critical_value_AD,
        '', '', '', '', '', '', ''),
    'conclusion': (
        conclusion_ShW,
        conclusion_EP,
        conclusion_K2,
        conclusion_AD,
        conclusion_KS,
        conclusion_L,
        conclusion_CvM,
        conclusion_Chi2,
        conclusion_JB,
        conclusion_As,
        conclusion_Es
        )})  
        
    return result