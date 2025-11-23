# TODO & Project Review

## 📋 Обзор проекта

**Цель**: Прогнозирование цены Bitcoin на основе новостных данных и исторических цен.

**Текущий статус**: Базовая реализация (baseline) с классическими ML-моделями работает.

**Достижения**:
- ✅ Собран датасет из 9,007 статей (2021-2025)
- ✅ Добавлены почасовые цены Bitcoin от Binance API
- ✅ Реализован baseline с 5 моделями (Ridge, Lasso, Decision Tree, Random Forest, LightGBM)
- ✅ Добавлены type hints во всех функциях
- ✅ Настроена кросс-валидация для временных рядов

---

## 🎯 TODO List

### 1. Оптимизация производительности

#### 1.1 Кэширование (низкий приоритет для baseline)
- [ ] **НЕ рекомендуется** использовать `lru_cache` для текущего pipeline
  - Функции вызываются один раз
  - DataFrame не hashable (нельзя кэшировать)
  - Overhead превысит пользу

#### 1.2 Реальные оптимизации (средний приоритет)
- [ ] Использовать `category` dtype для текстовых колонок (`source`, `sentiment`)
- [ ] Downcasting числовых типов (float64 → float32 где возможно)
- [ ] Ранняя фильтрация данных до feature engineering
- [ ] Параллелизация feature engineering с `joblib` или `multiprocessing`

**Пример оптимизации dtype:**
```python
df['source'] = df['source'].astype('category')
df['sentiment'] = df['sentiment'].astype('category')
df['close_price'] = df['close_price'].astype('float32')
```

---

### 2. Улучшение Feature Engineering

#### 2.1 Продвинутые ценовые признаки (высокий приоритет)
- [ ] Технические индикаторы:
  - RSI (Relative Strength Index)
  - MACD (Moving Average Convergence Divergence)
  - Bollinger Bands
  - ATR (Average True Range)
- [ ] Фрактальные признаки (Hurst exponent)
- [ ] Volume-based features (OBV, VWAP)

**Библиотека**: `ta` (Technical Analysis Library)
```bash
poetry add ta
```

#### 2.2 NLP признаки (высокий приоритет)
- [ ] **Заменить подсчёт ключевых слов на embeddings**:
  - Sentence-BERT для semantic similarity
  - FinBERT для финансового sentiment analysis
  - Word2Vec/GloVe для word embeddings

- [ ] Named Entity Recognition (NER):
  - Извлечение компаний (Tesla, MicroStrategy)
  - Извлечение стран (China, USA)
  - Извлечение регуляторов (SEC, CFTC)

- [ ] Topic Modeling:
  - LDA для определения тем новостей
  - Clustering похожих новостей

**Пример с Sentence-BERT:**
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
df['title_embedding'] = df['title'].apply(lambda x: model.encode(x))
```

#### 2.3 Временные признаки (средний приоритет)
- [ ] Циклические признаки (sin/cos для часа, дня):
  ```python
  df['hour_sin'] = np.sin(2 * np.pi * df['hour_of_day'] / 24)
  df['hour_cos'] = np.cos(2 * np.pi * df['hour_of_day'] / 24)
  ```
- [ ] Праздники (US holidays impact on trading)
- [ ] События (Bitcoin halvings, ETF approvals)

#### 2.4 Агрегационные признаки (средний приоритет)
- [ ] Количество новостей в разные временные окна (1h, 4h, 24h)
- [ ] Тренд sentiment (растёт/падает)
- [ ] Volatility sentiment (разброс мнений)

---

### 3. Улучшение моделей

#### 3.1 Гиперпараметры (высокий приоритет)
- [ ] Grid Search / Random Search для подбора параметров
- [ ] Bayesian Optimization (Optuna, Hyperopt)
- [ ] AutoML (AutoGluon, FLAML)

**Пример с Optuna:**
```python
import optuna

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1)
    }
    model = LGBMRegressor(**params)
    # ... обучение и валидация
    return mae
```

#### 3.2 Новые модели (высокий приоритет)
- [ ] **XGBoost** - часто лучше LightGBM
- [ ] **CatBoost** - хорош с категориальными признаками
- [ ] **Stacking** - комбинация моделей

#### 3.3 Deep Learning (средний-высокий приоритет)
- [ ] **LSTM/GRU** для временных рядов
- [ ] **Transformer-based** (Temporal Fusion Transformer)
- [ ] **CNN-LSTM** гибрид
- [ ] **Attention механизмы** для важности признаков

**Пример LSTM с PyTorch:**
```python
import torch
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
```

#### 3.4 Ансамбли (средний приоритет)
- [ ] Voting Regressor (среднее предсказаний)
- [ ] Stacking с мета-моделью
- [ ] Blending разных подходов

---

### 4. Валидация и метрики

#### 4.1 Улучшенная валидация (высокий приоритет)
- [ ] Walk-forward validation (двигающееся окно)
- [ ] Purged K-Fold для финансовых данных
- [ ] Out-of-time validation (разные периоды рынка)

#### 4.2 Дополнительные метрики (средний приоритет)
- [ ] Directional Accuracy (правильно ли предсказано направление)
- [ ] Sharpe Ratio (если конвертировать в торговую стратегию)
- [ ] Maximum Drawdown
- [ ] Profit Factor

**Пример Directional Accuracy:**
```python
def directional_accuracy(y_true, y_pred):
    direction_true = np.sign(y_true.diff())
    direction_pred = np.sign(y_pred.diff())
    return (direction_true == direction_pred).mean()
```

---

### 5. Данные и источники

#### 5.1 Больше данных (высокий приоритет)
- [ ] Twitter/X sentiment (потребуется API)
- [ ] Reddit r/Bitcoin, r/cryptocurrency
- [ ] Google Trends для "Bitcoin"
- [ ] Fear & Greed Index
- [ ] On-chain metrics (число транзакций, hash rate)

#### 5.2 Альтернативные цены (низкий приоритет)
- [ ] Order book data (глубина рынка)
- [ ] Funding rates (фьючерсы)
- [ ] Цены альткоинов (корреляция с BTC)

---

### 6. Production-ready улучшения

#### 6.1 Код (средний приоритет)
- [ ] Разделить на модули:
  - `data/` - загрузка и обработка
  - `features/` - feature engineering
  - `models/` - обучение моделей
  - `evaluation/` - метрики и визуализация

- [ ] Конфигурация через YAML/JSON
- [ ] Логирование (не print, а logging)
- [ ] Unit tests для критичных функций

#### 6.2 MLOps (низкий приоритет для начинающих)
- [ ] Tracking экспериментов (MLflow, Weights & Biases)
- [ ] Версионирование данных (DVC)
- [ ] CI/CD для модели
- [ ] Model serving (FastAPI endpoint)

---

### 7. Документация

- [ ] Добавить docstrings с примерами
- [ ] README с инструкциями запуска
- [ ] Jupyter notebook с EDA (исследовательский анализ)
- [ ] Визуализация важности признаков

---

## 📊 Приоритизация (для начинающего в ML)

### Начать с (1-2 недели):
1. ✅ **Улучшить NLP признаки** - простой способ поднять качество
   - Установить `sentence-transformers`
   - Добавить embeddings заголовков

2. ✅ **Подобрать гиперпараметры** - бесплатный буст качества
   - Использовать Optuna для LightGBM

3. ✅ **Добавить технические индикаторы** - стандарт в финансовом ML
   - Библиотека `ta`

### Затем (2-4 недели):
4. **Попробовать XGBoost/CatBoost**
5. **Добавить больше источников данных** (Twitter sentiment)
6. **Реализовать простой LSTM**

### Позже (для продвинутых):
7. Deep Learning архитектуры
8. Production deployment
9. Автоматическая торговая стратегия

---

## 🎓 Обучающие ресурсы

### Книги:
- "Hands-On Machine Learning" (Aurélien Géron) - главы про Time Series
- "Machine Learning for Asset Managers" (Marcos López de Prado)

### Курсы:
- Fast.ai - Practical Deep Learning
- Coursera "Machine Learning Specialization" (Andrew Ng)

### Библиотеки для изучения:
- `scikit-learn` - базовый ML
- `PyTorch` / `TensorFlow` - Deep Learning
- `optuna` - гиперпараметры
- `sentence-transformers` - NLP embeddings
- `ta` - технические индикаторы

---

## 🔍 Code Review Insights

### Что сделано хорошо:
✅ Правильная сортировка данных по времени
✅ Time Series Split вместо обычного train/test
✅ Несколько моделей для сравнения
✅ Визуализация результатов
✅ Type hints везде

### Потенциальные проблемы:
⚠️ **Data Leakage риски:**
- Rolling features используют `min_periods=1` - это может включать будущее
- Лучше использовать строгое окно или `.shift(1)` после расчёта

⚠️ **Масштабирование признаков:**
- Ridge/Lasso требуют нормализации
- Добавить `StandardScaler` или `MinMaxScaler`

⚠️ **Imbalanced target:**
- Если предсказываем направление (вверх/вниз), может быть дисбаланс
- Рассмотреть class weights или SMOTE

### Рекомендуемые изменения:

```python
# 1. Масштабирование
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Feature importance
import matplotlib.pyplot as plt

feature_importance = model.feature_importances_
plt.barh(feature_names, feature_importance)
plt.xlabel('Importance')
plt.title('Feature Importance')

# 3. Сохранение модели
import joblib

joblib.dump(best_model, 'models/best_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
```

---

## 📝 Следующие шаги

1. **Сегодня**: Добавить технические индикаторы (RSI, MACD)
2. **Эта неделя**: Реализовать Sentence-BERT embeddings
3. **Следующая неделя**: Подобрать гиперпараметры с Optuna
4. **Через 2 недели**: Попробовать простой LSTM

**Главное**: Не пытайтесь сделать всё сразу! Добавляйте по одному улучшению, измеряйте результат, учитесь на процессе. 🚀

---

## 💡 Полезные команды

```bash
# Установить новые зависимости
poetry add sentence-transformers ta optuna xgboost catboost

# Запустить baseline
poetry run python crypto_webscrapper/baseline/baseline.ipynb

# Заполнить недостающие часы (если нужно)
poetry run fill-missing-hours

# Проверить датасет
poetry run verify-dataset
```

---

**Последнее обновление**: 2025-01-18
**Автор**: Claude Code
**Статус проекта**: 🟢 В активной разработке
