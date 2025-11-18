"""Script for Bitcoin Price Prediction with Reproducible Results"""
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

SEED = 42

# ============================================================================
# ШАГ 1: ЗАГРУЗКА ДАННЫХ
# ============================================================================

def load_data(filepath):
    """Загружает и подготавливает данные"""
    print("\n" + "="*90)
    print("📂 ШАГ 1: ЗАГРУЗКА ДАННЫХ")
    print("="*90 + "\n")
    
    df = pd.read_csv(filepath)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    print(f"✅ Загружено: {len(df)} записей")
    print(f"📅 Период: {df['date'].min()} - {df['date'].max()}")
    
    # Диагностика по годам
    print("\n🔍 Диагностика наличия close_price по годам:")
    df['year'] = df['date'].dt.year
    for year in sorted(df['year'].unique()):
        year_data = df[df['year'] == year]
        missing = year_data['close_price'].isna().sum()
        total = len(year_data)
        pct = missing / total * 100
        status = "✅ OK" if pct < 10 else "❌ ПРОБЛЕМА"
        print(f"   {year}: {total:5d} записей, NaN: {missing:5d} ({pct:5.1f}%) {status}")
    
    df = df.drop('year', axis=1)
    return df


# ============================================================================
# ШАГ 2: АГРЕГАЦИЯ ПО ЧАСАМ
# ============================================================================

def aggregate_hourly(df):
    """Агрегирует данные по часам"""
    print("\n" + "="*90)
    print("📊 ШАГ 2: АГРЕГАЦИЯ ДАННЫХ ПО ЧАСАМ")
    print("="*90 + "\n")
    
    df['hour'] = df['date'].dt.floor('H')
    
    agg_dict = {
        'open_price': 'first',
        'high_price': 'max',
        'low_price': 'min',
        'close_price': 'last',
        'btc_volume': 'sum',
        'sentiment_positive': 'mean',
        'sentiment_negative': 'mean',
        'sentiment_neutral': 'mean',
        'sentiment': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'neutral',
        'title': 'count',
    }
    
    df = df.groupby('hour').agg(agg_dict).reset_index()
    df = df.rename(columns={'hour': 'datetime', 'title': 'news_count'})
    df = df.sort_values('datetime').reset_index(drop=True)
    
    print(f"✅ Агрегировано: {len(df)} часовых записей")
    return df


# ============================================================================
# ШАГ 3: РАЗДЕЛЕНИЕ НА TRAIN/TEST (ПЕРЕД feature engineering!)
# ============================================================================

def split_train_test(df, test_size=0.15):
    """
    КРИТИЧНО: Разделение ПО ИНДЕКСАМ (не датам) чтобы избежать gaps
    И ПЕРЕД feature engineering чтобы избежать утечки!
    """
    print("\n" + "="*90)
    print("✂️  ШАГ 3: РАЗДЕЛЕНИЕ ДАННЫХ (ПЕРЕД feature engineering)")
    print("="*90 + "\n")
    
    # Разделяем по индексам
    test_size_count = int(len(df) * test_size)
    split_idx = len(df) - test_size_count
    
    df_train = df.iloc[:split_idx].copy()
    df_test = df.iloc[split_idx:].copy()
    
    print(f"📊 Train set: {len(df_train)} записей ({df_train['datetime'].min()} - {df_train['datetime'].max()})")
    print(f"📊 Test set:  {len(df_test)} записей ({df_test['datetime'].min()} - {df_test['datetime'].max()})")
    
    if len(df_train) < 100 or len(df_test) < 50:
        raise ValueError("❌ Train или test слишком маленький!")
    
    print("✅ Разделение завершено")
    return df_train, df_test


# ============================================================================
# ШАГ 4: СОЗДАНИЕ ПРИЗНАКОВ (ОТДЕЛЬНО ДЛЯ TRAIN И TEST)
# ============================================================================

def create_features(df, name=""):
    """
    Создаёт признаки ОТДЕЛЬНО для train и test
    Это предотвращает утечку через rolling окна!
    """
    df = df.copy()
    
    # Shift признаки (безопасно)
    df['price_lag_1h'] = df['close_price'].shift(1)
    df['price_lag_4h'] = df['close_price'].shift(4)
    df['price_lag_24h'] = df['close_price'].shift(24)
    
    # Доходности (безопасно)
    df['return_1h'] = df['close_price'].pct_change(1) * 100
    df['return_4h'] = df['close_price'].pct_change(4) * 100
    df['return_24h'] = df['close_price'].pct_change(24) * 100
    
    # Скользящие средние (ТОЛЬКО для текущего набора - безопасно!)
    df['ma_24h'] = df['close_price'].rolling(window=24, min_periods=1).mean()
    df['ma_168h'] = df['close_price'].rolling(window=168, min_periods=1).mean()
    
    # Волатильность
    df['volatility_24h'] = df['return_1h'].rolling(window=24, min_periods=1).std()
    df['volatility_168h'] = df['return_1h'].rolling(window=168, min_periods=1).std()
    
    # Отношение к скользящей средней (с защитой от деления на 0)
    df['price_to_ma_24h'] = (df['close_price'] - df['ma_24h']) / (df['ma_24h'] + 1e-10) * 100
    df['price_to_ma_168h'] = (df['close_price'] - df['ma_168h']) / (df['ma_168h'] + 1e-10) * 100
    
    # Новостные признаки
    sentiment_map = {'positive': 1, 'neutral': 0, 'negative': -1}
    df['sentiment_score'] = df['sentiment'].map(sentiment_map).fillna(0)
    
    df['avg_sentiment_24h'] = df['sentiment_score'].rolling(window=24, min_periods=1).mean()
    df['news_count_24h'] = df['news_count'].rolling(window=24, min_periods=1).sum()
    
    # Временные признаки
    df['hour_of_day'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Volume признаки
    df['volume_ma_ratio'] = df['btc_volume'] / (df['btc_volume'].rolling(window=24, min_periods=1).mean() + 1e-10)
    
    return df


# ============================================================================
# ШАГ 5: СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ (ОТДЕЛЬНО ДЛЯ TRAIN И TEST)
# ============================================================================

def create_target(df, horizon_hours=1):
    """Создаёт целевую переменную"""
    df = df.copy()
    
    # Целевая переменная (из будущего)
    df['future_price'] = df['close_price'].shift(-horizon_hours)
    df['target'] = ((df['future_price'] - df['close_price']) / df['close_price']) * 100
    df = df.drop('future_price', axis=1)
    
    return df


# ============================================================================
# ШАГ 6: ОБРАБОТКА NaN (ОТДЕЛЬНО ДЛЯ TRAIN И TEST)
# ============================================================================

def handle_nan(df_train, df_test, feature_cols):
    """
    Обрабатывает NaN ОТДЕЛЬНО для train и test
    Это предотвращает утечку медианы из test в train!
    """
    # Для train: вычисляем медиану только из train
    for col in feature_cols:
        median_val = df_train[col].median()
        if pd.isna(median_val) or np.isnan(median_val):
            median_val = 0
        
        df_train[col] = df_train[col].fillna(median_val)
        df_test[col] = df_test[col].fillna(median_val)
    
    # Замена inf
    df_train[feature_cols] = df_train[feature_cols].replace([np.inf, -np.inf], np.nan)
    df_train[feature_cols] = df_train[feature_cols].fillna(df_train[feature_cols].median())
    
    df_test[feature_cols] = df_test[feature_cols].replace([np.inf, -np.inf], np.nan)
    df_test[feature_cols] = df_test[feature_cols].fillna(df_test[feature_cols].median())
    
    # Удаляем NaN в target
    df_train = df_train.dropna(subset=['target'])
    df_test = df_test.dropna(subset=['target'])
    
    return df_train, df_test


# ============================================================================
# ШАГ 7: ОБУЧЕНИЕ И ОЦЕНКА МОДЕЛЕЙ
# ============================================================================

def train_models(df_train, df_test):
    """Обучает модели и вычисляет метрики"""
    print("\n" + "="*90)
    print("🤖 ШАГ 7: ОБУЧЕНИЕ МОДЕЛЕЙ")
    print("="*90 + "\n")
    
    # Исключаемые колонки
    exclude = ['datetime', 'date', 'sentiment', 'title', 'content',
               'open_price', 'high_price', 'low_price', 'close_price']
    
    feature_cols = [c for c in df_train.columns 
                   if c not in exclude and c != 'target'
                   and df_train[c].dtype in ['int64', 'float64']]
    
    print(f"📊 Признаков: {len(feature_cols)}")
    print(f"📊 Train: {len(df_train)}, Test: {len(df_test)}\n")
    
    # Обработка NaN
    print("🔧 Обработка NaN (ОТДЕЛЬНО для train и test)...")
    df_train, df_test = handle_nan(df_train, df_test, feature_cols)
    print(f"✅ После обработки NaN: Train: {len(df_train)}, Test: {len(df_test)}\n")
    
    # Подготавливаем данные
    X_train = df_train[feature_cols].copy()
    y_train = df_train['target'].copy()
    X_test = df_test[feature_cols].copy()
    y_test = df_test['target'].copy()
    
    # Масштабирование (fit только на train!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Модели с SEED для воспроизводимости
    models = {
        'Ridge': Ridge(alpha=1.0, random_state=SEED),
        'Lasso': Lasso(alpha=0.1, max_iter=5000, random_state=SEED),
        'Decision Tree': DecisionTreeRegressor(max_depth=8, random_state=SEED),
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=8, 
                                              random_state=SEED, n_jobs=-1),
        'LightGBM': LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.05,
                                 num_leaves=32, random_state=SEED, verbose=-1)
    }
    
    results = {}
    tscv = TimeSeriesSplit(n_splits=5)
    
    for name, model in models.items():
        print(f"{'='*60}")
        print(f"📈 {name}")
        print(f"{'='*60}")
        
        # CV
        cv_maes = []
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
            X_tr = X_train_scaled[train_idx]
            y_tr = y_train.iloc[train_idx]
            X_val = X_train_scaled[val_idx]
            y_val = y_train.iloc[val_idx]
            
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            cv_maes.append(mae)
            print(f"  Fold {fold}: MAE = {mae:.3f}%")
        
        cv_mean = np.mean(cv_maes)
        cv_std = np.std(cv_maes)
        print(f"\n  CV Mean: {cv_mean:.3f}% ± {cv_std:.3f}%")
        
        # Train final
        model.fit(X_train_scaled, y_train)
        y_pred_test = model.predict(X_test_scaled)
        
        # Metrics (с защитой от NaN/inf)
        mae = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        
        # R² (с защитой от NaN)
        try:
            r2 = r2_score(y_test, y_pred_test)
        except:
            r2 = np.nan
        
        # Directional accuracy
        direction_true = (y_test > 0).astype(int)
        direction_pred = (y_pred_test > 0).astype(int)
        dir_acc = (direction_true == direction_pred).mean() * 100
        
        print(f"\n  TEST РЕЗУЛЬТАТЫ:")
        print(f"     MAE:  {mae:.3f}%")
        print(f"     RMSE: {rmse:.3f}%")
        print(f"     R²:   {r2:.4f}")
        print(f"     Dir. Acc.: {dir_acc:.1f}%")
        print()
        
        results[name] = {
            'model': model,
            'predictions': y_pred_test,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'dir_acc': dir_acc,
            'cv_mean': cv_mean
        }
    
    return results, y_test


# ============================================================================
# ШАГ 8: ВЫВОД РЕЗУЛЬТАТОВ
# ============================================================================

def print_results(results):
    """Выводит сравнительную таблицу"""
    print("\n" + "="*90)
    print("📋 ТАБЛИЦА СРАВНЕНИЯ МОДЕЛЕЙ")
    print("="*90 + "\n")
    
    data = []
    for name, metrics in results.items():
        data.append({
            'Модель': name,
            'MAE (%)': f"{metrics['mae']:.3f}",
            'RMSE (%)': f"{metrics['rmse']:.3f}",
            'R²': f"{metrics['r2']:.4f}",
            'Dir. Acc (%)': f"{metrics['dir_acc']:.1f}",
            'CV MAE (%)': f"{metrics['cv_mean']:.3f}"
        })
    
    df_results = pd.DataFrame(data)
    print(df_results.to_string(index=False))
    
    best = min(results, key=lambda x: results[x]['mae'])
    print(f"\n🏆 Лучшая модель: {best}")
    print(f"   MAE: {results[best]['mae']:.3f}%")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================

def main():
    """Главная функция"""
    print(f"\n🔧 SEED для воспроизводимости: {SEED}")
    print("\n")
    
    try:
        # Шаг 1: Загрузка
        df = load_data('bitcoin_news_final_dataset.csv')
        
        # Шаг 2: Агрегация
        df = aggregate_hourly(df)
        
        # Шаг 3: Разделение (ПЕРЕД feature engineering!)
        df_train, df_test = split_train_test(df, test_size=0.15)
        
        # Шаг 4: Создание признаков (ОТДЕЛЬНО!)
        print("\n" + "="*90)
        print("🔧 ШАГ 4: СОЗДАНИЕ ПРИЗНАКОВ (ОТДЕЛЬНО для train и test)")
        print("="*90 + "\n")
        
        df_train = create_features(df_train, "train")
        df_test = create_features(df_test, "test")
        print("✅ Признаки созданы")
        
        # Шаг 5: Целевая переменная
        print("\n" + "="*90)
        print("🎯 ШАГ 5: СОЗДАНИЕ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
        print("="*90 + "\n")
        
        df_train = create_target(df_train, horizon_hours=1)
        df_test = create_target(df_test, horizon_hours=1)
        print("✅ Целевая переменная создана")
        
        # Шаг 6+7: Обработка и обучение
        results, y_test = train_models(df_train, df_test)
        
        # Шаг 8: Результаты
        print_results(results)
        
        print("\n" + "="*90)
        print("✅ АНАЛИЗ ЗАВЕРШЁН БЕЗ ОШИБОК!")
        print("="*90)
        
        return results
        
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()
