import pandas as pd
import polars as pl

import warnings

warnings.filterwarnings('ignore')


class FraudDetectionEDA:
    def __init__(self, transaction_file, currency_file):
        self.transaction_file = transaction_file
        self.currency_file = currency_file
        self.df_transactions = None
        self.df_currency = None
        self.load_data()

    def load_data(self):
        print("📊 Загрузка данных...")

        self.df_transactions = pl.read_parquet(self.transaction_file)
        print(f"Транзакции: {self.df_transactions.shape[0]} строк, {self.df_transactions.shape[1]} столбцов")

        self.df_currency = pl.read_parquet(self.currency_file)
        print(f"Валютные курсы: {self.df_currency.shape[0]} строк, {self.df_currency.shape[1]} столбцов")

        print("✅ Данные успешно загружены!")

    def basic_info(self):
        print("\n" + "=" * 60)
        print("📋 БАЗОВАЯ ИНФОРМАЦИЯ О ДАННЫХ")
        print("=" * 60)

        print("\n🏦 ДАННЫЕ ТРАНЗАКЦИЙ:")
        print(f"Размер датасета: {self.df_transactions.shape}")
        print(f"Период данных: {self.df_transactions['timestamp'].min()} - {self.df_transactions['timestamp'].max()}")

        null_counts = self.df_transactions.null_count()
        print("\n🔍 Пропущенные значения:")
        for col in null_counts.columns:
            null_val = null_counts[col][0]
            if null_val > 0:
                print(f"  {col}: {null_val} ({null_val / len(self.df_transactions) * 100:.2f}%)")

        fraud_stats = self.df_transactions.group_by('is_fraud').agg([
            pl.count().alias('count'),
            pl.col('amount').mean().alias('avg_amount'),
            pl.col('amount').sum().alias('total_amount')
        ])
        print(f"\n🚨 Распределение мошенничества:")
        print(fraud_stats.to_pandas())

        print(f"\n💱 ВАЛЮТНЫЕ КУРСЫ:")
        print(f"Период: {self.df_currency['date'].min()} - {self.df_currency['date'].max()}")
        print(f"Валюты: {[col for col in self.df_currency.columns if col != 'date']}")

    def fraud_overview(self):
        """Обзор мошеннических транзакций"""
        print("\n" + "=" * 60)
        print("🚨 АНАЛИЗ МОШЕННИЧЕСТВА")
        print("=" * 60)

        df_pd = self.df_transactions.to_pandas()

        total_transactions = len(df_pd)
        fraud_transactions = df_pd['is_fraud'].sum()
        fraud_rate = fraud_transactions / total_transactions * 100

        print(f"\n📊 Общая статистика:")
        print(f"  Всего транзакций: {total_transactions:,}")
        print(f"  Мошеннических транзакций: {fraud_transactions:,}")
        print(f"  Уровень мошенничества: {fraud_rate:.2f}%")

        fraud_amount = df_pd[df_pd['is_fraud']]['amount'].sum()
        total_amount = df_pd['amount'].sum()
        fraud_loss_rate = fraud_amount / total_amount * 100

        print(f"\n💰 Финансовые потери:")
        print(f"  Общий оборот: ${total_amount:,.2f}")
        print(f"  Потери от мошенничества: ${fraud_amount:,.2f}")
        print(f"  Доля потерь: {fraud_loss_rate:.2f}%")

        avg_legit = df_pd[~df_pd['is_fraud']]['amount'].mean()
        avg_fraud = df_pd[df_pd['is_fraud']]['amount'].mean()

        print(f"\n💳 Средние суммы транзакций:")
        print(f"  Легитимные: ${avg_legit:,.2f}")
        print(f"  Мошеннические: ${avg_fraud:,.2f}")
        print(f"  Разница: {((avg_fraud / avg_legit - 1) * 100):+.1f}%")

    def analyze_by_dimensions(self):
        print("\n" + "=" * 60)
        print("🔍 АНАЛИЗ ПО ИЗМЕРЕНИЯМ")
        print("=" * 60)

        df_pd = self.df_transactions.to_pandas()

        print("\n🏪 По категориям вендоров:")
        vendor_analysis = df_pd.groupby('vendor_category').agg({
            'is_fraud': ['count', 'sum', 'mean'],
            'amount': ['sum', 'mean']
        }).round(4)
        vendor_analysis.columns = ['total_trans', 'fraud_trans', 'fraud_rate', 'total_amount', 'avg_amount']
        vendor_analysis['fraud_rate_pct'] = vendor_analysis['fraud_rate'] * 100
        print(vendor_analysis.sort_values('fraud_rate', ascending=False))

        print("\n💳 По типам карт:")
        card_analysis = df_pd.groupby('card_type').agg({
            'is_fraud': ['count', 'sum', 'mean'],
            'amount': ['mean']
        }).round(4)
        card_analysis.columns = ['total_trans', 'fraud_trans', 'fraud_rate', 'avg_amount']
        card_analysis['fraud_rate_pct'] = card_analysis['fraud_rate'] * 100
        print(card_analysis.sort_values('fraud_rate', ascending=False))

        print("\n📱 По каналам:")
        channel_analysis = df_pd.groupby('channel').agg({
            'is_fraud': ['count', 'sum', 'mean'],
            'amount': ['mean']
        }).round(4)
        channel_analysis.columns = ['total_trans', 'fraud_trans', 'fraud_rate', 'avg_amount']
        channel_analysis['fraud_rate_pct'] = channel_analysis['fraud_rate'] * 100
        print(channel_analysis.sort_values('fraud_rate', ascending=False))

        print("\n🌍 Топ-10 стран по уровню мошенничества:")
        country_analysis = df_pd.groupby('country').agg({
            'is_fraud': ['count', 'sum', 'mean']
        }).round(4)
        country_analysis.columns = ['total_trans', 'fraud_trans', 'fraud_rate']
        country_analysis = country_analysis[country_analysis['total_trans'] >= 100]  # Минимум 100 транзакций
        top_fraud_countries = country_analysis.sort_values('fraud_rate', ascending=False).head(10)
        top_fraud_countries['fraud_rate_pct'] = top_fraud_countries['fraud_rate'] * 100
        print(top_fraud_countries)

    def temporal_analysis(self):
        print("\n" + "=" * 60)
        print("⏰ ВРЕМЕННОЙ АНАЛИЗ")
        print("=" * 60)

        df_pd = self.df_transactions.to_pandas()
        df_pd['timestamp'] = pd.to_datetime(df_pd['timestamp'])
        df_pd['hour'] = df_pd['timestamp'].dt.hour
        df_pd['day_of_week'] = df_pd['timestamp'].dt.day_name()
        df_pd['date'] = df_pd['timestamp'].dt.date

        print("\n🕐 По часам дня:")
        hourly_analysis = df_pd.groupby('hour').agg({
            'is_fraud': ['count', 'sum', 'mean']
        }).round(4)
        hourly_analysis.columns = ['total_trans', 'fraud_trans', 'fraud_rate']
        hourly_analysis['fraud_rate_pct'] = hourly_analysis['fraud_rate'] * 100

        dangerous_hours = hourly_analysis.sort_values('fraud_rate', ascending=False).head(5)
        print("Топ-5 самых опасных часов:")
        print(dangerous_hours)

        print("\n📅 Выходные vs будни:")
        weekend_analysis = df_pd.groupby('is_weekend').agg({
            'is_fraud': ['count', 'sum', 'mean']
        }).round(4)
        weekend_analysis.columns = ['total_trans', 'fraud_trans', 'fraud_rate']
        weekend_analysis['fraud_rate_pct'] = weekend_analysis['fraud_rate'] * 100
        weekend_analysis.index = ['Будни', 'Выходные']
        print(weekend_analysis)

    def risk_factors_analysis(self):
        print("\n" + "=" * 60)
        print("⚠️ АНАЛИЗ ФАКТОРОВ РИСКА")
        print("=" * 60)

        df_pd = self.df_transactions.to_pandas()

        risk_factors = ['is_high_risk_vendor', 'is_outside_home_country', 'is_card_present']

        for factor in risk_factors:
            print(f"\n🎯 Фактор: {factor}")
            factor_analysis = df_pd.groupby(factor).agg({
                'is_fraud': ['count', 'sum', 'mean']
            }).round(4)
            factor_analysis.columns = ['total_trans', 'fraud_trans', 'fraud_rate']
            factor_analysis['fraud_rate_pct'] = factor_analysis['fraud_rate'] * 100
            print(factor_analysis)

        print("\n⚡ Анализ активности за последний час:")

        df_pd['last_hour_num_trans'] = df_pd['last_hour_activity'].apply(
            lambda x: x.get('num_transactions', 0) if x else 0)
        df_pd['last_hour_total_amount'] = df_pd['last_hour_activity'].apply(
            lambda x: x.get('total_amount', 0) if x else 0)
        df_pd['last_hour_unique_merchants'] = df_pd['last_hour_activity'].apply(
            lambda x: x.get('unique_merchants', 0) if x else 0)

        activity_corr = \
        df_pd[['is_fraud', 'last_hour_num_trans', 'last_hour_total_amount', 'last_hour_unique_merchants']].corr()[
            'is_fraud'].sort_values(ascending=False)
        print("Корреляция активности с мошенничеством:")
        print(activity_corr)

    def currency_analysis(self):
        """Анализ по валютам"""
        print("\n" + "=" * 60)
        print("💱 АНАЛИЗ ПО ВАЛЮТАМ")
        print("=" * 60)

        df_pd = self.df_transactions.to_pandas()

        # Анализ по валютам
        currency_analysis = df_pd.groupby('currency').agg({
            'is_fraud': ['count', 'sum', 'mean'],
            'amount': ['sum', 'mean']
        }).round(4)
        currency_analysis.columns = ['total_trans', 'fraud_trans', 'fraud_rate', 'total_amount', 'avg_amount']
        currency_analysis['fraud_rate_pct'] = currency_analysis['fraud_rate'] * 100
        currency_analysis = currency_analysis.sort_values('fraud_rate', ascending=False)

        print("Анализ мошенничества по валютам:")
        print(currency_analysis)

    def device_fingerprint_analysis(self):
        """Анализ отпечатков устройств"""
        print("\n" + "=" * 60)
        print("📱 АНАЛИЗ УСТРОЙСТВ")
        print("=" * 60)

        df_pd = self.df_transactions.to_pandas()

        # Анализ устройств
        device_analysis = df_pd.groupby('device').agg({
            'is_fraud': ['count', 'sum', 'mean']
        }).round(4)
        device_analysis.columns = ['total_trans', 'fraud_trans', 'fraud_rate']
        device_analysis['fraud_rate_pct'] = device_analysis['fraud_rate'] * 100
        device_analysis = device_analysis[device_analysis['total_trans'] >= 50]  # Минимум 50 транзакций

        print("Топ-10 самых рискованных устройств:")
        print(device_analysis.sort_values('fraud_rate', ascending=False).head(10))

        # Анализ отпечатков устройств - подозрительные паттерны
        fingerprint_analysis = df_pd.groupby('device_fingerprint').agg({
            'is_fraud': ['count', 'sum', 'mean'],
            'customer_id': 'nunique'
        }).round(4)
        fingerprint_analysis.columns = ['total_trans', 'fraud_trans', 'fraud_rate', 'unique_customers']

        # Подозрительные отпечатки (высокий fraud rate или много клиентов на одном устройстве)
        suspicious_fingerprints = fingerprint_analysis[
            (fingerprint_analysis['fraud_rate'] > 0.5) |  # Более 50% мошенничества
            (fingerprint_analysis['unique_customers'] > 10)  # Более 10 клиентов на одном устройстве
            ].sort_values('fraud_rate', ascending=False)

        print(f"\n🚨 Подозрительные отпечатки устройств: {len(suspicious_fingerprints)}")
        if len(suspicious_fingerprints) > 0:
            print("Топ-5 самых подозрительных:")
            print(suspicious_fingerprints.head())

    def generate_hypotheses(self):
        """Генерация гипотез на основе анализа"""
        print("\n" + "=" * 80)
        print("💡 ПРОДУКТОВЫЕ И ТЕХНИЧЕСКИЕ ГИПОТЕЗЫ")
        print("=" * 80)

        # Вычисляем текущие потери для подстановки в гипотезы
        df_pd = self.df_transactions.to_pandas()
        fraud_amount_billions = df_pd[df_pd['is_fraud']]['amount'].sum() / 1e9  # Переводим в миллиарды

        print(f"""
🏢 ПРОДУКТОВЫЕ ГИПОТЕЗЫ:

1. ДИНАМИЧЕСКОЕ ЦЕНООБРАЗОВАНИЕ СТРАХОВКИ ОТ МОШЕННИЧЕСТВА
   • Взимать различную комиссию за страховку в зависимости от риск-профиля транзакции
   • Высокорискованные категории (путешествия, развлечения) = выше комиссия
   • Потенциальный доход: 0.1-0.5% от оборота высокорискованных транзакций

2. СИСТЕМА ВРЕМЕННЫХ ОГРАНИЧЕНИЙ
   • Ограничивать транзакции в ночные часы для новых карт/устройств
   • Требовать дополнительную аутентификацию в пиковые часы мошенничества
   • Снижение мошенничества на 15-25%

3. ПРЕМИУМ ТАРИФЫ ДЛЯ БЕЗОПАСНОСТИ
   • Предлагать клиентам расширенную защиту за дополнительную плату
   • Мгновенные уведомления, биометрия, страховка возврата средств
   • Потенциальная маржа: $5-15/месяц с премиум клиентов

4. ГЕОЛОКАЦИОННЫЕ ПРОДУКТЫ
   • Автоматическая блокировка карт при подозрительных геолокациях
   • Услуга "Travel Mode" для уведомления о поездках
   • Снижение false positives на 20-30%

🔧 ТЕХНИЧЕСКИЕ ГИПОТЕЗЫ:

1. АНСАМБЛЕВАЯ МОДЕЛЬ РЕАЛЬНОГО ВРЕМЕНИ
   • Комбинирование: XGBoost + Neural Network + Правила
   • Фичи: device fingerprinting, behavioral patterns, network analysis
   • Ожидаемая точность: 95%+ с 2%- false positive rate

2. СИСТЕМА СКОРИНГА УСТРОЙСТВ
   • Присваивать trust score каждому device fingerprint
   • Машинное обучение на истории устройства
   • Блокировка новых устройств с подозрительными паттернами

3. ГРАФ АНАЛИЗ СЕТЕЙ
   • Построение графов связей: IP → Device → Customer → Merchant
   • Выявление мошеннических кластеров и organized fraud
   • Community detection для выявления fraud rings

4. REAL-TIME FEATURE ENGINEERING
   • Динамические фичи: velocity checks, amount patterns, location jumps
   • Streaming обработка с Apache Kafka + Apache Flink
   • Латенси принятия решения < 100ms

5. ANOMALY DETECTION ДЛЯ НОВЫХ ВИДОВ МОШЕННИЧЕСТВА
   • Unsupervised learning для выявления новых паттернов
   • Автоматическое обновление правил на основе аномалий
   • Система алертов для fraud analysts

📊 ПОТЕНЦИАЛЬНАЯ ЦЕННОСТЬ ДЛЯ БИЗНЕСА:

1. СНИЖЕНИЕ ПОТЕРЬ: текущие потери ~${fraud_amount_billions:.1f} млрд/год → цель снижения на 40-60%
2. УВЕЛИЧЕНИЕ ДОХОДОВ: новые продукты безопасности → +5-10% revenue
3. ПОВЫШЕНИЕ TRUST: снижение false positives → улучшение UX
4. OPERATIONAL EFFICIENCY: автоматизация → снижение manual review на 70%
5. COMPLIANCE: улучшенный risk management → снижение regulatory costs
        """)

    def run_full_analysis(self):
        """Запуск полного анализа"""
        print("🚀 ЗАПУСК ПОЛНОГО РАЗВЕДОЧНОГО АНАЛИЗА ДАННЫХ")
        print("=" * 80)

        self.basic_info()
        self.fraud_overview()
        self.analyze_by_dimensions()
        self.temporal_analysis()
        self.risk_factors_analysis()
        self.currency_analysis()
        self.device_fingerprint_analysis()
        self.generate_hypotheses()

        print("\n✅ АНАЛИЗ ЗАВЕРШЕН!")
        print("📋 Рекомендации сохранены в виде гипотез выше")


if __name__ == "__main__":
    eda = FraudDetectionEDA(
        transaction_file="transaction_fraud_data.parquet",
        currency_file="historical_currency_exchange.parquet"
    )

    eda.run_full_analysis()
