"""
Exploratory Data Analysis (EDA) Script
Project: AI-Based Electricity Demand Projection Including Peak Demand Projection for Delhi Power System

Dataset: cleaned_delhi_powerdemand_5min_2021_to_2024.csv
Output Directory: outputs/eda/

Analyses Performed:
1. Descriptive statistics for electricity demand.
2. Complete time series visualization (2021-2024).
3. Hourly demand profile & diurnal cycle.
4. Day-of-week profile (weekday vs weekend).
5. Monthly demand profile & seasonal variations.
6. Identification of Top 10 peak and Top 10 trough demand events.
7. Weather-demand relationships (temp, dwpt, rhum, wdir, wspd, pres).
8. Correlation matrix and heatmap.
9. Comprehensive pattern synthesis.
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating figures
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['figure.autolayout'] = True

OUTPUT_DIR = os.path.join("outputs", "eda")
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATA_PATH = "cleaned_delhi_powerdemand_5min_2021_to_2024.csv"

def run_eda():
    print(f"Loading cleaned dataset: {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['day_of_week'] = df['datetime'].dt.day_name()
    df['day_of_week_num'] = df['datetime'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = df['day_of_week_num'].isin([5, 6])
    
    demand_col = "Power demand"

    # ==========================================
    # 1. Descriptive Statistics for Power Demand
    # ==========================================
    desc = df[demand_col].describe(percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    skew = float(df[demand_col].skew())
    kurt = float(df[demand_col].kurt())
    iqr = float(desc['75%'] - desc['25%'])

    stats_summary = {
        "count": int(desc['count']),
        "mean": float(desc['mean']),
        "std": float(desc['std']),
        "min": float(desc['min']),
        "1%": float(desc['1%']),
        "5%": float(desc['5%']),
        "25%": float(desc['25%']),
        "50% (median)": float(desc['50%']),
        "75%": float(desc['75%']),
        "95%": float(desc['95%']),
        "99%": float(desc['99%']),
        "max": float(desc['max']),
        "iqr": iqr,
        "skewness": skew,
        "kurtosis": kurt
    }
    
    print("\n--- 1. DESCRIPTIVE STATISTICS (Power demand in MW) ---")
    for k, v in stats_summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # ==========================================
    # 2. Complete Time Series Plot
    # ==========================================
    fig, ax = plt.subplots(figsize=(15, 6), dpi=150)
    # Downsample daily max, mean, min for crystal clear visualization of 4 years
    daily_stats = df.set_index('datetime')[demand_col].resample('D').agg(['mean', 'max', 'min'])
    
    ax.fill_between(daily_stats.index, daily_stats['min'], daily_stats['max'], color='#90CAF9', alpha=0.4, label='Daily Range (Min-Max)')
    ax.plot(daily_stats.index, daily_stats['mean'], color='#0D47A1', linewidth=1.2, label='Daily Mean Demand')
    
    # Highlight historical maximum peak
    max_idx = df[demand_col].idxmax()
    max_row = df.loc[max_idx]
    ax.scatter(max_row['datetime'], max_row[demand_col], color='#D32F2F', s=70, zorder=5, label=f"Historic Peak: {max_row[demand_col]:.1f} MW ({max_row['datetime'].strftime('%Y-%m-%d')})")
    ax.annotate(f"All-Time Peak: {max_row[demand_col]:.1f} MW\n({max_row['datetime'].strftime('%Y-%m-%d %H:%M')})",
                xy=(max_row['datetime'], max_row[demand_col]),
                xytext=(max_row['datetime'] - pd.Timedelta(days=120), max_row[demand_col] + 250),
                arrowprops=dict(facecolor='#D32F2F', shrink=0.05, width=1.5, headwidth=6),
                fontweight='bold', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", fc="#FFEBEE", ec="#D32F2F", lw=1.2))

    ax.set_title("Delhi Power System - Complete Electricity Demand Time Series (2021 - 2024)", fontsize=14, pad=12, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Power Demand (MW)", fontsize=12)
    ax.set_ylim(1000, 9500)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    ts_plot_path = os.path.join(OUTPUT_DIR, "1_power_demand_time_series.png")
    plt.savefig(ts_plot_path)
    plt.close()
    print(f"Saved: {ts_plot_path}")

    # ==========================================
    # 3. Hourly Profile & Diurnal Cycle
    # ==========================================
    hourly_overall = df.groupby('hour')[demand_col].agg(['mean', 'std', 'min', 'max']).reset_index()
    hourly_weekday = df[~df['is_weekend']].groupby('hour')[demand_col].mean().reset_index()
    hourly_weekend = df[df['is_weekend']].groupby('hour')[demand_col].mean().reset_index()

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    ax.plot(hourly_overall['hour'], hourly_overall['mean'], marker='o', color='#1565C0', linewidth=2.5, label='Overall Average Demand')
    ax.plot(hourly_weekday['hour'], hourly_weekday[demand_col], linestyle='--', color='#2E7D32', linewidth=2.0, label='Weekday Average')
    ax.plot(hourly_weekend['hour'], hourly_weekend[demand_col], linestyle=':', color='#E65100', linewidth=2.0, label='Weekend Average')
    ax.fill_between(hourly_overall['hour'], hourly_overall['mean'] - hourly_overall['std'], hourly_overall['mean'] + hourly_overall['std'], color='#BBDEFB', alpha=0.35, label='±1 Standard Deviation')

    ax.set_title("Diurnal Electricity Demand Profile by Hour of the Day in Delhi", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Hour of Day (0 - 23)", fontsize=11)
    ax.set_ylabel("Average Power Demand (MW)", fontsize=11)
    ax.set_xticks(range(0, 24))
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    hourly_plot_path = os.path.join(OUTPUT_DIR, "2_hourly_demand_profile.png")
    plt.savefig(hourly_plot_path)
    plt.close()
    print(f"Saved: {hourly_plot_path}")

    # ==========================================
    # 4. Day of Week Profile
    # ==========================================
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    dow_stats = df.groupby('day_of_week')[demand_col].agg(['mean', 'median', 'std']).reindex(days_order)

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=150)
    colors = ['#1976D2' if d not in ['Saturday', 'Sunday'] else '#F57C00' for d in days_order]
    bars = ax.bar(dow_stats.index, dow_stats['mean'], yerr=dow_stats['std'], capsize=5, color=colors, alpha=0.85, edgecolor='black', linewidth=0.8)
    
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f} MW',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points",
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_title("Average Electricity Demand by Day of the Week (Weekdays vs Weekends)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Day of Week", fontsize=11)
    ax.set_ylabel("Average Power Demand (MW)", fontsize=11)
    ax.set_ylim(0, 6000)
    plt.tight_layout()
    dow_plot_path = os.path.join(OUTPUT_DIR, "3_day_of_week_demand.png")
    plt.savefig(dow_plot_path)
    plt.close()
    print(f"Saved: {dow_plot_path}")

    # ==========================================
    # 5. Monthly Demand Profile & Seasonality
    # ==========================================
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_stats = df.groupby('month')[demand_col].agg(['mean', 'max', 'min']).reindex(range(1, 13))
    monthly_stats.index = month_names

    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    ax.plot(monthly_stats.index, monthly_stats['mean'], marker='s', color='#1565C0', linewidth=2.5, label='Monthly Average Demand')
    ax.plot(monthly_stats.index, monthly_stats['max'], marker='^', color='#C62828', linewidth=2.0, linestyle='--', label='Monthly Maximum Peak')
    ax.plot(monthly_stats.index, monthly_stats['min'], marker='v', color='#2E7D32', linewidth=2.0, linestyle=':', label='Monthly Minimum Trough')

    ax.fill_between(monthly_stats.index, monthly_stats['min'], monthly_stats['max'], color='#FFE0B2', alpha=0.3)
    ax.set_title("Seasonal Electricity Demand Pattern by Month in Delhi (2021-2024)", fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel("Month", fontsize=11)
    ax.set_ylabel("Power Demand (MW)", fontsize=11)
    ax.legend(loc="upper left", frameon=True)
    plt.tight_layout()
    monthly_plot_path = os.path.join(OUTPUT_DIR, "4_monthly_demand_profile.png")
    plt.savefig(monthly_plot_path)
    plt.close()
    print(f"Saved: {monthly_plot_path}")

    # ==========================================
    # 6. Top 10 Highest Actual Demand Observations
    # ==========================================
    top_10_high = df.nlargest(10, demand_col)[['datetime', demand_col, 'temp', 'rhum', 'month', 'hour']].copy()
    print("\n--- 6. TOP 10 HIGHEST DEMAND OBSERVATIONS ---")
    print(top_10_high.to_string(index=False))

    # ==========================================
    # 7. Top 10 Lowest Actual Demand Observations
    # ==========================================
    top_10_low = df.nsmallest(10, demand_col)[['datetime', demand_col, 'temp', 'rhum', 'month', 'hour']].copy()
    print("\n--- 7. TOP 10 LOWEST DEMAND OBSERVATIONS ---")
    print(top_10_low.to_string(index=False))

    # ==========================================
    # 8. Weather vs Demand Relationships
    # ==========================================
    weather_vars = ['temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres']
    titles = [
        'Temperature (°C) vs Demand (Cooling Load Driver)',
        'Dew Point (°C) vs Demand',
        'Relative Humidity (%) vs Demand',
        'Wind Direction (°) vs Demand',
        'Wind Speed (km/h) vs Demand',
        'Atmospheric Pressure (hPa) vs Demand'
    ]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=150)
    axes = axes.flatten()

    for i, (var, title) in enumerate(zip(weather_vars, titles)):
        ax = axes[i]
        # Binned average curve to highlight true non-linear relationship through noise
        binned = df.groupby(pd.qcut(df[var], q=30, duplicates='drop', retbins=False))[demand_col].mean()
        bin_x = df.groupby(pd.qcut(df[var], q=30, duplicates='drop', retbins=False))[var].mean()

        # Sample scatter for density
        sampled = df.sample(n=5000, random_state=42)
        ax.scatter(sampled[var], sampled[demand_col], alpha=0.15, s=8, color='#90CAF9')
        ax.plot(bin_x, binned, color='#D32F2F', linewidth=2.8, label='Binned Mean Trend')

        r = df[var].corr(df[demand_col])
        ax.set_title(f"{title}\n(Pearson r = {r:+.3f})", fontsize=10.5, fontweight='bold')
        ax.set_xlabel(var, fontsize=10)
        ax.set_ylabel("Power Demand (MW)", fontsize=10)
        ax.legend(loc='upper left', fontsize=9)

    plt.tight_layout()
    weather_plot_path = os.path.join(OUTPUT_DIR, "5_weather_vs_demand_relationships.png")
    plt.savefig(weather_plot_path)
    plt.close()
    print(f"Saved: {weather_plot_path}")

    # ==========================================
    # 9. Correlation Analysis
    # ==========================================
    numeric_cols = [demand_col, 'temp', 'dwpt', 'rhum', 'wdir', 'wspd', 'pres', 'month', 'hour']
    corr_matrix = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    sns.heatmap(corr_matrix, annot=True, fmt=".3f", cmap='coolwarm', vmin=-1, vmax=1,
                linewidths=0.5, cbar_kws={'label': 'Pearson Correlation Coefficient'}, ax=ax)
    ax.set_title("Correlation Matrix: Power Demand, Weather, and Temporal Variables", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    corr_plot_path = os.path.join(OUTPUT_DIR, "6_correlation_heatmap.png")
    plt.savefig(corr_plot_path)
    plt.close()
    print(f"Saved: {corr_plot_path}")

    # ==========================================
    # 10. Temporal Patterns Detailed Checks
    # ==========================================
    # Peak hour and trough hour
    peak_hour = int(hourly_overall.loc[hourly_overall['mean'].idxmax(), 'hour'])
    peak_hour_val = float(hourly_overall['mean'].max())
    trough_hour = int(hourly_overall.loc[hourly_overall['mean'].idxmin(), 'hour'])
    trough_hour_val = float(hourly_overall['mean'].min())

    # Weekday vs Weekend means
    weekday_mean = float(df[~df['is_weekend']][demand_col].mean())
    weekend_mean = float(df[df['is_weekend']][demand_col].mean())
    weekday_weekend_diff = weekday_mean - weekend_mean

    # Peak Month vs Trough Month
    monthly_means = df.groupby('month')[demand_col].mean()
    peak_month = int(monthly_means.idxmax())
    peak_month_val = float(monthly_means.max())
    trough_month = int(monthly_means.idxmin())
    trough_month_val = float(monthly_means.min())

    patterns_summary = {
        "peak_hour": peak_hour,
        "peak_hour_avg_demand": peak_hour_val,
        "trough_hour": trough_hour,
        "trough_hour_avg_demand": trough_hour_val,
        "diurnal_swing_mw": peak_hour_val - trough_hour_val,
        "weekday_avg_demand": weekday_mean,
        "weekend_avg_demand": weekend_mean,
        "weekday_weekend_diff_mw": weekday_weekend_diff,
        "weekday_weekend_diff_pct": (weekday_weekend_diff / weekday_mean) * 100,
        "peak_month": peak_month,
        "peak_month_avg_demand": peak_month_val,
        "trough_month": trough_month,
        "trough_month_avg_demand": trough_month_val,
        "seasonal_swing_mw": peak_month_val - trough_month_val
    }

    print("\n--- 10. TEMPORAL PATTERN SUMMARY ---")
    for k, v in patterns_summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Save summary as json for reporting
    results_path = os.path.join(OUTPUT_DIR, "eda_numerical_findings.json")
    results_dict = {
        "statistics": stats_summary,
        "patterns": patterns_summary,
        "correlations_with_demand": corr_matrix[demand_col].to_dict(),
        "top_10_high": top_10_high.to_dict(orient='records'),
        "top_10_low": top_10_low.to_dict(orient='records')
    }
    with open(results_path, "w") as f:
        json.dump(results_dict, f, default=str, indent=2)
    print(f"\nAll numerical findings saved to: {results_path}")
    print("EDA completed successfully!")

if __name__ == "__main__":
    run_eda()
