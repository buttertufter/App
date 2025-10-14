# How to Get Full Historical Data (35+ Quarters)

## 🔥 Quick Fix

If you're only seeing **5 quarters** of data, here's what to do:

### Option 1: Clear Cache (Already Done!)
```bash
rm -rf /workspaces/App/data_store/quarterlies/*.csv
```
✅ **I've already done this for you!** The next time you load a stock, it will fetch fresh data.

### Option 2: Use Force Refresh Checkbox
1. In the app, check the box: **"🔄 Force refresh data (ignore cache)"**
2. Click **Compute**
3. You'll now get **35-43 quarters** from Finnhub API

## 📊 What You'll See Now

### Before (Cached Yahoo Data):
- ❌ Only 5-6 quarters (2024-2025)
- ❌ Limited historical context
- ❌ Can't see long-term trends

### After (Fresh Finnhub Data):
- ✅ **35-43 quarters** (2010-2025)
- ✅ **15 years** of historical data
- ✅ Complete business cycles visible
- ✅ Long-term trends and patterns

## 🎯 New Features Added

### 1. **Prominent Cache Warning**
- Shows age of cached data
- Example: "ℹ️ Using cached data (5 days old). Check '🔄 Force refresh' for latest data."

### 2. **Limited Data Alert**
- If fewer than 10 quarters detected, you'll see:
  > ⚠️ Only 5 quarters found. Enable '🔄 Force refresh data' above to fetch more historical data from Finnhub API (~35+ quarters).

### 3. **Data Source Display**
- Expandable section showing which APIs provided data
- Example:
  ```
  ✓ Finnhub data: 35 quarters (2010-03-31 to 2025-03-31)
  ✓ Yahoo data: 5 quarters (2024-06-30 to 2025-06-30)
  ✓ Combined data: 35 quarters (2010-03-31 to 2025-03-31)
  ```

## 🔍 How the Cache Works

### Cache Lifetime: **30 days**
- Data fetched from APIs is cached for 30 days
- After 30 days, it automatically refetches
- "Force refresh" bypasses this and always fetches fresh data

### Cache Location:
```
/workspaces/App/data_store/quarterlies/[SYMBOL].csv
```

### To Manually Clear Cache:
```bash
# Clear all quarterly data
rm -rf /workspaces/App/data_store/quarterlies/*.csv

# Clear specific stock (e.g., MSFT)
rm /workspaces/App/data_store/quarterlies/MSFT.csv

# Clear all cached data
rm -rf /workspaces/App/data_store/*/*.csv
```

## 📈 What Each Source Provides

| Source | Quarters | Date Range | Status |
|--------|----------|------------|--------|
| **Finnhub** | 35-43 | 2010-2025 | ✅ Active |
| **Yahoo Finance** | 5-6 | 2024-2025 | ✅ Active |
| **FMP** | 0-400 | Varies | ❌ 403 Error (expired key) |

## 🚀 Best Practice

### For Maximum Data:
1. Select **"Max (All Data)"** from History dropdown (default)
2. Check **"🔄 Force refresh data"** checkbox
3. Click **Compute**
4. Enjoy **15 years** of historical financial analysis!

### For Quick Analysis:
1. Keep cache enabled (uncheck Force refresh)
2. Select desired time period (1y, 3y, 5y, 10y, or Max)
3. Click **Compute**
4. Uses cached data for faster load

## ⚡ Performance Tips

- **First load**: May take 5-10 seconds to fetch all data
- **Cached loads**: Instant (< 1 second)
- **Force refresh**: Recommended once per week for active trading
- **Max history**: Best for long-term fundamental analysis

## 🐛 Troubleshooting

### Still seeing only 5 quarters?
1. ✅ Check **"🔄 Force refresh data"** checkbox
2. ✅ Make sure **"Max (All Data)"** is selected
3. ✅ Click **Compute**
4. ✅ Look at terminal output to confirm Finnhub fetch

### Terminal shows "Finnhub data retrieved: 35 quarters" but UI shows 5?
- This was the caching issue - now fixed!
- The warning will alert you if this happens again

### Want even more data?
- Get a paid FMP API key (supports 400 quarters / 100 years)
- Update the key in `.env` file
- Provides even more historical depth

---

**Note**: I've already cleared your cache, so the next stock you load will automatically fetch the full Finnhub dataset (35+ quarters)! 🎉
