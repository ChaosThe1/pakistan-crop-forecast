
import ee
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, recall_score, roc_auc_score,
                             confusion_matrix, classification_report,
                             precision_recall_curve)

# fill these in with your own GEE service account — see README
SERVICE_ACCOUNT = 'YOUR-SA@your-project.iam.gserviceaccount.com'
KEY_PATH = 'path/to/your-key.json'
PROJECT = 'your-gee-project'

DISTRICT_BBOXES = {
    'Bahawalpur':     [71.3, 28.8, 72.4, 29.8],
    'Rahim Yar Khan': [69.8, 28.0, 70.9, 28.9],
    'Bahawalnagar':   [72.6, 29.6, 73.6, 30.4],
    'Multan':         [71.2, 30.0, 72.0, 30.6],
    'Vehari':         [71.8, 29.9, 72.7, 30.5],
    'Lodhran':        [71.4, 29.3, 72.0, 29.9],
    'Khanewal':       [71.6, 30.1, 72.3, 30.7],
}

# yields are maund/acre, "Total" column from punjab CRS final kharif estimates
COTTON_YIELDS = {
    2017: {'Bahawalpur': 21.88, 'Rahim Yar Khan': 20.22, 'Bahawalnagar': 18.76,
           'Multan': 20.58, 'Vehari': 21.72, 'Lodhran': 19.68, 'Khanewal': 22.29},
    2018: {'Bahawalpur': 19.45, 'Rahim Yar Khan': 21.16, 'Bahawalnagar': 16.84,
           'Multan': 19.72, 'Vehari': 17.85, 'Lodhran': 21.69, 'Khanewal': 19.39},
    2019: {'Bahawalpur': 18.90, 'Rahim Yar Khan': 20.05, 'Bahawalnagar': 19.05,
           'Multan': 18.35, 'Vehari': 15.54, 'Lodhran': 17.43, 'Khanewal': 15.69},
    2020: {'Bahawalpur': 17.27, 'Rahim Yar Khan': 15.63, 'Bahawalnagar': 19.62,
           'Multan': 13.68, 'Vehari': 15.79, 'Lodhran': 15.88, 'Khanewal': 13.96},
    2021: {'Bahawalpur': 19.43, 'Rahim Yar Khan': 19.85, 'Bahawalnagar': 19.88,
           'Multan': 21.19, 'Vehari': 18.73, 'Lodhran': 23.86, 'Khanewal': 18.32},
    2022: {'Bahawalpur': 10.91, 'Rahim Yar Khan': 9.28, 'Bahawalnagar': 10.92,
           'Multan': 9.41, 'Vehari': 12.39, 'Lodhran': 11.93, 'Khanewal': 11.70},
    2023: {'Bahawalpur': 18.74, 'Rahim Yar Khan': 16.04, 'Bahawalnagar': 19.25,
           'Multan': 17.91, 'Vehari': 18.33, 'Lodhran': 16.12, 'Khanewal': 20.54},
    2024: {'Bahawalpur': 14.58, 'Rahim Yar Khan': 11.44, 'Bahawalnagar': 14.89,
           'Multan': 16.72, 'Vehari': 18.10, 'Lodhran': 15.96, 'Khanewal': 18.36},
}

# wheat harvest year = sown nov of (year-1)
WHEAT_YIELDS = {
    2018: {'Bahawalpur': 34.93, 'Rahim Yar Khan': 32.21, 'Bahawalnagar': 33.28,
           'Multan': 34.83, 'Vehari': 35.36, 'Lodhran': 37.04, 'Khanewal': 37.64},
    2019: {'Bahawalpur': 32.70, 'Rahim Yar Khan': 34.51, 'Bahawalnagar': 35.19,
           'Multan': 33.82, 'Vehari': 33.17, 'Lodhran': 36.73, 'Khanewal': 33.86},
    2020: {'Bahawalpur': 35.91, 'Rahim Yar Khan': 37.82, 'Bahawalnagar': 40.25,
           'Multan': 36.86, 'Vehari': 37.69, 'Lodhran': 36.50, 'Khanewal': 35.67},
    2021: {'Bahawalpur': 33.70, 'Rahim Yar Khan': 35.37, 'Bahawalnagar': 37.61,
           'Multan': 34.50, 'Vehari': 35.29, 'Lodhran': 34.20, 'Khanewal': 33.77},
    2022: {'Bahawalpur': 35.98, 'Rahim Yar Khan': 37.52, 'Bahawalnagar': 36.04,
           'Multan': 36.73, 'Vehari': 35.26, 'Lodhran': 38.07, 'Khanewal': 37.33},
    2023: {'Bahawalpur': 37.8, 'Rahim Yar Khan': 38.2, 'Bahawalnagar': 36.0,
           'Multan': 36.7, 'Vehari': 35.3, 'Lodhran': 38.1, 'Khanewal': 37.3},
    2024: {'Bahawalpur': 38.9, 'Rahim Yar Khan': 41.1, 'Bahawalnagar': 38.8,
           'Multan': 39.9, 'Vehari': 40.7, 'Lodhran': 40.1, 'Khanewal': 39.6},
}


def init_ee():
    creds = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, KEY_PATH)
    ee.Initialize(creds, project=PROJECT)


def get_kharif_ndvi(year, geom):
    wc = ee.ImageCollection('ESA/WorldCover/v200').first()
    mask = wc.eq(40)  # 40 = cropland

    # 40% cloud — 30 left too many gaps in 2022 monsoon
    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(f'{year}-05-01', f'{year}-12-31')
          .filterBounds(geom)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))

    def add_ndvi(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return img.addBands(ndvi.updateMask(mask)).set(
            'system:time_start', img.get('system:time_start'))

    s2_ndvi = s2.map(add_ndvi).select('NDVI')

    def make_composite(offset):
        start = ee.Date(f'{year}-05-01').advance(offset, 'day')
        end = start.advance(10, 'day')
        window = s2_ndvi.filterDate(start, end)
        # If() because empty windows break reduceRegion later
        composite = ee.Image(ee.Algorithms.If(
            window.size().gt(0),
            window.median().rename('NDVI'),
            ee.Image.constant(0).rename('NDVI').updateMask(ee.Image.constant(0))))
        return composite.set('date', start.format('YYYY-MM-dd'))

    composites = ee.List.sequence(0, 240, 10).map(make_composite)

    def extract(img):
        img = ee.Image(img)
        r = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom,
                             scale=200, maxPixels=1e9, bestEffort=True)
        ndvi = ee.Algorithms.If(r.contains('NDVI'), r.get('NDVI'), None)
        return ee.Feature(None, {'date': img.get('date'), 'ndvi': ndvi})

    return ee.FeatureCollection(composites.map(extract))


def get_rabi_ndvi(harvest_year, geom):
    # basically a copy of get_kharif_ndvi w/ different dates, didn't bother refactoring
    wc = ee.ImageCollection('ESA/WorldCover/v200').first()
    mask = wc.eq(40)

    s2 = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterDate(f'{harvest_year - 1}-11-01', f'{harvest_year}-04-30')
          .filterBounds(geom)
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)))

    def add_ndvi(img):
        ndvi = img.normalizedDifference(['B8', 'B4']).rename('NDVI')
        return img.addBands(ndvi.updateMask(mask)).set(
            'system:time_start', img.get('system:time_start'))

    s2_ndvi = s2.map(add_ndvi).select('NDVI')

    def make_composite(offset):
        start = ee.Date(f'{harvest_year - 1}-11-01').advance(offset, 'day')
        end = start.advance(10, 'day')
        window = s2_ndvi.filterDate(start, end)
        composite = ee.Image(ee.Algorithms.If(
            window.size().gt(0),
            window.median().rename('NDVI'),
            ee.Image.constant(0).rename('NDVI').updateMask(ee.Image.constant(0))))
        return composite.set('date', start.format('YYYY-MM-dd'))

    composites = ee.List.sequence(0, 170, 10).map(make_composite)

    def extract(img):
        img = ee.Image(img)
        r = img.reduceRegion(reducer=ee.Reducer.mean(), geometry=geom,
                             scale=200, maxPixels=1e9, bestEffort=True)
        ndvi = ee.Algorithms.If(r.contains('NDVI'), r.get('NDVI'), None)
        return ee.Feature(None, {'date': img.get('date'), 'ndvi': ndvi})

    return ee.FeatureCollection(composites.map(extract))


def get_era5_features(year, geom, months):
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/DAILY_AGGR")
    out = {'year': year}
    for name, start, end in months:
        coll = era5.filterDate(start, end).filterBounds(geom)
        stats = ee.Image.cat([
            coll.select('temperature_2m_max').max().rename(f'{name}_tmax_K'),
            coll.select('temperature_2m').mean().rename(f'{name}_tmean_K'),
            coll.select('total_precipitation_sum').sum().rename(f'{name}_precip_m'),
            coll.select('volumetric_soil_water_layer_1').mean().rename(f'{name}_soil_m3m3'),
        ]).reduceRegion(reducer=ee.Reducer.mean(), geometry=geom,
                        scale=10000, maxPixels=1e9, bestEffort=True)
        out.update(stats.getInfo())
    return out


def pull_all_satellite_data():
    init_ee()

    # cotton kharif NDVI
    rows = []
    for district, bbox in DISTRICT_BBOXES.items():
        geom = ee.Geometry.Rectangle(bbox)
        for y in range(2017, 2025):
            d = get_kharif_ndvi(y, geom).getInfo()
            for f in d['features']:
                p = f['properties']
                if p.get('ndvi') is not None:
                    rows.append({'district': district, 'year': y,
                                 'date': p['date'], 'ndvi': p['ndvi']})
            time.sleep(3)  # rate limit otherwise
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df['doy'] = df['date'].dt.dayofyear
    df.to_csv('cotton_kharif_ndvi.csv', index=False)

    # wheat rabi NDVI
    rows = []
    for district, bbox in DISTRICT_BBOXES.items():
        geom = ee.Geometry.Rectangle(bbox)
        for y in range(2018, 2025):
            d = get_rabi_ndvi(y, geom).getInfo()
            for f in d['features']:
                p = f['properties']
                if p.get('ndvi') is not None:
                    rows.append({'district': district, 'year': y,
                                 'date': p['date'], 'ndvi': p['ndvi']})
            time.sleep(3)
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df['date'])
    df['doy'] = df['date'].dt.dayofyear
    df.to_csv('wheat_rabi_ndvi.csv', index=False)

    # cotton weather
    kharif_months = [('may', '{Y}-05-01', '{Y}-06-01'),
                     ('jun', '{Y}-06-01', '{Y}-07-01'),
                     ('jul', '{Y}-07-01', '{Y}-08-01'),
                     ('aug', '{Y}-08-01', '{Y}-09-01'),
                     ('sep', '{Y}-09-01', '{Y}-10-01'),
                     ('oct', '{Y}-10-01', '{Y}-11-01')]
    rows = []
    for district, bbox in DISTRICT_BBOXES.items():
        geom = ee.Geometry.Rectangle(bbox)
        for y in range(2017, 2025):
            months = [(n, s.format(Y=y), e.format(Y=y)) for n, s, e in kharif_months]
            feats = get_era5_features(y, geom, months)
            feats['district'] = district
            rows.append(feats)
            time.sleep(2)
    _save_weather(rows, 'cotton_kharif_weather.csv')

    # wheat weather
    rows = []
    for district, bbox in DISTRICT_BBOXES.items():
        geom = ee.Geometry.Rectangle(bbox)
        for y in range(2018, 2025):
            months = [('dec', f'{y-1}-12-01', f'{y}-01-01'),
                      ('jan', f'{y}-01-01', f'{y}-02-01'),
                      ('feb', f'{y}-02-01', f'{y}-03-01'),
                      ('mar', f'{y}-03-01', f'{y}-04-01'),
                      ('apr', f'{y}-04-01', f'{y}-05-01')]
            feats = get_era5_features(y, geom, months)
            feats['district'] = district
            rows.append(feats)
            time.sleep(2)
    _save_weather(rows, 'wheat_rabi_weather.csv')


def _save_weather(rows, path):
    df = pd.DataFrame(rows)
    for col in list(df.columns):
        if '_tmax_K' in col or '_tmean_K' in col:
            df[col.replace('_K', '_C')] = df[col] - 273.15
            df = df.drop(columns=[col])
        if '_precip_m' in col:
            df[col.replace('_m', '_mm')] = df[col] * 1000
            df = df.drop(columns=[col])
    df.to_csv(path, index=False)


def build_dataset():
    # yields -> long form
    rows = []
    for y, dists in COTTON_YIELDS.items():
        for d, v in dists.items():
            rows.append({'year': y, 'district': d, 'crop': 'cotton', 'yield': v})
    for y, dists in WHEAT_YIELDS.items():
        for d, v in dists.items():
            rows.append({'year': y, 'district': d, 'crop': 'wheat', 'yield': v})
    yield_df = pd.DataFrame(rows)

    # cotton NDVI: aug = peak boll-formation, jul = pre-peak
    cn = pd.read_csv('cotton_kharif_ndvi.csv')
    cotton_feat = []
    for (district, year), g in cn.groupby(['district', 'year']):
        g = g.sort_values('doy')
        aug = g[(g['doy'] >= 213) & (g['doy'] <= 243)]
        jul = g[(g['doy'] >= 182) & (g['doy'] <= 212)]
        cotton_feat.append({
            'district': district, 'year': year, 'crop': 'cotton',
            'peak_month_ndvi': aug['ndvi'].mean() if len(aug) else np.nan,
            'prior_month_ndvi': jul['ndvi'].mean() if len(jul) else np.nan,
        })
    cotton_ndvi_df = pd.DataFrame(cotton_feat)

    cw = pd.read_csv('cotton_kharif_weather.csv')
    cw['crop'] = 'cotton'
    cw['peak_tmean_C'] = cw['aug_tmean_C']
    cw['season_precip_mm'] = cw['jun_precip_mm'] + cw['jul_precip_mm'] + cw['aug_precip_mm']
    cw['peak_soil_m3m3'] = cw['aug_soil_m3m3']
    cw_slim = cw[['year', 'district', 'crop', 'peak_tmean_C',
                  'season_precip_mm', 'peak_soil_m3m3']]

    # wheat NDVI: feb-mar heading stage
    wn = pd.read_csv('wheat_rabi_ndvi.csv')
    wn['date'] = pd.to_datetime(wn['date'])
    wn['month'] = wn['date'].dt.month
    wheat_feat = []
    for (district, year), g in wn.groupby(['district', 'year']):
        peak = g[g['month'].isin([2, 3])]
        prior = g[g['month'].isin([12, 1])]
        wheat_feat.append({
            'district': district, 'year': year, 'crop': 'wheat',
            'peak_month_ndvi': peak['ndvi'].mean() if len(peak) else np.nan,
            'prior_month_ndvi': prior['ndvi'].mean() if len(prior) else np.nan,
        })
    wheat_ndvi_df = pd.DataFrame(wheat_feat)

    ww = pd.read_csv('wheat_rabi_weather.csv')
    ww['crop'] = 'wheat'
    ww['peak_tmean_C'] = ww['mar_tmean_C']  # mar = grain filling, heat-sensitive
    ww['season_precip_mm'] = (ww['dec_precip_mm'] + ww['jan_precip_mm']
                              + ww['feb_precip_mm'] + ww['mar_precip_mm'])
    ww['peak_soil_m3m3'] = ww['mar_soil_m3m3']
    ww_slim = ww[['year', 'district', 'crop', 'peak_tmean_C',
                  'season_precip_mm', 'peak_soil_m3m3']]

    ndvi_all = pd.concat([cotton_ndvi_df, wheat_ndvi_df], ignore_index=True)
    weather_all = pd.concat([cw_slim, ww_slim], ignore_index=True)
    data = (yield_df.merge(ndvi_all, on=['year', 'district', 'crop'])
                    .merge(weather_all, on=['year', 'district', 'crop'])
                    .dropna()
                    .reset_index(drop=True))

    # baseline = mean of PRIOR years only — using current year would leak
    def baseline(row, df):
        prior = df[(df['district'] == row['district'])
                   & (df['crop'] == row['crop'])
                   & (df['year'] < row['year'])]
        return prior['yield'].mean() if len(prior) > 0 else np.nan

    data['baseline'] = data.apply(lambda r: baseline(r, data), axis=1)
    data = data.dropna(subset=['baseline']).copy()
    data['pct_dev'] = (data['yield'] - data['baseline']) / data['baseline']
    data['label'] = data['pct_dev'].apply(
        lambda p: 'below' if p < -0.10 else ('above' if p > 0.10 else 'normal'))
    data['binary'] = (data['label'] == 'below').astype(int)
    data['crop_code'] = (data['crop'] == 'wheat').astype(int)

    # print(data['label'].value_counts())
    return data


FEATURES = ['peak_month_ndvi', 'prior_month_ndvi',
            'peak_tmean_C', 'season_precip_mm', 'peak_soil_m3m3',
            'crop_code']


def walk_forward(df, label_col='binary', seed=42):
    all_p, all_t, all_prob = [], [], []
    for test_year in sorted(df['year'].unique()):
        train = df[df['year'] < test_year]
        test = df[df['year'] == test_year]
        if len(test) == 0 or len(train) < 10:
            continue
        m = GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                       learning_rate=0.05, random_state=seed)
        m.fit(train[FEATURES].values, train[label_col].values)
        all_p.extend(m.predict(test[FEATURES].values))
        all_prob.extend(m.predict_proba(test[FEATURES].values)[:, 1])
        all_t.extend(test[label_col].values)
    return np.array(all_p), np.array(all_t), np.array(all_prob)


def evaluate(data):
    preds, truths, probs = walk_forward(data)
    print(f"Accuracy:      {accuracy_score(truths, preds):.3f}")
    print(f"Below recall:  {recall_score(truths, preds):.3f}")
    print(f"AUC-ROC:       {roc_auc_score(truths, probs):.3f}")
    print(f"\nNaive (always not-below): {(truths == 0).mean():.3f}")
    print()
    print(classification_report(truths, preds, target_names=['not-below', 'below']))
    return preds, truths, probs


def diagnostics(data, n_seeds=10, n_permutations=20):
    # 1. seed sensitivity — does the result wobble with the RNG?
    seed_accs = []
    for seed in range(n_seeds):
        p, t, _ = walk_forward(data, seed=seed)
        seed_accs.append(accuracy_score(t, p))
    seed_accs = np.array(seed_accs)
    print(f"Seed sensitivity: mean={seed_accs.mean():.3f}  std={seed_accs.std():.3f}")

    # 2. permutation test
    rng = np.random.RandomState(42)
    shuffled_accs = []
    for _ in range(n_permutations):
        sh = data.copy()
        sh['binary'] = rng.permutation(sh['binary'].values)
        p, t, _ = walk_forward(sh)
        shuffled_accs.append(accuracy_score(t, p))
    shuffled_accs = np.array(shuffled_accs)
    p_value = (shuffled_accs >= seed_accs.mean()).mean()
    print(f"Permutation test: real={seed_accs.mean():.3f}  shuffled={shuffled_accs.mean():.3f}  p={p_value:.3f}")

    # 3. training-size scaling
    print("\nTraining-size scaling:")
    for cutoff in [2020, 2021, 2022, 2023]:
        train = data[data['year'] <= cutoff]
        test = data[data['year'] > cutoff]
        if len(test) == 0 or len(train) < 10:
            continue
        m = GradientBoostingClassifier(n_estimators=100, max_depth=2,
                                       learning_rate=0.05, random_state=42)
        m.fit(train[FEATURES].values, train['binary'].values)
        p = m.predict(test[FEATURES].values)
        t = test['binary'].values
        print(f"  train<={cutoff}: acc={accuracy_score(t, p):.3f}  recall={recall_score(t, p):.3f}")

    return seed_accs, shuffled_accs


# pull_all_satellite_data()  # run once, expensive
data = build_dataset()
print(f"Dataset: {len(data)} district-year-crop rows")
print(f"  Cotton: {(data['crop']=='cotton').sum()}")
print(f"  Wheat:  {(data['crop']=='wheat').sum()}")
print(f"  Below:  {(data['binary']==1).sum()}")
print()
evaluate(data)
print()
diagnostics(data)
