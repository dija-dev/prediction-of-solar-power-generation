# -*- coding: utf-8 -*-
import datetime
import json
import numpy as np
import os
import pandas as pd
import string
from collections import Counter
from pathlib import Path
from pyts.bag_of_words import BagOfWords
from pyts.classification.time_series_forest import WindowFeatureExtractor
from typing import List, Tuple


class MyDataset:
    def __init__(self, data_dir: str) -> None:
        data_dir = Path(data_dir)

        image_paths = []
        for root, dirs, files in os.walk(data_dir / "radar"):
            for file in files:
                image_paths.append(os.path.join(root, file))

        self.dict_path = {
            "pv": data_dir / "PV.csv",
            "amd": data_dir / "AMeDAS.csv",
            "gsr": data_dir / "GSR.csv",
            "radar": image_paths,
        }

        self.df_pv, self.df_amd, self.df_gsr = self.read_pv(), self.read_amd(), self.read_gsr()
        return None

    def read_pv(self) -> pd.DataFrame:
        df = pd.read_csv(self.dict_path["pv"], encoding="cp932", index_col=0, parse_dates=True)
        df = df.sort_index(ascending=True)
        df = df.set_index(df.index.date)

        assert all(df.isnull().sum()) == 0
        return df

    def read_amd(self) -> pd.DataFrame:
        def convert_to_numeric(x):
            try:
                return pd.to_numeric(x)
            except ValueError:
                return x  # 数値に変換できない場合はそのまま返す

        def convert_to_radian(x):
            dirs = [
                "北",
                "北北東",
                "北東",
                "東北東",
                "東",
                "東南東",
                "南東",
                "南南東",
                "南",
                "南南西",
                "南西",
                "西南西",
                "西",
                "西北西",
                "北西",
                "北北西",
            ]
            dir_to_rad = {el: i * (2 * np.pi) / len(dirs) for i, el in enumerate(dirs)}
            return dir_to_rad.get(x, np.nan)

        # データの読込
        df = pd.read_csv(self.dict_path["amd"], encoding="cp932", index_col=0, parse_dates=True)
        df = df.sort_index(ascending=True)  # 念のため日付順にソート

        # 日付と時間のマルチインデックスを指定
        df = df.set_index([df.index.date, df.index.time])

        df = df.drop(index=[datetime.date(2023, 9, 1)])  # 対象外の行を削除
        df = df.rename({"\nPrecipitation_mm": "Precipitation_mm"}, axis=1)  # 列名を修正
        df = df.replace({"///": np.nan, "0.0 ]": 0, "1.8 )": 1.8, "東北東 )": "東北東"})  # タイポを修正

        df = df.apply(convert_to_numeric)  # 数値に変換

        # 日付を周期的に扱う
        df["Date"] = [(el - datetime.date(el.year, 1, 1)).days + 1 for el in df.index.get_level_values(0)]
        df["Date_Rad"] = df["Date"].apply(lambda x: (x - 1) * (2 * np.pi) / 365)
        df["Date_Cos"] = df["Date_Rad"].apply(np.cos)
        df["Date_Sin"] = df["Date_Rad"].apply(np.sin)

        # 時間を周期的に扱う
        df["Time"] = [el.hour for el in df.index.get_level_values(1)]
        df["Time_Rad"] = df["Time"].apply(lambda x: x * (2 * np.pi) / 24)
        df["Time_Cos"] = df["Time_Rad"].apply(np.cos)
        df["Time_Sin"] = df["Time_Rad"].apply(np.sin)

        # 風向を周期的に扱う
        df["WindDirection_Rad"] = df["WindDirection"].apply(convert_to_radian)
        df["WindDirection_Cos"] = df["WindDirection_Rad"].apply(lambda x: np.cos(x) if x != "静穏" else 0)  # 静穏も処理
        df["WindDirection_Sin"] = df["WindDirection_Rad"].apply(lambda x: np.sin(x) if x != "静穏" else 0)  # 静穏も処理

        assert all(df.isnull().sum()) == 0
        return df

    def read_gsr(self) -> pd.DataFrame:
        df = pd.read_csv(self.dict_path["gsr"], encoding="cp932", index_col=0, parse_dates=True)
        df = df.sort_index(ascending=True)

        assert all(df.isnull().sum()) == 0
        return df

    def select_columns(
        self,
        use_columns: List[str] = None,
        exclude_columns: List[str] = None,
        **kwargs,
    ) -> List[str]:
        if use_columns is not None:
            pass
        elif exclude_columns is not None:
            use_columns = [col for col in self.df_amd.columns if col not in exclude_columns]
        elif not all([use_columns, exclude_columns]):
            exclude_columns = [
                "Date",
                "Date_Rad",
                "Date_Cos",
                "Date_Sin",
                "Time",
                "Time_Rad",
                "Time_Cos",
                "Time_Sin",
                "WindDirection",
                "WindDirection_Rad",
            ]
            use_columns = [col for col in self.df_amd.columns if col not in exclude_columns]
        return use_columns

    def select_dates(
        self,
        use_dates: List[int] = None,
        exclude_dates: List[int] = None,
        **kwargs,
    ) -> List[datetime.date]:
        if use_dates is not None:
            pass
        elif exclude_dates is not None:
            use_dates = [d for d in self.df_amd.index.get_level_values(0).unique() if d not in exclude_dates]
        elif not all([use_dates, exclude_dates]):
            exclude_dates = []
            use_dates = [d for d in self.df_amd.index.get_level_values(0).unique() if d not in exclude_dates]
        return use_dates

    def select_hours(
        self,
        use_hours: List[int] = None,
        exclude_hours: List[int] = None,
        **kwargs,
    ) -> List[datetime.time]:
        if use_hours is not None:
            pass
        elif exclude_hours is not None:
            use_hours = [datetime.time(hour=h) for h in range(24) if h not in exclude_hours]
        elif not all([use_hours, exclude_hours]):
            exclude_hours = [0, 1, 2, 3, 21, 22, 23]
            use_hours = [datetime.time(hour=h) for h in range(24) if h not in exclude_hours]
        return use_hours

    def apply_hour(self, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        use_columns = self.select_columns(**kwargs)
        use_dates, use_hours = self.select_dates(**kwargs), self.select_hours(**kwargs)

        X = np.zeros(shape=(len(use_dates), len(use_columns), len(use_hours)))
        for i, d in enumerate(use_dates):
            hourly = self.df_amd.loc[d, use_columns]

            no_hours = set(use_hours) - set(hourly.index)
            for h in list(no_hours):
                hourly.loc[h, :] = np.nan

            hourly = hourly.fillna(method="ffill", axis=0).fillna(method="bfill", axis=0)
            hourly = hourly.loc[use_hours, :].sort_index(ascending=True)
            X[i] = hourly.T.values

        columns = []
        for h in use_hours:
            for col in use_columns:
                columns.append(f"{str(h.hour).zfill(2)}_{col}")

        return X, columns, use_dates

    def apply_stat(self, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        use_columns = self.select_columns(**kwargs)
        use_dates, use_hours = self.select_dates(**kwargs), self.select_hours(**kwargs)

        use_stats = kwargs.get("use_stats", ["mean", "std", "min", "max"])
        if isinstance(use_stats, str):
            use_stats = json.loads(use_stats)

        X = np.zeros(shape=(len(use_dates), len(use_columns), len(use_stats)))
        for i, d in enumerate(use_dates):
            hourly = self.df_amd.loc[d, use_columns]

            no_hours = set(use_hours) - set(hourly.index)
            for h in list(no_hours):
                hourly.loc[h, :] = np.nan

            hourly = hourly.fillna(method="ffill", axis=0).fillna(method="bfill", axis=0)
            hourly = hourly.loc[use_hours, :].sort_index(ascending=True)

            X[i] = hourly.describe().loc[use_stats, :].T.values

        columns = []
        for col in use_columns:
            for s in use_stats:
                columns.append(f"{s.capitalize()}_{col}")

        return X, columns, use_dates

    def apply_wfe(self, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        use_columns = self.select_columns(**kwargs)
        use_dates, use_hours = self.select_dates(**kwargs), self.select_hours(**kwargs)

        wfe = WindowFeatureExtractor(**kwargs)
        wfe_features = ["mean", "std", "slope"]

        X = []
        for i, d in enumerate(use_dates):
            hourly = self.df_amd.loc[d, use_columns]

            no_hours = set(use_hours) - set(hourly.index)
            for h in list(no_hours):
                hourly.loc[h, :] = np.nan

            hourly = hourly.fillna(method="ffill", axis=0).fillna(method="bfill", axis=0)
            hourly = hourly.loc[use_hours, :].sort_index(ascending=True)

            if i == 0:
                wfe.fit(hourly.T.values)
            X.append(wfe.transform(hourly.T.values))

        columns = []
        for col in use_columns:
            for f in wfe_features:
                for n in range(len(wfe.indices_)):
                    columns.append(f"{f.capitalize()}_{n}_{col}")

        return np.array(X), columns, use_dates

    def apply_bow(self, **kwargs) -> Tuple[pd.DataFrame, List[str]]:
        use_columns = self.select_columns(**kwargs)
        use_dates, use_hours = self.select_dates(**kwargs), self.select_hours(**kwargs)

        bow = BagOfWords(**kwargs)
        use_words = list(string.ascii_lowercase)[: bow.n_bins]

        X = np.zeros(shape=(len(use_dates), len(use_columns), len(use_words)))
        for i, col in enumerate(use_columns):
            variable = self.df_amd.loc[:, [col]].unstack(level=1).fillna(method="ffill", axis=1).fillna(method="bfill", axis=1)

            texts = bow.transform(variable.values)
            uniques = [Counter(el.replace(" ", "")) for el in texts]
            uniques = [dict(sorted(el.items(), key=lambda x: x[0])) for el in uniques]
            uniques = [[el.get(key, 0) for key in use_words] for el in uniques]

            X[:, i, :] = np.array(uniques)

        columns = []
        for col in use_columns:
            for w in use_words:
                columns.append(f"Word_{w}_{col}")

        return X, columns, use_dates

    def get_dataset(self, return_type: str = "hour", **kwargs) -> pd.DataFrame:
        if return_type == "hour":
            X, columns, dates = self.apply_hour(**kwargs)
        elif return_type == "stat":
            X, columns, dates = self.apply_stat(**kwargs)
        elif return_type == "wfe":
            X, columns, dates = self.apply_wfe(**kwargs)
        elif return_type == "bow":
            X, columns, dates = self.apply_bow(**kwargs)

        Y = np.zeros(shape=(X.shape[0], X.shape[1] * X.shape[2]))
        for i in range(X.shape[1]):
            for j in range(X.shape[2]):
                Y[:, X.shape[2] * i + j] = X[:, i, j]

        df = pd.DataFrame(Y, columns=columns, index=dates)

        # 列を追加
        df["Date_Cos"] = self.df_amd["Date_Cos"].groupby(level=0).max()
        df["Date_Sin"] = self.df_amd["Date_Sin"].groupby(level=0).max()

        df = df.reindex(columns=sorted(df.columns))
        return df
