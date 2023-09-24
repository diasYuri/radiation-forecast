from sklearn.preprocessing import MinMaxScaler
from pandas import DataFrame


class Scaler:
    scales: {}
    df: DataFrame
    df_scaled: DataFrame
    scaled: bool

    def __init__(self, df: DataFrame, feature_range=(0, 1)):
        self.df = df.copy()
        self.scales = {}
        for c in df.columns:
            self.scales[c] = MinMaxScaler(feature_range=feature_range)
        self.scaled = False

    def get_dataframe_scaled(self):
        if self.scaled:
            return self.df_scaled
        self.scaled = True
        self.df_scaled = self.df.copy()
        for c in self.df.columns:
            self.df_scaled[c] = self.scales[c] \
                .fit_transform(self.df_scaled[c].values.reshape(-1, 1))
        return self.df_scaled
