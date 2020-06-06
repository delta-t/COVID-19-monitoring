import pandas as pd
from datetime import timedelta, datetime
import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


class CovidMonitoring:
    def __init__(self, url: str) -> None:
        self.url = url
        self.confirmed = None
        self.recovered = None
        self.deaths = None
        self.confirmed_daily = None
        self.deaths_daily = None
        self.recovered_daily = None
        self.last_date = None
        self.countries = None
        self.smooth_confirmed_daily = None
        self.confirmed_total_ratio = None
        self.confirmed_top = None
        self.l_confirmed = None
        self.l_recovered = None
        self.l_deaths = None
        self.label_confirmed = None
        self.label_recovered = None
        self.label_deaths = None

    def get_data(self):
        self.confirmed = pd.read_csv(self.url + 'time_series_covid19_confirmed_global.csv', sep=',')
        self.deaths = pd.read_csv(self.url + 'time_series_covid19_deaths_global.csv', sep=',')
        self.recovered = pd.read_csv(self.url + 'time_series_covid19_recovered_global.csv', sep=',')

    def draw_graphics(self):
        self.get_data()

        # Reconstruct dates to 'dd.mm.yy'
        new_cols = list(self.confirmed.columns[:4]) + list(
            self.confirmed.columns[4:].map(lambda x: '{0:02d}.{1:02d}.{2:d}'.format(int(x.split(sep='/')[1]),
                                                                                    int(x.split(sep='/')[0]),
                                                                                    int(x.split(sep='/')[2]))))
        self.confirmed.columns = new_cols
        self.recovered.columns = new_cols
        self.deaths.columns = new_cols

        self.confirmed_daily = self.confirmed.copy()
        self.confirmed_daily.iloc[:, 4:] = self.confirmed_daily.iloc[:, 4:].diff(axis=1)

        self.deaths_daily = self.deaths.copy()
        self.deaths_daily.iloc[:, 4:] = self.deaths_daily.iloc[:, 4:].diff(axis=1)

        self.recovered_daily = self.recovered.copy()
        self.recovered_daily.iloc[:, 4:] = self.recovered_daily.iloc[:, 4:].diff(axis=1)

        self.smooth_confirmed_daily = self.confirmed_daily.copy()
        self.smooth_confirmed_daily.iloc[:, 4:] = self.smooth_confirmed_daily.iloc[:, 4:].rolling(window=8,
                                                                                                  min_periods=2,
                                                                                                  center=True,
                                                                                                  axis=1).mean()
        self.smooth_confirmed_daily.iloc[:, 4:] = self.smooth_confirmed_daily.iloc[:, 4:].round(1)

        self.last_date = self.confirmed.columns[-1]

        # top-20 countries with max confirmed cases
        self.confirmed_top = self.confirmed.iloc[:, [1, -1]].groupby('Country/Region').sum().sort_values(self.last_date,
                                                                                                         ascending=False).head(
            20)
        self.countries = list(self.confirmed_top.index)

        self.confirmed_total_ratio = self.confirmed_top.sum() / self.confirmed.iloc[:, 4:].sum()[-1]

        self.l_confirmed = 'Infected at ' + self.last_date + ' - ' + str(self.confirmed.iloc[:, 4:].sum()[-1])
        self.l_recovered = 'Recovered at ' + self.last_date + ' - ' + str(self.recovered.iloc[:, 4:].sum()[-1])
        self.l_deaths = 'Dead at ' + self.last_date + ' - ' + str(self.deaths.iloc[:, 4:].sum()[-1])

        fig, ax = plt.subplots(figsize=[20, 10])
        ax.plot(self.confirmed.iloc[:, 4:].sum(), '-', alpha=0.6, color='orange', label=self.l_confirmed)
        ax.plot(self.recovered.iloc[:, 4:].sum(), '-', alpha=0.6, color='green', label=self.l_recovered)
        ax.plot(self.deaths.iloc[:, 4:].sum(), '-', alpha=0.6, color='red', label=self.l_deaths)

        ax.legend(loc='upper left', prop=dict(size=12))
        ax.xaxis.grid(which='minor')
        ax.yaxis.grid()
        ax.tick_params(axis='x', labelrotation=90)
        plt.title(
            'COVID-19 in all countries. Top 20 countries consists {:.2%} of total confirmed infected cases'.format(
                self.confirmed_total_ratio[0]))
        plt.show()
        fig.savefig('total_cases.png')

        self.confirmed_top = self.confirmed_top.rename(columns={str(self.last_date): 'total_confirmed'})
        dates = [i for i in self.confirmed.columns[4:]]

        for country in self.countries:
            self.confirmed_top.loc[country, 'total_confirmed'] = \
                self.confirmed.loc[self.confirmed['Country/Region'] == country, :].sum()[4:][
                    -1]
            self.confirmed_top.loc[country, 'last_day_confirmed'] = \
                self.confirmed.loc[self.confirmed['Country/Region'] == country, :].sum()[
                    -2]
            self.confirmed_top.loc[country, 'mortality, %'] = round(
                self.deaths.loc[self.deaths['Country/Region'] == country, :].sum()[4:][-1] /
                self.confirmed.loc[self.confirmed['Country/Region'] == country, :].sum()[4:][-1] * 100, 1)

            smoothed_confirmed_max = round(
                self.smooth_confirmed_daily[self.smooth_confirmed_daily['Country/Region'] == country].iloc[:,
                4:].sum().max(), 2)
            peak_date = self.smooth_confirmed_daily[self.smooth_confirmed_daily['Country/Region'] == country].iloc[:,
                        4:].sum().idxmax()
            peak_day = dates.index(peak_date)

            start_day = (self.smooth_confirmed_daily[self.smooth_confirmed_daily['Country/Region'] == country].iloc[:,
                         4:].sum() < smoothed_confirmed_max / 100).sum()
            start_date = dates[start_day - 1]

            self.confirmed_top.loc[country, 'trend_max'] = smoothed_confirmed_max
            self.confirmed_top.loc[country, 'start_date'] = start_date
            self.confirmed_top.loc[country, 'peak_date'] = peak_date
            self.confirmed_top.loc[country, 'peak_passed'] = round(
                self.smooth_confirmed_daily.loc[
                    self.smooth_confirmed_daily['Country/Region'] == country, self.last_date].sum(),
                2) != smoothed_confirmed_max
            self.confirmed_top.loc[country, 'days_to_peak'] = peak_day - start_day

            if self.confirmed_top.loc[country, 'peak_passed']:
                self.confirmed_top.loc[country, 'end_date'] = (
                        datetime.strptime(self.confirmed_top.loc[country, 'peak_date'] + '20',
                                          '%d.%m.%Y').date() + timedelta(
                    self.confirmed_top.loc[country, 'days_to_peak'])).strftime('%d.%m%Y')

            self.label_confirmed = 'Infected at ' + self.last_date + ' - ' + str(
                self.confirmed.loc[self.confirmed['Country/Region'] == country, :].sum()[-1])
            self.label_recovered = 'Recovered at ' + self.last_date + ' - ' + str(
                self.recovered.loc[self.recovered['Country/Region'] == country, :].sum()[-1])
            self.label_deaths = 'Dead at ' + self.last_date + ' - ' + str(
                self.deaths.loc[self.deaths['Country/Region'] == country, :].sum()[-1])

            self.plot_by_country(country)

    def plot_by_country(self, country='Russia'):
        df = pd.DataFrame(self.confirmed_daily.loc[self.confirmed_daily['Country/Region'] == country, :].sum()[4:])
        df.columns = ['confirmed_daily']
        df['recovered_daily'] = self.recovered_daily.loc[self.recovered_daily['Country/Region'] == country, :].sum()[4:]
        df['deaths_daily'] = self.deaths_daily.loc[self.deaths_daily['Country/Region'] == country, :].sum()[4:]
        df['smooth_confirmed_daily'] = self.smooth_confirmed_daily.loc[
                                       self.smooth_confirmed_daily['Country/Region'] == country,
                                       :].sum()[4:]

        fig, ax = plt.subplots(figsize=[16, 6], nrows=1)
        plt.title(f'COVID-19 dynamics daily in {country}')

        ax.bar(df.index, df.confirmed_daily, alpha=0.5, color='orange', label=self.label_confirmed)
        ax.bar(df.index, df.recovered_daily, alpha=0.6, color='green', label=self.label_recovered)
        ax.bar(df.index, df.deaths_daily, alpha=0.7, color='red', label=self.label_deaths)
        ax.plot(df.index, df.smooth_confirmed_daily, alpha=0.7, color='black')

        start_date = self.confirmed_top[self.confirmed_top.index == country].start_date.iloc[0]
        start_point = self.smooth_confirmed_daily.loc[
            self.smooth_confirmed_daily['Country/Region'] == country, start_date].sum()
        ax.plot_date(start_date, start_point, 'o', alpha=0.7, color='black')
        shift = self.confirmed_top.loc[self.confirmed_top.index == country, 'trend_max'].iloc[0] / 40
        plt.text(start_date, start_point + shift, f'Start at {start_date}', ha='right', fontsize=20)

        peak_date = self.confirmed_top[self.confirmed_top.index == country].peak_date.iloc[0]
        peak_point = self.smooth_confirmed_daily.loc[
            self.smooth_confirmed_daily['Country/Region'] == country, peak_date].sum()
        ax.plot_date(peak_date, peak_point, 'o', alpha=0.7, color='black')
        plt.text(peak_date, peak_point + shift, f'Peak at {peak_date}', ha='right', fontsize=20)

        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        ax.tick_params(axis='x', labelrotation=90)
        ax.legend(loc='upper left', prop=dict(size=12))

        max_pos = max(df['confirmed_daily'].max(), df['recovered_daily'].max())
        if self.confirmed_top[self.confirmed_top.index == country].peak_passed.iloc[0]:
            estimation = f'peak is passed {self.confirmed_top[self.confirmed_top.index == country].end_date.iloc[0]}'
        else:
            estimation = 'peak is not passed'
        plt.text(df.index[len(df.index) // 2], 3 * max_pos // 4, country, ha='center', fontsize=50)
        plt.text(df.index[len(df.index) // 2], 2 * max_pos // 3, estimation, ha='center', fontsize=20)

        plt.show()
        fig.savefig(f'{country} statistics.png')


if __name__ == '__main__':
    data = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
    monitoring = CovidMonitoring(data)
    monitoring.draw_graphics()
