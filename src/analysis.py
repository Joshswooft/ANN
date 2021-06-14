import matplotlib.dates as mdates
import matplotlib.pyplot as plt

def plot_month_vs_col(df, colName, seriesName):
    fig, ax = plt.subplots()
    ax.plot(df['Date'], df[colName])
    months = mdates.MonthLocator()  # every month
    ax.xaxis.set_minor_locator(months)
    fig.autofmt_xdate()
    plt.xlabel('$Date$')
    plt.ylabel('$Col$')
    plt.title(seriesName)
    plt.show()


def plot_every_col(df):
    columns = df.columns
    for col in columns:
        plot_month_vs_col(df, col, col)
