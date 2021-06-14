import data_preprocessing
import analysis

if __name__ == "__main__":
    df = data_preprocessing.import_data()
    print(df.head())

    df = data_preprocessing.clean_errorenous_data(df)
    analysis.plot_every_col(df)

    df = data_preprocessing.remove_outliers(df)

    analysis.plot_every_col(df)