import data_preprocessing


if __name__ == "__main__":
    df = data_preprocessing.import_data()
    print(df.head())