import pandas as pd
import os

def bindf(df):

    #df = pd.read_csv('data_process/'+datasetname+'.csv', index_col=0)

    # Find continuous numerical columns
    continuous_columns = []
    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            # Check if the values are continuous
            if df[column].nunique() > 15:  # Ensure more than one unique value
                continuous_columns.append(column)

    print("Continuous numerical columns:")
    print(continuous_columns)

    # Loop through each continuous column and bin into tertiles
    for column in continuous_columns:
        # Calculate tertiles dynamically
        bins, bins_edges = pd.qcut(df[column], q=3, labels=False, retbins=True)

        print(bins_edges)
        
        # Create labels for the bins
        labels = [f'{bins_edges[0]} - {bins_edges[1]}',f'{bins_edges[1]} - {bins_edges[2]}',f'{bins_edges[2]} - {bins_edges[3]}']
        
        # Assign bins and labels to a new column
        df[column] = pd.cut(df[column], bins=3, labels=labels)


    return df

    #print(df)

    #df.to_csv('data_process/' + datasetname +'.csv', index=False)
