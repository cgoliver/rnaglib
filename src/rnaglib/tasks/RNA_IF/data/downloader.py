import pandas as pd
import gdown

# File ID and URL
file_id = '1lbdiE1LfWPReo5VnZy0zblvhVl5QhaF4'
url = f'https://drive.google.com/uc?id={file_id}'
output = 'downloaded_file.csv'

# Download the file
gdown.download(url, output, quiet=False)

# Read the CSV file with pandas
df = pd.read_csv(output)

# Display the DataFrame
print(df.head())