# Define the years and regions
years=(2018 2019 2020 2021 2022)
regions=(seoul daegu gwangju jeonju andong)

# Loop through each year and region to copy the files
for year in "${years[@]}"; do
    for region in "${regions[@]}"; do
        cp "../../collect_data/filtered/kma/$year/kma_${year}_${region}.csv" ./
    done
done