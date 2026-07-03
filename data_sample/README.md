# Data Sample Policy

This project expects the NYC Motor Vehicle Collisions crash CSV to be available locally at:

```text
datasets/h9gi-nx95_2021_2024.csv
```

The full raw dataset is intentionally not committed to this repository. The pipeline reads the local CSV in place, and `.gitignore` excludes `datasets/*.csv` to reduce the risk of accidentally committing raw data.
