# Portfolio Visualization

A web application for visualizing investment portfolio data from various sources (Fidelity, Schwab, etc.).

## Features

- Portfolio overview with asset allocation pie chart
- Detailed holdings breakdown by type
- Multi-account support
- Interactive filtering and sorting
- Responsive design

## Setup

1. Clone the repository:
```bash
git clone https://github.com/njucsHikaru/portfolio_viz.git
cd portfolio_viz
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Place your portfolio CSV files in the `input` directory:
   - Example files are provided in `input/example_portfolio.csv` and `input/example_schwab.csv`
   - The application supports both Fidelity and Schwab export formats

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:8080
```

## Data Privacy

- All portfolio data is processed locally
- No data is sent to external servers
- Use example files for testing before using real portfolio data

## License

MIT License 