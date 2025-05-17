# Portfolio Visualization

A web application for visualizing investment portfolio data from various sources (Fidelity, Schwab, etc.).
<img width="1690" alt="image" src="https://github.com/user-attachments/assets/1be5b999-4d94-4e78-b417-ab58a07ba14c" />

## Features

- Portfolio overview with asset allocation pie chart
- Detailed holdings breakdown by type
- Multi-account support
- Interactive filtering and sorting
- Responsive design
- Web-based file upload interface
- File management system with backup support

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

3. [Optional] Place your portfolio CSV files in the `input` directory:
   - Example files are provided in `input/example_portfolio.csv` and `input/schwab.csv`
   - The application supports both Fidelity and Schwab export formats
   - Please rename your Schwab file to schwab.csv

4. Run the application:
```bash
python app.py
```

5. Open your browser and navigate to:
```
http://localhost:8080
```

## File Management

- Upload portfolio files directly through the web interface
- Supports both Fidelity and Schwab CSV formats
- Automatic file backup before any changes
- View and manage uploaded files
- Maximum file size: 16MB
- File list shows size and last modified date
- One-click file deletion with confirmation

## Data Privacy

- All portfolio data is processed locally
- No data is sent to external servers
- Use example files for testing before using real portfolio data
- Automatic backup of files before deletion

## License

MIT License 
