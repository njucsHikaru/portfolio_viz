import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, jsonify
from pathlib import Path
import traceback

app = Flask(__name__)

def load_and_process_data():
    # Get all CSV files from input directory
    input_dir = Path("input")
    csv_files = list(input_dir.glob("*.csv"))
    
    print(f"\nFound {len(csv_files)} CSV files in input directory:")
    for f in csv_files:
        print(f"  - {f}")
    
    all_positions = []
    account_totals = []
    
    # Define exact phrases to skip
    skip_phrases = [
        'Cash & Cash Investments',
        'Pending Activity',
        'Account Total',
        'nan'
    ]
    
    for csv_file in csv_files:
        print(f"\nProcessing file: {csv_file}")
        
        try:
            # Check if this is the 652 file
            is_652_file = '652' in str(csv_file)
            
            # Read the CSV file
            if is_652_file:
                # Skip the first 3 rows for 652 file
                df = pd.read_csv(csv_file, skiprows=3)
            else:
                df = pd.read_csv(csv_file)
            
            print(f"File columns: {df.columns.tolist()}")
            print(f"Number of rows: {len(df)}")
            
            # Process each row
            for _, row in df.iterrows():
                try:
                    # Skip rows without Symbol or with empty Symbol
                    symbol = str(row.get('Symbol', '')).strip()
                    if pd.isna(symbol) or symbol == '':
                        continue
                    
                    # Get description for checking
                    description = str(row.get('Description', row.get('Security Description', ''))).strip()
                    
                    # Skip if description exactly matches any of the skip phrases
                    if symbol in skip_phrases:
                        print(f"Skipping entry: {description}")
                        continue
                    
                    # Extract and clean numeric values
                    def clean_numeric(value):
                        if pd.isna(value) or str(value).strip() in ['--', 'N/A', '', 'N/A%']:
                            return 0.0
                        return float(str(value).replace('$', '').replace(',', '').replace('%', '').strip())
                    
                    # Function to get value from multiple possible column names
                    def get_value_from_columns(row, possible_columns, default=0):
                        for col in possible_columns:
                            if col in row and pd.notna(row[col]):
                                return clean_numeric(row[col])
                        return default
                    
                    # Define column mappings for different file formats
                    if is_652_file:
                        value_cols = ['Mkt Val (Market Value)']
                        cost_basis_cols = ['Cost Basis']
                        gain_loss_cols = ['Gain $ (Gain/Loss $)']
                        quantity_cols = ['Qty (Quantity)']
                        price_cols = ['Price']
                    else:
                        value_cols = ['Current Value', 'Mkt Val (Market Value)']
                        cost_basis_cols = ['Cost Basis Total', 'Cost Basis']
                        gain_loss_cols = ['Gain/Loss Total', 'Total Gain/Loss Dollar', 'Gain/Loss $']
                        quantity_cols = ['Quantity', 'Qty (Quantity)']
                        price_cols = ['Last Price', 'Price']
                    
                    # Create position dictionary
                    position = {
                        'symbol': symbol,
                        'description': description,
                        'type': str(row.get('Type', row.get('Security Type', 'Stock'))),
                        'value': get_value_from_columns(row, value_cols),
                        'cost_basis': get_value_from_columns(row, cost_basis_cols),
                        'gain_loss': get_value_from_columns(row, gain_loss_cols)
                    }
                    
                    # Add quantity and price
                    position.update({
                        'quantity': get_value_from_columns(row, quantity_cols),
                        'price': get_value_from_columns(row, price_cols)
                    })
                    
                    # Add to positions list
                    all_positions.append(position)
                    print(f"Added Position: {position['symbol']} - {position['description']} - Value: ${position['value']:,.2f} - Gain/Loss: ${position['gain_loss']:,.2f}")
                
                except Exception as e:
                    print(f"Error processing row: {e}")
                    print(f"Row data: {row.to_dict()}")
        
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
            traceback.print_exc()
    
    # Convert to DataFrames
    positions_df = pd.DataFrame(all_positions)
    account_totals_df = pd.DataFrame(account_totals)
    
    print(f"\nProcessed data summary:")
    print(f"Number of positions: {len(positions_df)}")
    print(f"Number of account totals: {len(account_totals_df)}")
    
    if not positions_df.empty:
        print("\nPositions DataFrame columns:", positions_df.columns.tolist())
        print("\nSample of positions:")
        print(positions_df[['symbol', 'description', 'value', 'gain_loss']].head().to_string())
    
    return positions_df, account_totals_df

def create_portfolio_overview():
    df = load_and_process_data()[0]
    
    # Clean up and standardize asset types
    type_mapping = {
        'Stock': 'Stocks',
        'Equity': 'Stocks',
        'ETF': 'ETFs',
        'CASH': 'Cash',
        'Bond': 'Bonds',
        'Fixed Income': 'Bonds',
        'Mutual Fund': 'Mutual Funds',
        'Money Market': 'Cash'
    }
    
    # Print unique types before mapping
    print("Unique types before mapping:", df['type'].unique())
    
    df['type'] = df['type'].apply(lambda x: type_mapping.get(str(x).strip(), str(x).strip()))
    
    # Print unique types after mapping
    print("Unique types after mapping:", df['type'].unique())
    
    # Calculate total portfolio value by type
    portfolio_by_type = df.groupby('type').agg({
        'value': 'sum',
        'symbol': 'count'
    }).reset_index()
    
    total_value = portfolio_by_type['value'].sum()
    portfolio_by_type['percentage'] = (portfolio_by_type['value'] / total_value * 100).round(2)
    
    # Sort by value descending
    portfolio_by_type = portfolio_by_type.sort_values('value', ascending=False)
    
    print("Portfolio by type:")
    for _, row in portfolio_by_type.iterrows():
        print(f"{row['type']}: ${row['value']:,.2f} ({row['percentage']}%), {row['symbol']} holdings")
    
    # Create pie chart with improved styling
    fig = px.pie(portfolio_by_type, 
                 values='value', 
                 names='type',
                 title='Asset Allocation',
                 custom_data=['percentage', 'symbol'])
    
    fig.update_traces(
        textposition='inside',
        texttemplate='%{label}<br>%{percent:.1%}',
        hovertemplate='<b>%{label}</b><br>' +
                     'Value: $%{value:,.2f}<br>' +
                     'Allocation: %{percent:.1%}<br>' +
                     'Holdings: %{customdata[1]}<extra></extra>'
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig.to_json()

def create_stock_distribution():
    df = load_and_process_data()[0]
    
    # Filter for stocks and ETFs
    stocks_df = df[df['type'].str.lower().isin(['stocks', 'etfs'])]
    
    print(f"\nProcessing stock distribution. Found {len(stocks_df)} stock/ETF positions")
    print("Types included:", stocks_df['type'].unique())
    
    if len(stocks_df) == 0:
        print("No stock holdings found")
        return "{}"
    
    # Group by symbol and calculate metrics
    stock_dist = stocks_df.groupby('symbol').agg({
        'value': 'sum',
        'description': 'first',
        'quantity': 'sum',
        'price': 'first',
        'type': 'first'
    }).reset_index()
    
    # Calculate percentage of total stock/ETF value
    total_stock_value = stock_dist['value'].sum()
    stock_dist['percentage'] = (stock_dist['value'] / total_stock_value * 100).round(2)
    
    # Sort by value descending
    stock_dist = stock_dist.sort_values('value', ascending=False)
    
    print("\nTop 10 holdings:")
    for _, row in stock_dist.head(10).iterrows():
        print(f"{row['symbol']} ({row['type']}): ${row['value']:,.2f} ({row['percentage']}%)")
    
    # Create treemap with improved styling
    fig = px.treemap(
        stock_dist,
        path=['type', 'symbol'],  # Group by type first, then symbol
        values='value',
        custom_data=['description', 'quantity', 'price', 'percentage'],
        title='Stock & ETF Holdings Distribution'
    )
    
    fig.update_traces(
        textinfo='label+value+percent parent',
        hovertemplate="""
            <b>%{label}</b><br>
            %{customdata[0]}<br>
            Shares: %{customdata[1]:,.0f}<br>
            Price: $%{customdata[2]:,.2f}<br>
            Value: $%{value:,.2f}<br>
            Portfolio: %{customdata[3]:.1f}%
            <extra></extra>
        """
    )
    
    fig.update_layout(
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig.to_json()

@app.route('/api/holdings')
def get_holdings():
    try:
        positions_df, account_totals_df = load_and_process_data()
        
        print("\nPreparing holdings data...")
        holdings = []
        
        # First add Account Total
        if not account_totals_df.empty:
            for _, row in account_totals_df.iterrows():
                holding = {
                    'symbol': row['symbol'],
                    'description': row['description'],
                    'type': row['type'],
                    'value': float(row['value']),
                    'cost_basis': float(row['cost_basis']),
                    'gain_loss': float(row['gain_loss']),
                    'portfolio_percentage': 100.0
                }
                holdings.append(holding)
                print(f"Added Account Total: {holding['symbol']} - Value: ${holding['value']:,.2f}")
        
        # Then add individual positions
        if not positions_df.empty:
            total_value = positions_df['value'].sum()
            print(f"Total value for percentage calculation: ${total_value:,.2f}")
            
            for _, row in positions_df.iterrows():
                try:
                    value = float(row['value'])
                    portfolio_percentage = (value / total_value * 100) if total_value > 0 else 0
                    
                    holding = {
                        'symbol': str(row['symbol']),
                        'description': str(row['description']),
                        'type': str(row['type']),
                        'quantity': float(row['quantity']) if 'quantity' in row else None,
                        'price': float(row['price']) if 'price' in row else None,
                        'value': value,
                        'cost_basis': float(row['cost_basis']),
                        'gain_loss': float(row['gain_loss']),
                        'portfolio_percentage': round(portfolio_percentage, 2)
                    }
                    holdings.append(holding)
                    print(f"Added Position: {holding['symbol']} - Value: ${holding['value']:,.2f} - {holding['portfolio_percentage']}%")
                except Exception as e:
                    print(f"Error processing position: {e}")
                    continue
        
        print(f"\nTotal holdings to return: {len(holdings)}")
        return jsonify(holdings)
    
    except Exception as e:
        print(f"Error in get_holdings: {e}")
        traceback.print_exc()
        return jsonify([])

@app.route('/api/portfolio_summary')
def get_portfolio_summary():
    try:
        positions_df, _ = load_and_process_data()
        
        print("\nCalculating Portfolio Summary:")
        
        # Calculate total portfolio value from all positions
        total_value = positions_df['value'].sum()
        print(f"Total Portfolio Value: ${total_value:,.2f}")
        
        # Calculate total gain/loss
        total_gain_loss = positions_df['gain_loss'].sum()
        print(f"Total Gain/Loss: ${total_gain_loss:,.2f}")
        
        # Calculate cash balance (if any cash positions exist)
        cash_positions = positions_df[positions_df['type'].str.contains('Cash', case=False, na=False)]
        cash_balance = cash_positions['value'].sum() if not cash_positions.empty else 0.0
        print(f"Cash Balance: ${cash_balance:,.2f}")
        
        # Print all positions for debugging
        print("\nAll positions:")
        for _, row in positions_df.iterrows():
            print(f"{row['symbol']}: Value=${row['value']:,.2f}, Gain/Loss=${row['gain_loss']:,.2f}")
        
        return jsonify({
            'total_value': float(total_value),
            'total_gain': float(total_gain_loss),
            'day_change': 0.0,  # Placeholder for now
            'cash_balance': float(cash_balance)
        })
    except Exception as e:
        print(f"Error in get_portfolio_summary: {e}")
        traceback.print_exc()
        return jsonify({
            'total_value': 0,
            'total_gain': 0,
            'day_change': 0,
            'cash_balance': 0
        })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/portfolio_overview')
def portfolio_overview():
    return jsonify(create_portfolio_overview())

@app.route('/api/stock_distribution')
def stock_distribution():
    return jsonify(create_stock_distribution())

if __name__ == '__main__':
    app.run(debug=True) 