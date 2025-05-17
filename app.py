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
    
    print(f"Found {len(csv_files)} CSV files in input directory")
    
    all_positions = []
    account_totals = []
    
    for csv_file in csv_files:
        print(f"\nProcessing file: {csv_file}")
        
        try:
            # Try reading first few lines to check format
            with open(csv_file, 'r') as f:
                first_line = f.readline().strip()
            
            if "Account Name" in first_line:  # Positions file format
                df = pd.read_csv(csv_file)
                print(f"Processing as positions file. Columns: {df.columns}")
                
                # First, get Account Total
                total_rows = df[df['Description'].str.contains('Account Total', na=False)]
                for _, row in total_rows.iterrows():
                    try:
                        total = {
                            'symbol': 'TOTAL',
                            'description': 'Account Total',
                            'type': 'Account Total',
                            'value': float(str(row['Current Value']).replace('$', '').replace(',', '')) if pd.notna(row['Current Value']) else 0.0,
                            'cost_basis': float(str(row['Cost Basis Total']).replace('$', '').replace(',', '')) if pd.notna(row['Cost Basis Total']) else 0.0,
                            'gain_loss': float(str(row.get('Gain/Loss Total', 0)).replace('$', '').replace(',', '')) if pd.notna(row.get('Gain/Loss Total')) else 0.0
                        }
                        account_totals.append(total)
                        print(f"Found account total: {total}")
                    except Exception as e:
                        print(f"Error processing account total: {e}")
                
                # Then process individual positions
                position_rows = df[~df['Description'].str.contains('Account Total', na=False)]
                for _, row in position_rows.iterrows():
                    try:
                        if pd.notna(row['Symbol']):  # Include all positions, including cash
                            position = {
                                'symbol': str(row['Symbol']),
                                'description': str(row.get('Description', '')),
                                'type': str(row['Type']) if pd.notna(row['Type']) else 'Stock',
                                'value': float(str(row['Current Value']).replace('$', '').replace(',', '')) if pd.notna(row['Current Value']) else 0.0,
                                'cost_basis': float(str(row['Cost Basis Total']).replace('$', '').replace(',', '')) if pd.notna(row['Cost Basis Total']) else 0.0,
                                'gain_loss': float(str(row.get('Gain/Loss Total', 0)).replace('$', '').replace(',', '')) if pd.notna(row.get('Gain/Loss Total')) else 0.0,
                                'quantity': float(str(row['Quantity']).replace(',', '')) if pd.notna(row['Quantity']) else 0,
                                'price': float(str(row['Last Price']).replace('$', '').replace(',', '')) if pd.notna(row['Last Price']) else 0.0
                            }
                            all_positions.append(position)
                    except Exception as e:
                        print(f"Error processing position row: {e}")
            
            else:  # Account file format
                df = pd.read_csv(csv_file, skiprows=3)
                print(f"Processing as account file. Columns: {df.columns}")
                
                # Process all rows
                for _, row in df.iterrows():
                    try:
                        if pd.notna(row['Symbol']):
                            is_total = 'Account Total' in str(row.get('Security Description', ''))
                            
                            # Common data processing
                            value_str = str(row['Mkt Val (Market Value)']).strip().replace('$', '').replace(',', '')
                            cost_basis_str = str(row['Cost Basis']).strip().replace('$', '').replace(',', '')
                            gain_loss_str = str(row.get('Gain/Loss', 0)).strip().replace('$', '').replace(',', '')
                            
                            position = {
                                'symbol': 'TOTAL' if is_total else str(row['Symbol']),
                                'description': 'Account Total' if is_total else str(row.get('Security Description', '')),
                                'type': 'Account Total' if is_total else str(row['Security Type']),
                                'value': float(value_str) if value_str not in ['--', 'N/A'] else 0.0,
                                'cost_basis': float(cost_basis_str) if cost_basis_str not in ['--', 'N/A'] else 0.0,
                                'gain_loss': float(gain_loss_str) if gain_loss_str not in ['--', 'N/A'] else 0.0
                            }
                            
                            if not is_total:
                                position.update({
                                    'quantity': float(str(row['Qty (Quantity)']).strip().replace(',', '')) if pd.notna(row['Qty (Quantity)']) else 0,
                                    'price': float(str(row['Price']).strip().replace('$', '').replace(',', '')) if pd.notna(row['Price']) else 0.0
                                })
                                all_positions.append(position)
                            else:
                                account_totals.append(position)
                                
                    except Exception as e:
                        print(f"Error processing row: {e}")
        
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
    
    positions_df = pd.DataFrame(all_positions)
    account_totals_df = pd.DataFrame(account_totals)
    
    print(f"\nProcessed {len(positions_df)} positions and {len(account_totals_df)} account totals")
    
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
        
        # Debug prints
        print("\nAccount Totals:")
        print(account_totals_df.to_string())
        
        print("\nPositions DataFrame Shape:", positions_df.shape)
        print("\nSample of Positions:")
        if not positions_df.empty:
            print(positions_df[['symbol', 'type', 'value', 'cost_basis', 'gain_loss']].head().to_string())
        
        holdings = []
        
        # Add Account Total first
        for _, row in account_totals_df.iterrows():
            holding = {
                'symbol': row['symbol'],
                'description': row['description'],
                'type': row['type'],
                'quantity': None,
                'price': None,
                'value': float(row['value']),
                'cost_basis': float(row['cost_basis']),
                'gain_loss': float(row['gain_loss']),
                'portfolio_percentage': 100.0
            }
            holdings.append(holding)
            print(f"\nAccount Total added: {holding}")
        
        # Calculate percentages for individual positions using Account Total value
        total_value = account_totals_df['value'].sum()
        
        # Then add individual positions
        for _, row in positions_df.iterrows():
            try:
                value = float(row['value']) if pd.notna(row['value']) else 0.0
                portfolio_percentage = (value / total_value * 100) if total_value > 0 else 0
                
                holding = {
                    'symbol': str(row['symbol']),
                    'description': str(row['description']),
                    'type': str(row['type']),
                    'quantity': float(row['quantity']) if pd.notna(row['quantity']) else None,
                    'price': float(row['price']) if pd.notna(row['price']) else None,
                    'value': value,
                    'cost_basis': float(row['cost_basis']) if pd.notna(row['cost_basis']) else 0.0,
                    'gain_loss': float(row['gain_loss']) if pd.notna(row['gain_loss']) else 0.0,
                    'portfolio_percentage': round(portfolio_percentage, 2)
                }
                holdings.append(holding)
            except Exception as e:
                print(f"Error processing individual holding: {e}")
                continue
        
        print(f"\nTotal holdings to return: {len(holdings)}")
        if holdings:
            print("First holding:", holdings[0])
            print("Last holding:", holdings[-1])
        
        return jsonify(holdings)
    except Exception as e:
        print(f"Error in get_holdings: {e}")
        traceback.print_exc()
        return jsonify([])

@app.route('/api/portfolio_summary')
def get_portfolio_summary():
    try:
        _, account_totals_df = load_and_process_data()
        
        # Debug prints
        print("\nPortfolio Summary Calculation:")
        print("Account Totals DataFrame:")
        print(account_totals_df.to_string())
        
        # Use Account Total values directly
        total_value = account_totals_df['value'].sum()
        total_gain_loss = account_totals_df['gain_loss'].sum()
        
        print(f"Total Value from Account Total: ${total_value:,.2f}")
        print(f"Total Gain/Loss from Account Total: ${total_gain_loss:,.2f}")
        
        # Calculate cash balance
        positions_df, _ = load_and_process_data()
        cash_positions = positions_df[positions_df['symbol'].isin(['SPAXX**', 'FDRXX**'])]
        cash_balance = cash_positions['value'].sum()
        
        print("\nCash Positions:")
        print(cash_positions[['symbol', 'value']].to_string())
        print(f"Total Cash Balance: ${cash_balance:,.2f}")
        
        return jsonify({
            'total_value': float(total_value),
            'total_gain': float(total_gain_loss),
            'day_change': 0.0,  # Placeholder
            'cash_balance': float(cash_balance)
        })
    except Exception as e:
        print(f"Error in get_portfolio_summary: {e}")
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