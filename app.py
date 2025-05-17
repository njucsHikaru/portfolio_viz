import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import Flask, render_template, jsonify, request, send_from_directory
from pathlib import Path
import traceback
import os
from werkzeug.utils import secure_filename
import shutil
from datetime import datetime

app = Flask(__name__)

# Configure upload settings
UPLOAD_FOLDER = Path('input')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def backup_existing_files():
    """Backup existing CSV files to a backup directory"""
    backup_dir = UPLOAD_FOLDER / 'backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    for csv_file in UPLOAD_FOLDER.glob('*.csv'):
        if not csv_file.name.startswith('example_'):
            shutil.copy2(csv_file, backup_dir / csv_file.name)

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        file_type = request.form.get('file_type', 'fidelity')
        
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Please upload a CSV file.'}), 400
        
        # Backup existing files
        backup_existing_files()
        
        # Save the file with appropriate name based on type
        if file_type == 'fidelity':
            filename = 'Portfolio_Positions.csv'
        else:
            filename = 'schwab.csv'
        
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        return jsonify({
            'message': 'File uploaded successfully. Portfolio data has been updated.',
            'filename': filename
        })
        
    except Exception as e:
        print(f"Error in upload_file: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

def standardize_type(type_str):
    """Standardize asset type names"""
    type_mapping = {
        'Stock': 'Stocks',
        'Equity': 'Stocks',
        'ETF': 'ETFs',
        'ETFs & Closed End Funds': 'ETFs',
        'CASH': 'Cash',
        'Bond': 'Bonds',
        'Fixed Income': 'Bonds',
        'Mutual Fund': 'Mutual Funds',
        'Money Market': 'Cash',
        'Cash and Money Market': 'Cash',
        'Security Type': 'Unknown'
    }
    return type_mapping.get(str(type_str).strip(), type_str)

def determine_asset_type(row):
    symbol = str(row['symbol']).upper()
    description = str(row['description']).upper()
    
    # Cash/Money Market
    if any(cash_sym in symbol for cash_sym in ['SPAXX', 'FDRXX', 'SNSXX', 'SWVXX']):
        return 'Cash'
    
    # ETFs - Common ETF symbols and keywords
    etf_keywords = ['ETF', 'ISHARES', 'VANGUARD ETF', 'SPDR', 'PROSHARES', 'INVESCO']
    if (symbol in ['VOO', 'QQQ', 'TLT', 'IBIT', 'VTI', 'SPY', 'IVV', 'VEA', 'BND', 'VWO'] or
        any(keyword in description for keyword in etf_keywords)):
        return 'ETFs'
    
    # Mutual Funds - Common keywords and patterns
    mutual_fund_keywords = ['FUND', 'VANGUARD TARGET', 'INDEX FUND', 'FIDELITY', 'BALANCED FUND']
    if (any(keyword in description for keyword in mutual_fund_keywords) or
        (len(symbol) == 5 and symbol.endswith('X'))):  # Common mutual fund symbol pattern
        return 'Mutual Funds'
    
    # Bonds - Treasury and corporate bond patterns
    if (symbol.startswith('91282') or  # Treasury bond pattern
        'TREASURY' in description or
        'BOND' in description or
        'NOTE' in description or
        'GOVT' in description):
        return 'Bonds'
    
    # Everything else is considered as Stocks
    return 'Stocks'

def load_and_process_data():
    # Get all CSV files from input directory
    input_dir = Path("input")
    csv_files = list(input_dir.glob("*.csv"))
    
    print(f"\nFound {len(csv_files)} CSV files in input directory:")
    for f in csv_files:
        print(f"  - {f.stem}: {f}")
    
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
        account_name = csv_file.stem  # Get filename without extension
        print(f"\nProcessing file for {account_name}: {csv_file}")
        
        try:
            # Check if this is the 652 file
            is_schwab_file = 'schwab' in str(csv_file)
            
            # Read the CSV file
            if is_schwab_file:
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
                    if is_schwab_file:
                        value_cols = ['Mkt Val (Market Value)']
                        cost_basis_cols = ['Cost Basis']
                        gain_loss_cols = ['Gain $ (Gain/Loss $)']
                        quantity_cols = ['Qty (Quantity)']
                        price_cols = ['Price']
                        type_cols = ['Security Type']
                        account_cols = ['Account Name']
                    else:
                        value_cols = ['Current Value']
                        cost_basis_cols = ['Cost Basis Total']
                        gain_loss_cols = ['Total Gain/Loss Dollar']
                        quantity_cols = ['Quantity']
                        price_cols = ['Last Price']
                        type_cols = ['Type']
                        account_cols = ['Account Name']
                    
                    # Determine initial type and account
                    initial_type = str(row.get(type_cols[0], 'Unknown'))
                    account_name = str(row.get(account_cols[0], csv_file.stem))
                    
                    # Create position dictionary
                    position = {
                        'symbol': symbol,
                        'description': description,
                        'type': initial_type,  # We'll standardize this later
                        'value': get_value_from_columns(row, value_cols),
                        'cost_basis': get_value_from_columns(row, cost_basis_cols),
                        'gain_loss': get_value_from_columns(row, gain_loss_cols),
                        'quantity': get_value_from_columns(row, quantity_cols),
                        'price': get_value_from_columns(row, price_cols),
                        'account': account_name  # Use account name from file if available
                    }
                    
                    # Only add positions with valid values
                    if position['value'] > 0:
                        # Add to positions list
                        all_positions.append(position)
                        print(f"Added Position: {position['symbol']} - {position['description']} - Type: {position['type']} - Account: {position['account']} - Value: ${position['value']:,.2f}")
                
                except Exception as e:
                    print(f"Error processing row: {e}")
                    print(f"Row data: {row.to_dict()}")
        
        except Exception as e:
            print(f"Error processing file {csv_file}: {e}")
            traceback.print_exc()
    
    # Convert to DataFrame
    positions_df = pd.DataFrame(all_positions)
    
    # Print the data before merging
    print("\nBefore merging - number of positions:", len(positions_df))
    
    # Group by account and symbol to identify duplicates
    duplicates = positions_df.groupby(['account', 'symbol']).size()
    duplicates = duplicates[duplicates > 1]
    if not duplicates.empty:
        print("\nFound duplicate positions to merge:")
        for (account, symbol), count in duplicates.items():
            print(f"{account} - {symbol}: {count} positions")
    
    # Merge positions with the same symbol within the same account
    merged_positions = []
    for (account, symbol), group in positions_df.groupby(['account', 'symbol']):
        if len(group) > 1:
            # Sum numeric values
            total_value = group['value'].sum()
            total_cost_basis = group['cost_basis'].sum()
            total_gain_loss = group['gain_loss'].sum()
            total_quantity = group['quantity'].sum()
            
            # Use the first row for non-numeric fields
            merged_position = {
                'account': account,
                'symbol': symbol,
                'description': group.iloc[0]['description'],
                'type': group.iloc[0]['type'],
                'price': group.iloc[0]['price'],  # Use the latest price
                'value': total_value,
                'cost_basis': total_cost_basis,
                'gain_loss': total_gain_loss,
                'quantity': total_quantity
            }
            merged_positions.append(merged_position)
            print(f"\nMerged positions for {account} - {symbol}:")
            print(f"Total Value: ${total_value:,.2f}")
            print(f"Total Quantity: {total_quantity:,.2f}")
        else:
            # Single position, no need to merge
            merged_positions.append(group.iloc[0].to_dict())
    
    # Convert merged positions back to DataFrame
    positions_df = pd.DataFrame(merged_positions)
    
    # Print the data after merging
    print("\nAfter merging - number of positions:", len(positions_df))
    
    # Apply type determination
    positions_df['type'] = positions_df.apply(determine_asset_type, axis=1)
    
    print(f"\nProcessed data summary:")
    print(f"Number of positions: {len(positions_df)}")
    
    if not positions_df.empty:
        print("\nPositions DataFrame columns:", positions_df.columns.tolist())
        print("\nSample of positions:")
        print(positions_df[['symbol', 'description', 'type', 'account', 'value', 'gain_loss']].head().to_string())
        print("\nUnique types after processing:", positions_df['type'].unique())
        
        # Print type distribution
        type_distribution = positions_df.groupby('type').agg({
            'symbol': 'count',
            'value': 'sum'
        }).round(2)
        type_distribution['percentage'] = (type_distribution['value'] / type_distribution['value'].sum() * 100).round(2)
        print("\nType distribution:")
        print(type_distribution.to_string())
    
    return positions_df, pd.DataFrame(account_totals)

def create_portfolio_overview():
    df = load_and_process_data()[0]
    
    # Calculate total gain/loss
    total_gain_loss = df['gain_loss'].sum()
    print(f"\nTotal Gain/Loss: ${total_gain_loss:,.2f}")  # Debug print
    
    gain_loss_color = '#22c55e' if total_gain_loss >= 0 else '#ef4444'  # Bright green if positive, bright red if negative
    gain_loss_sign = '+' if total_gain_loss >= 0 else ''
    gain_loss_text = f"{gain_loss_sign}${abs(total_gain_loss):,.0f}"
    print(f"Formatted Gain/Loss Text: {gain_loss_text}")  # Debug print
    
    # Define colors for each asset type
    type_colors = {
        'Stocks': '#00b894',    # Teal
        'ETFs': '#0984e3',      # Blue
        'Mutual Funds': '#6c5ce7', # Purple
        'Bonds': '#fdcb6e',     # Yellow
        'Cash': '#a8e6cf'       # Mint
    }
    
    # Calculate total portfolio value by type
    portfolio_by_type = df.groupby('type').agg({
        'value': 'sum',
        'symbol': 'count'
    }).reset_index()
    
    # Sort by value descending
    portfolio_by_type = portfolio_by_type.sort_values('value', ascending=False)
    
    total_value = portfolio_by_type['value'].sum()
    portfolio_by_type['percentage'] = (portfolio_by_type['value'] / total_value * 100).round(2)
    
    # Create figure with subplots
    fig = go.Figure()
    
    # Add pie chart to the left side
    fig.add_trace(go.Pie(
        values=portfolio_by_type['value'],
        labels=portfolio_by_type['type'],
        domain={'x': [0.02, 0.45], 'y': [0, 0.95]},
        textposition='none',
        hovertemplate='<b>%{label}</b><br>' +
                     'Value: $%{value:,.0f}<br>' +
                     'Allocation: %{percent}<br>' +
                     '<extra></extra>',
        direction='clockwise',
        sort=False,
        marker=dict(
            colors=[type_colors.get(t, '#95a5a6') for t in portfolio_by_type['type']]
        ),
        showlegend=False
    ))
    
    # Add annotations
    annotations = []
    
    # Add total portfolio value at the top
    annotations.append(dict(
        x=0.73,
        y=0.95,
        text=f"<b>Total Portfolio Value</b><br>${total_value:,.0f}",
        showarrow=False,
        align='center',
        xanchor='center',
        yanchor='top',
        font=dict(size=16)  # Increased font size
    ))
    
    # Add total gain/loss below total value
    annotations.append(dict(
        x=0.73,
        y=0.87,  # Adjusted position to be closer to total value
        text=f"<b>Total Gain/Loss</b><br><span style='color: {gain_loss_color}; font-weight: bold'>{gain_loss_text}</span>",
        showarrow=False,
        align='center',
        xanchor='center',
        yanchor='top',
        font=dict(size=14),
        bgcolor='rgba(255, 255, 255, 0)'  # Transparent background
    ))
    
    # Add individual asset type details
    num_types = len(portfolio_by_type)
    spacing = 0.65 / (num_types + 1)  # Further reduced spacing
    
    for i, (_, row) in enumerate(portfolio_by_type.iterrows()):
        y_pos = 0.60 - (i + 1) * spacing  # Adjusted starting position lower to make room for gain/loss
        value_text = f"${row['value']:,.0f}"
        percent_text = f"{row['percentage']:.1f}%"
        color = type_colors.get(row['type'], '#95a5a6')
        
        # Add color square
        annotations.append(dict(
            x=0.55,
            y=y_pos,
            text='■',
            showarrow=False,
            align='center',
            xanchor='center',
            yanchor='middle',
            font=dict(
                size=16,
                color=color
            )
        ))
        
        # Add type and values
        annotations.append(dict(
            x=0.73,
            y=y_pos,
            text=f"<b>{row['type']}</b><br>{value_text} ({percent_text})",
            showarrow=False,
            align='center',
            xanchor='center',
            yanchor='middle',
            font=dict(size=12)
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Asset Allocation',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        annotations=annotations,
        height=500,
        margin=dict(l=50, r=50, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
        plot_bgcolor='rgba(0,0,0,0)'    # Transparent background
    )

    return fig.to_json()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/portfolio_overview')
def portfolio_overview():
    return jsonify(create_portfolio_overview())
    return;

@app.route('/api/accounts')
def get_accounts():
    try:
        positions_df, _ = load_and_process_data()
        accounts = sorted(positions_df['account'].unique().tolist())
        return jsonify(accounts)
    except Exception as e:
        print(f"Error in get_accounts: {e}")
        traceback.print_exc()
        return jsonify([])

@app.route('/api/symbol_details/<symbol>')
def get_symbol_details(symbol):
    try:
        positions_df, _ = load_and_process_data()
        
        # Filter for the specific symbol
        symbol_positions = positions_df[positions_df['symbol'] == symbol]
        
        if symbol_positions.empty:
            return jsonify([])
        
        # Group by account to get details from each account
        details = []
        for account, group in symbol_positions.groupby('account'):
            quantity = float(group['quantity'].sum())
            cost_basis = float(group['cost_basis'].sum())
            # Calculate unit cost (cost per share)
            unit_cost = cost_basis / quantity if quantity != 0 else 0.0
            
            position = {
                'account': account,
                'symbol': symbol,
                'description': group.iloc[0]['description'],
                'type': group.iloc[0]['type'],
                'quantity': quantity,
                'price': float(group.iloc[0]['price']),
                'value': float(group['value'].sum()),
                'cost_basis': cost_basis,
                'gain_loss': float(group['gain_loss'].sum()),
                'unit_cost': unit_cost
            }
            details.append(position)
        
        # Sort details by unit cost in descending order
        details.sort(key=lambda x: x['unit_cost'], reverse=True)
        
        # Add total row
        total_quantity = float(symbol_positions['quantity'].sum())
        total_cost_basis = float(symbol_positions['cost_basis'].sum())
        total_unit_cost = total_cost_basis / total_quantity if total_quantity != 0 else 0.0
        
        total = {
            'account': 'Total',
            'symbol': symbol,
            'description': symbol_positions.iloc[0]['description'],
            'type': symbol_positions.iloc[0]['type'],
            'quantity': total_quantity,
            'price': float(symbol_positions.iloc[0]['price']),
            'value': float(symbol_positions['value'].sum()),
            'cost_basis': total_cost_basis,
            'gain_loss': float(symbol_positions['gain_loss'].sum()),
            'unit_cost': total_unit_cost
        }
        details.append(total)
        
        return jsonify(details)
    except Exception as e:
        print(f"Error in get_symbol_details: {e}")
        traceback.print_exc()
        return jsonify([])

@app.route('/api/holdings/<asset_type>')
def get_holdings_by_type(asset_type):
    try:
        positions_df, _ = load_and_process_data()
        
        # Get account filter from query parameter
        account_filter = request.args.get('account', 'all')
        
        print("\nAll unique types in positions_df:")
        print(positions_df['type'].unique())
        
        # If showing all accounts, merge positions with the same symbol first
        if account_filter == 'all':
            # Group by symbol
            merged_positions = []
            for symbol, group in positions_df.groupby('symbol'):
                accounts = sorted(group['account'].unique())
                total_value = group['value'].sum()
                total_cost_basis = group['cost_basis'].sum()
                total_gain_loss = group['gain_loss'].sum()
                total_quantity = group['quantity'].sum()
                
                merged_position = {
                    'symbol': symbol,
                    'description': group.iloc[0]['description'],
                    'type': group.iloc[0]['type'],
                    'accounts': accounts,  # List of accounts that hold this symbol
                    'account': ', '.join(accounts),  # Display string of accounts
                    'quantity': total_quantity,
                    'price': float(group.iloc[0]['price']),
                    'value': total_value,
                    'cost_basis': total_cost_basis,
                    'gain_loss': total_gain_loss
                }
                merged_positions.append(merged_position)
            
            # Convert back to DataFrame
            positions_df = pd.DataFrame(merged_positions)
        
        # Apply type filter after merging
        if asset_type != 'all':
            positions_df = positions_df[positions_df['type'].str.lower() == asset_type.lower()]
            print(f"\nFiltered for asset_type: {asset_type}")
            print(f"Number of positions after type filtering: {len(positions_df)}")
        
        # Apply account filter if specified
        if account_filter != 'all':
            # For merged positions, check if the account is in the accounts list
            if 'accounts' in positions_df.columns:
                positions_df = positions_df[positions_df['accounts'].apply(lambda x: account_filter in x)]
            else:
                positions_df = positions_df[positions_df['account'] == account_filter]
            print(f"\nFiltered for account: {account_filter}")
            print(f"Number of positions after account filtering: {len(positions_df)}")
        
        print(f"\nFiltering holdings for type: {asset_type} and account: {account_filter}")
        print(f"Found {len(positions_df)} positions")
        
        holdings = []
        total_value = positions_df['value'].sum()
        
        for _, row in positions_df.iterrows():
            try:
                value = float(row['value'])
                portfolio_percentage = (value / total_value * 100) if total_value > 0 else 0
                
                holding = {
                    'symbol': str(row['symbol']),
                    'description': str(row['description']),
                    'type': str(row['type']),
                    'account': str(row['account']),
                    'quantity': float(row['quantity']) if 'quantity' in row else None,
                    'price': float(row['price']) if 'price' in row else None,
                    'value': value,
                    'cost_basis': float(row['cost_basis']),
                    'gain_loss': float(row['gain_loss']),
                    'portfolio_percentage': round(portfolio_percentage, 2)
                }
                holdings.append(holding)
                print(f"Added Position: {holding['symbol']} - Type: {holding['type']} - Account: {holding['account']} - Value: ${holding['value']:,.2f}")
            except Exception as e:
                print(f"Error processing position: {e}")
                continue
        
        return jsonify(holdings)
    except Exception as e:
        print(f"Error in get_holdings_by_type: {e}")
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
        
        # Calculate cash balance (only include money market funds)
        cash_symbols = ['SPAXX', 'FDRXX', 'SNSXX']
        cash_positions = positions_df[positions_df['symbol'].str.contains('|'.join(cash_symbols), case=False, na=False)]
        cash_balance = cash_positions['value'].sum() if not cash_positions.empty else 0.0
        print(f"\nCash positions found:")
        for _, row in cash_positions.iterrows():
            print(f"{row['symbol']}: ${row['value']:,.2f}")
        print(f"\nTotal Cash Balance: ${cash_balance:,.2f}")
        
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

@app.route('/api/files', methods=['GET'])
def list_files():
    try:
        files = []
        for file in UPLOAD_FOLDER.glob('*.csv'):
            # Skip example files
            if not file.name.startswith('example_'):
                files.append({
                    'name': file.name,
                    'size': os.path.getsize(file) / 1024,  # Convert to KB
                    'modified': datetime.fromtimestamp(os.path.getmtime(file)).strftime('%Y-%m-%d %H:%M:%S')
                })
        return jsonify(files)
    except Exception as e:
        print(f"Error listing files: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/files/<filename>', methods=['DELETE'])
def delete_file(filename):
    try:
        if filename.startswith('example_'):
            return jsonify({'error': 'Cannot delete example files'}), 403
            
        file_path = UPLOAD_FOLDER / filename
        if file_path.exists():
            # Backup the file before deletion
            backup_dir = UPLOAD_FOLDER / 'backup' / datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, backup_dir / filename)
            
            # Delete the file
            os.remove(file_path)
            return jsonify({'message': f'File {filename} deleted successfully'})
        else:
            return jsonify({'error': 'File not found'}), 404
    except Exception as e:
        print(f"Error deleting file: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Print pie chart data
    print("\nGenerating pie chart data...")
    create_portfolio_overview()
    
    # Start the Flask app on port 8080
    app.run(debug=True, host='0.0.0.0', port=8080) 