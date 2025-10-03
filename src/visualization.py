import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# create a pivoted dataframe with monthly KPIs 
def key_metrics_monthly(df, df2, df3, start_date=None, end_date=None):
    """
    Create a pivoted dataframe with monthly invoice statistics by broker type.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The invoice dataframe
    df2 : pandas DataFrame
        The payments dataframe
    df3 : pandas DataFrame
        The time reporting dataframe
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas DataFrame: Pivoted dataframe with months as columns and invoice metrics as rows
    """
    # create copies to avoid modifying the originals
    plot_df = df.copy()
    payments_df = df2.copy()
    
    # convert dates to datetime if they're not already
    plot_df['final_pay_date'] = pd.to_datetime(plot_df['final_pay_date'])
    payments_df['final_pay_date'] = pd.to_datetime(payments_df['final_pay_date'])
    
    # parse date filters once
    start_date_dt = pd.to_datetime(start_date) if start_date else None
    end_date_dt = pd.to_datetime(end_date) if end_date else None
    
    # apply date filters to invoices
    plot_df = plot_df.dropna(subset=['final_pay_date'])
    if start_date_dt:
        plot_df = plot_df[plot_df['final_pay_date'] >= start_date_dt]
    if end_date_dt:
        plot_df = plot_df[plot_df['final_pay_date'] <= end_date_dt]
    
    # apply date filters to payments
    payments_df = payments_df.dropna(subset=['final_pay_date'])
    if start_date_dt:
        payments_df = payments_df[payments_df['final_pay_date'] >= start_date_dt]
    if end_date_dt:
        payments_df = payments_df[payments_df['final_pay_date'] <= end_date_dt]
    
    # check if we have data after filtering
    if plot_df.empty:
        print("No invoice data available after applying date filters.")
        return pd.DataFrame()
    
    # create month column for grouping
    plot_df['month'] = plot_df['final_pay_date'].dt.to_period('M')
    plot_df['month_str'] = plot_df['month'].dt.strftime('%Y-%m')
    
    # create a broker type column with standardized categories
    plot_df['broker_type'] = plot_df['broker'].apply(
        lambda x: 'Broker' if 'broker' in str(x).lower() else 
                 ('Partner' if 'partner' in str(x).lower() else 'Direct')
    )
    
    # group by month and calculate totals
    monthly_totals = plot_df.groupby('month_str').agg(
        total_net=('invoice_amount_net', 'sum'),
        total_total=('invoice_amount_total', 'sum')
    )
    
    # group by month and broker type
    broker_data = plot_df.groupby(['month_str', 'broker_type']).agg(
        net_amount=('invoice_amount_net', 'sum'),
        total_amount=('invoice_amount_total', 'sum')
    ).reset_index()
    
    # calculate percentage of total net for each broker type
    broker_data = broker_data.merge(monthly_totals, on='month_str')
    broker_data['pct_of_net'] = (broker_data['net_amount'] / broker_data['total_net']) * 100
    
    # create a list to store the rows for our pivot table
    pivot_rows = []
    
    # add total rows (independent of broker type)
    pivot_rows.append(('Net Amount', plot_df.groupby('month_str')['invoice_amount_net'].sum()))
    pivot_rows.append(('Total Invoice Amount', plot_df.groupby('month_str')['invoice_amount_total'].sum()))
    
    # add rows for each broker type
    for broker_type in ['Broker', 'Direct', 'Partner']:
        # filter for the current broker type
        broker_subset = broker_data[broker_data['broker_type'] == broker_type]
        
        if not broker_subset.empty:
            # set the month_str as index for easy pivoting and add rows for this broker type
            broker_subset = broker_subset.set_index('month_str')
            pivot_rows.append((f'{broker_type} Net Amount', broker_subset['net_amount']))
            pivot_rows.append((f'{broker_type} Total Amount', broker_subset['total_amount']))
            pivot_rows.append((f'{broker_type} % of Net', broker_subset['pct_of_net']))
    
    # add Payments row from payments_df (already filtered)
    payments_df['month_str'] = payments_df['final_pay_date'].dt.strftime('%Y-%m')
    payments_by_month = payments_df.groupby('month_str')['invoice_payment'].sum()
    pivot_rows.append(('Payments', payments_by_month))

    # create Revenue row
    total_net_series = plot_df.groupby('month_str')['invoice_amount_net'].sum()
    # align indexes and fill missing months with 0
    revenue_series = total_net_series.subtract(payments_by_month, fill_value=0)
    pivot_rows.append(('Revenue', revenue_series))
    
    # add time reports from df3 - apply same date filter
    if df3 is not None:
        df3_copy = df3.copy()
        df3_copy = df3_copy.dropna(subset=['date'])
        
        # apply date filters to time reporting data
        df3_copy['date'] = pd.to_datetime(df3_copy['date'])
        if start_date_dt:
            df3_copy = df3_copy[df3_copy['date'] >= start_date_dt]
        if end_date_dt:
            df3_copy = df3_copy[df3_copy['date'] <= end_date_dt]
            
        df3_copy['month_str'] = df3_copy['date'].dt.strftime('%Y-%m')
        hours_by_month = df3_copy.groupby('month_str').agg(
            billable_hours=('billable_hours', 'sum'),
            non_billable_hours=('non_billable_hours', 'sum'),
            total_hours=('total_hours', 'sum')
        )
        # utilization percentage
        hours_by_month['Utilization Percentage'] = (
            (hours_by_month['billable_hours'] / hours_by_month['total_hours']).replace([np.inf, -np.inf], 0).fillna(0) * 100
        )
        pivot_rows.append(('Billable Hours', hours_by_month['billable_hours']))
        pivot_rows.append(('Non-Billable Hours', hours_by_month['non_billable_hours']))
        pivot_rows.append(('Total Hours', hours_by_month['total_hours']))
        pivot_rows.append(('Utilization Percentage', hours_by_month['Utilization Percentage']))

    # create pivot table clean up and format the DataFrame
    result = pd.DataFrame({row_name: data for row_name, data in pivot_rows})
    result = result.fillna(0)
    result = result.applymap(lambda x: int(round(x)) if not isinstance(x, str) else x)

    # format percentage columns as strings with % and no decimals
    percentage_rows = ['Broker % of Net', 'Direct % of Net', 'Partner % of Net', 'Utilization Percentage']
    for row in percentage_rows:
        if row in result.index:
            result.loc[row] = result.loc[row].apply(lambda x: f"{int(round(x))}%")
    
    # format other columns with thousand separator
    for idx in result.index:
        if idx not in percentage_rows:
            result.loc[idx] = result.loc[idx].apply(lambda x: f"{x:,}" if isinstance(x, int) else x)

    # ensure columns are sorted chronologically
    result = result.reindex(sorted(result.columns), axis=1)

    # after formatting, transpose so months are columns and format from YYYY-MM to MMM YYYY 
    result = result.T
    formatted_columns = {}
    for col in result.columns:
        try:
            date_obj = pd.to_datetime(col.split()[0] + '-01')
            formatted_col = date_obj.strftime('%b %Y')
            formatted_columns[col] = formatted_col
        except Exception:
            formatted_columns[col] = col
    result = result.rename(columns=formatted_columns)

    return result

# Replace the existing highlight_revenue_trend with the version below.
def highlight_revenue_trend(row, pivot_df=None, color=True):
    """
    Applies:
      - Adds % sign ONLY here (not in key_metrics_monthly) for:
        Broker % of Net, Direct % of Net, Partner % of Net, Utilization Percentage
      - Colors:
          Revenue & Payments: green if Revenue >= 50% of Net (same month) else red
          Utilization Percentage: green >=80 else red
          Direct % of Net: green >=50 else red
          Broker % of Net & Partner % of Net: green <=25 else red
          Net Amount: green >= 1,600,000 else red
    Returns list of CSS styles for the row.
    """
    percentage_rows = {
        'Broker % of Net',
        'Direct % of Net',
        'Partner % of Net',
        'Utilization Percentage'
    }

    if pivot_df is None or 'Net Amount' not in pivot_df.index:
        return [''] * len(row)

    def to_number(val):
        try:
            return float(str(val).replace(',', '').replace('%', '').strip())
        except Exception:
            return 0.0

    # Capture numeric values BEFORE we add % signs (so we can evaluate thresholds)
    numeric_values = {col: to_number(val) for col, val in row.items()}

    # If this is a percentage row, add % signs (mutate pivot_df so they display)
    if row.name in percentage_rows:
        # Only add if not already formatted
        def fmt_percent(v):
            # v expected numeric (int/float or already string); parse again safely
            num = to_number(v)
            return f"{int(round(num))}%"
        # mutate the full DataFrame so styling shows updated values
        pivot_df.loc[row.name] = [fmt_percent(v) for v in row.values]
        # refresh local row view to reflect mutation (Styler works with original df; but row
        # copy won't change – that's fine, we already have numeric_values)

    # Stop early if no color requested (but still ensured % above)
    if not color:
        return [''] * len(row)

    # Pre-fetch comparison rows (need raw numbers, so parse after potential prior formatting)
    net_row_nums = {
        col: to_number(pivot_df.loc['Net Amount', col]) for col in pivot_df.columns
    }
    revenue_row_nums = None
    if 'Revenue' in pivot_df.index:
        revenue_row_nums = {
            col: to_number(pivot_df.loc['Revenue', col]) for col in pivot_df.columns
        }

    styles = []
    rn = row.name

    for col in row.index:
        val = numeric_values[col]
        if rn in ('Revenue', 'Payments'):
            net_val = net_row_nums.get(col, 0)
            rev_val = revenue_row_nums.get(col, 0) if revenue_row_nums else val
            if net_val > 0 and rev_val >= 0.5 * net_val:
                styles.append('color: green; font-weight: bold;')
            elif net_val > 0:
                styles.append('color: red; font-weight: bold;')
            else:
                styles.append('')
        elif rn == 'Net Amount':
            styles.append('color: green; font-weight: bold;' if val >= 1_600_000 else 'color: red; font-weight: bold;')
        elif rn == 'Utilization Percentage':
            styles.append('color: green; font-weight: bold;' if val >= 80 else 'color: red; font-weight: bold;')
        elif rn == 'Direct % of Net':
            styles.append('color: green; font-weight: bold;' if val >= 50 else 'color: red; font-weight: bold;')
        elif rn in ('Broker % of Net', 'Partner % of Net'):
            styles.append('color: green; font-weight: bold;' if val <= 25 else 'color: red; font-weight: bold;')
        else:
            styles.append('')  # No styling for other rows
    return styles

# plot net, total, payments or revenue by month with optional broker hue
def plot_invoice_amounts(df, df2=None, start_date=None, end_date=None, amount_type='net', hue=False, show_trend=False):
    """
    Plot invoice amounts by month based on final payment date.
    Supports plotting net, total, payments, or revenue.

    Parameters:
    -----------
    df : pandas DataFrame
        The invoice dataframe
    df2 : pandas DataFrame, optional
        The payments dataframe (required for payments/revenue)
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format
    amount_type : str, default 'net'
        Type of amount to plot: 'net', 'total', 'payments', or 'revenue'
    hue : bool, default False
        If True, group and color by broker column

    Returns:
    --------
    None: Displays the plotly figure
    """
    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=['final_pay_date'])
    
    # convert dates to datetime objects once
    start_date_dt = pd.to_datetime(start_date) if start_date else None
    end_date_dt = pd.to_datetime(end_date) if end_date else None

    # apply date filters to invoice data
    if start_date_dt:
        plot_df = plot_df[plot_df['final_pay_date'] >= start_date_dt]
    if end_date_dt:
        plot_df = plot_df[plot_df['final_pay_date'] <= end_date_dt]
    
    if plot_df.empty:
        print("No invoice data available after applying date filters.")
        return None

    plot_df['month'] = plot_df['final_pay_date'].dt.to_period('M')
    plot_df['month_str'] = plot_df['month'].dt.strftime('%Y-%m')

    # prepare payments and revenue if needed
    payments_by_month = None
    revenue_by_month = None
    if amount_type.lower() in ['payments', 'revenue']:
        if df2 is None:
            raise ValueError("df2 (payments dataframe) must be provided for payments or revenue plots.")
        
        # apply the same date filters to payments data
        payments_df = df2.dropna(subset=['final_pay_date']).copy()
        
        # apply date filters to payments data too
        if start_date_dt:
            payments_df = payments_df[payments_df['final_pay_date'] >= start_date_dt]
        if end_date_dt:
            payments_df = payments_df[payments_df['final_pay_date'] <= end_date_dt]
            
        # check if we have payments data after filtering
        if payments_df.empty and amount_type.lower() == 'payments':
            print("No payment data available after applying date filters.")
            return None
        
        payments_df['month_str'] = payments_df['final_pay_date'].dt.strftime('%Y-%m')
        payments_by_month = payments_df.groupby('month_str')['invoice_payment'].sum()
        
        # for revenue, we need both invoices and payments
        if amount_type.lower() == 'revenue':
            total_net_series = plot_df.groupby('month_str')['invoice_amount_net'].sum()
            revenue_by_month = total_net_series.subtract(payments_by_month, fill_value=0)
            
            # if there's no revenue data after calculation
            if revenue_by_month.empty:
                print("No revenue data available after calculations.")
                return None

    # select the amount column and title
    if amount_type.lower() == 'net':
        amount_col = 'invoice_amount_net'
        amount_title = 'Net Invoice Amount'
        monthly_data = plot_df.groupby('month_str')[amount_col].sum().reset_index()
    elif amount_type.lower() == 'total':
        amount_col = 'invoice_amount_total'
        amount_title = 'Total Invoice Amount'
        monthly_data = plot_df.groupby('month_str')[amount_col].sum().reset_index()
    elif amount_type.lower() == 'payments':
        amount_title = 'Payments'
        monthly_data = payments_by_month.reset_index()
        monthly_data.columns = ['month_str', 'Payments']
    elif amount_type.lower() == 'revenue':
        amount_title = 'Revenue'
        monthly_data = revenue_by_month.reset_index()
        monthly_data.columns = ['month_str', 'Revenue']
    else:
        raise ValueError("amount_type must be 'net', 'total', 'payments', or 'revenue'")

    # make sure we have data to plot or error out 
    if monthly_data.empty:
        print(f"No {amount_type} data available for the selected date range.")
        return None

    # plot
    if hue and amount_type.lower() in ['net', 'total']:
        monthly_data = plot_df.groupby(['month_str', 'broker'])[amount_col].sum().reset_index()
        fig = px.bar(
            monthly_data,
            x='month_str',
            y=amount_col,
            color='broker',
            labels={
                'month_str': 'Month',
                amount_col: f'{amount_title} (SEK)',
                'broker': 'Broker'
            },
            barmode='group'
        )
        fig.update_traces(
            hovertemplate='Month: %{x}<br>Broker: %{marker.color}<br>' + f'{amount_title}: %{{y:,.2f}} SEK'
        )
    else:
        y_col = monthly_data.columns[1]
        fig = px.bar(
            monthly_data,
            x='month_str',
            y=y_col,
            labels={
                'month_str': 'Month',
                y_col: f'{amount_title} (SEK)'
            },
            text_auto='.2s'
        )
        fig.update_traces(
            hovertemplate='Month: %{x}<br>' + f'{amount_title}: %{{y:,.2f}} SEK'
        )

    # multiple regression lines for when hue is true
    if show_trend and len(monthly_data) > 1:
        if hue and amount_type.lower() in ['net', 'total']:
            for broker in monthly_data['broker'].unique():
                broker_data = monthly_data[monthly_data['broker'] == broker].copy()
                if len(broker_data) > 1:
                    broker_data = broker_data.sort_values('month_str')
                    broker_data['x_numeric'] = range(len(broker_data))
                    coeffs = np.polyfit(broker_data['x_numeric'], broker_data[amount_col], 1)
                    trend = np.polyval(coeffs, broker_data['x_numeric'])
                    fig.add_traces(
                        go.Scatter(
                            x=broker_data['month_str'],
                            y=trend,
                            mode='lines',
                            name=f'{broker} Trend',
                            line=dict(dash='dash'),
                            showlegend=True
                        )
                    )
        # single regression line for when hue is false
        elif not hue:
            y_col = monthly_data.columns[1]
            monthly_data['x_numeric'] = range(len(monthly_data))
            coeffs = np.polyfit(monthly_data['x_numeric'], monthly_data[y_col], 1)
            trend = np.polyval(coeffs, monthly_data['x_numeric'])
            fig.add_traces(
                go.Scatter(
                    x=monthly_data['month_str'],
                    y=trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                )
            )
    
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title=f'{amount_title} (SEK)',
        height=400,
        xaxis={'categoryorder': 'category ascending'}
    )
    return fig

def plot_invoice_amounts_line(df, df2=None, start_date=None, end_date=None, amount_type='net', hue=False, show_trend=False):
    """
    Plot invoice amounts by month based on final payment date.
    Supports plotting net, total, payments, or revenue.

    Parameters:
    -----------
    df : pandas DataFrame
        The invoice dataframe
    df2 : pandas DataFrame, optional
        The payments dataframe (required for payments/revenue)
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format
    amount_type : str, default 'net'
        Type of amount to plot: 'net', 'total', 'payments', or 'revenue'
    hue : bool, default False
        If True, group and color by broker column

    Returns:
    --------
    None: Displays the plotly figure
    """
    plot_df = df.copy()
    plot_df = plot_df.dropna(subset=['final_pay_date'])
    
    # convert dates to datetime objects once
    start_date_dt = pd.to_datetime(start_date) if start_date else None
    end_date_dt = pd.to_datetime(end_date) if end_date else None

    # apply date filters to invoice data
    if start_date_dt:
        plot_df = plot_df[plot_df['final_pay_date'] >= start_date_dt]
    if end_date_dt:
        plot_df = plot_df[plot_df['final_pay_date'] <= end_date_dt]
    
    if plot_df.empty:
        print("No invoice data available after applying date filters.")
        return None

    plot_df['month'] = plot_df['final_pay_date'].dt.to_period('M')
    plot_df['month_str'] = plot_df['month'].dt.strftime('%Y-%m')

    # prepare payments and revenue if needed
    payments_by_month = None
    revenue_by_month = None
    if amount_type.lower() in ['payments', 'revenue']:
        if df2 is None:
            raise ValueError("df2 (payments dataframe) must be provided for payments or revenue plots.")
        
        # apply the same date filters to payments data
        payments_df = df2.dropna(subset=['final_pay_date']).copy()
        
        # apply date filters to payments data too
        if start_date_dt:
            payments_df = payments_df[payments_df['final_pay_date'] >= start_date_dt]
        if end_date_dt:
            payments_df = payments_df[payments_df['final_pay_date'] <= end_date_dt]
            
        # check if we have payments data after filtering
        if payments_df.empty and amount_type.lower() == 'payments':
            print("No payment data available after applying date filters.")
            return None
        
        payments_df['month_str'] = payments_df['final_pay_date'].dt.strftime('%Y-%m')
        payments_by_month = payments_df.groupby('month_str')['invoice_payment'].sum()
        
        # for revenue, we need both invoices and payments
        if amount_type.lower() == 'revenue':
            total_net_series = plot_df.groupby('month_str')['invoice_amount_net'].sum()
            revenue_by_month = total_net_series.subtract(payments_by_month, fill_value=0)
            
            # if there's no revenue data after calculation
            if revenue_by_month.empty:
                print("No revenue data available after calculations.")
                return None

    # select the amount column and title
    if amount_type.lower() == 'net':
        amount_col = 'invoice_amount_net'
        amount_title = 'Net Invoice Amount'
        monthly_data = plot_df.groupby('month_str')[amount_col].sum().reset_index()
    elif amount_type.lower() == 'total':
        amount_col = 'invoice_amount_total'
        amount_title = 'Total Invoice Amount'
        monthly_data = plot_df.groupby('month_str')[amount_col].sum().reset_index()
    elif amount_type.lower() == 'payments':
        amount_title = 'Payments'
        monthly_data = payments_by_month.reset_index()
        monthly_data.columns = ['month_str', 'Payments']
    elif amount_type.lower() == 'revenue':
        amount_title = 'Revenue'
        monthly_data = revenue_by_month.reset_index()
        monthly_data.columns = ['month_str', 'Revenue']
    else:
        raise ValueError("amount_type must be 'net', 'total', 'payments', or 'revenue'")

    # make sure we have data to plot or error out 
    if monthly_data.empty:
        print(f"No {amount_type} data available for the selected date range.")
        return None

    # plot - CHANGED FROM BAR TO LINE
    if hue and amount_type.lower() in ['net', 'total']:
        monthly_data = plot_df.groupby(['month_str', 'broker'])[amount_col].sum().reset_index()
        fig = px.line(
            monthly_data,
            x='month_str',
            y=amount_col,
            color='broker',
            labels={
                'month_str': 'Month',
                amount_col: f'{amount_title} (SEK)',
                'broker': 'Broker'
            },
            markers=True  # Add markers to make data points visible
        )
        fig.update_traces(
            hovertemplate='Month: %{x}<br>Broker: %{fullData.name}<br>' + f'{amount_title}: %{{y:,.2f}} SEK'
        )
    else:
        y_col = monthly_data.columns[1]
        fig = px.line(
            monthly_data,
            x='month_str',
            y=y_col,
            labels={
                'month_str': 'Month',
                y_col: f'{amount_title} (SEK)'
            },
            markers=True  # Add markers to make data points visible
        )
        fig.update_traces(
            hovertemplate='Month: %{x}<br>' + f'{amount_title}: %{{y:,.2f}} SEK'
        )
    
    # dynamic net amount target, 15% increase over previous year's monthly average 
    target_value = None
    base_df = df.copy()
    base_df['final_pay_date'] = pd.to_datetime(base_df['final_pay_date'], errors='coerce')
    base_df = base_df.dropna(subset=['final_pay_date'])
    latest_year = plot_df['final_pay_date'].dt.year.max()
    if not base_df.empty and pd.notna(latest_year):
        previous_year = int(latest_year) - 1
        prev_year_df = base_df[base_df['final_pay_date'].dt.year == previous_year]
        if not prev_year_df.empty:
            previous_year_total_net = prev_year_df['invoice_amount_net'].sum()
            target_value = (previous_year_total_net * 1.15) / 12
            target_value = float(np.ceil(target_value / 100000.0) * 100000)
    
    if amount_type.lower() == 'net' and not hue and target_value is not None:
        fig.add_hline(
            y=target_value, 
            #y=1_600_000,    
            line_dash='dash',
            line_color='orange',    
            annotation_text=f'Monthly Target: {target_value:,.0f} SEK',
            #annotation_text='Monthly Target: 1.6M SEK',
            annotation_position='top right'
    )

    # multiple regression lines for when hue is true
    if show_trend and len(monthly_data) > 1:
        if hue and amount_type.lower() in ['net', 'total']:
            for broker in monthly_data['broker'].unique():
                broker_data = monthly_data[monthly_data['broker'] == broker].copy()
                if len(broker_data) > 1:
                    broker_data = broker_data.sort_values('month_str')
                    broker_data['x_numeric'] = range(len(broker_data))
                    coeffs = np.polyfit(broker_data['x_numeric'], broker_data[amount_col], 1)
                    trend = np.polyval(coeffs, broker_data['x_numeric'])
                    fig.add_traces(
                        go.Scatter(
                            x=broker_data['month_str'],
                            y=trend,
                            mode='lines',
                            name=f'{broker} Trend',
                            line=dict(dash='dash'),
                            showlegend=True
                        )
                    )
        # single regression line for when hue is false
        elif not hue:
            y_col = monthly_data.columns[1]
            monthly_data['x_numeric'] = range(len(monthly_data))
            coeffs = np.polyfit(monthly_data['x_numeric'], monthly_data[y_col], 1)
            trend = np.polyval(coeffs, monthly_data['x_numeric'])
            fig.add_traces(
                go.Scatter(
                    x=monthly_data['month_str'],
                    y=trend,
                    mode='lines',
                    name='Trend',
                    line=dict(color='red', dash='dash')
                )
            )
    
    fig.update_layout(
        xaxis_title='Month',
        yaxis_title=f'{amount_title} (SEK)',
        height=400,
   
        xaxis={'categoryorder': 'category ascending'}
    )
    return fig

# total, billable and non billable hours
def plot_monthly_hours(df_time, start_date=None, end_date=None, hours_type='total', show_trend=False):
    """
    Aggregate and plot monthly billable, non-billable, total hours, or utilization percentage, with optional regression line.

    Parameters:
    -----------
    df_time : pandas DataFrame
        The time reporting dataframe (must have 'date', 'billable_hours', 'non_billable_hours', 'total_hours')
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format
    hours_type : str, default 'total'
        Which metric to plot: 'billable', 'non_billable', 'total', or 'utilization'
    show_trend : bool, default False
        Whether to show a regression (trend) line

    Returns:
    --------
    plotly Figure or None
    """
    df = df_time.copy()
    df['date'] = pd.to_datetime(df['date'])
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    df['month_str'] = df['date'].dt.strftime('%Y-%m')

    if hours_type == 'billable':
        col = 'billable_hours'
        hours_title = 'Billable Hours'
        y_axis_title = 'Hours'
        hover_format = ':,.2f'
    elif hours_type == 'non billable':
        col = 'non_billable_hours'
        hours_title = 'Non-Billable Hours'
        y_axis_title = 'Hours'
        hover_format = ':,.2f'
    elif hours_type == 'utilization':
        # Calculate utilization percentage monthly
        monthly = df.groupby('month_str').agg(
            billable_hours=('billable_hours', 'sum'),
            total_hours=('total_hours', 'sum')
        ).reset_index()
        monthly['utilization_percentage'] = (
            (monthly['billable_hours'] / monthly['total_hours']).replace([np.inf, -np.inf], 0).fillna(0) * 100
        ).round(0)  
        col = 'utilization_percentage'
        hours_title = 'Utilization Percentage'
        y_axis_title = 'Percentage (%)'
        hover_format = ':,.1f}%'
    else:
        col = 'total_hours'
        hours_title = 'Total Hours'
        y_axis_title = 'Hours'
        hover_format = ':,.2f'

    # for non-utilization metrics, aggregate normally
    if hours_type != 'utilization':
        monthly = df.groupby('month_str')[col].sum().reset_index()
    
    if monthly.empty:
        return None

    fig = px.bar(
        monthly,
        x='month_str',
        y=col,
        labels={'month_str': 'Month', col: hours_title},
        text_auto='.2s' if hours_type != 'utilization' else '.0f'
    )
    
    if hours_type == 'utilization':
        fig.update_traces(
            hovertemplate='Month: %{x}<br>' + f'{hours_title}: %{{y{hover_format}<extra></extra>'
        )
        fig.update_traces(texttemplate='%{y:.0f}%', textposition="outside")
    else:
        fig.update_traces(
            hovertemplate='Month: %{x}<br>' + f'{hours_title}: %{{y{hover_format}<extra></extra>'
        )

    # add regression line
    if show_trend and len(monthly) > 1:
        # Numeric x for regression
        monthly['x_numeric'] = range(len(monthly))
        coeffs = np.polyfit(monthly['x_numeric'], monthly[col], 1)
        trend = np.polyval(coeffs, monthly['x_numeric'])
        fig.add_traces(
            go.Scatter(
                x=monthly['month_str'],
                y=trend,
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            )
        )

    fig.update_layout(
        xaxis_title='Month',
        yaxis_title=y_axis_title,
        height=400,
        xaxis={'categoryorder': 'category ascending'}
    )
    return fig

def plot_monthly_hours_line(df_time, start_date=None, end_date=None, hours_type='total', show_trend=False):
    """
    Aggregate and plot monthly billable, non-billable, total hours, or utilization percentage, with optional regression line.

    Parameters:
    -----------
    df_time : pandas DataFrame
        The time reporting dataframe (must have 'date', 'billable_hours', 'non_billable_hours', 'total_hours')
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format
    hours_type : str, default 'total'
        Which metric to plot: 'billable', 'non_billable', 'total', or 'utilization'
    show_trend : bool, default False
        Whether to show a regression (trend) line

    Returns:
    --------
    plotly Figure or None
    """
    df = df_time.copy()
    df['date'] = pd.to_datetime(df['date'])
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]

    df['month_str'] = df['date'].dt.strftime('%Y-%m')

    if hours_type == 'billable':
        col = 'billable_hours'
        hours_title = 'Billable Hours'
        y_axis_title = 'Hours'
        hover_format = ':,.2f'
    elif hours_type == 'non billable':
        col = 'non_billable_hours'
        hours_title = 'Non-Billable Hours'
        y_axis_title = 'Hours'
        hover_format = ':,.2f'
    elif hours_type == 'utilization':
        # Calculate utilization percentage monthly
        monthly = df.groupby('month_str').agg(
            billable_hours=('billable_hours', 'sum'),
            total_hours=('total_hours', 'sum')
        ).reset_index()
        monthly['utilization_percentage'] = (
            (monthly['billable_hours'] / monthly['total_hours']).replace([np.inf, -np.inf], 0).fillna(0) * 100
        ).round(0)  
        col = 'utilization_percentage'
        hours_title = 'Utilization Percentage'
        y_axis_title = 'Percentage (%)'
        hover_format = ':,.1f}%'
    else:
        col = 'total_hours'
        hours_title = 'Total Hours'
        y_axis_title = 'Hours'
        hover_format = ':,.2f'

    # for non-utilization metrics, aggregate normally
    if hours_type != 'utilization':
        monthly = df.groupby('month_str')[col].sum().reset_index()
    
    if monthly.empty:
        return None

    fig = px.line(
        monthly,
        x='month_str',
        y=col,
        labels={'month_str': 'Month', col: hours_title},
        markers=True  # Add markers to make data points visible
    )
    
    if hours_type == 'utilization':
        fig.update_traces(
            hovertemplate='Month: %{x}<br>' + f'{hours_title}: %{{y{hover_format}<extra></extra>'
        )
    else:
        fig.update_traces(
            hovertemplate='Month: %{x}<br>' + f'{hours_title}: %{{y{hover_format}<extra></extra>'
        )

    # Add 80% reference line for utilization
    if hours_type == 'utilization':
        fig.add_hline(
            y=80,
            line_dash="dash",
            line_color="orange",
            annotation_text="Monthly Target: 80%",
            annotation_position="bottom right"
        )
        
    # add regression line
    if show_trend and len(monthly) > 1:
        # Numeric x for regression
        monthly['x_numeric'] = range(len(monthly))
        # Fit a linear regression
        coeffs = np.polyfit(monthly['x_numeric'], monthly[col], 1)
        trend = np.polyval(coeffs, monthly['x_numeric'])
        fig.add_traces(
            go.Scatter(
                x=monthly['month_str'],
                y=trend,
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            )
        )

    # Set y-axis range for utilization to 0-100%
    if hours_type == 'utilization':
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title=y_axis_title,
            height=400,
            xaxis={'categoryorder': 'category ascending'},
            yaxis={'range': [0, 100]}  # Fix y-axis range for utilization
        )
    else:
        fig.update_layout(
            xaxis_title='Month',
            yaxis_title=y_axis_title,
            height=400,
            xaxis={'categoryorder': 'category ascending'}
        )
    
    return fig

def get_metric_row(df, df2, df3, row_name, start_date=None, end_date=None):
    """
    Extract a specific row from the key_metrics_monthly pivot table.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The invoice dataframe
    df2 : pandas DataFrame
        The payments dataframe
    df3 : pandas DataFrame
        The time reporting dataframe
    row_name : str
        Name of the row to extract (e.g., 'Revenue', 'Net Amount', 'Utilization Percentage')
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas Series or None: The requested row data, or None if row doesn't exist
    """
    # Get the full pivot table
    pivot_data = key_metrics_monthly(df, df2, df3, start_date, end_date)
    
    # Check if the row exists in the pivot table
    if row_name in pivot_data.index:
        return pivot_data.loc[row_name]
    else:
        print(f"Row '{row_name}' not found in the data.")
        return None

def create_average_kpi_card(df, df2, df3, row_name, start_date=None, end_date=None):
    """
    Create a KPI card showing the average value of a specific metric row.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The invoice dataframe
    df2 : pandas DataFrame
        The payments dataframe
    df3 : pandas DataFrame
        The time reporting dataframe
    row_name : str
        Name of the row to calculate average for
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format
        
    Returns:
    --------
    dict: Contains the metric name, average value, and formatted display value
    """
    # get the specific row data
    row_data = get_metric_row(df, df2, df3, row_name, start_date, end_date)
    
    if row_data is None:
        return None
    
    # convert values to numeric (remove commas and % signs)
    numeric_values = []
    for value in row_data:
        try:
            # remove commas and % signs, then convert to float
            clean_value = str(value).replace(',', '').replace('%', '')
            numeric_values.append(float(clean_value))
        except (ValueError, TypeError):
            continue
    
    if not numeric_values:
        return None
    
    # calculate average
    avg_value = sum(numeric_values) / len(numeric_values)
    
    # format the display value based on metric type
    percentage_metrics = ['Broker % of Net', 'Direct % of Net', 'Partner % of Net', 'Utilization Percentage']
    
    if row_name in percentage_metrics:
        formatted_value = f"{avg_value:.0f}%"
    else:
        formatted_value = f"{avg_value:,.0f}"
    
    return {
        'metric_name': row_name,
        'value': avg_value,
        'formatted_value': formatted_value,
        'type': 'average'
    }

def create_sum_kpi_card(df, df2, df3, row_name, start_date=None, end_date=None):
    """
    Create a KPI card showing the sum of a specific metric row.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The invoice dataframe
    df2 : pandas DataFrame
        The payments dataframe
    df3 : pandas DataFrame
        The time reporting dataframe
    row_name : str
        Name of the row to calculate sum for
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format
        
    Returns:
    --------
    dict: Contains the metric name, sum value, and formatted display value
    """
    # get the specific row data
    row_data = get_metric_row(df, df2, df3, row_name, start_date, end_date)
    
    if row_data is None:
        return None
    
    # convert values to numeric (remove commas and % signs)
    numeric_values = []
    for value in row_data:
        try:
            # remove commas and % signs, then convert to float
            clean_value = str(value).replace(',', '').replace('%', '')
            numeric_values.append(float(clean_value))
        except (ValueError, TypeError):
            continue
    
    if not numeric_values:
        return None
    
    # calculate sum
    sum_value = sum(numeric_values)
    
    # format the display value based on metric type
    percentage_metrics = ['Broker % of Net', 'Direct % of Net', 'Partner % of Net', 'Utilization Percentage']
    
    if row_name in percentage_metrics:
        formatted_value = f"{sum_value:.0f}%"
    else:
        formatted_value = f"{sum_value:,.0f}"
    
    return {
        'metric_name': row_name,
        'value': sum_value,
        'formatted_value': formatted_value,
        'type': 'sum'
    }

def create_kpi_cards(df, df2, df3, metric_type, start_date=None, end_date=None, show_broker=False):
    """
    Create KPI cards showing sum and average for a specific metric, with optional broker breakdown.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The invoice dataframe
    df2 : pandas DataFrame
        The payments dataframe
    df3 : pandas DataFrame
        The time reporting dataframe
    metric_type : str
        Type of metric: 'net', 'payments', 'revenue', 'billable', 'non billable', 'total', 'utilization'
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format
    show_broker : bool, default False
        Whether to show broker breakdown
        
    Returns:
    --------
    dict: Contains metric cards data
    """
    # map metric types to row names
    metric_mapping = {
        'net': 'Net Amount',
        'payments': 'Payments', 
        'revenue': 'Revenue',
        'billable': 'Billable Hours',
        'non billable': 'Non-Billable Hours',
        'total': 'Total Hours',
        'utilization': 'Utilization Percentage'
    }
    
    row_name = metric_mapping.get(metric_type)
    if not row_name:
        return None
    
    # get the main metric cards
    avg_card = create_average_kpi_card(df, df2, df3, row_name, start_date, end_date)
    sum_card = create_sum_kpi_card(df, df2, df3, row_name, start_date, end_date)
    
    result = {
        'main_avg': avg_card,
        'main_sum': sum_card,
        'broker_breakdown': None
    }
    
    # add broker breakdown if requested and metric is 'net'
    if show_broker and metric_type == 'net':
        broker_cards = {}
        broker_types = ['Broker', 'Direct', 'Partner']
        
        # for net amount, we need to use the broker-specific row names
        for broker in broker_types:
            broker_net_row = f'{broker} Net Amount'
            broker_pct_row = f'{broker} % of Net'
            
            avg_net = create_average_kpi_card(df, df2, df3, broker_net_row, start_date, end_date)
            sum_net = create_sum_kpi_card(df, df2, df3, broker_net_row, start_date, end_date)
            avg_pct = create_average_kpi_card(df, df2, df3, broker_pct_row, start_date, end_date)
            
            broker_cards[broker.lower()] = {
                'avg_amount': avg_net,
                'sum_amount': sum_net,
                'avg_percentage': avg_pct
            }
        
        result['broker_breakdown'] = broker_cards
    
    return result

# plot percentage increase/decrease in net amount month over month, with a dynamic target line that shows required increase/decrease to hit 20m end of year
def plot_net_amount_mom_growth(invoices_df, payments_df, time_df,
                                start_date=None, end_date=None,
                                annual_target=20_000_000,
                                use_remaining_months=True,
                                show_required_line=True,
                                show_first_month=True,
                                first_month_as_zero=True):
    """
    Net Amount Month-over-Month Growth vs (optional) Required Growth line.

    New params:
      show_first_month: keep the first selected month visible on the chart
      first_month_as_zero: if True, set first month's MoM growth to 0%
                           (otherwise keep it as None which hides the point)

    """
    pivot = key_metrics_monthly(invoices_df, payments_df, time_df,
                                start_date=start_date, end_date=end_date)
    if pivot.empty or 'Net Amount' not in pivot.index:
        return None

    net_row = pivot.loc['Net Amount']

    def to_number(v):
        try:
            return float(str(v).replace(',', '').replace('%', '').strip())
        except Exception:
            return 0.0

    net_numeric = net_row.map(to_number)

    df = net_numeric.reset_index()
    if len(df.columns) != 2:
        return None
    df.columns = ['MonthLabel', 'NetAmount']

    # Parse month formats
    df['Month_dt'] = pd.to_datetime(df['MonthLabel'], format='%b %Y', errors='coerce')
    if df['Month_dt'].isna().all():
        df['Month_dt'] = pd.to_datetime(df['MonthLabel'] + '-01', format='%Y-%m-%d', errors='coerce')

    df = df.dropna(subset=['Month_dt']).sort_values('Month_dt')
    if df.empty:
        return None

    # Actual MoM growth
    df['MoM_Growth_Pct'] = df['NetAmount'].pct_change() * 100

    # Handle first month visibility
    first_idx = df.index.min()
    if show_first_month:
        if first_month_as_zero:
            df.loc[first_idx, 'MoM_Growth_Pct'] = 0.0
        else:
            # keep None so point would be hidden
            df.loc[first_idx, 'MoM_Growth_Pct'] = None
    else:
        # If explicitly not showing, remove first row (rare use case)
        df = df.loc[df.index != first_idx]

    # required growth line (optional)
    if show_required_line:
        df['Year'] = df['Month_dt'].dt.year
        target_year = df['Year'].max()
        year_mask = df['Year'] == target_year
        df.loc[year_mask, 'CumulativeNet'] = (
            df.loc[year_mask].sort_values('Month_dt')['NetAmount'].cumsum()
        )

        req_vals = []
        for _, row in df.iterrows():
            if row.get('Year') != target_year:
                req_vals.append(None)
                continue
            cumulative_net = row.get('CumulativeNet')
            current_net = row['NetAmount']
            if cumulative_net is None or current_net <= 0:
                req_vals.append(None)
                continue
            remaining_net = annual_target - cumulative_net
            if remaining_net <= 0:
                req_vals.append(0.0)
                continue

            if use_remaining_months:
                month_number = row['Month_dt'].month
                remaining_months = max(12 - month_number, 1)
                required_avg_remaining = remaining_net / remaining_months
            else:
                required_avg_remaining = remaining_net / 12.0

            req_growth = (required_avg_remaining / current_net) - 1
            req_vals.append(req_growth * 100)
        df['Required_MoM_Growth_Pct'] = req_vals
    
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df['Month_dt'],
            y=df['MoM_Growth_Pct'],
            mode='lines+markers',
            name='Actual MoM Growth %',
            line=dict(color='#1f77b4', width=2),
            hovertemplate=(
                'Month: %{x|%b %Y}<br>'
                'Actual: %{y:.1f}%<br>'
                'Net: %{customdata[0]:,.0f} SEK<extra></extra>'
            ),
            customdata=np.stack([df['NetAmount']], axis=-1)
        )
    )

    if show_required_line and 'Required_MoM_Growth_Pct' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Month_dt'],
                y=df['Required_MoM_Growth_Pct'],
                mode='lines+markers',
                name='Required MoM Growth % to hit target of 20M SEK',
                line=dict(color='orange', width=2, dash='dash'),
                hovertemplate=(
                    'Month: %{x|%b %Y}<br>'
                    'Required: %{y:.1f}%<br>'
                    'Cumulative Net: %{customdata[0]:,.0f} SEK<extra></extra>'
                ),
                customdata=np.stack([df.get('CumulativeNet', df['NetAmount'].cumsum())], axis=-1)
            )
        )

    fig.add_hline(y=0, line_color='lightgray', line_dash='dash')

    # dynamic y-range
    series = [df['MoM_Growth_Pct'].dropna()]
    if show_required_line and 'Required_MoM_Growth_Pct' in df.columns:
        series.append(df['Required_MoM_Growth_Pct'].dropna())
    combined = pd.concat(series) if series else pd.Series(dtype=float)
    if not combined.empty:
        span = combined.abs().quantile(0.95)
        if span > 0:
            fig.update_yaxes(range=[-1.2 * span, 1.2 * span])

    fig.update_layout(
        xaxis=dict(title='Month', tickformat='%b %Y'),
        yaxis=dict(title='MoM Growth %'),
        legend=dict(orientation='h', y=1.1, x=0),
        height=300,
        margin=dict(l=50, r=20, t=60, b=40)
    )

    return fig