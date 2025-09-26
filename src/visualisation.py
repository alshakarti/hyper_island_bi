import pandas as pd
import os
from pathlib import Path
import io
import plotly.graph_objects as go
import plotly.express as px
import numpy as np


# first version only with weighted amount filter 
def sales_funnel_viz(df, weighted_amount=True):
    """
    Sales funnel visualization using Plotly
    
    Arguments:
    df (DataFrame): Sales pipeline dataframe
    weighted_amount (bool): If True, use weighted_deal_amount, otherwise use deal_amount
    
    Returns:
    None: Displays the plotly figure
    """
    # select the appropriate amount column based on the parameter
    amount_col = 'weighted_deal_amount' if weighted_amount else 'deal_amount'
    amount_title = 'Weighted Deal Amount' if weighted_amount else 'Deal Amount'
    
    # group data by pipeline stage
    funnel_data = df.groupby(['pipeline_stage_order', 'pipeline_stage']).agg(
        amount=(amount_col, 'sum'),
        deal_count=('deal_id', 'count')
    ).reset_index().sort_values('pipeline_stage_order')
    
    # create stage labels
    funnel_data['stage_label'] = funnel_data.apply(
        lambda x: f"Stage {int(x['pipeline_stage_order'])}: {x['pipeline_stage']}", axis=1
    )
    
    # format amount for hover text
    funnel_data['amount_fmt'] = funnel_data['amount'].apply(lambda x: f"{x:,.2f} SEK")
    
    # create hover text
    funnel_data['hover_text'] = funnel_data.apply(
        lambda x: f"<b>{x['stage_label']}</b><br>" +
                 f"{amount_title}: {x['amount_fmt']}<br>" +
                 f"Number of Deals: {x['deal_count']}", 
        axis=1
    )
    
    # create funnel chart
    fig = go.Figure()
    fig.add_trace(go.Funnel(
        name='Sales Funnel',
        y=funnel_data['stage_label'],
        x=funnel_data['amount'],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.7,
        marker={
            "color": ["#1f77b4", "#7fc3e5", "#4292c6", "#2171b5", "#08519c"][:len(funnel_data)],
            "line": {"width": [1] * len(funnel_data), "color": ["white"] * len(funnel_data)}
        },
        hovertext=funnel_data['hover_text'],
        hoverinfo='text'
    ))
    
    # add title and labels
    fig.update_layout(
        title={
            'text': f'Sales Pipeline Funnel ({amount_title})',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        height=500,
        width=800
    )
    
    return fig
    
# second version with weighted amount and time range filter 
def sales_funnel_viz2(df, weighted_amount=True, time_filter_start=None, time_filter_stop=None):
    """
    Sales funnel visualization using Plotly with optional time filtering.
    
    Arguments:
    df (DataFrame): Sales pipeline dataframe
    weighted_amount (bool): If True, use weighted_deal_amount, otherwise use deal_amount
    time_filter_start (str): Start date for filtering in format 'YYYY-MM-DD' 
    time_filter_stop (str): End date for filtering in format 'YYYY-MM-DD' 
    
    Returns:
    None: Displays the plotly figure
    """
    # create a copy of the dataframe to avoid modifying the original
    filtered_df = df.copy()
    
    # apply time filters if provided
    if time_filter_start or time_filter_stop:
        # get a sample date to check timezone info
        date_col = filtered_df['create_date']
        
        # alternative approach for filtering with timezone-aware dates
        if time_filter_start:
            # convert to datetime and localize to match dataframe timezone
            start_date = pd.to_datetime(time_filter_start)
            start_date_str = start_date.strftime('%Y-%m-%d')
            # Create a mask by converting to string format first for comparison
            filtered_df = filtered_df[filtered_df['create_date'].dt.strftime('%Y-%m-%d') >= start_date_str]
        
        if time_filter_stop:
            # convert to datetime and localize to match dataframe timezone
            end_date = pd.to_datetime(time_filter_stop)
            end_date_str = end_date.strftime('%Y-%m-%d')
            # Create a mask by converting to string format first for comparison
            filtered_df = filtered_df[filtered_df['create_date'].dt.strftime('%Y-%m-%d') <= end_date_str]
        
        # create date range text for title
        if time_filter_start and time_filter_stop:
            date_range_text = f" (Created {time_filter_start} to {time_filter_stop})"
        elif time_filter_start:
            date_range_text = f" (Created from {time_filter_start})"
        elif time_filter_stop:
            date_range_text = f" (Created until {time_filter_stop})"
    else:
        date_range_text = ""
    
    # check if we have data after filtering
    if filtered_df.empty:
        print("No data available after applying time filters.")
        return
    
    # select the appropriate amount column based on the parameter
    amount_col = 'weighted_deal_amount' if weighted_amount else 'deal_amount'
    amount_title = 'Weighted Deal Amount' if weighted_amount else 'Deal Amount'
    
    # group data by pipeline stage
    funnel_data = filtered_df.groupby(['pipeline_stage_order', 'pipeline_stage']).agg(
        amount=(amount_col, 'sum'),
        deal_count=('deal_id', 'count')
    ).reset_index().sort_values('pipeline_stage_order')
    
    # create stage labels
    funnel_data['stage_label'] = funnel_data.apply(
        lambda x: f"Stage {int(x['pipeline_stage_order'])}: {x['pipeline_stage']}", axis=1
    )
    
    # format amount for hover text
    funnel_data['amount_fmt'] = funnel_data['amount'].apply(lambda x: f"{x:,.2f} SEK")
    
    # create hover text
    funnel_data['hover_text'] = funnel_data.apply(
        lambda x: f"<b>{x['stage_label']}</b><br>" +
                 f"{amount_title}: {x['amount_fmt']}<br>" +
                 f"Number of Deals: {x['deal_count']}", 
        axis=1
    )
    
    # create funnel chart
    fig = go.Figure()
    fig.add_trace(go.Funnel(
        name='Sales Funnel',
        y=funnel_data['stage_label'],
        x=funnel_data['amount'],
        textposition="inside",
        textinfo="value+percent initial",
        opacity=0.7,
        marker={
            "color": ["#1f77b4", "#7fc3e5", "#4292c6", "#2171b5", "#08519c"][:len(funnel_data)],
            "line": {"width": [1] * len(funnel_data), "color": ["white"] * len(funnel_data)}
        },
        hovertext=funnel_data['hover_text'],
        hoverinfo='text'
    ))
    
    # add title and labels
    fig.update_layout(
        title={
            'text': f'Sales Pipeline Funnel ({amount_title}){date_range_text}',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
        },
        height=500,
        width=800
    )    
    return fig

def get_monthly_invoice_pivot(df, df2, start_date=None, end_date=None):
    """
    Create a pivoted dataframe with monthly invoice statistics by broker type.
    
    Parameters:
    -----------
    df : pandas DataFrame
        The invoice dataframe
    start_date : str, optional
        Start date for filtering in 'YYYY-MM-DD' format
    end_date : str, optional
        End date for filtering in 'YYYY-MM-DD' format
        
    Returns:
    --------
    pandas DataFrame: Pivoted dataframe with months as columns and invoice metrics as rows
    """
    # Create a copy to avoid modifying the original
    plot_df = df.copy()
    
    # Drop rows where final_pay_date is NaT
    plot_df = plot_df.dropna(subset=['final_pay_date'])
    
    # Convert dates to datetime if they're not already
    plot_df['final_pay_date'] = pd.to_datetime(plot_df['final_pay_date'])
    
    # Apply date filters if provided
    if start_date:
        start_date = pd.to_datetime(start_date)
        plot_df = plot_df[plot_df['final_pay_date'] >= start_date]
        
    if end_date:
        end_date = pd.to_datetime(end_date)
        plot_df = plot_df[plot_df['final_pay_date'] <= end_date]
    
    # Check if we have data after filtering
    if plot_df.empty:
        print("No data available after applying date filters.")
        return pd.DataFrame()
    
    # Create month column for grouping
    plot_df['month'] = plot_df['final_pay_date'].dt.to_period('M')
    plot_df['month_str'] = plot_df['month'].dt.strftime('%Y-%m')
    
    # Create a broker type column with standardized categories
    plot_df['broker_type'] = plot_df['broker'].apply(
        lambda x: 'Broker' if 'broker' in str(x).lower() else 
                 ('Partner' if 'partner' in str(x).lower() else 'Direct')
    )
    
    # Group by month and calculate totals
    monthly_totals = plot_df.groupby('month_str').agg(
        total_net=('invoice_amount_net', 'sum'),
        total_total=('invoice_amount_total', 'sum')
    )
    
    # Group by month and broker type
    broker_data = plot_df.groupby(['month_str', 'broker_type']).agg(
        net_amount=('invoice_amount_net', 'sum'),
        total_amount=('invoice_amount_total', 'sum')
    ).reset_index()
    
    # Calculate percentage of total net for each broker type
    broker_data = broker_data.merge(monthly_totals, on='month_str')
    broker_data['pct_of_net'] = (broker_data['net_amount'] / broker_data['total_net']) * 100
    
    # Create a list to store the rows for our pivot table
    pivot_rows = []
    
    # Add total rows (independent of broker type)
    pivot_rows.append(('Total Net Amount', plot_df.groupby('month_str')['invoice_amount_net'].sum()))
    pivot_rows.append(('Total Invoice Amount', plot_df.groupby('month_str')['invoice_amount_total'].sum()))
    
    # Add rows for each broker type
    for broker_type in ['Broker', 'Direct', 'Partner']:
        # Filter for the current broker type
        broker_subset = broker_data[broker_data['broker_type'] == broker_type]
        
        if not broker_subset.empty:
            # Set the month_str as index for easy pivoting
            broker_subset = broker_subset.set_index('month_str')
            
            # Add rows for this broker type
            pivot_rows.append((f'{broker_type} Net Amount', broker_subset['net_amount']))
            pivot_rows.append((f'{broker_type} Total Amount', broker_subset['total_amount']))
            pivot_rows.append((f'{broker_type} % of Total Net', broker_subset['pct_of_net']))
    
    # add Payments row from df2 
    payments_df = df2.dropna(subset=['final_pay_date']).copy()
    payments_df['month_str'] = payments_df['final_pay_date'].dt.strftime('%Y-%m')
    payments_by_month = payments_df.groupby('month_str')['invoice_payment'].sum()
    pivot_rows.append(('Payments', payments_by_month))

    # - create Revenue row
    total_net_series = plot_df.groupby('month_str')['invoice_amount_net'].sum()
    payments_series = payments_by_month
    # Align indexes and fill missing months with 0
    revenue_series = total_net_series.subtract(payments_series, fill_value=0)
    pivot_rows.append(('Revenue', revenue_series))

    # Create the pivot table
    result = pd.DataFrame({row_name: data for row_name, data in pivot_rows})

    # Clean up and format the DataFrame
    result = result.fillna(0)

    # Format all values to integers (no decimals)
    result = result.applymap(lambda x: int(round(x)) if not isinstance(x, str) else x)

    # Format percentage columns as strings with % and no decimals
    percentage_rows = ['Broker % of Total Net', 'Direct % of Total Net', 'Partner % of Total Net']
    for col in percentage_rows:
        if col in result.columns:
            result[col] = result[col].apply(lambda x: f"{int(round(x))}%")
    
        # Format other columns with thousand separator
    for col in result.columns:
        if col not in percentage_rows:
            result[col] = result[col].apply(lambda x: f"{x:,}" if isinstance(x, int) else x)

    # Ensure columns are sorted chronologically
    result = result.reindex(sorted(result.columns), axis=1)

    # AFTER formatting, transpose so months are columns
    result = result.T

    # Format column names from YYYY-MM to MMM YYYY
    formatted_columns = {}
    for col in result.columns:
        try:
            date_obj = pd.to_datetime(col.split()[0] + '-01')
            formatted_col = date_obj.strftime('%b %Y')
            formatted_columns[col] = formatted_col
        except:
            formatted_columns[col] = col

    result = result.rename(columns=formatted_columns)

    return result

def plot_invoice_amounts(df, df2=None, start_date=None, end_date=None, amount_type='net', hue=False):
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

    if start_date:
        start_date = pd.to_datetime(start_date)
        plot_df = plot_df[plot_df['final_pay_date'] >= start_date]
    if end_date:
        end_date = pd.to_datetime(end_date)
        plot_df = plot_df[plot_df['final_pay_date'] <= end_date]
    if plot_df.empty:
        print("No data available after applying date filters.")
        return

    plot_df['month'] = plot_df['final_pay_date'].dt.to_period('M')
    plot_df['month_str'] = plot_df['month'].dt.strftime('%Y-%m')

    # Prepare payments and revenue if needed
    payments_by_month = None
    revenue_by_month = None
    if amount_type.lower() in ['payments', 'revenue']:
        if df2 is None:
            raise ValueError("df2 (payments dataframe) must be provided for payments or revenue plots.")
        payments_df = df2.dropna(subset=['final_pay_date']).copy()
        payments_df['month_str'] = payments_df['final_pay_date'].dt.strftime('%Y-%m')
        payments_by_month = payments_df.groupby('month_str')['invoice_payment'].sum()
        total_net_series = plot_df.groupby('month_str')['invoice_amount_net'].sum()
        revenue_by_month = total_net_series.subtract(payments_by_month, fill_value=0)

    # Select the amount column and title
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

    # Date range text for title
    if start_date and end_date:
        date_range_text = f" ({start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')})"
    elif start_date:
        date_range_text = f" (from {start_date.strftime('%Y-%m-%d')})"
    elif end_date:
        date_range_text = f" (until {end_date.strftime('%Y-%m-%d')})"
    else:
        date_range_text = ""

    # Plot
    if hue and amount_type.lower() in ['net', 'total']:
        monthly_data = plot_df.groupby(['month_str', 'broker'])[amount_col].sum().reset_index()
        fig = px.bar(
            monthly_data,
            x='month_str',
            y=amount_col,
            color='broker',
            title=f'Monthly {amount_title} by Broker{date_range_text}',
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
            title=f'Monthly {amount_title}{date_range_text}',
            labels={
                'month_str': 'Month',
                y_col: f'{amount_title} (SEK)'
            },
            text_auto='.2s'
        )
        fig.update_traces(
            hovertemplate='Month: %{x}<br>' + f'{amount_title}: %{{y:,.2f}} SEK'
        )

    fig.update_layout(
        xaxis_title='Month',
        yaxis_title=f'{amount_title} (SEK)',
        height=500,
        width=900,
        xaxis={'categoryorder': 'category ascending'}
    )
    return fig

def highlight_revenue_trend(row):
    
    target_rows = [
    'Total Net Amount',
    'Payments',
    'Revenue'
    ]
    if row.name not in target_rows:
        return [''] * len(row)
    # convert values to int (remove commas and % if present)
    values = []
    for v in row:
        try:
            values.append(int(str(v).replace(',', '').replace('%', '')))
        except Exception:
            values.append(0)
    # compare each month to previous (first column has no previous month)
    styles = ['']
    for i in range(1, len(values)):
        if values[i] > values[i-1]:
            styles.append('color: green; font-weight: bold;')
        elif values[i] < values[i-1]:
            styles.append('color: red; font-weight: bold;')
        else:
            styles.append('')
    return styles